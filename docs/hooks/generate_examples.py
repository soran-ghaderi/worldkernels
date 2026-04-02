import logging
import re
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Any

from mkdocs.structure.files import File

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT_DIR / "examples"
DOCS_DIR = ROOT_DIR / "docs"
SUPPORTED_PATTERNS = (
    "*.py",
    "*.md",
    "*.sh",
    "*.json",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.txt",
)
LANGUAGE_ALIASES = {
    "yml": "yaml",
    "md": "markdown",
    "py": "python",
    "sh": "bash",
}

_virtual_pages: dict[str, str] = {}


def _title(text: str) -> str:
    text = text.replace("_", " ").replace("/", " - ").title()
    subs = {
        "api": "API",
        "cli": "CLI",
        "cpu": "CPU",
        "gpu": "GPU",
        "diffusion": "Diffusion",
        "lora": "LoRA",
        "vae": "VAE",
        "dit": "DiT",
        "vram": "VRAM",
        "json": "JSON",
        "yaml": "YAML",
        "toml": "TOML",
        r"fp\d+": lambda x: x.group(0).upper(),
        r"int\d+": lambda x: x.group(0).upper(),
    }
    for pattern, repl in subs.items():
        text = re.sub(rf"\b{pattern}\b", repl, text, flags=re.IGNORECASE)
    return text


def _fix_relative_links(content: str, src_file: Path) -> str:
    link_pattern = r"\[([^\]]*)\]\((?!(?:https?|ftp)://|#)([^)]+)\)"

    def replace_link(match: re.Match) -> str:
        link_text = match.group(1)
        relative_path = match.group(2)
        resolved = (src_file.parent / relative_path).resolve()

        if not resolved.exists():
            return match.group(0)

        try:
            rel = resolved.relative_to(ROOT_DIR)
        except ValueError:
            return match.group(0)

        slug = "tree/main" if resolved.is_dir() else "blob/main"
        return f"[{link_text}](https://github.com/soran-ghaderi/worldkernels/{slug}/{rel.as_posix()})"

    return re.sub(link_pattern, replace_link, content)


@dataclass
class Example:
    path: Path
    category: str

    @cached_property
    def main_file(self) -> Path | None:
        if self.path.is_file():
            return self.path
        for name in ("README.md", "readme.md"):
            readme = self.path / name
            if readme.exists():
                return readme
        md_files = sorted(self.path.glob("*.md"))
        if md_files:
            return md_files[0]
        for pattern in SUPPORTED_PATTERNS:
            files = sorted(f for f in self.path.glob(pattern) if f.is_file())
            if files:
                return files[0]
        return None

    @cached_property
    def other_files(self) -> list[Path]:
        if self.path.is_file():
            return []
        return sorted(
            f
            for f in self.path.rglob("*")
            if f.is_file()
            and f != self.main_file
            and any(f.match(p) for p in SUPPORTED_PATTERNS)
        )

    @cached_property
    def is_code(self) -> bool:
        return self.main_file is not None and self.main_file.suffix.lower() != ".md"

    @cached_property
    def title(self) -> str:
        if self.main_file is None or self.is_code:
            return _title(self.path.stem)
        with open(self.main_file, encoding="utf-8") as f:
            first_line = f.readline().strip()
        match = re.match(r"^#\s+(?P<title>.+)$", first_line)
        return match.group("title") if match else _title(self.path.stem)

    def _render_code_block(self, file: Path) -> str:
        ext = file.suffix[1:].lower()
        lang = LANGUAGE_ALIASES.get(ext, ext)
        rel_path = file.relative_to(ROOT_DIR).as_posix()
        return f'``````{lang}\n--8<-- "{rel_path}"\n``````\n'

    def _source_url(self) -> str:
        prefix = "tree/main" if self.path.is_dir() else "blob/main"
        rel = self.path.relative_to(ROOT_DIR).as_posix()
        return f"https://github.com/soran-ghaderi/worldkernels/{prefix}/{rel}"

    def generate(self) -> str:
        parts: list[str] = [f"# {self.title}", "", f"Source <{self._source_url()}>.", ""]

        if self.main_file is not None:
            if self.is_code:
                parts.append(self._render_code_block(self.main_file))
            else:
                with open(self.main_file, encoding="utf-8") as f:
                    lines = f.readlines()
                if lines and lines[0].lstrip().startswith("#"):
                    lines = lines[1:]
                parts.append(_fix_relative_links("".join(lines), self.main_file))
                parts.append("")
        elif self.path.is_dir() and self.other_files:
            for file in self.other_files:
                file_title = _title(str(file.relative_to(self.path).with_suffix("")))
                parts.extend([f"## {file_title}", "", self._render_code_block(file), ""])
            return "\n".join(parts)

        if self.other_files:
            parts.extend(["## Example materials", ""])
            for file in self.other_files:
                rel = file.relative_to(self.path)
                parts.append(f'??? abstract "{rel}"')
                if file.suffix.lower() != ".md":
                    code_lines = self._render_code_block(file).rstrip().splitlines()
                    parts.extend(f"    {line}" for line in code_lines)
                else:
                    rel_path = file.relative_to(ROOT_DIR).as_posix()
                    parts.append(f'    --8<-- "{rel_path}"')
                parts.append("")

        return "\n".join(parts)


def _discover_examples() -> list[Example]:
    if not EXAMPLES_DIR.exists():
        return []

    examples: list[Example] = []
    categories = sorted(p for p in EXAMPLES_DIR.iterdir() if p.is_dir())

    for category in categories:
        logger.info("Processing example category: %s", category.stem)
        file_globs = [category.glob(pattern) for pattern in SUPPORTED_PATTERNS]
        for path in chain(*file_globs):
            if path.is_file() and path.name.lower() != "readme.md":
                examples.append(Example(path=path, category=category.stem))

        for nested_dir in sorted(p for p in category.iterdir() if p.is_dir()):
            has_files = any(
                f
                for pattern in SUPPORTED_PATTERNS
                for f in nested_dir.rglob(pattern)
                if f.is_file()
            )
            if has_files:
                examples.append(Example(path=nested_dir, category=category.stem))

    root_file_globs = [EXAMPLES_DIR.glob(pattern) for pattern in SUPPORTED_PATTERNS]
    for path in chain(*root_file_globs):
        if path.is_file() and path.name.lower() != "readme.md":
            examples.append(Example(path=path, category="general"))

    seen: set[tuple[str, str]] = set()
    unique: list[Example] = []
    for ex in sorted(examples, key=lambda e: (e.category, e.path.as_posix())):
        key = (ex.category, ex.path.as_posix())
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def _doc_src_path(example: Example) -> str:
    r"""Compute the virtual src_path under docs/ for an example."""
    if example.category == "general":
        rel = example.path.relative_to(EXAMPLES_DIR)
    else:
        rel = example.path.relative_to(EXAMPLES_DIR / example.category)
    prefix = "" if example.category == "general" else f"{example.category}/"
    return f"examples/{prefix}{rel.with_suffix('.md').as_posix()}"


def _generate_category_page(category: str, entries: list[tuple[Example, str]]) -> str:
    lines = [f"# {_title(category)} Examples", ""]

    for name in ("README.md", "readme.md"):
        readme = EXAMPLES_DIR / category / name
        if readme.exists():
            with open(readme, encoding="utf-8") as f:
                readme_lines = f.readlines()
            if readme_lines and readme_lines[0].lstrip().startswith("#"):
                readme_lines = readme_lines[1:]
            content = _fix_relative_links("".join(readme_lines).strip(), readme)
            if content:
                lines.extend([content, ""])
            break

    rel = (EXAMPLES_DIR / category).relative_to(ROOT_DIR).as_posix()
    lines.extend([f"Generated from `{rel}`.", ""])

    for example, src_path in sorted(entries, key=lambda item: item[0].title.lower()):
        link = Path(src_path).name
        lines.append(f"- [{example.title}]({link})")

    return "\n".join(lines) + "\n"


def _generate_links_page(grouped: dict[str, list[tuple[Example, str]]]) -> str:
    lines: list[str] = []

    if "general" in grouped:
        for example, src_path in grouped["general"]:
            link = Path(src_path).relative_to(Path("examples")).as_posix()
            lines.append(f"- [{example.title}]({link})")
        if len(grouped) > 1 and lines:
            lines.append("")

    for category in sorted(k for k in grouped if k != "general"):
        entries = grouped[category]
        if not entries:
            continue
        lines.extend([f"## [{_title(category)}]({category}/index.md)", ""])
        for example, src_path in entries[:5]:
            link = Path(src_path).relative_to(Path("examples")).as_posix()
            lines.append(f"- [{example.title}]({link})")
        remaining = len(entries) - 5
        if remaining > 0:
            lines.append(f"- ... and {remaining} more in {_title(category)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n" if lines else "- No examples found.\n"


def _build_examples_nav() -> list[dict[str, Any]]:
    inner: list[dict[str, Any]] = []

    if "examples/index.md" in _virtual_pages or (DOCS_DIR / "examples/index.md").exists():
        inner.append({"Overview": "examples/index.md"})

    categorized: dict[str, list[str]] = {}
    for src_path in sorted(_virtual_pages):
        if src_path in ("examples/index.md", "examples/_generated_links.md"):
            continue
        parts = Path(src_path).parts
        if len(parts) == 2:
            categorized.setdefault("_general", []).append(src_path)
        elif len(parts) >= 3 and parts[2] != "index.md":
            categorized.setdefault(parts[1], []).append(src_path)

    for src_path in categorized.get("_general", []):
        inner.append({_title(Path(src_path).stem): src_path})

    for category in sorted(k for k in categorized if k != "_general"):
        section: list[dict[str, Any]] = []
        index_path = f"examples/{category}/index.md"
        if index_path in _virtual_pages:
            section.append({"Overview": index_path})
        for src_path in categorized[category]:
            section.append({_title(Path(src_path).stem): src_path})
        if section:
            inner.append({_title(category): section})

    return inner


def on_config(config):
    r"""Discover examples and populate virtual pages. No files written to disk."""
    _virtual_pages.clear()

    examples = _discover_examples()
    logger.info("Discovered %d examples for virtual generation", len(examples))

    grouped: dict[str, list[tuple[Example, str]]] = {}
    for example in examples:
        src_path = _doc_src_path(example)
        _virtual_pages[src_path] = example.generate()
        grouped.setdefault(example.category, []).append((example, src_path))

    for category, entries in grouped.items():
        if category == "general":
            continue
        _virtual_pages[f"examples/{category}/index.md"] = _generate_category_page(
            category, entries
        )

    _virtual_pages["examples/_generated_links.md"] = _generate_links_page(grouped)

    nav = config.get("nav") or []
    new_nav: list[Any] = []
    replaced = False
    for item in nav:
        if isinstance(item, dict) and "User Guide" in item:
            new_nav.append({"User Guide": _build_examples_nav()})
            replaced = True
        else:
            new_nav.append(item)
    if not replaced:
        new_nav.append({"User Guide": _build_examples_nav()})
    config["nav"] = new_nav
    logger.debug("User Guide nav injected with %d virtual pages", len(_virtual_pages))
    return config


def on_files(files, config):
    r"""Add virtual File entries for generated example pages."""
    docs_dir = config["docs_dir"]
    site_dir = config["site_dir"]
    use_directory_urls = config.get("use_directory_urls", True)

    existing = {f.src_path for f in files}
    for src_path in _virtual_pages:
        if src_path not in existing:
            files.append(File(src_path, docs_dir, site_dir, use_directory_urls))
            logger.debug("Virtual example page: %s", src_path)

    return files


def on_page_read_source(page, config):
    r"""Serve generated content for virtual example pages instead of reading from disk."""
    if page.file.src_path in _virtual_pages:
        return _virtual_pages[page.file.src_path]
    return None
