import logging
import re
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("mkdocs")


def _write_if_changed(path: Path, content: str) -> bool:
    r"""Write file only if content differs from existing, avoiding spurious mtime changes."""
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == content:
                return False
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True

ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT_DIR / "examples"
EXAMPLES_DOC_DIR = ROOT_DIR / "docs/examples"
EXAMPLES_LINKS_FILE = EXAMPLES_DOC_DIR / "_generated_links.md"
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


def title(text: str) -> str:
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
        gh_url = f"https://github.com/soran-ghaderi/worldkernels/{slug}/{rel.as_posix()}"
        return f"[{link_text}]({gh_url})"

    return re.sub(link_pattern, replace_link, content)


@dataclass
class Example:
    path: Path
    category: str

    @cached_property
    def main_file(self) -> Path | None:
        if self.path.is_file():
            return self.path
        readme_candidates = [
            self.path / "README.md",
            self.path / "readme.md",
        ]
        for readme in readme_candidates:
            if readme.exists():
                return readme
        md_files = sorted(self.path.glob("*.md"))
        if md_files:
            return md_files[0]
        for pattern in SUPPORTED_PATTERNS:
            files = sorted(file for file in self.path.glob(pattern) if file.is_file())
            if files:
                return files[0]
        return None

    @cached_property
    def other_files(self) -> list[Path]:
        if self.path.is_file():
            return []

        def is_supported(file: Path) -> bool:
            return any(file.match(pattern) for pattern in SUPPORTED_PATTERNS)

        return sorted(
            file
            for file in self.path.rglob("*")
            if file.is_file() and file != self.main_file and is_supported(file)
        )

    @cached_property
    def is_code(self) -> bool:
        return self.main_file is not None and self.main_file.suffix.lower() != ".md"

    @cached_property
    def title(self) -> str:
        if self.main_file is None or self.is_code:
            return title(self.path.stem)
        with open(self.main_file, encoding="utf-8") as f:
            first_line = f.readline().strip()
        match = re.match(r"^#\s+(?P<title>.+)$", first_line)
        if match:
            return match.group("title")
        return title(self.path.stem)

    def _render_code_block(self, file: Path) -> str:
        code_fence = "``````"
        ext = file.suffix[1:].lower()
        lang = LANGUAGE_ALIASES.get(ext, ext)
        rel_path = file.relative_to(ROOT_DIR).as_posix()
        return f"{code_fence}{lang}\n--8<-- \"{rel_path}\"\n{code_fence}\n"

    def _source_url(self) -> str:
        url_prefix = "tree/main" if self.path.is_dir() else "blob/main"
        rel = self.path.relative_to(ROOT_DIR).as_posix()
        return f"https://github.com/soran-ghaderi/worldkernels/{url_prefix}/{rel}"

    def generate(self) -> str:
        content: list[str] = [f"# {self.title}", "", f"Source <{self._source_url()}>.", ""]

        if self.main_file is not None:
            if self.is_code:
                content.append(self._render_code_block(self.main_file))
            else:
                with open(self.main_file, encoding="utf-8") as f:
                    main_content_lines = f.readlines()
                if main_content_lines and main_content_lines[0].lstrip().startswith("#"):
                    main_content_lines = main_content_lines[1:]
                main_content = _fix_relative_links("".join(main_content_lines), self.main_file)
                content.append(main_content)
                content.append("")
        elif self.path.is_dir() and self.other_files:
            for file in self.other_files:
                file_title = title(str(file.relative_to(self.path).with_suffix("")))
                content.append(f"## {file_title}")
                content.append("")
                content.append(self._render_code_block(file))
                content.append("")
            return "\n".join(content)

        if self.other_files:
            content.append("## Example materials")
            content.append("")
            for file in self.other_files:
                rel = file.relative_to(self.path)
                content.append(f'??? abstract "{rel}"')
                if file.suffix.lower() != ".md":
                    code = self._render_code_block(file).rstrip().splitlines()
                    content.extend([f"    {line}" for line in code])
                else:
                    rel_path = file.relative_to(ROOT_DIR).as_posix()
                    content.append(f'    --8<-- "{rel_path}"')
                content.append("")

        return "\n".join(content)


class ExampleGenerator:
    def __init__(self, root_dir: Path, examples_dir: Path, output_dir: Path):
        self.root_dir = root_dir
        self.examples_dir = examples_dir
        self.output_dir = output_dir

    @staticmethod
    def _is_readme(path: Path) -> bool:
        return path.is_file() and path.name.lower() == "readme.md"

    def _discover_category_examples(self, category_path: Path) -> list[Example]:
        examples: list[Example] = []
        file_globs = [category_path.glob(pattern) for pattern in SUPPORTED_PATTERNS]
        for path in chain(*file_globs):
            if path.is_file() and not self._is_readme(path):
                examples.append(Example(path=path, category=category_path.stem))

        nested_dirs = sorted(p for p in category_path.iterdir() if p.is_dir())
        for nested_dir in nested_dirs:
            nested_files = [
                file
                for pattern in SUPPORTED_PATTERNS
                for file in nested_dir.rglob(pattern)
                if file.is_file()
            ]
            if nested_files:
                examples.append(Example(path=nested_dir, category=category_path.stem))
        return examples

    def discover_examples(self) -> list[Example]:
        if not self.examples_dir.exists():
            return []

        examples: list[Example] = []
        categories = sorted(path for path in self.examples_dir.iterdir() if path.is_dir())

        for category in categories:
            logger.info("Processing category: %s", category.stem)
            examples.extend(self._discover_category_examples(category))

        root_file_globs = [self.examples_dir.glob(pattern) for pattern in SUPPORTED_PATTERNS]
        for path in chain(*root_file_globs):
            if path.is_file() and not self._is_readme(path):
                examples.append(Example(path=path, category="general"))

        seen: set[tuple[str, str]] = set()
        unique_examples: list[Example] = []
        for example in sorted(examples, key=lambda e: (e.category, e.path.as_posix())):
            key = (example.category, example.path.as_posix())
            if key in seen:
                continue
            seen.add(key)
            unique_examples.append(example)
        return unique_examples

    def _doc_output_path(self, example: Example) -> Path:
        category_prefix = Path() if example.category == "general" else Path(example.category)
        if example.category == "general":
            rel = example.path.relative_to(self.examples_dir)
        else:
            rel = example.path.relative_to(self.examples_dir / example.category)
        return (self.output_dir / category_prefix / rel).with_suffix(".md")

    def write_examples(self, examples: list[Example]) -> list[tuple[Example, Path]]:
        generated: list[tuple[Example, Path]] = []
        for example in examples:
            doc_path = self._doc_output_path(example)
            if _write_if_changed(doc_path, example.generate()):
                logger.debug("Example generated: %s", doc_path.relative_to(self.root_dir))
            generated.append((example, doc_path))
        return generated

    def clean_stale(self, generated: list[tuple[Example, Path]]) -> None:
        r"""Remove generated docs that no longer correspond to a source example."""
        keep = {doc_path.resolve() for _, doc_path in generated}
        preserved = {"index.md", "_generated_links.md"}
        for md in self.output_dir.rglob("*.md"):
            if md.name in preserved:
                continue
            if md.resolve() not in keep:
                logger.info("Removing stale doc: %s", md.relative_to(self.root_dir))
                md.unlink()

    def write_links_file(self, generated: list[tuple[Example, Path]]) -> None:
        lines: list[str] = []
        grouped: dict[str, list[tuple[Example, Path]]] = {}
        for example, doc_path in generated:
            grouped.setdefault(example.category, []).append((example, doc_path))

        if "general" in grouped:
            for example, doc_path in grouped["general"]:
                link_target = doc_path.relative_to(self.output_dir).as_posix()
                lines.append(f"- [{example.title}]({link_target})")
            if len(grouped) > 1 and lines:
                lines.append("")

        for category in sorted(key for key in grouped if key != "general"):
            category_entries = grouped[category]
            if not category_entries:
                continue
            category_index = f"{category}/index.md"
            lines.append(f"## [{title(category)}]({category_index})")
            lines.append("")
            preview_count = 5
            for example, doc_path in category_entries[:preview_count]:
                link_target = doc_path.relative_to(self.output_dir).as_posix()
                lines.append(f"- [{example.title}]({link_target})")
            remaining = len(category_entries) - preview_count
            if remaining > 0:
                lines.append(f"- ... and {remaining} more in {title(category)}")
            lines.append("")

        if not lines:
            lines = ["- No examples found."]

        _write_if_changed(EXAMPLES_LINKS_FILE, "\n".join(lines).rstrip() + "\n")

    def _read_category_readme(self, category: str) -> str | None:
        r"""Read the category README.md and return content (minus the title line)."""
        for name in ("README.md", "readme.md"):
            readme = self.examples_dir / category / name
            if readme.exists():
                with open(readme, encoding="utf-8") as f:
                    lines = f.readlines()
                if lines and lines[0].lstrip().startswith("#"):
                    lines = lines[1:]
                content = _fix_relative_links("".join(lines).strip(), readme)
                return content if content else None
        return None

    def write_category_pages(self, generated: list[tuple[Example, Path]]) -> None:
        grouped: dict[str, list[tuple[Example, Path]]] = {}
        for example, doc_path in generated:
            if example.category == "general":
                continue
            grouped.setdefault(example.category, []).append((example, doc_path))

        for category, entries in grouped.items():
            if not entries:
                continue
            category_dir = self.output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            category_index = category_dir / "index.md"

            lines = [
                f"# {title(category)} Examples",
                "",
            ]

            readme_content = self._read_category_readme(category)
            if readme_content:
                lines.append(readme_content)
                lines.append("")

            lines.append(f"Generated from `{(self.examples_dir / category).relative_to(self.root_dir).as_posix()}`.",)
            lines.append("")

            for example, doc_path in sorted(
                entries,
                key=lambda item: item[0].title.lower(),
            ):
                link_target = doc_path.relative_to(category_dir).as_posix()
                lines.append(f"- [{example.title}]({link_target})")

            if _write_if_changed(category_index, "\n".join(lines) + "\n"):
                logger.debug("Category index generated: %s", category_index.relative_to(self.root_dir))


def _generate_examples_docs() -> int:
    logger.debug("Root directory: %s", ROOT_DIR.resolve())
    logger.debug("Examples directory: %s", EXAMPLES_DIR.resolve())
    logger.debug("Output directory: %s", EXAMPLES_DOC_DIR.resolve())

    EXAMPLES_DOC_DIR.mkdir(parents=True, exist_ok=True)
    generator = ExampleGenerator(
        root_dir=ROOT_DIR,
        examples_dir=EXAMPLES_DIR,
        output_dir=EXAMPLES_DOC_DIR,
    )
    examples = generator.discover_examples()
    generated = generator.write_examples(examples)
    generator.clean_stale(generated)
    generator.write_category_pages(generated)
    generator.write_links_file(generated)
    return len(generated)


def _build_directory_nav(directory: Path, docs_prefix: str) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []

    index_file = directory / "index.md"
    if index_file.exists():
        children.append({"Overview": f"{docs_prefix}/index.md"})

    pages = sorted(
        page
        for page in directory.glob("*.md")
        if page.name not in {"index.md", "_generated_links.md", "README.md"}
    )
    for page in pages:
        children.append({title(page.stem): f"{docs_prefix}/{page.name}"})

    subdirs = sorted(path for path in directory.iterdir() if path.is_dir())
    for subdir in subdirs:
        sub_prefix = f"{docs_prefix}/{subdir.name}"
        sub_children = _build_directory_nav(subdir, sub_prefix)
        if sub_children:
            children.append({title(subdir.name): sub_children})

    return children


def _build_examples_nav() -> list[dict[str, Any]]:
    inner: list[dict[str, Any]] = []

    index_file = EXAMPLES_DOC_DIR / "index.md"
    if index_file.exists():
        inner.append({"Overview": "examples/index.md"})

    general_pages = sorted(
        p
        for p in EXAMPLES_DOC_DIR.glob("*.md")
        if p.name not in {"index.md", "_generated_links.md", "README.md"}
    )
    for page in general_pages:
        inner.append({title(page.stem): f"examples/{page.name}"})

    for category_dir in sorted(path for path in EXAMPLES_DOC_DIR.iterdir() if path.is_dir()):
        section = _build_directory_nav(category_dir, f"examples/{category_dir.name}")
        if section:
            inner.append({title(category_dir.name): section})

    return inner


def on_config(config):
    generated_count = _generate_examples_docs()
    logger.info("Total examples generated: %d", generated_count)

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
    logger.debug("User Guide nav updated with generated pages")
    return config