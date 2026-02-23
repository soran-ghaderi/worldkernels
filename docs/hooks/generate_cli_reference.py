r"""Auto-generate CLI reference docs from worldkernels.cli source files.

Scans ``worldkernels/cli/*.py`` using AST to extract commands, options,
subcommands, and help text. Generates markdown pages in ``docs/cli/``
and injects nav entries under "CLI Reference".

Visual options are read from ``extra.cli_reference`` in ``mkdocs.yml``,
mirroring mkdocstrings naming conventions (``show_source``,
``show_symbol_type_heading``, etc.).
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).resolve().parents[2]
CLI_PACKAGE_DIR = ROOT_DIR / "worldkernels" / "cli"
CLI_DOC_DIR = ROOT_DIR / "docs" / "cli"

HELP_ALIASES = {"help"}
_MDASH = "\u2014"
_TOC_SYMBOLS: dict[str, str] = {}

_OPT_DEFAULTS: dict[str, Any] = {
    "heading_level": 1,
    "show_source": True,
    "show_usage": True,
    "show_examples": True,
    "show_global_options": True,
    "show_subcommands": True,
    "show_symbol_type_heading": True,
    "show_symbol_type_toc": True,
    "show_options_table": True,
    "show_description": True,
    "code_block_style": "fenced",
    "members_order": "alphabetical",
}


@dataclass
class CLIOptions:
    r"""Visual options for CLI reference generation, read from ``extra.cli_reference``."""

    heading_level: int = 1
    show_source: bool = True
    show_usage: bool = True
    show_examples: bool = True
    show_global_options: bool = True
    show_subcommands: bool = True
    show_symbol_type_heading: bool = True
    show_symbol_type_toc: bool = True
    show_options_table: bool = True
    show_description: bool = True
    code_block_style: str = "fenced"
    members_order: str = "alphabetical"

    @classmethod
    def from_config(cls, config: dict) -> "CLIOptions":
        extra = config.get("extra", {}).get("cli_reference", {})
        merged = {**_OPT_DEFAULTS, **extra}
        valid = {k: v for k, v in merged.items() if k in _OPT_DEFAULTS}
        return cls(**valid)

    def heading(self, level_offset: int = 0) -> str:
        return "#" * (self.heading_level + level_offset)

    def heading_symbol(self, kind: str) -> str:
        r"""Return ``<code class="doc-symbol ...">`` HTML badge for headings."""
        if self.show_symbol_type_heading:
            return (
                f'<code class="doc-symbol doc-symbol-heading'
                f' doc-symbol-{kind}"></code> '
            )
        return ""


def _slugify(text: str) -> str:
    r"""Simple slugify matching Python-Markdown's toc default."""
    slug = text.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')


def _write_if_changed(path: Path, content: str) -> bool:
    r"""Write file only if content differs from existing."""
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == content:
                return False
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CLIOption:
    flags: list[str]
    metavar: str | None = None
    description: str = ""
    default: str | None = None


@dataclass
class CLICommand:
    name: str
    description: str = ""
    usage: str = ""
    options: list[CLIOption] = field(default_factory=list)
    subcommands: list["CLICommand"] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    source_file: Path | None = None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


class HelpTextParser:
    r"""Parse structured CLI help text to extract commands and options."""

    USAGE_LINE = re.compile(r"^\s{4}worldkernels\s+(?P<rest>.+)$")
    OPTION_RE = re.compile(r"\[--(?P<name>[\w-]+)(?:\s+(?P<meta>[A-Z_]+))?\]")

    def parse(self, help_text: str) -> tuple[str, list[CLICommand], list[str]]:
        r"""Return ``(description, commands, examples)`` from help text."""
        lines = help_text.strip().splitlines()
        description = ""
        commands: list[CLICommand] = []
        examples: list[str] = []
        section: str | None = None

        for line in lines:
            stripped = line.strip()

            if stripped.lower().startswith("usage"):
                section = "usage"
                continue
            elif stripped.lower().startswith("example"):
                section = "examples"
                continue
            elif not line.startswith(" ") and stripped and section is None:
                description = re.sub(r"^worldkernels\s*[-\u2013\u2014]\s*", "", stripped)
                continue

            if section == "usage":
                m = self.USAGE_LINE.match(line)
                if m:
                    rest = m.group("rest").strip()
                    parts = re.split(r"\s{2,}", rest, maxsplit=1)
                    cmd_opts = parts[0]
                    desc = parts[1].strip() if len(parts) > 1 else ""

                    options = [
                        CLIOption(
                            flags=[f"--{om.group('name')}"],
                            metavar=om.group("meta"),
                        )
                        for om in self.OPTION_RE.finditer(cmd_opts)
                    ]

                    cmd_token = cmd_opts.split()[0] if cmd_opts.split() else ""
                    if cmd_token.startswith("-"):
                        continue

                    commands.append(
                        CLICommand(
                            name=cmd_token,
                            description=desc,
                            usage=f"worldkernels {cmd_opts}",
                            options=options,
                        )
                    )
                elif not stripped:
                    section = None

            elif section == "examples":
                if stripped.startswith("worldkernels"):
                    examples.append(stripped)
                elif not stripped:
                    section = None

        return description, commands, examples


class CLIASTExtractor(ast.NodeVisitor):
    r"""Extract CLI command names and help strings from Python source via AST."""

    def __init__(self):
        self.help_texts: list[str] = []
        self.command_names: list[str] = []
        self.module_docstring: str = ""

    def extract(self, source_path: Path) -> None:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        self.module_docstring = ast.get_docstring(tree) or ""
        self.visit(tree)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "print" and node.args:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if "usage" in arg.value.lower():
                        self.help_texts.append(arg.value)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if self._is_args_subscript(node.left):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.In)):
                    self._collect_names(comparator)
        self.generic_visit(node)

    def _is_args_subscript(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == 0
        )

    def _collect_names(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if not node.value.startswith("-") and node.value not in HELP_ALIASES:
                self.command_names.append(node.value)
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                self._collect_names(elt)


class DocstringSubcommandParser:
    r"""Parse subcommand definitions from module docstrings.

    Matches lines like ``- worldkernels bench latency : description``.
    """

    PATTERN = re.compile(
        r"^\s*-\s*worldkernels\s+(?P<parent>\w+)\s+(?P<name>\w+)\s*:\s*(?P<desc>.+)$",
        re.MULTILINE,
    )

    def parse(self, docstring: str) -> list[CLICommand]:
        return [
            CLICommand(
                name=m.group("name"),
                description=m.group("desc").strip(),
                usage=f"worldkernels {m.group('parent')} {m.group('name')}",
            )
            for m in self.PATTERN.finditer(docstring)
        ]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class CLIReferenceGenerator:
    r"""Discover and document the worldkernels CLI from source files."""

    def __init__(self, opts: CLIOptions):
        self.opts = opts
        self.help_parser = HelpTextParser()
        self.sub_parser = DocstringSubcommandParser()
        self.root_description = ""
        self.commands: dict[str, CLICommand] = {}

    def discover(self) -> None:
        if not CLI_PACKAGE_DIR.exists():
            logger.warning("CLI package not found: %s", CLI_PACKAGE_DIR)
            return

        for py_file in sorted(CLI_PACKAGE_DIR.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            self._process_file(py_file)

    def _process_file(self, path: Path) -> None:
        extractor = CLIASTExtractor()
        extractor.extract(path)

        for help_text in extractor.help_texts:
            desc, commands, examples = self.help_parser.parse(help_text)
            if desc and not self.root_description:
                self.root_description = desc
            for cmd in commands:
                cmd.source_file = path
                cmd.examples = examples
                self._merge_command(cmd)

        if extractor.module_docstring:
            subcommands = self.sub_parser.parse(extractor.module_docstring)
            if subcommands:
                parent_name = path.stem
                parent = self.commands.get(parent_name)
                if parent is None:
                    first_line = extractor.module_docstring.strip().splitlines()[0]
                    parent = CLICommand(
                        name=parent_name,
                        description=first_line.strip().rstrip("."),
                        usage=f"worldkernels {parent_name}",
                        source_file=path,
                    )
                    self.commands[parent_name] = parent
                parent.subcommands = subcommands

    def _merge_command(self, cmd: CLICommand) -> None:
        if cmd.name in self.commands:
            existing = self.commands[cmd.name]
            if not existing.description:
                existing.description = cmd.description
            if not existing.usage:
                existing.usage = cmd.usage
            existing.options.extend(cmd.options)
            existing.examples.extend(cmd.examples)
        else:
            self.commands[cmd.name] = cmd

    def _sorted_commands(self) -> list[str]:
        if self.opts.members_order == "source":
            return list(self.commands.keys())
        return sorted(self.commands.keys())

    # -- Markdown generation ------------------------------------------------

    def _generate_index(self) -> str:
        o = self.opts
        h = o.heading
        lines = [
            "---",
            "icon: material/console",
            "---",
            "",
            f"{h()} CLI Reference",
            "",
        ]

        if o.show_description:
            lines.extend([
                self.root_description or "Command-line interface for WorldKernels.",
                "",
            ])

        if o.show_usage:
            lines.extend([
                f"{h(1)} Usage",
                "",
                "```bash",
                "worldkernels <command> [options]",
                "```",
                "",
            ])

        lines.extend([
            f"{h(1)} Commands",
            "",
            "| Command | Description |",
            "|---------|-------------|",
        ])
        for name in self._sorted_commands():
            cmd = self.commands[name]
            desc = cmd.description or _MDASH
            lines.append(f"| [`{name}`]({name}.md) | {desc} |")
        lines.append("")

        if o.show_global_options:
            lines.extend([
                f"{h(1)} Global Options",
                "",
                "| Flag | Description |",
                "|------|-------------|",
                "| `--help`, `-h` | Show help message and exit |",
                "| `--version`, `-V` | Show version and exit |",
                "",
            ])

        return "\n".join(lines)

    def _generate_command_page(self, cmd: CLICommand) -> str:
        o = self.opts
        h = o.heading
        heading_text = f"worldkernels {cmd.name}"
        symbol = o.heading_symbol("command")
        lines = [f"{h()} {symbol}{heading_text}", ""]

        if o.show_symbol_type_toc:
            _TOC_SYMBOLS[_slugify(heading_text)] = "command"

        if o.show_description and cmd.description:
            lines.extend([cmd.description, ""])

        if o.show_usage:
            lines.extend([
                f"{h(1)} Usage",
                "",
                "```bash",
                cmd.usage or f"worldkernels {cmd.name}",
                "```",
                "",
            ])

        if o.show_options_table and cmd.options:
            lines.extend([
                f"{h(1)} Options",
                "",
                "| Flag | Metavar | Description |",
                "|------|---------|-------------|",
            ])
            for opt in cmd.options:
                flags = ", ".join(f"`{f}`" for f in opt.flags)
                meta = f"`{opt.metavar}`" if opt.metavar else _MDASH
                desc = opt.description or _MDASH
                lines.append(f"| {flags} | {meta} | {desc} |")
            lines.append("")

        if o.show_subcommands and cmd.subcommands:
            lines.extend([
                f"{h(1)} Subcommands",
                "",
                "| Subcommand | Description |",
                "|------------|-------------|",
            ])
            for sub in cmd.subcommands:
                lines.append(f"| `{sub.name}` | {sub.description} |")
            lines.append("")

            for sub in cmd.subcommands:
                sub_text = f"worldkernels {cmd.name} {sub.name}"
                sub_symbol = o.heading_symbol("subcommand")
                lines.extend([
                    f"{h(2)} {sub_symbol}{sub_text}",
                    "",
                ])
                if o.show_symbol_type_toc:
                    _TOC_SYMBOLS[_slugify(sub_text)] = "subcommand"
                if o.show_description:
                    lines.extend([sub.description, ""])
                if o.show_usage:
                    lines.extend([
                        "```bash",
                        sub.usage,
                        "```",
                        "",
                    ])

        if o.show_examples and cmd.examples:
            lines.extend([f"{h(1)} Examples", "", "```bash"])
            lines.extend(cmd.examples)
            lines.extend(["```", ""])

        if o.show_source and cmd.source_file:
            rel = cmd.source_file.relative_to(ROOT_DIR).as_posix()
            gh = f"https://github.com/soran-ghaderi/worldkernels/blob/main/{rel}"
            lines.extend([f"{h(1)} Source", "", f"Defined in [`{rel}`]({gh}).", ""])

        return "\n".join(lines)

    # -- File I/O -----------------------------------------------------------

    def write(self) -> list[Path]:
        CLI_DOC_DIR.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        index_path = CLI_DOC_DIR / "index.md"
        _write_if_changed(index_path, self._generate_index())
        written.append(index_path)

        for name in self._sorted_commands():
            cmd = self.commands[name]
            page_path = CLI_DOC_DIR / f"{name}.md"
            _write_if_changed(page_path, self._generate_command_page(cmd))
            written.append(page_path)

        for md in CLI_DOC_DIR.glob("*.md"):
            if md not in set(written):
                logger.info("Removing stale CLI doc: %s", md.relative_to(ROOT_DIR))
                md.unlink()

        return written

    def build_nav(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [{"Overview": "cli/index.md"}]
        for name in self._sorted_commands():
            items.append({name.capitalize(): f"cli/{name}.md"})
        return items


# ---------------------------------------------------------------------------
# MkDocs hook
# ---------------------------------------------------------------------------


def on_post_page(output: str, page, config) -> str | None:
    r"""Inject doc-symbol badges into TOC entries for CLI pages."""
    if not _TOC_SYMBOLS:
        return None
    modified = output
    for slug, kind in _TOC_SYMBOLS.items():
        badge = (
            f'<code class="doc-symbol doc-symbol-toc'
            f' doc-symbol-{kind}"></code>\u00a0'
        )
        pattern = re.compile(
            rf'(href="#{re.escape(slug)}"[^>]*?class="md-nav__link"[^>]*?>\s*'
            rf'<span[^>]*?>\s*)',
            re.DOTALL,
        )
        modified = pattern.sub(rf'\g<1>{badge}', modified)
    return modified if modified != output else None


def on_config(config: dict) -> dict:
    _TOC_SYMBOLS.clear()
    opts = CLIOptions.from_config(config)
    generator = CLIReferenceGenerator(opts)
    generator.discover()
    written = generator.write()
    logger.info("CLI reference: %d pages generated", len(written))

    nav = config.get("nav") or []
    new_nav: list[Any] = []
    replaced = False

    for item in nav:
        if isinstance(item, dict) and "CLI Reference" in item:
            new_nav.append({"CLI Reference": generator.build_nav()})
            replaced = True
        else:
            new_nav.append(item)

    if not replaced:
        new_nav.append({"CLI Reference": generator.build_nav()})

    config["nav"] = new_nav
    return config
