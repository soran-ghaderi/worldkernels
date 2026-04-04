r"""Auto-generate CLI reference docs from tyro-based dataclass commands (in-memory)."""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import _virtual_registry as registry

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).resolve().parents[2]
CLI_MAIN = ROOT_DIR / "worldkernels" / "cli" / "main.py"

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
    r"""Visual options for CLI reference generation."""

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
        if self.show_symbol_type_heading:
            return f'<code class="doc-symbol doc-symbol-heading doc-symbol-{kind}"></code> '
        return ""


def _slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug.strip("-")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CLIField:
    name: str
    type_str: str = "str"
    default: str | None = None
    description: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class CLICommand:
    name: str
    class_name: str = ""
    description: str = ""
    fields: list[CLIField] = field(default_factory=list)
    source_file: Path | None = None


# ---------------------------------------------------------------------------
# AST extraction from tyro dataclasses
# ---------------------------------------------------------------------------


def _ast_to_str(node: ast.AST) -> str:
    """Best-effort conversion of an AST node to a readable string."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_ast_to_str(node.value)}.{node.attr}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return f"{_ast_to_str(node.left)} | {_ast_to_str(node.right)}"
    if isinstance(node, ast.Subscript):
        return f"{_ast_to_str(node.value)}[{_ast_to_str(node.slice)}]"
    if isinstance(node, ast.Tuple):
        return ", ".join(_ast_to_str(e) for e in node.elts)
    return ast.dump(node)


def _extract_subcommand_name(annotation_node: ast.AST) -> str | None:
    """Extract the name from ``Annotated[SomeClass, tyro.conf.subcommand("name")]``."""
    if not isinstance(annotation_node, ast.Subscript):
        return None
    if not isinstance(annotation_node.value, ast.Name):
        return None
    if annotation_node.value.id != "Annotated":
        return None
    if not isinstance(annotation_node.slice, ast.Tuple):
        return None
    for elt in annotation_node.slice.elts[1:]:
        if isinstance(elt, ast.Call):
            func_str = _ast_to_str(elt.func)
            if "subcommand" in func_str and elt.args:
                if isinstance(elt.args[0], ast.Constant):
                    return str(elt.args[0].value)
    return None


def _extract_alias(annotation_node: ast.AST) -> list[str]:
    """Extract aliases from ``Annotated[..., tyro.conf.arg(aliases=("-k",))]``."""
    if not isinstance(annotation_node, ast.Subscript):
        return []
    if not isinstance(annotation_node.slice, ast.Tuple):
        return []
    aliases: list[str] = []
    for elt in annotation_node.slice.elts[1:]:
        if isinstance(elt, ast.Call):
            for kw in elt.keywords:
                if kw.arg == "aliases" and isinstance(kw.value, ast.Tuple):
                    for a in kw.value.elts:
                        if isinstance(a, ast.Constant):
                            aliases.append(str(a.value))
    return aliases


def _get_field_type(annotation: ast.AST | None) -> str:
    if annotation is None:
        return "str"
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        if annotation.value.id == "Annotated" and isinstance(annotation.slice, ast.Tuple):
            return _ast_to_str(annotation.slice.elts[0])
    return _ast_to_str(annotation)


class TyroASTExtractor(ast.NodeVisitor):
    r"""Extract command dataclasses and the Command Union from cli/main.py."""

    def __init__(self) -> None:
        self.dataclasses: dict[str, CLICommand] = {}
        self.subcommand_map: dict[str, str] = {}
        self._source: str = ""

    def extract(self, source_path: Path) -> None:
        self._source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(self._source, filename=str(source_path))
        self.visit(tree)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id == "dataclass"
            )
            for d in node.decorator_list
        )
        if not is_dataclass:
            self.generic_visit(node)
            return

        docstring = ast.get_docstring(node) or ""
        fields: list[CLIField] = []

        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if not isinstance(item.target, ast.Name):
                continue

            fname = item.target.id
            if fname.startswith("_"):
                continue

            ftype = _get_field_type(item.annotation)
            aliases = _extract_alias(item.annotation) if item.annotation else []

            default_val = None
            if item.value is not None:
                if isinstance(item.value, ast.Constant):
                    default_val = repr(item.value.value)
                elif isinstance(item.value, ast.Call):
                    default_val = None
                else:
                    default_val = _ast_to_str(item.value)

            fields.append(
                CLIField(
                    name=fname,
                    type_str=ftype,
                    default=default_val,
                    aliases=aliases,
                )
            )

        self.dataclasses[node.name] = CLICommand(
            name=node.name,
            class_name=node.name,
            description=docstring.strip(),
            fields=fields,
            source_file=None,
        )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Find Command = ... Union with subcommand annotations."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "Command":
                self._extract_union_subcommands(node.value)
        self.generic_visit(node)

    def _extract_union_subcommands(self, node: ast.AST) -> None:
        """Walk nested Subscript nodes to find Annotated[Cls, subcommand("name")]."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Tuple):
                for elt in node.slice.elts:
                    name = _extract_subcommand_name(elt)
                    if name is not None and isinstance(elt, ast.Subscript):
                        if isinstance(elt.slice, ast.Tuple) and elt.slice.elts:
                            cls_node = elt.slice.elts[0]
                            if isinstance(cls_node, ast.Name):
                                self.subcommand_map[cls_node.id] = name
                    else:
                        self._extract_union_subcommands(elt)
            else:
                self._extract_union_subcommands(node.slice)


def _visit_assign_name(node: ast.Assign) -> str | None:
    for target in node.targets:
        if isinstance(target, ast.Name):
            return target.id
    return None


def discover_commands(source_path: Path) -> list[CLICommand]:
    """Extract CLI commands from tyro-based main.py."""
    extractor = TyroASTExtractor()
    extractor.extract(source_path)

    subcommand_map = extractor.subcommand_map

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            name = _visit_assign_name(node)
            if name == "Command":
                _walk_for_subcommands(node.value, subcommand_map)

    commands: list[CLICommand] = []
    for cls_name, cmd in extractor.dataclasses.items():
        if cls_name in subcommand_map:
            cmd.name = subcommand_map[cls_name]
            cmd.source_file = source_path
            commands.append(cmd)

    commands.sort(key=lambda c: c.name)
    return commands


def _walk_for_subcommands(node: ast.AST, result: dict[str, str]) -> None:
    """Recursively find Annotated[Cls, tyro.conf.subcommand("name")] in Union."""
    if isinstance(node, ast.Subscript):
        name = _extract_subcommand_name(node)
        if name is not None and isinstance(node.slice, ast.Tuple) and node.slice.elts:
            cls_node = node.slice.elts[0]
            if isinstance(cls_node, ast.Name):
                result[cls_node.id] = name
        if isinstance(node.slice, ast.Tuple):
            for elt in node.slice.elts:
                _walk_for_subcommands(elt, result)
        else:
            _walk_for_subcommands(node.slice, result)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class CLIReferenceGenerator:
    r"""Discover and document the worldkernels CLI from tyro dataclasses."""

    def __init__(self, opts: CLIOptions) -> None:
        self.opts = opts
        self.commands: list[CLICommand] = []
        self.grouped: dict[str, list[CLICommand]] = {}

    def discover(self) -> None:
        if not CLI_MAIN.exists():
            logger.warning("CLI main module not found: %s", CLI_MAIN)
            return
        self.commands = discover_commands(CLI_MAIN)
        for cmd in self.commands:
            prefix = cmd.name.split(":")[0] if ":" in cmd.name else cmd.name
            self.grouped.setdefault(prefix, []).append(cmd)

    def _all_top_level_names(self) -> list[str]:
        names = list(self.grouped.keys())
        if self.opts.members_order == "alphabetical":
            names.sort()
        return names

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
            lines.extend(["Command-line interface for WorldKernels.", ""])

        if o.show_usage:
            lines.extend(
                [
                    f"{h(1)} Usage",
                    "",
                    "```bash",
                    "worldkernels <command> [options]",
                    "```",
                    "",
                ]
            )

        lines.extend(
            [
                f"{h(1)} Commands",
                "",
                "| Command | Description |",
                "|---------|-------------|",
            ]
        )

        for name in self._all_top_level_names():
            cmds = self.grouped[name]
            if len(cmds) == 1 and ":" not in cmds[0].name:
                desc = cmds[0].description.splitlines()[0] if cmds[0].description else _MDASH
                lines.append(f"| [`{name}`]({name}.md) | {desc} |")
            else:
                descs = []
                for c in cmds:
                    sub = c.name.split(":")[-1] if ":" in c.name else c.name
                    first_line = c.description.splitlines()[0] if c.description else ""
                    descs.append(f"`{sub}`: {first_line}")
                combined = "; ".join(descs)
                lines.append(f"| [`{name}`]({name}.md) | {combined} |")

        lines.append("")

        if o.show_global_options:
            lines.extend(
                [
                    f"{h(1)} Global Options",
                    "",
                    "| Flag | Description |",
                    "|------|-------------|",
                    "| `--help`, `-h` | Show help message and exit |",
                    "",
                ]
            )

        return "\n".join(lines)

    def _generate_command_page(self, group_name: str, cmds: list[CLICommand]) -> str:
        o = self.opts
        h = o.heading

        if len(cmds) == 1 and ":" not in cmds[0].name:
            return self._generate_single_command(cmds[0])

        heading_text = f"worldkernels {group_name}"
        symbol = o.heading_symbol("command")
        lines = [f"{h()} {symbol}{heading_text}", ""]

        if o.show_symbol_type_toc:
            _TOC_SYMBOLS[_slugify(heading_text)] = "command"

        if o.show_subcommands:
            lines.extend(
                [
                    f"{h(1)} Subcommands",
                    "",
                    "| Subcommand | Description |",
                    "|------------|-------------|",
                ]
            )
            for cmd in cmds:
                sub = cmd.name.split(":")[-1] if ":" in cmd.name else cmd.name
                first_line = cmd.description.splitlines()[0] if cmd.description else _MDASH
                lines.append(f"| `{sub}` | {first_line} |")
            lines.append("")

        for cmd in cmds:
            sub = cmd.name.split(":")[-1] if ":" in cmd.name else cmd.name
            sub_text = f"worldkernels {group_name} {sub}"
            sub_symbol = o.heading_symbol("subcommand")
            lines.extend([f"{h(1)} {sub_symbol}{sub_text}", ""])

            if o.show_symbol_type_toc:
                _TOC_SYMBOLS[_slugify(sub_text)] = "subcommand"

            if o.show_description and cmd.description:
                lines.extend([cmd.description, ""])

            if o.show_usage:
                usage = f"worldkernels {cmd.name.replace(':', ' ')}"
                opt_parts = []
                for f in cmd.fields:
                    flag = f"--{f.name.replace('_', '-')}"
                    opt_parts.append(f"[{flag} {f.type_str.upper()}]")
                if opt_parts:
                    usage += " " + " ".join(opt_parts)
                lines.extend(["```bash", usage, "```", ""])

            if o.show_options_table and cmd.fields:
                self._append_options_table(lines, cmd, h, level=2)

        if cmds and cmds[0].source_file and o.show_source:
            rel = cmds[0].source_file.relative_to(ROOT_DIR).as_posix()
            gh = f"https://github.com/soran-ghaderi/worldkernels/blob/main/{rel}"
            lines.extend([f"{h(1)} Source", "", f"Defined in [`{rel}`]({gh}).", ""])

        return "\n".join(lines)

    def _generate_single_command(self, cmd: CLICommand) -> str:
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
            usage = f"worldkernels {cmd.name}"
            opt_parts = []
            for f in cmd.fields:
                flag = f"--{f.name.replace('_', '-')}"
                opt_parts.append(f"[{flag} {f.type_str.upper()}]")
            if opt_parts:
                usage += " " + " ".join(opt_parts)
            lines.extend([f"{h(1)} Usage", "", "```bash", usage, "```", ""])

        if o.show_options_table and cmd.fields:
            self._append_options_table(lines, cmd, h, level=1)

        if cmd.source_file and o.show_source:
            rel = cmd.source_file.relative_to(ROOT_DIR).as_posix()
            gh = f"https://github.com/soran-ghaderi/worldkernels/blob/main/{rel}"
            lines.extend([f"{h(1)} Source", "", f"Defined in [`{rel}`]({gh}).", ""])

        return "\n".join(lines)

    def _append_options_table(self, lines: list[str], cmd: CLICommand, h: Any, level: int) -> None:
        o = self.opts
        lines.extend(
            [
                f"{h(level)} Options",
                "",
                "| Flag | Type | Default | Description |",
                "|------|------|---------|-------------|",
            ]
        )
        for fld in cmd.fields:
            flag = f"`--{fld.name.replace('_', '-')}`"
            if fld.aliases:
                alias_str = ", ".join(f"`{a}`" for a in fld.aliases)
                flag = f"{flag}, {alias_str}"
            type_str = f"`{fld.type_str}`"
            default = f"`{fld.default}`" if fld.default is not None else _MDASH
            desc = fld.description or _MDASH

            if o.show_symbol_type_toc:
                opt_slug = _slugify(f"--{fld.name.replace('_', '-')}")
                _TOC_SYMBOLS[opt_slug] = "option"

            lines.append(f"| {flag} | {type_str} | {default} | {desc} |")
        lines.append("")

    def populate_virtual_pages(self) -> int:
        r"""Register generated pages in the shared virtual registry."""
        registry.register("cli/index.md", self._generate_index())
        count = 1

        for group_name in self._all_top_level_names():
            cmds = self.grouped[group_name]
            registry.register(
                f"cli/{group_name}.md",
                self._generate_command_page(group_name, cmds),
            )
            count += 1

        return count

    def build_nav(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [{"Overview": "cli/index.md"}]
        for name in self._all_top_level_names():
            items.append({name.capitalize(): f"cli/{name}.md"})
        return items


# ---------------------------------------------------------------------------
# MkDocs hooks
# ---------------------------------------------------------------------------


def on_post_page(output: str, page, config) -> str | None:
    if not _TOC_SYMBOLS:
        return None
    modified = output
    for slug, kind in _TOC_SYMBOLS.items():
        badge = f'<code class="doc-symbol doc-symbol-toc doc-symbol-{kind}"></code>\u00a0'
        pattern = re.compile(
            rf'(href="#{re.escape(slug)}"[^>]*?class="md-nav__link"[^>]*?>\s*'
            rf"<span[^>]*?>\s*)",
            re.DOTALL,
        )
        modified = pattern.sub(rf"\g<1>{badge}", modified)
    return modified if modified != output else None


def on_config(config: dict) -> dict:
    r"""Discover CLI commands and populate virtual pages. No files written to disk."""
    _TOC_SYMBOLS.clear()
    registry.clear_prefix("cli/")

    opts = CLIOptions.from_config(config)
    generator = CLIReferenceGenerator(opts)
    generator.discover()
    count = generator.populate_virtual_pages()
    logger.info("CLI reference: %d virtual pages generated", count)

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
