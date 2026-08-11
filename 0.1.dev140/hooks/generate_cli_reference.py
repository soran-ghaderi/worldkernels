r"""Auto-generate CLI reference docs by introspecting the argparse parser (in-memory)."""

import argparse
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
_GLOBAL_DESTS = {"help", "quiet", "verbose", "debug", "output", "version", "dispatch"}

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
    description: str = ""
    fields: list[CLIField] = field(default_factory=list)
    source_file: Path | None = None


def _action_type_str(action: argparse.Action) -> str:
    if isinstance(action, argparse.BooleanOptionalAction | argparse._StoreTrueAction):
        return "bool"
    if isinstance(action, argparse._CountAction):
        return "int"
    if callable(action.type):
        return getattr(action.type, "__name__", "str")
    return "str"


def _field_from_action(action: argparse.Action) -> CLIField:
    if action.option_strings:
        longs = [o for o in action.option_strings if o.startswith("--")]
        shorts = [o for o in action.option_strings if not o.startswith("--")]
        name = (longs[0] if longs else action.option_strings[0]).lstrip("-")
        aliases = shorts + longs[1:]
        if isinstance(action, argparse.BooleanOptionalAction):
            aliases = [o for o in aliases if o not in (f"--{name}", f"--no-{name}")]
    else:
        name = action.dest
        aliases = []

    description = action.help or ""
    if action.choices:
        choices = ", ".join(str(c) for c in action.choices)
        description = f"{description} Choices: {choices}.".strip()

    default = None
    if action.default not in (None, argparse.SUPPRESS):
        default = repr(action.default)

    return CLIField(
        name=name,
        type_str=_action_type_str(action),
        default=default,
        description=description,
        aliases=aliases,
    )


def _command_from_parser(name: str, parser: argparse.ArgumentParser) -> CLICommand:
    fields = [
        _field_from_action(a)
        for a in parser._actions
        if a.dest not in _GLOBAL_DESTS and not isinstance(a, argparse._SubParsersAction)
    ]
    return CLICommand(
        name=name,
        description=(parser.description or "").strip(),
        fields=fields,
        source_file=CLI_MAIN,
    )


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def discover_commands() -> tuple[list[CLICommand], list[CLIField]]:
    r"""Build the real parser and walk its subcommand tree.

    Returns:
        ``(commands, global_fields)`` where nested subcommands are named
        ``parent:child`` (e.g. ``bench:latency``).
    """
    from worldkernels.cli.main import build_parser

    parser = build_parser()
    sub = _subparsers_action(parser)
    commands: list[CLICommand] = []
    if sub is not None:
        for name, sp in sub.choices.items():
            nested = _subparsers_action(sp)
            if nested is not None:
                for child_name, child in nested.choices.items():
                    commands.append(_command_from_parser(f"{name}:{child_name}", child))
            else:
                commands.append(_command_from_parser(name, sp))

    global_fields = [
        _field_from_action(a)
        for a in parser._actions
        if a.option_strings and a.dest in _GLOBAL_DESTS and a.dest not in ("dispatch", "help")
    ]
    commands.sort(key=lambda c: c.name)
    return commands, global_fields


class CLIReferenceGenerator:
    r"""Discover and document the worldkernels CLI from the argparse tree."""

    def __init__(self, opts: CLIOptions) -> None:
        self.opts = opts
        self.commands: list[CLICommand] = []
        self.global_fields: list[CLIField] = []
        self.grouped: dict[str, list[CLICommand]] = {}

    def discover(self) -> None:
        try:
            self.commands, self.global_fields = discover_commands()
        except Exception as exc:
            logger.warning("CLI reference: could not build parser: %s", exc)
            return
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
                ]
            )
            for fld in self.global_fields:
                flags = ", ".join([f"`--{fld.name}`", *(f"`{a}`" for a in fld.aliases)])
                lines.append(f"| {flags} | {fld.description or _MDASH} |")
            lines.append("| `--help`, `-h` | Show help message and exit |")
            lines.append(
                "| `--help=<group\\|keyword\\|all>` | Search help: one config group, "
                "a keyword across options, or everything |"
            )
            lines.append("")

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
                lines.extend(["```bash", self._usage_line(cmd), "```", ""])

            if o.show_options_table and cmd.fields:
                self._append_options_table(lines, cmd, h, level=2)

        if cmds and cmds[0].source_file and o.show_source:
            rel = cmds[0].source_file.relative_to(ROOT_DIR).as_posix()
            gh = f"https://github.com/soran-ghaderi/worldkernels/blob/main/{rel}"
            lines.extend([f"{h(1)} Source", "", f"Defined in [`{rel}`]({gh}).", ""])

        return "\n".join(lines)

    def _usage_line(self, cmd: CLICommand) -> str:
        usage = f"worldkernels {cmd.name.replace(':', ' ')}"
        opt_parts = []
        for f in cmd.fields:
            flag = f"--{f.name.replace('_', '-')}"
            opt_parts.append(f"[{flag} {f.type_str.upper()}]")
        if opt_parts:
            usage += " " + " ".join(opt_parts)
        return usage

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
            lines.extend([f"{h(1)} Usage", "", "```bash", self._usage_line(cmd), "```", ""])

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
