r"""Owned argparse layer: help search, flag aliasing, dotted JSON args, human ints."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "FlexibleArgumentParser",
    "SortedHelpFormatter",
    "human_readable_int",
    "WK_SUBCMD_EPILOG",
]

log = logging.getLogger(__name__)

WK_SUBCMD_EPILOG = (
    "Use `--help=all` to show every option, `--help=<group>` for one config group "
    "(e.g. `--help=CacheConfig`), or `--help=<keyword>` to search options by name."
)

_DECIMAL_SUFFIXES = {"k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
_BINARY_SUFFIXES = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
_HUMAN_INT_RE = re.compile(r"(\d+(?:\.\d+)?)([kmgtKMGT]?)")


def human_readable_int(value: str) -> int:
    r"""Parse ints with suffixes: lowercase decimal (``25.6k``), uppercase binary (``16K``)."""
    value = value.strip()
    match = _HUMAN_INT_RE.fullmatch(value)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid integer {value!r}")
    number, suffix = match.group(1), match.group(2)
    if suffix in _DECIMAL_SUFFIXES:
        return int(float(number) * _DECIMAL_SUFFIXES[suffix])
    if "." in number:
        raise argparse.ArgumentTypeError(
            f"decimals are only allowed with decimal suffixes (k/m/g/t), got {value!r}"
        )
    if suffix in _BINARY_SUFFIXES:
        return int(number) * _BINARY_SUFFIXES[suffix]
    return int(number)


def check_port(value: str) -> int:
    port = int(value)
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be in [1024, 65535], got {port}")
    return port


class SortedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    r"""Sorts options alphabetically and always shows defaults."""

    def add_arguments(self, actions: Iterable[argparse.Action]) -> None:
        actions = list(actions)
        for action in actions:
            if (
                not action.help
                and action.option_strings
                and action.default not in (None, argparse.SUPPRESS)
            ):
                action.help = "(default: %(default)s)"
        super().add_arguments(sorted(actions, key=lambda a: tuple(a.option_strings) or ("",)))

    def _get_help_string(self, action: argparse.Action) -> str:
        text = action.help or ""
        if (
            "%(default)" not in text
            and action.default is not argparse.SUPPRESS
            and action.default is not None
            and action.option_strings
        ):
            text = f"{text} (default: %(default)s)".strip()
        return text


def _coerce_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return human_readable_int(value)
    except argparse.ArgumentTypeError:
        return value


def _assign_nested(target: dict[str, Any], keys: list[str], value: Any, append: bool) -> None:
    for key in keys[:-1]:
        node = target.setdefault(key, {})
        if not isinstance(node, dict):
            log.warning("overwriting non-dict value at %r with a dict", key)
            node = target[key] = {}
        target = node
    leaf = keys[-1]
    if append:
        existing = target.setdefault(leaf, [])
        if not isinstance(existing, list):
            existing = target[leaf] = [existing]
        existing.append(value)
    else:
        if leaf in target:
            log.warning("duplicate key %r, keeping the last value", leaf)
        target[leaf] = value


def _normalize_option(key: str) -> str:
    head, dot, tail = key.partition(".")
    return head.replace("_", "-") + dot + tail


class FlexibleArgumentParser(argparse.ArgumentParser):
    r"""ArgumentParser with ``--help=<group|keyword|all>``, ``--under_score`` aliasing,
    dotted JSON args (``--x.y v``, ``--x.y+ v``), and one-time deprecation warnings.
    """

    _warned_deprecated: set[str] = set()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", SortedHelpFormatter)
        super().__init__(*args, **kwargs)
        self._search_keyword: str | None = None
        self._deprecated_options: set[str] = set()

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        deprecated = kwargs.pop("deprecated", False)
        action = super().add_argument(*args, **kwargs)
        if deprecated:
            self._deprecated_options.update(action.option_strings)
        return action

    def parse_known_args(  # type: ignore[override]
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        argv = list(sys.argv[1:] if args is None else args)
        argv = self._preprocess(argv)
        self._warn_deprecated(argv)
        return super().parse_known_args(argv, namespace)

    def _subcommand_choices(self) -> set[str]:
        for action in self._actions:
            if type(action).__name__ == "_SubParsersAction":
                return set(action.choices or ())
        return set()

    def _preprocess(self, argv: list[str]) -> list[str]:
        choices = self._subcommand_choices()
        owns_help = True
        out: list[str] = []
        for tok in argv:
            if owns_help and tok in choices:
                owns_help = False
            if tok.startswith("--help=") and owns_help:
                self._search_keyword = tok.split("=", 1)[1].strip().lower()
                tok = "--help"
            elif tok.startswith("--") and len(tok) > 2:
                key, eq, val = tok.partition("=")
                tok = _normalize_option(key) + eq + val
            out.append(tok)
        return self._repack_dotted(out)

    def _repack_dotted(self, argv: list[str]) -> list[str]:
        merged: dict[str, dict[str, Any]] = {}
        out: list[str] = []
        i = 0
        while i < len(argv):
            tok = argv[i]
            key, eq, inline = tok.partition("=")
            if tok.startswith("--") and "." in key:
                if eq:
                    value, i = inline, i + 1
                elif i + 1 < len(argv):
                    value, i = argv[i + 1], i + 2
                else:
                    self.error(f"missing value for {tok}")
                root, _, path = key.partition(".")
                append = path.endswith("+")
                _assign_nested(
                    merged.setdefault(root, {}),
                    path.rstrip("+").split("."),
                    _coerce_value(value),
                    append,
                )
            else:
                out.append(tok)
                i += 1
        for root, obj in merged.items():
            out.extend((root, json.dumps(obj)))
        return out

    def _warn_deprecated(self, argv: list[str]) -> None:
        for tok in argv:
            opt = tok.partition("=")[0]
            if opt in self._deprecated_options and opt not in self._warned_deprecated:
                FlexibleArgumentParser._warned_deprecated.add(opt)
                log.warning("argument '%s' is deprecated", opt.lstrip("-"))

    def format_help(self) -> str:
        keyword = self._search_keyword
        if keyword is None:
            return super().format_help()
        if keyword == "all":
            return self._format_groups(self._action_groups)
        norm = keyword.replace("-", "").replace("_", "")
        for group in self._action_groups:
            title = (group.title or "").replace("-", "").replace("_", "").lower()
            if title == norm and group._group_actions:
                return self._format_groups([group])
        matches = [
            a for a in self._actions if any(keyword in opt.lower() for opt in a.option_strings)
        ]
        if matches:
            return self._format_matches(keyword, matches)
        return (
            f"No group or option matches {keyword!r}. "
            "Use '--help' for a summary or '--help=all' for every option.\n"
        )

    def _format_groups(self, groups: Sequence[Any]) -> str:
        formatter = self._get_formatter()
        actions = [a for g in groups for a in g._group_actions]
        formatter.add_usage(self.usage, actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        for group in groups:
            formatter.start_section(group.title)
            formatter.add_text(group.description)
            formatter.add_arguments(group._group_actions)
            formatter.end_section()
        formatter.add_text(self.epilog)
        return formatter.format_help()

    def _format_matches(self, keyword: str, matches: list[argparse.Action]) -> str:
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, matches, [])
        formatter.start_section(f"options matching {keyword!r}")
        formatter.add_arguments(matches)
        formatter.end_section()
        return formatter.format_help()
