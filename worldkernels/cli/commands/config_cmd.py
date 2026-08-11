r"""`worldkernels config-show` — print the resolved runtime config + source of each flag."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worldkernels.config.runtime import RuntimeConfig


def _config_rows(cfg: "RuntimeConfig") -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    from dataclasses import fields

    from worldkernels.config.profiles import NESTED_CONFIGS
    from worldkernels.config.runtime import ALL_TOGGLE_FIELDS

    flat = [(f, getattr(cfg, f)) for f in ("device", "max_sessions", *ALL_TOGGLE_FIELDS)]
    nested = [
        (f"{group}.{f.name}", getattr(getattr(cfg, group), f.name))
        for group in NESTED_CONFIGS
        for f in fields(getattr(cfg, group))
    ]
    return flat, nested


def print_runtime_config(
    cfg: "RuntimeConfig", sources: dict[str, str], title: str = "runtime config"
) -> None:
    r"""Render flat + nested config tables with per-field source attribution."""
    from worldkernels import ui

    flat, nested = _config_rows(cfg)

    def add_rows(t: Any, rows: list[tuple[str, Any]]) -> None:
        for name, value in rows:
            src = sources.get(name, "default")
            t.add_row(name, str(value), "" if src == "default" else f"[green]{src}[/green]")

    ui.rule(title)
    t = ui.table("flag", "value", "source")
    add_rows(t, flat)
    ui.print_table(t)

    ui.rule("nested")
    t = ui.table("field", "value", "source")
    add_rows(t, nested)
    ui.print_table(t)


def run_config_show(
    profile: str | None = None,
    as_json: bool = False,
    config_file: str | None = None,
    overrides: dict | None = None,
) -> None:
    from worldkernels.config.profiles import resolve_runtime_config

    cfg, sources = resolve_runtime_config(
        profile=profile, cli_overrides=overrides, config_file=config_file
    )

    if as_json:
        import json

        flat, nested = _config_rows(cfg)
        data = {
            name: {"value": value, "source": sources.get(name, "default")}
            for name, value in (*flat, *nested)
        }
        print(json.dumps(data, indent=2))
        return

    print_runtime_config(cfg, sources, title=f"runtime config · {profile or 'default'}")
