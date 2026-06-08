r"""List entry_points plugins discovered for worldkernels."""

from __future__ import annotations


def run_list() -> None:
    from importlib.metadata import entry_points

    from worldkernels import ui

    eps = entry_points()
    groups = ("worldkernels.worlds", "worldkernels.pipelines")

    for group in groups:
        if hasattr(eps, "select"):
            items = list(eps.select(group=group))
        else:
            items = list(eps.get(group, []))
        ui.rule(group)
        if not items:
            ui.info("(none)")
            continue
        t = ui.table("name", "value")
        for ep in items:
            t.add_row(ep.name, ep.value)
        ui.print_table(t)
