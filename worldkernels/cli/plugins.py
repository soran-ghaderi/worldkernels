r"""List entry_points plugins discovered for worldkernels."""

from __future__ import annotations


def run_list() -> None:
    from importlib.metadata import entry_points

    eps = entry_points()
    groups = ("worldkernels.worlds", "worldkernels.pipelines")

    for group in groups:
        if hasattr(eps, "select"):
            items = list(eps.select(group=group))
        else:
            items = list(eps.get(group, []))
        print(f"[{group}]")
        if not items:
            print("  (none)")
        for ep in items:
            print(f"  {ep.name:24s}  {ep.value}")
