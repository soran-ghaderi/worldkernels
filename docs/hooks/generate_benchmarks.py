r"""Auto-generate benchmark and metrics documentation from source files.

Discovers benchmark scripts in ``benchmarks/`` and extracts Prometheus
metric definitions from ``worldkernels/runtime/metrics.py`` using AST.
Generates markdown pages in ``docs/benchmarks/`` and injects nav entries.

Visual options are read from ``extra.benchmarks`` in ``mkdocs.yml``,
mirroring mkdocstrings naming conventions.
"""

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = ROOT_DIR / "benchmarks"
METRICS_SOURCES = [
    ROOT_DIR / "worldkernels" / "runtime" / "metrics.py",
]
BENCHMARKS_DOC_DIR = ROOT_DIR / "docs" / "benchmarks"

SUPPORTED_EXTENSIONS = {".py", ".sh", ".yaml", ".yml", ".json", ".toml"}
LANGUAGE_ALIASES = {"yml": "yaml", "py": "python", "sh": "bash", "md": "markdown"}

_METRIC_TYPE_ICONS = {
    "counter": ":material-counter:",
    "gauge": ":material-speedometer:",
    "histogram": ":material-chart-bar:",
    "info": ":material-information:",
    "summary": ":material-sigma:",
}

_MDASH = "\u2014"

_BENCH_DEFAULTS: dict[str, Any] = {
    "heading_level": 1,
    "show_source": True,
    "show_docstring": True,
    "show_code": True,
    "show_symbol_type_heading": True,
    "show_symbol_type_toc": True,
    "code_block_style": "fenced",
    "code_annotations": True,
}

_METRICS_DEFAULTS: dict[str, Any] = {
    "show_table": True,
    "show_type_badge": True,
    "show_description": True,
    "group_by_type": True,
    "members_order": "alphabetical",
}


@dataclass
class MetricsOptions:
    r"""Visual options for metrics tables, read from ``extra.benchmarks.metrics``."""

    show_table: bool = True
    show_type_badge: bool = True
    show_description: bool = True
    group_by_type: bool = True
    members_order: str = "alphabetical"


@dataclass
class BenchOptions:
    r"""Visual options for benchmarks generation, read from ``extra.benchmarks``."""

    heading_level: int = 1
    show_source: bool = True
    show_docstring: bool = True
    show_code: bool = True
    show_symbol_type_heading: bool = True
    show_symbol_type_toc: bool = True
    code_block_style: str = "fenced"
    code_annotations: bool = True
    metrics: MetricsOptions = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = MetricsOptions()

    @classmethod
    def from_config(cls, config: dict) -> "BenchOptions":
        extra = config.get("extra", {}).get("benchmarks", {})
        metrics_raw = extra.pop("metrics", {}) if isinstance(extra, dict) else {}
        merged = {**_BENCH_DEFAULTS, **{k: v for k, v in extra.items() if k in _BENCH_DEFAULTS}}
        m_merged = {**_METRICS_DEFAULTS, **{k: v for k, v in metrics_raw.items() if k in _METRICS_DEFAULTS}}
        return cls(**merged, metrics=MetricsOptions(**m_merged))

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


def _title(text: str) -> str:
    text = text.replace("_", " ").replace("-", " ").title()
    subs = {
        "gpu": "GPU",
        "cpu": "CPU",
        "vram": "VRAM",
        "fps": "FPS",
        "api": "API",
        "cli": "CLI",
        "dit": "DiT",
        "vae": "VAE",
        "wk": "WK",
    }
    for pattern, repl in subs.items():
        text = re.sub(rf"\b{pattern}\b", repl, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Metrics extraction (adapted from vLLM's generate_metrics.py)
# ---------------------------------------------------------------------------


@dataclass
class Metric:
    name: str
    type: str
    documentation: str = ""


class MetricExtractor(ast.NodeVisitor):
    r"""Extract Prometheus metric definitions from source via AST and docstrings.

    Handles three patterns:
    1. Direct constructor: ``Histogram("name", "doc")``
    2. Factory method (vLLM-style): ``self._histogram_cls(name=..., documentation=...)``
    3. Docstring listing: ``- wk_metric_name (histogram)``
    """

    CONSTRUCTORS = {
        "Histogram": "histogram",
        "Counter": "counter",
        "Gauge": "gauge",
        "Info": "info",
        "Summary": "summary",
    }
    FACTORY_METHODS = {
        "_histogram_cls": "histogram",
        "_counter_cls": "counter",
        "_gauge_cls": "gauge",
    }
    DOCSTRING_RE = re.compile(
        r"^\s*-\s+(?P<name>\w+)\s+\((?P<type>\w+)\)",
        re.MULTILINE,
    )

    def __init__(self):
        self.metrics: list[Metric] = []

    def extract_from_file(self, filepath: Path) -> list[Metric]:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))

        self.visit(tree)

        if not self.metrics:
            module_doc = ast.get_docstring(tree) or ""
            for m in self.DOCSTRING_RE.finditer(module_doc):
                self.metrics.append(
                    Metric(name=m.group("name"), type=m.group("type"))
                )

        return self.metrics

    def visit_Call(self, node: ast.Call) -> None:
        metric_type = None

        if isinstance(node.func, ast.Name):
            metric_type = self.CONSTRUCTORS.get(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            metric_type = self.FACTORY_METHODS.get(node.func.attr)

        if metric_type:
            name = self._extract_str(node, "name", pos=0)
            doc = self._extract_str(node, "documentation", pos=1)
            if name:
                self.metrics.append(
                    Metric(name=name, type=metric_type, documentation=doc or "")
                )

        self.generic_visit(node)

    def _extract_str(self, node: ast.Call, kwarg: str, pos: int) -> str | None:
        if len(node.args) > pos and isinstance(node.args[pos], ast.Constant):
            return str(node.args[pos].value)
        for kw in node.keywords:
            if kw.arg == kwarg and isinstance(kw.value, ast.Constant):
                return str(kw.value.value)
        return None


def _generate_metrics_table(metrics: list[Metric], opts: MetricsOptions) -> str:
    if not metrics:
        return "*No metrics defined yet.*\n"

    if opts.members_order == "source":
        sorted_metrics = metrics
    elif opts.group_by_type:
        sorted_metrics = sorted(metrics, key=lambda m: (m.type, m.name))
    else:
        sorted_metrics = sorted(metrics, key=lambda m: m.name)

    header_cols = ["Metric Name"]
    if opts.show_type_badge:
        header_cols.append("Type")
    if opts.show_description:
        header_cols.append("Description")

    sep_cols = ["---"] * len(header_cols)
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(sep_cols) + " |",
    ]

    for m in sorted_metrics:
        row = [f"`{m.name}`"]
        if opts.show_type_badge:
            icon = _METRIC_TYPE_ICONS.get(m.type, "")
            row.append(f"{m.type.capitalize()} {icon}".strip())
        if opts.show_description:
            doc = m.documentation.replace("\n", " ").strip() or _MDASH
            row.append(doc)
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Benchmark script discovery
# ---------------------------------------------------------------------------


@dataclass
class Benchmark:
    path: Path
    opts: BenchOptions
    docstring: str = ""
    title: str = ""

    def __post_init__(self):
        if not self.title:
            self.title = _title(self.path.stem)
        if self.path.suffix == ".py" and not self.docstring:
            self.docstring = self._extract_docstring()

    def _extract_docstring(self) -> str:
        try:
            source = self.path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(self.path))
            return ast.get_docstring(tree) or ""
        except Exception:
            return ""

    def render(self) -> str:
        o = self.opts
        h = o.heading
        ext = self.path.suffix[1:].lower()
        lang = LANGUAGE_ALIASES.get(ext, ext)
        rel = self.path.relative_to(ROOT_DIR).as_posix()
        source_url = f"https://github.com/soran-ghaderi/worldkernels/blob/main/{rel}"

        suffix = o.heading_symbol("benchmark") if o.show_symbol_type_heading else ""
        lines = [f"{h()} {suffix}{self.title}", ""]
        if o.show_source:
            lines.extend([f"Source: <{source_url}>", ""])
        if o.show_docstring and self.docstring:
            lines.extend([self.docstring.strip(), ""])
        if o.show_code:
            anno = ' hl_lines=""' if o.code_annotations else ""
            lines.extend([f"``````{lang}{anno}", f'--8<-- "{rel}"', "``````", ""])
        return "\n".join(lines)


def _discover_benchmarks(opts: BenchOptions) -> list[Benchmark]:
    if not BENCHMARKS_DIR.exists():
        return []

    benchmarks = []
    for path in sorted(BENCHMARKS_DIR.rglob("*")):
        if (
            path.is_file()
            and path.suffix in SUPPORTED_EXTENSIONS
            and not path.name.startswith("_")
        ):
            benchmarks.append(Benchmark(path=path, opts=opts))
    return benchmarks


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class BenchmarksGenerator:
    def __init__(self, opts: BenchOptions):
        self.opts = opts
        self.benchmarks: list[Benchmark] = []
        self.metrics: list[Metric] = []

    def discover(self) -> None:
        self.benchmarks = _discover_benchmarks(self.opts)

        for source in METRICS_SOURCES:
            if source.exists():
                extractor = MetricExtractor()
                self.metrics.extend(extractor.extract_from_file(source))

        logger.info(
            "Benchmarks: %d scripts, %d metrics found",
            len(self.benchmarks),
            len(self.metrics),
        )

    def _generate_index(self) -> str:
        o = self.opts
        h = o.heading
        lines = [
            "---",
            "icon: material/chart-line",
            "---",
            "",
            f"{h()} Benchmarks & Metrics",
            "",
            "Auto-generated from `benchmarks/` scripts and"
            " `worldkernels/runtime/metrics.py`.",
            "",
        ]

        if self.metrics and o.metrics.show_table:
            lines.extend([
                f"{h(1)} Prometheus Metrics",
                "",
                "The following metrics are exported on the `/metrics` endpoint.",
                "",
                _generate_metrics_table(self.metrics, o.metrics),
                "",
            ])

        if self.benchmarks:
            lines.extend([
                f"{h(1)} Benchmark Scripts",
                "",
                "| Script | Description |",
                "|--------|-------------|",
            ])
            for b in self.benchmarks:
                desc = b.docstring.split("\n")[0].strip() if b.docstring else _MDASH
                lines.append(f"| [{b.title}]({b.path.stem}.md) | {desc} |")
            lines.append("")

        if not self.benchmarks and not self.metrics:
            lines.extend([
                "*No benchmarks or metrics available yet.*",
                "",
                "Add benchmark scripts to `benchmarks/` or define metrics in",
                "`worldkernels/runtime/metrics.py` and they will appear here"
                " automatically.",
                "",
            ])

        return "\n".join(lines)

    def _generate_metrics_page(self) -> str:
        o = self.opts
        h = o.heading
        lines = [
            f"{h()} Metrics Reference",
            "",
            "Prometheus-style metrics exported by WorldKernels on the"
            " `/metrics` endpoint.",
            "",
            "Auto-generated from source files. Do not edit manually.",
            "",
            _generate_metrics_table(self.metrics, o.metrics),
        ]
        return "\n".join(lines)

    def write(self) -> list[Path]:
        BENCHMARKS_DOC_DIR.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        index_path = BENCHMARKS_DOC_DIR / "index.md"
        _write_if_changed(index_path, self._generate_index())
        written.append(index_path)

        if self.metrics and self.opts.metrics.show_table:
            metrics_path = BENCHMARKS_DOC_DIR / "metrics.md"
            _write_if_changed(metrics_path, self._generate_metrics_page())
            written.append(metrics_path)

        for b in self.benchmarks:
            page_path = BENCHMARKS_DOC_DIR / f"{b.path.stem}.md"
            _write_if_changed(page_path, b.render())
            written.append(page_path)

        keep = set(written)
        for md in BENCHMARKS_DOC_DIR.glob("*.md"):
            if md not in keep:
                logger.info(
                    "Removing stale benchmark doc: %s", md.relative_to(ROOT_DIR)
                )
                md.unlink()

        return written

    def build_nav(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [{"Overview": "benchmarks/index.md"}]
        if self.metrics:
            items.append({"Metrics Reference": "benchmarks/metrics.md"})
        for b in self.benchmarks:
            items.append({b.title: f"benchmarks/{b.path.stem}.md"})
        return items


# ---------------------------------------------------------------------------
# MkDocs hook
# ---------------------------------------------------------------------------


def on_config(config: dict) -> dict:
    opts = BenchOptions.from_config(config)
    generator = BenchmarksGenerator(opts)
    generator.discover()
    written = generator.write()
    logger.info("Benchmarks: %d pages generated", len(written))

    nav = config.get("nav") or []
    new_nav: list[Any] = []
    replaced = False

    for item in nav:
        if isinstance(item, dict) and "Benchmarks" in item:
            new_nav.append({"Benchmarks": generator.build_nav()})
            replaced = True
        else:
            new_nav.append(item)

    if not replaced:
        new_nav.append({"Benchmarks": generator.build_nav()})

    config["nav"] = new_nav
    return config
