r"""MkDocs hook to inject navigation tab icons into page metadata.

Maps top-level section paths to Material for MkDocs icons so that
``navigation.tabs`` renders icons alongside tab titles.
"""

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

SECTION_ICONS: dict[str, str] = {
    "index.md": "octicons/home-fill-16",
    "examples/": "material/school",
    "cli/": "material/console",
    "benchmarks/": "material/chart-line",
    "blog/": "material/post",
    "api/": "material/file-document",
}


def on_page_markdown(markdown: str, *, page: Page, config: MkDocsConfig, files: Files) -> str:
    if "icon" in page.meta:
        return markdown

    src = page.file.src_path
    for prefix, icon in SECTION_ICONS.items():
        if src == prefix or src == f"{prefix}index.md":
            page.meta["icon"] = icon
            break
        if (
            prefix.endswith("/")
            and src.startswith(prefix)
            and src.count("/") <= 2
            and src.endswith("index.md")
        ):
            page.meta["icon"] = icon
            break

    return markdown
