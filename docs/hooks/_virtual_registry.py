r"""Shared virtual page registry for in-memory MkDocs hooks.

Generation hooks call ``register()`` during ``on_config`` to store pages.
Listed after all generators in mkdocs.yml, this hook serves them via
``on_files`` and ``on_page_read_source``.

Self-registers in ``sys.modules["_virtual_registry"]`` so sibling hooks
can ``import _virtual_registry as registry`` regardless of how MkDocs
names the module internally.
"""

import logging
import sys

from mkdocs.structure.files import File

sys.modules.setdefault("_virtual_registry", sys.modules[__name__])

logger = logging.getLogger("mkdocs")

_pages: dict[str, str] = {}


def register(src_path: str, content: str) -> None:
    _pages[src_path] = content


def clear_prefix(prefix: str) -> None:
    r"""Remove all pages whose src_path starts with *prefix*."""
    to_remove = [k for k in _pages if k.startswith(prefix)]
    for k in to_remove:
        del _pages[k]


def contains(src_path: str) -> bool:
    return src_path in _pages


def on_files(files, config):
    docs_dir = config["docs_dir"]
    site_dir = config["site_dir"]
    use_directory_urls = config.get("use_directory_urls", True)

    existing = {f.src_path for f in files}
    added = 0
    for src_path in _pages:
        if src_path not in existing:
            files.append(File(src_path, docs_dir, site_dir, use_directory_urls))
            added += 1

    if added:
        logger.debug("Virtual registry: added %d file entries", added)
    return files


def on_page_read_source(page, config):
    if page.file.src_path in _pages:
        return _pages[page.file.src_path]
    return None
