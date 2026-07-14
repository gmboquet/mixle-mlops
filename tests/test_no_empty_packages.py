"""Guard against importable-but-empty package directories under mixle_mlops/.

A directory that holds only a stale `__pycache__` (no `__init__.py`, no `.py` source)
is still importable as a PEP 420 implicit namespace package. That's a footgun: `import
mixle_mlops.whatever` silently "succeeds" against an empty shell instead of raising
ModuleNotFoundError, hiding the fact that nothing real lives there. This test scans the
package tree directly (filesystem, not the import system) so it also catches directories
that a stale editable install's meta-path finder might otherwise paper over.
"""

import os

import mixle_mlops


def _package_root() -> str:
    return os.path.dirname(os.path.abspath(mixle_mlops.__file__))


def _non_pycache_entries(dirpath: str) -> list[str]:
    return [name for name in os.listdir(dirpath) if name != "__pycache__"]


def test_no_pycache_only_directories_under_mixle_mlops():
    root = _package_root()
    offenders = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if name == "__pycache__":
            continue
        entries = _non_pycache_entries(path)
        if not entries:
            offenders.append(name)
    assert not offenders, (
        f"mixle_mlops/{{{', '.join(offenders)}}} hold only __pycache__ (or nothing) — "
        "delete the directory, or give it real content (__init__.py / a module), before "
        "merging: an empty dir is still an importable namespace package."
    )


def test_known_retired_mlops_packages_stay_gone():
    # These eight were removed as empty, pycache-only leftovers (no tracked source, no
    # live references). If one of them reappears, it should show up as a real package
    # with real content — not silently resurrect as an empty namespace package.
    retired = [
        "verification",
        "substrate",
        "promotion",
        "jobs",
        "contexts",
        "capabilities",
        "artifacts",
        "applications",
    ]
    root = _package_root()
    for name in retired:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        assert _non_pycache_entries(path), (
            f"mixle_mlops/{name}/ exists but is still empty (pycache-only); either remove "
            "it or land the real module content for it in the same change."
        )
