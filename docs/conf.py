"""Sphinx configuration for the mixle-mlops documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_CORE = ROOT.parent / "mixle"
for path in (ROOT, WORKSPACE_CORE):
    if path.exists():
        sys.path.insert(0, str(path))

import fastapi  # noqa: F401,E402

project = "mixle-mlops"
author = "Grant Boquet"
copyright = "2026, Grant Boquet"

pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
release = pyproject["project"]["version"]
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "frontend/**/node_modules"]

autosummary_generate = False
autodoc_default_options = {
    "members": True,
    "no-index": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
add_module_names = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "boto3",
    "botocore",
    "fakeredis",
    "jsonschema",
    "mcp",
    "openai",
    "pandas",
    "psycopg",
    "pyarrow",
    "pypdf",
    "redis",
    "reportlab",
    "s3fs",
    "transformers",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

html_theme = "furo"
html_title = "mixle-mlops"
html_static_path = []
todo_include_todos = False
