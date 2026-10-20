from pathlib import Path


ROOT = Path(__file__).parents[1]
project = "KORA"
author = "Varun Karthic"
copyright = "2026, Varun Karthic"
release = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
version = release

extensions = ["myst_parser", "sphinx_design", "sphinxcontrib.mermaid"]
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = True

html_theme = "furo"
html_title = f"KORA {release}"
html_static_path = []
html_theme_options = {
    "source_repository": "https://github.com/varunkarthic/kora/",
    "source_branch": "main",
    "source_directory": "docs/",
}
