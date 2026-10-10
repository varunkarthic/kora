import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


stamp_module = load("stamp_changelog", ROOT / "tools/release/stamp_changelog.py")
render_module = load("render_release", ROOT / "tools/release/render_release.py")


class StampTests(unittest.TestCase):
    def test_stamps_unreleased_and_comparison_links(self):
        source = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Work.\n\n[Unreleased]: old\n"
        result = stamp_module.stamp(source, "0.1.3", "v0.1.2", "2026-10-10")
        self.assertIn("## [0.1.3] - 2026-10-10", result)
        self.assertIn("compare/v0.1.3...HEAD", result)
        self.assertIn("compare/v0.1.2...v0.1.3", result)


class RenderTests(unittest.TestCase):
    def test_open_issue_makes_beta_without_assignee_column(self):
        issues = [
            {"number": 71, "title": "Docs", "html_url": "https://example/71", "state": "closed"},
            {"number": 72, "title": "Security", "html_url": "https://example/72", "state": "open"},
        ]
        body, stable = render_module.render("o/r", "0.1.0", "changes", issues)
        self.assertFalse(stable)
        self.assertIn("**Beta.**", body)
        self.assertNotIn("Assigned", body)

    def test_all_closed_makes_stable(self):
        issues = [{"number": 71, "title": "Docs", "html_url": "https://example/71", "state": "closed"}]
        _, stable = render_module.render("o/r", "0.1.0", "changes", issues)
        self.assertTrue(stable)


if __name__ == "__main__":
    unittest.main()
