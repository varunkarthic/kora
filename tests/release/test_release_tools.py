import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_empty_milestone_remains_beta_with_explicit_explanation(self):
        body, stable = render_module.render("o/r", "0.1.0", "changes", [])
        self.assertFalse(stable)
        self.assertIn("empty milestone is not eligible for Stable", body)

    def test_malformed_issue_cannot_be_marked_stable(self):
        with self.assertRaisesRegex(RuntimeError, "malformed issue data"):
            render_module.render("o/r", "0.1.0", "changes", [{"number": 1, "state": "closed"}])


class PaginationTests(unittest.TestCase):
    class Response(io.BytesIO):
        def __init__(self, payload, link=""):
            super().__init__(json_bytes(payload))
            self.headers = {"Link": link}

    def test_later_issue_page_keeps_release_beta(self):
        closed = [
            {"number": number, "title": f"Issue {number}", "state": "closed", "html_url": f"https://example/{number}"}
            for number in range(1, 101)
        ]
        later_open = [{"number": 101, "title": "Open gate", "state": "open", "html_url": "https://example/101"}]
        responses = [
            self.Response([{"number": 7, "title": "v0.2"}]),
            self.Response(
                closed,
                '<https://api.github.com/repositories/1407643890/issues?state=all&milestone=7&sort=created&direction=asc&per_page=100&after=cursor&page=2>; rel="next", '
                '<https://api.github.com/repositories/1407643890/issues?state=all&milestone=7&sort=created&direction=asc&per_page=100&after=cursor&page=2>; rel="last"',
            ),
            self.Response(later_open),
        ]
        requested = []

        def open_url(request, timeout):
            requested.append(request.full_url)
            return responses.pop(0)

        with patch.object(render_module.urllib.request, "urlopen", side_effect=open_url):
            issues = render_module.milestone_issues("o/r", "v0.2", "token")
        self.assertEqual(len(requested), 3)
        self.assertIn("page=2", requested[-1])
        self.assertEqual(len(issues), 101)
        _, stable = render_module.render("o/r", "0.2.1", "changes", issues)
        self.assertFalse(stable)

    def test_milestone_on_later_page_is_found(self):
        first_page = [{"number": number, "title": f"other-{number}"} for number in range(1, 101)]
        responses = [
            self.Response(
                first_page,
                '<https://api.github.com/repositories/1407643890/milestones?state=all&per_page=100&page=2>; rel="next", '
                '<https://api.github.com/repositories/1407643890/milestones?state=all&per_page=100&page=2>; rel="last"',
            ),
            self.Response([{"number": 101, "title": "v0.2"}]),
            self.Response([]),
        ]
        requested = []

        def open_url(request, timeout):
            requested.append(request.full_url)
            return responses.pop(0)

        with patch.object(render_module.urllib.request, "urlopen", side_effect=open_url):
            issues = render_module.milestone_issues("o/r", "v0.2", "token")
        self.assertEqual(issues, [])
        self.assertIn("page=2", requested[1])

    def test_full_page_without_next_link_fetches_followup_until_exhausted(self):
        page = [{"number": number, "state": "closed"} for number in range(1, 101)]
        responses = [self.Response(page), self.Response([])]
        requested = []

        def open_url(request, timeout):
            requested.append(request.full_url)
            return responses.pop(0)

        with patch.object(render_module.urllib.request, "urlopen", side_effect=open_url):
            records = render_module.api_get_pages("o/r", "/issues", "token")
        self.assertEqual(len(records), 100)
        self.assertEqual(len(requested), 2)
        self.assertIn("page=2", requested[1])

    def test_full_final_page_with_last_link_is_complete(self):
        page = [{"number": number} for number in range(1, 101)]
        response = self.Response(
            page, '<https://api.github.com/repositories/1/issues?per_page=100&page=1>; rel="last"'
        )
        with patch.object(render_module.urllib.request, "urlopen", return_value=response) as open_url:
            records = render_module.api_get_pages("o/r", "/issues", "token")
        self.assertEqual(len(records), 100)
        open_url.assert_called_once()

    def test_duplicate_page_identity_fails_closed(self):
        responses = [
            self.Response(
                [{"number": 5}],
                '<https://api.github.com/repositories/1/issues?per_page=100&page=2>; rel="next"',
            ),
            self.Response([{"number": 5}]),
        ]
        with patch.object(render_module.urllib.request, "urlopen", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "duplicate item identity"):
                render_module.api_get_pages("o/r", "/issues", "token")

    def test_nonsequential_or_cross_host_next_link_fails_closed(self):
        links = [
            '<https://attacker.invalid/repositories/1/issues?state=all&per_page=100&page=2>; rel="next"',
            '<https://api.github.com/repositories/1/issues?state=all&per_page=100&page=3>; rel="next"',
            '<https://api.github.com/repositories/1/issues?state=closed&per_page=100&page=2>; rel="next"',
        ]
        for link in links:
            with self.subTest(link=link):
                response = self.Response([{"number": 5}], link)
                with patch.object(render_module.urllib.request, "urlopen", return_value=response):
                    with self.assertRaisesRegex(RuntimeError, "invalid or non-sequential"):
                        render_module.api_get_pages("o/r", "/issues?state=all", "token")

    def test_last_page_link_without_next_fetches_missing_intermediate_page(self):
        response = self.Response(
            [{"number": 5}], '<https://api.github.com/repositories/1/issues?per_page=100&page=3>; rel="last"'
        )
        responses = [response, self.Response([{"number": 6}]), self.Response([{"number": 7}])]
        with patch.object(render_module.urllib.request, "urlopen", side_effect=responses):
            records = render_module.api_get_pages("o/r", "/issues", "token")
        self.assertEqual([item["number"] for item in records], [5, 6, 7])

    def test_empty_intermediate_page_cannot_skip_declared_later_page(self):
        responses = [
            self.Response([{"number": 5}], '<https://api.github.com/repositories/1/issues?per_page=100&page=3>; rel="last"'),
            self.Response([]),
        ]
        with patch.object(render_module.urllib.request, "urlopen", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "empty page before the declared last page"):
                render_module.api_get_pages("o/r", "/issues", "token")

    def test_malformed_next_link_fails_closed(self):
        response = self.Response(
            [{"number": 5}], 'https://api.github.com/repos/o/r/issues?page=2; rel="next"'
        )
        with patch.object(render_module.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "malformed pagination link"):
                render_module.api_get_pages("o/r", "/issues", "token")

    def test_api_failure_propagates_instead_of_returning_partial_items(self):
        with patch.object(render_module.urllib.request, "urlopen", side_effect=OSError("API unavailable")):
            with self.assertRaisesRegex(OSError, "API unavailable"):
                render_module.api_get_pages("o/r", "/issues", "token")

    def test_later_page_api_failure_does_not_return_first_page(self):
        first = self.Response(
            [{"number": 5}], '<https://api.github.com/repositories/1/issues?per_page=100&page=2>; rel="next"'
        )
        with patch.object(
            render_module.urllib.request, "urlopen", side_effect=[first, OSError("page 2 unavailable")]
        ):
            with self.assertRaisesRegex(OSError, "page 2 unavailable"):
                render_module.api_get_pages("o/r", "/issues", "token")

    def test_empty_page_linked_as_next_is_incomplete(self):
        first = self.Response(
            [{"number": 5}],
            '<https://api.github.com/repositories/1/issues?per_page=100&page=2>; rel="next"',
        )
        with patch.object(render_module.urllib.request, "urlopen", side_effect=[first, self.Response([])]):
            with self.assertRaisesRegex(RuntimeError, "empty page before the declared last page"):
                render_module.api_get_pages("o/r", "/issues", "token")

    def test_non_list_page_is_malformed(self):
        with patch.object(render_module.urllib.request, "urlopen", return_value=self.Response({"items": []})):
            with self.assertRaisesRegex(RuntimeError, "malformed paginated response"):
                render_module.api_get_pages("o/r", "/issues", "token")


def json_bytes(payload):
    import json

    return json.dumps(payload).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
