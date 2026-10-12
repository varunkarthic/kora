import importlib.util
import io
import subprocess
import tempfile
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
freshness_module = load("check_pr_branch_freshness", ROOT / "tools/ci/check_pr_branch_freshness.py")
changelog_sections_module = load("check_changelog_sections", ROOT / "tools/ci/check_changelog_sections.py")


class StampTests(unittest.TestCase):
    def test_stamps_unreleased_and_comparison_links(self):
        source = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Work.\n\n[Unreleased]: old\n"
        result = stamp_module.stamp(source, "0.1.3", "v0.1.2", "2026-10-10")
        self.assertIn("## [0.1.3] - 2026-10-10", result)
        self.assertIn("compare/v0.1.3...HEAD", result)
        self.assertIn("compare/v0.1.2...v0.1.3", result)

    def test_empty_unreleased_section_cannot_produce_a_release(self):
        source = "# Changelog\n\n## [Unreleased]\n\n## [0.1.2]\n- Previous.\n\n[Unreleased]: old\n"
        with self.assertRaisesRegex(ValueError, "must contain at least one changelog entry"):
            stamp_module.stamp(source, "0.1.3", "v0.1.2", "2026-10-10")

    def test_heading_without_a_changelog_bullet_is_not_release_content(self):
        source = "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n## [0.1.2]\n- Previous.\n\n[Unreleased]: old\n"
        with self.assertRaisesRegex(ValueError, "must contain at least one changelog entry"):
            stamp_module.stamp(source, "0.1.3", "v0.1.2", "2026-10-10")


class BranchFreshnessTests(unittest.TestCase):
    @staticmethod
    def git(repo: str, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", repo, *args], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    def test_intervening_release_commit_requires_branch_synchronization(self):
        with tempfile.TemporaryDirectory() as repo:
            self.git(repo, "init", "-b", "main")
            self.git(repo, "config", "user.name", "KORA test")
            self.git(repo, "config", "user.email", "kora-test@example.invalid")
            self.git(repo, "commit", "--allow-empty", "-m", "initial main")
            self.git(repo, "switch", "-c", "issue/stale")
            self.git(repo, "commit", "--allow-empty", "-m", "issue change")
            stale_head = self.git(repo, "rev-parse", "HEAD")
            self.git(repo, "switch", "main")
            self.git(repo, "commit", "--allow-empty", "-m", "automatic release stamp")
            release_tip = self.git(repo, "rev-parse", "HEAD")

            self.assertFalse(freshness_module.branch_contains_base(release_tip, stale_head, repo))
            self.git(repo, "switch", "issue/stale")
            self.git(repo, "merge", "--no-edit", "main")
            synchronized_head = self.git(repo, "rev-parse", "HEAD")
            self.assertTrue(freshness_module.branch_contains_base(release_tip, synchronized_head, repo))

    def test_missing_git_refs_fail_loudly(self):
        with tempfile.TemporaryDirectory() as repo:
            self.git(repo, "init", "-b", "main")
            with self.assertRaisesRegex(RuntimeError, "git merge-base failed"):
                freshness_module.branch_contains_base("missing", "also-missing", repo)


class ChangelogSectionTests(unittest.TestCase):
    def test_user_visible_change_requires_new_unreleased_entry(self):
        base = "# Changelog\n\n## [Unreleased]\n\n## [0.2.6] - 2026-10-12\n- Existing.\n"
        with self.assertRaisesRegex(ValueError, r"must add a bullet under \[Unreleased\]"):
            changelog_sections_module.validate(base, base, require_unreleased_entry=True)

        changed = base.replace(
            "## [Unreleased]\n\n",
            "## [Unreleased]\n\n### Changed\n- Document benchmark JSON schema version 3.\n\n",
        )
        changelog_sections_module.validate(base, changed, require_unreleased_entry=True)

    def test_unreleased_entry_must_be_categorized_and_new(self):
        base = "# Changelog\n\n## [Unreleased]\n\n## [0.2.6]\n- Existing.\n"
        uncategorized = base.replace("## [Unreleased]\n\n", "## [Unreleased]\n- Not categorized.\n\n")
        with self.assertRaisesRegex(ValueError, r"must add a bullet under \[Unreleased\]"):
            changelog_sections_module.validate(base, uncategorized, require_unreleased_entry=True)

        same_entry = base.replace("## [Unreleased]\n\n", "## [Unreleased]\n\n### Fixed\n- Existing.\n\n")
        base_with_entry = base.replace("## [Unreleased]\n\n", "## [Unreleased]\n\n### Fixed\n- Existing.\n\n")
        with self.assertRaisesRegex(ValueError, r"must add a bullet under \[Unreleased\]"):
            changelog_sections_module.validate(
                base_with_entry, same_entry, require_unreleased_entry=True
            )

    def test_issue_change_cannot_be_added_to_an_already_published_section(self):
        base = (
            "# Changelog\n\n## [Unreleased]\n\n"
            "## [0.2.5] - 2026-10-12\n\n### Fixed\n- Existing release fix.\n"
        )
        stale_merge = (
            "# Changelog\n\n## [Unreleased]\n\n"
            "## [0.2.5] - 2026-10-12\n\n### Fixed\n"
            "- Existing release fix.\n- Late issue-branch fix.\n"
        )
        with self.assertRaisesRegex(ValueError, r"published changelog section\(s\): 0\.2\.5"):
            changelog_sections_module.validate(base, stale_merge)

    def test_issue_change_under_unreleased_preserves_published_sections(self):
        base = (
            "# Changelog\n\n## [Unreleased]\n\n"
            "## [0.2.5] - 2026-10-12\n\n### Fixed\n- Existing release fix.\n"
        )
        updated = (
            "# Changelog\n\n## [Unreleased]\n\n### Fixed\n- New issue fix.\n\n"
            "## [0.2.5] - 2026-10-12\n\n### Fixed\n- Existing release fix.\n"
        )
        changelog_sections_module.validate(base, updated)

    def test_explicit_historical_correction_allows_only_reviewed_release_section(self):
        base = (
            "# Changelog\n\n## [Unreleased]\n\n"
            "## [0.2.5] - 2026-10-12\n\n### Fixed\n- Existing release fix.\n"
            "- Synchronize CPU executor task-drain counter updates with the worker wait mutex so graceful shutdown\n"
            "  cannot lose the final wakeup and strand idle workers.\n"
            "## [0.2.4] - 2026-10-12\n- Older release fix.\n"
        )
        correction = (
            "# Changelog\n\n## [Unreleased]\n\n### Fixed\n"
            "- Synchronize CPU executor task-drain counter updates with the worker wait mutex so graceful shutdown\n"
            "  cannot lose the final wakeup and strand idle workers.\n"
            "- Reject published changelog edits.\n\n"
            "## [0.2.5] - 2026-10-12\n\n### Fixed\n- Existing release fix.\n"
            "## [0.2.4] - 2026-10-12\n- Older release fix.\n"
        )
        changelog_sections_module.validate(base, correction, {"0.2.5"})
        unexpected_v025_edit = correction.replace(
            "- Existing release fix.\n", "- Existing release fix.\n- Another claim.\n"
        )
        with self.assertRaisesRegex(ValueError, r"published changelog section\(s\): 0\.2\.5"):
            changelog_sections_module.validate(base, unexpected_v025_edit, {"0.2.5"})
        changed_history = correction.replace("Older release fix.", "Changed history.")
        with self.assertRaisesRegex(ValueError, r"published changelog section\(s\): 0\.2\.4"):
            changelog_sections_module.validate(base, changed_history, {"0.2.5"})


class RenderTests(unittest.TestCase):
    def test_release_highlights_preserve_wrapped_lines_and_stop_at_boundaries(self):
        changelog = """## [0.2.10]

### Added
- First highlight with context
  continued on the next line.
- Second highlight.

### Fixed
- Third highlight.
- Fourth highlight.
"""
        highlights = render_module.release_highlights(changelog)
        self.assertEqual(
            highlights,
            [
                "First highlight with context continued on the next line.",
                "Second highlight.",
                "Third highlight.",
            ],
        )
        self.assertEqual(render_module.release_highlights(changelog, 2), highlights[:2])
        self.assertEqual(render_module.release_highlights(changelog, 0), [])

    def test_v0210_release_highlight_includes_wrapped_continuation(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        section = render_module.release_changelog(changelog, "0.2.10")
        self.assertIn(
            "Generate deterministic, timing-domain-separated benchmark SVG charts from retained schema-3 "
            "reports and document their methodology and limitations.",
            render_module.release_highlights(section)[0],
        )

    def test_release_page_contains_versioned_platform_downloads_and_source_summary(self):
        changelog = """## [0.2.10] - 2026-10-12

### Added
- Add useful release notes.
- Add another improvement.

### Fixed
- Repair a behavior.
"""
        body, stable = render_module.render("o/r", "0.2.10", changelog, [])
        self.assertFalse(stable)
        self.assertIn("# KORA v0.2.10", body)
        self.assertIn("**Beta.**", body)
        self.assertIn("[Source at this tag](https://github.com/o/r/tree/v0.2.10)", body)
        self.assertIn("- Add useful release notes.", body)
        self.assertIn("kora-0.2.10-linux-x86_64.tar.gz", body)
        self.assertIn("kora-0.2.10-macos-arm64.tar.gz", body)
        self.assertIn("kora-0.2.10-windows-x86_64.zip", body)
        self.assertIn("CMAKE_PREFIX_PATH", body)
        self.assertEqual(body.count("git clone --branch"), 1)

    def test_release_page_groups_every_gate_and_escapes_markdown_titles(self):
        issues = [
            {"number": 71, "title": "Open | [gate]", "html_url": "https://example/71", "state": "open"},
            {"number": 72, "title": "Closed gate", "html_url": "https://example/72", "state": "closed"},
        ]
        body, stable = render_module.render("o/r", "0.2.10", "- Change.", issues)
        self.assertFalse(stable)
        self.assertIn("### Remaining gates (1)", body)
        self.assertIn("### Completed gates (1)", body)
        self.assertIn(r"[Open \| \[gate\]](https://example/71)", body)
        self.assertIn("[Closed gate](https://example/72)", body)
        self.assertIn("🟡 Open", body)
        self.assertIn("✅ Closed", body)
        self.assertIn("1 of 2 release-validation gates closed", body)

    def test_release_asset_urls_match_release_workflow_names(self):
        assets = render_module.release_assets("o/r", "0.2.10")
        self.assertEqual(
            [asset[2] for asset in assets],
            [
                "kora-0.2.10-linux-x86_64.tar.gz",
                "kora-0.2.10-macos-arm64.tar.gz",
                "kora-0.2.10-windows-x86_64.zip",
            ],
        )
        self.assertTrue(all("/releases/download/v0.2.10/" in asset[3] for asset in assets))

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

    def test_empty_version_changelog_is_rejected(self):
        changelog = "## [0.2.2] - 2026-10-12\n\n## [0.2.1] - 2026-10-11\n- Previous.\n"
        with self.assertRaisesRegex(ValueError, "section for 0.2.2 has no release entries"):
            render_module.release_changelog(changelog, "0.2.2")

    def test_v022_benchmark_changes_are_not_attributed_to_v021(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        v021 = render_module.release_changelog(changelog, "0.2.1")
        v022 = render_module.release_changelog(changelog, "0.2.2")
        benchmark_change = "same Runtime-selected physical device"
        self.assertNotIn(benchmark_change, v021)
        self.assertIn(benchmark_change, v022)


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
