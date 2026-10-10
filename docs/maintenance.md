# Maintenance workflow

KORA is a solo-maintained project and does not accept external pull requests, feature requests, or
support issues. This page documents the maintainer workflow rather than inviting contributions.

## Issue lifecycle

1. Select the next dependency-ordered GitHub issue.
2. Create `issue/<number>-<slug>` from an updated `main`.
3. Implement only that issue and update tests, docs, and `[Unreleased]` changelog entries.
4. Run focused tests, the full Debug suite, and risk-specific sanitizer/package/docs checks.
5. Open a pull request containing `Closes #<number>`.
6. Merge with a merge commit after the local gate is green.
7. Distinguish local verification from hosted Actions and published-package verification.

## Documentation gate

```sh
python3 -m venv build/docs-venv
build/docs-venv/bin/pip install -r docs/requirements.txt
build/docs-venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
```

Warnings fail the documentation build locally, in CI, and on Read the Docs.

## Release maintenance

Maintain release gates in the `vMAJOR.MINOR` milestone. Do not manually claim Stable while a gate is
open, move published tags, or reuse a failed version. Correct failures through a new issue branch and
allow the next successful push to receive the next patch number.

## Private data

Do not store credentials, private reports, proprietary artifacts, or device dumps in the repository,
test fixtures, workflow logs, releases, or local context files.
