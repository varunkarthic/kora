# Continuous integration and releases

KORA uses GitHub Actions for pull-request validation, cross-platform main-branch testing, package
creation, semantic versioning, GitHub Releases, and milestone-driven release status.

## Continuous integration

`.github/workflows/ci.yml` runs for pull requests and pushes to `main`.

Pull requests run the Linux job with GCC and Mesa lavapipe. The job configures and builds Debug,
runs CTest, validates the installed CMake package with a standalone consumer, parses every workflow,
and runs the release-tool unit tests. It discovers the lavapipe ICD manifest installed by Mesa and
installs the Vulkan validation layer so the integration suite exercises both software-device
enumeration and validation-enabled instance creation.

Pushes to `main` additionally run macOS AppleClang and Windows MSVC jobs. These jobs use the same
checked-in Debug preset and CTest suite. This separates fast pull-request feedback from the complete
platform gate that precedes a release.

The Windows job initializes the Visual Studio developer environment before CMake runs and then
checks `CMakeCache.txt` for `cl.exe`. This assertion prevents a runner image's unrelated MinGW tools
from silently changing the compiler behind the **Windows / MSVC** label. When Vulkan-Headers is
downloaded as a fallback, KORA disables the dependency's optional C++ module and tests: KORA uses
only its headers, so those extra targets must not widen the supported compiler contract.

The workflows grant read-only repository permissions unless a job must publish or update a release.
Third-party actions are limited to checkout, Ninja and MSVC environment setup, and artifact
upload/download.

## Version calculation

[`VERSION`](../VERSION) stores `MAJOR.MINOR`, for example `0.1`. After CI succeeds for a push to
`main`, `.github/workflows/release.yml` lists tags matching `vMAJOR.MINOR.*` and increments the highest
patch number. With no prior matching tag, the first version is `MAJOR.MINOR.0`.

The release workflow stamps the `[Unreleased]` changelog entries into a dated version section,
creates a bot-authored release commit, creates an annotated `vMAJOR.MINOR.PATCH` tag, and pushes both.
Pushes made with the workflow token do not recursively create another release run.

To begin a new incompatible or feature release line, update `VERSION` in an ordinary issue branch and
create a matching `vMAJOR.MINOR` GitHub milestone before merging.

## Packages

Release jobs build, test, and install KORA independently on:

- Ubuntu x86-64;
- macOS ARM64;
- Windows x86-64 using the MSVC ABI.

The installed prefix is archived as a platform-specific `.tar.gz` or `.zip` file. Each archive
contains the library, public headers, generated version header, and CMake package metadata. GitHub's
automatically generated source archives remain available alongside these binary development
packages.

The Windows packaging job initializes the Visual Studio developer environment and verifies that
CMake selected `cl.exe` before compiling. The resulting archive contains the MSVC `.lib` library;
MinGW applications must build KORA from source with their own toolchain instead of consuming that
binary archive.

An archive is published only after its platform build and tests pass. The workflow does not claim a
platform package exists merely because another platform succeeded.

## Beta and Stable status

Each release line has a milestone named `vMAJOR.MINOR`. Issues in that milestone are release gates.
The release note generator reads every issue in the milestone and produces a table containing only
the issue number, title, link, and open/closed status; assignees are intentionally omitted.

- **Beta:** one or more release-gate issues are open.
- **Stable:** the milestone contains at least one issue and all its issues are closed.

The initial `v0.1` milestone tracks issues #71 through #75. The release is deliberately Beta until
documentation, security, cross-device validation, invariants, and shutdown semantics are complete.

`.github/workflows/release-status.yml` runs when an issue is opened, closed, reopened, edited, added
to a milestone, or removed from one. It regenerates the latest release body and switches the title,
prerelease flag, and latest-release flag without creating a new version.

Before the first semantic release exists, status refresh is intentionally a successful no-op. The
first successful `main` CI run creates the initial release; later issue events can then update it.

## Changelog maintenance

Every user-visible pull request must update `[Unreleased]` in
[`CHANGELOG.md`](https://github.com/varunkarthic/kora/blob/main/CHANGELOG.md) under
an appropriate Added, Changed, Deprecated, Removed, Fixed, or Security heading. The release workflow
moves that content under the exact version and date and rewrites comparison links.

`tools/release/stamp_changelog.py` performs the deterministic file transformation.
`tools/release/render_release.py` queries the GitHub API and renders usage instructions, status
badges, the Beta/Stable warning, validation table, and version-specific changelog.

## Local validation

Workflow YAML and release algorithms can be checked without publishing anything:

```sh
python3 -m pip install PyYAML
python3 tools/ci/check_workflows.py
python3 -m unittest discover -s tests/release -v
```

`check_workflows.py` parses each YAML file and verifies that it contains triggers and executable
jobs. It does not emulate GitHub-hosted runners. A successful local parse is distinct from a hosted
Actions run and from verification of a published release.

## Recovery and maintenance

If package creation fails, fix the cause in a new issue branch and push it through normal CI. The
next successful main push receives the next patch version; do not reuse or move a published tag.

If a release body becomes stale, dispatch **Refresh release status** manually. If the release
milestone is missing, create it rather than bypassing the renderer. Never mark a release Stable by
hand while a tracked gate remains open.

For a failing hosted job, open its log and identify the first failed command rather than relying on
the final summary alone. Confirm that the job actually started—account or repository restrictions
can prevent all steps from running—and keep these states distinct:

- a local workflow syntax check;
- a hosted build and test result on each operating system;
- a completed release workflow;
- downloaded and inspected release archives.

Release jobs use the repository-provided `GITHUB_TOKEN`; no personal token or package credential is
stored in the repository. Keep workflow permissions minimal and never print tokens in commands or
diagnostic output.
