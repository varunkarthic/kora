# Releases and packages

KORA releases use `MAJOR.MINOR.PATCH`. The repository `VERSION` file selects the release line and
automation increments the patch for every successfully validated push to `main`.

`VERSION` is `0.2`, so successful main-branch releases start at `v0.2.0` and increment the patch
number for every validated commit (`v0.2.1`, `v0.2.2`, and so on). The workflow looks up the matching `v0.2`
milestone. An empty release milestone is never Stable; any open issue keeps the release Beta.
Milestone and issue lists are read across every GitHub API page. The release renderer fails closed if
pagination is malformed, incomplete, repeated, or unavailable, so it never treats a partial issue list
as proof of Stable status.

## Choosing a release

Use Stable releases when all published validation gates are closed. Use Beta releases only when you
can evaluate the remaining open gates against your workload. Neither channel replaces application-
specific benchmarking, driver qualification, or security review.

## Reading the GitHub release page

Generated release notes show the version and Beta/Stable channel first, followed by a short highlights
list derived from that tag's changelog. The platform table links to the Linux x86-64, macOS ARM64, and
Windows x86-64 archives produced by the required release matrix. The release is published only after
all three package jobs succeed, so these links refer to assets uploaded with that tag. The page also
links to the exact source tag and its changelog.

Release-validation issues are split into remaining and completed gates so the release maturity is
visible without losing the full issue list. An empty milestone remains Beta: lack of tracked gates is
not treated as proof that a release is Stable. Issue changes regenerate the latest release body but
do not create a new version or change the stability rule.

For a package install, download and extract the archive for your platform, then point
`CMAKE_PREFIX_PATH` at the extracted installation prefix before configuring a consumer project:

```cmake
find_package(kora CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE kora::kora)
```

For a source build, use the exact tag shown on the page and follow the **Source build** commands below.

## Package contents

Platform archives contain an installed development prefix: static library, public headers, generated
version header, and CMake package files. They do not bundle a Vulkan driver, MoltenVK, Python runtime,
or system C/C++ runtime.

## Release sign-off checklist

Use the exact release commit when recording a sign-off. A check is complete only with the corresponding
evidence; do not infer a passing platform or artifact from another job's success.

1. [ ] Configure, build, and run the full CTest suite on the release commit. Record skipped tests and
   the reason for each skip.
2. [ ] On a host exposing the reference Vulkan implementations, run the full suite against each
   available target:

   ```sh
   python3 tools/ci/run_device_matrix.py --build-dir build/debug --require-all
   ```

   The runner discovers KosmicKrisp, llvmpipe, and MoltenVK using `kora-info`'s reported driver and
   portability metadata, then sets `KORA_DEVICE` to each reported enumeration index before invoking
   CTest. It never uses a vendor ID. Without `--require-all`, unavailable reference devices are
   reported as skips; a release qualification host should use `--require-all` so missing hardware
   cannot be mistaken for a passing run.
3. [ ] Confirm the hosted main-branch CI matrix passed on Linux/GCC/lavapipe, macOS/AppleClang, and
   Windows/MSVC for the exact commit. Confirm the strict documentation job also passed.
4. [ ] Confirm the release workflow completed its Linux x86-64, macOS ARM64, and Windows x86-64
   package jobs and published the expected Beta/Stable channel for the milestone state.
5. [ ] Download every published archive, record SHA-256 digests, list its contents, and inspect the
   library/tool binary format and architecture. Package-job success alone does not prove the
   published archive has the intended ABI or contents.
6. [ ] Reconcile the milestone's release-validation issues with the release notes. A release remains
   Beta while any required gate is open; do not hand-edit the channel to Stable.
7. [ ] Check changelog/version stamping and release links, then retain workflow URLs, commit SHA,
   device-matrix results, archive digests, and any expected skips in the release record.

For a milestone sign-off, retain the completed checklist with the release record; the source tree
does not claim that a checklist template is evidence for a run that has not taken place.

## Source build

```sh
git clone --branch v0.2.0 https://github.com/varunkarthic/kora.git
cd kora
cmake --preset release -DKORA_VERSION_PATCH=0
cmake --build --preset release --parallel
ctest --preset release --output-on-failure
```

The example targets the first v0.2.x patch. For another patch, substitute its tag and the final
numeric component in `KORA_VERSION_PATCH`. See [CI and releases](ci-cd.md) for automation and
failure recovery.
