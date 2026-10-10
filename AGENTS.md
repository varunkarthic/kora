# Agent workflow

These instructions apply to the entire repository.

## Source of work

- Read `context/PLAN.md`, the matching `context/issues/NN.md`, and the live GitHub issue before
  changing code.
- List active work with `gh issue list --repo varunkarthic/kora --assignee @me`.
- Respect the dependency order recorded in `context/PLAN.md`.
- Keep `context/PROGRESS.md` and `context/ARCHITECTURE.md` current. The `context/` directory is local
  working context and is intentionally not committed.

## Branch and review policy

- Use one branch per issue: `issue/<number>-<short-slug>`.
- Do not combine unrelated issues in one pull request.
- Use conventional commit subjects such as `build:`, `test:`, `docs:`, `feat:`, and `fix:`.
- Include `Closes #<number>` in the pull request body.
- Merge with a merge commit only after all relevant local checks pass. Do not wait for hosted CI when
  the recorded workflow explicitly permits local verification as the merge gate.

## Implementation rules

- Preserve unrelated and pre-existing changes in a dirty worktree.
- Keep KORA vendor-neutral; do not add CUDA, ROCm/HIP, oneAPI, OpenCL, or vendor-specific correctness
  dependencies.
- Use queried capabilities and measured behavior, never GPU vendor IDs, for runtime decisions.
- Keep platform-specific code isolated and optional features behind conservative fallbacks.
- Expected failures use `Result<T>` or `Status`; exceptions belong only at explicitly throwing API
  boundaries.
- Never place credentials, private data, proprietary kernels, or sensitive device details in source,
  tests, logs, workflow arguments, release notes, or local context.

## Verification before push

Run checks proportional to the changed layer. The default native gate is:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure
```

For build-system changes, also run:

```sh
tools/ci/check_presets.sh --build
tools/ci/check_install.sh
```

Format C++ sources with the repository `.clang-format`, validate workflow YAML, build documentation
with warnings as errors, and run sanitizer presets when the change touches memory, concurrency, or
public low-level contracts. Report skipped or unavailable checks accurately.

## Documentation and releases

- Update public documentation and `CHANGELOG.md` with user-visible behavior.
- Keep commands executable and state their expected outcomes.
- Separate locally tested behavior, hosted CI results, and published-package verification.
- A release is Stable only when all release-validation issues in its matching milestone are closed;
  otherwise it is Beta.
- Do not create `CONTRIBUTING.md` or `CODE_OF_CONDUCT.md`; this is a solo-maintained project that does
  not accept external contributions.
