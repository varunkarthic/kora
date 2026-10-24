# Building and testing

```sh
git clone https://github.com/varunkarthic/kora.git
cd kora
cargo build --workspace
cargo test --workspace
```

Expected result: every test passes. Tests that need a Vulkan device print `skipping: ...` and pass
when none is available.

## Workspace

| Crate | Contents |
| --- | --- |
| `crates/kora` | The runtime library |
| `crates/kora-capi` | C ABI: `libkora` and `include/kora.h` |
| `crates/kora-python` | Python extension `kora._core` (built with maturin) |
| `crates/kora-tools` | `kora-info`, `kora-top`, `kora-bench`, `kora-worker` |

SPIR-V test fixtures are committed under `crates/kora/tests/fixtures`, so no shader compiler is
needed. Regenerate them with `tools/shaders/compile_fixtures.sh` (needs `glslangValidator` and
SPIRV-Tools).

## The pre-merge gate

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
tools/ci/check_vendor_neutrality.sh
```

Plus, for the bindings and tools:

```sh
tools/ci/check_capi.sh                                            # C ABI install + example
python3 tests/check_kora_info_json.py target/release/kora-info
python3 tests/check_kora_bench_json.py target/release/kora-bench --gpu --cluster
python3 -m pip install ".[numpy]" pytest && python3 -m pytest tests/python
mkdocs build --strict                                             # this site
```

## Running against validation layers

```sh
KORA_VALIDATION=1 cargo test --workspace
```

The runtime, pipeline, and Cluster integration tests capture validation messages and fail on any
error. On macOS with Homebrew's layer, add `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`.

## Miri

Concurrency and `unsafe`-heavy modules have tests that run under Miri; loops and property tests run
fewer rounds there (`cfg!(miri)`), so the set takes about six minutes on an Apple M4:

```sh
rustup +nightly component add miri
MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-permissive-provenance" \
  cargo +nightly miri test -p kora --lib -- event cpu_executor buffer task_graph checked suballocator
```

Expected result: `test result: ok. 73 passed`. Permissive provenance allows the fake Vulkan handles
that the queue and command-pool mocks create from integers.

## macOS notes

Homebrew's lavapipe can crash in `vkCreateDevice` on some machines; exclude it with
`VK_LOADER_DRIVERS_DISABLE='*lvp*'`.
