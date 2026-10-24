# KORA 1.0 (Rust) vs. 0.3 (C++), Apple M4

Ten ABBA-interleaved pairs per device of `kora-bench --gpu --cluster` (100 samples, 10 warm-ups per
run), macOS on Apple M4, with lavapipe excluded (`VK_LOADER_DRIVERS_DISABLE='*lvp*'`):

- `a-NN.json.gz`: C++ 0.3 `kora-bench` (schema 3), built from `e07fd51` with `cmake --preset release`.
- `b-NN.json.gz`: Rust 1.0 `kora-bench` (schema 4), built from `f03431c` with `cargo build --release`.
- `moltenvk/`: `KORA_DEVICE=1` (MoltenVK). `kosmickrisp/`: `KORA_DEVICE=0` (KosmicKrisp, no GPU timestamps).

Pair *i* ran A then B for even *i* and B then A for odd *i* (`tools/kora-bench/run_ab.py`).

## Summaries

- `families-<device>.md`: per-family median of every per-pair B/A p50 ratio (below 1.00 = Rust faster).
- `compute-<device>.md`: matched compute workloads per size (`render_ab.py`).
- The README's and documentation's charts and tables: `python3 tools/bench-report/render.py`.

Regenerate:

```sh
python3 tools/kora-bench/render_ab.py --families \
  --before benchmarks/results/rust-vs-cpp-apple-m4/moltenvk/a-*.json.gz \
  --after benchmarks/results/rust-vs-cpp-apple-m4/moltenvk/b-*.json.gz
python3 tools/kora-bench/render_ab.py \
  --before benchmarks/results/rust-vs-cpp-apple-m4/moltenvk/a-*.json.gz \
  --after benchmarks/results/rust-vs-cpp-apple-m4/moltenvk/b-*.json.gz > compute-moltenvk.md
python3 tools/bench-report/render.py
```

## Reading the results

On this machine a family moves by up to about ±10% between repeated A/B sets: raw-Vulkan families,
whose code is identical in both builds, land between 0.98 and 1.06, and the CPU-only families differ
between the MoltenVK and KosmicKrisp sets although both ran the same binaries.

- Faster in Rust, beyond the noise: task-graph finalize and replay, allocator families, staging-ring
  reservations (constant time with live ranges), the CPU stencil baselines, direct transfers, and
  `task_graph_replay_64_reuse_plan`.
- At parity within the noise: the KORA runtime dispatch families (vector add, SAXPY, reduction, scan,
  stencil, matmul; default and deferred readback) and their device times.
- Slower in both sets: `cluster_local_saxpy_dispatch_wait` (1.24 MoltenVK, 1.08 KosmicKrisp; the
  Cluster's own per-dispatch work measures 5–9 µs), `gpu_tiny_dispatch` (1.05–1.12; it reads back a
  16 MiB output buffer), and MoltenVK SAXPY at 1M with default readback (per-case +45%, 1/10 wins).
  Phase profiling of that case shows equal host submission and differences only in 4 MiB host copies
  and device completion, within C++'s own run-to-run spread.
- `host_allocation_4k` (3.05) is not comparable: the C++ compiler may elide the allocation, while the
  Rust benchmark keeps it with `black_box`.
- Issue #229 (KORA kernels ~3–4× longer than raw Vulkan at 1M by device time on MoltenVK) is
  reproduced identically by both builds, so it predates the rewrite. Dedicated allocations were tested
  and do not close it.
