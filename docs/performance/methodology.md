# Methodology

## Timing domains

Each `kora-bench` result names its `timing_domain`; results in different domains are never compared.

| Domain | What is timed |
| --- | --- |
| `host_submit_completion` | From building a dispatch to its event completing (including default readback), on the host clock |
| `gpu_execution_timestamp` | Kernel time from GPU timestamps around the dispatch |
| `host_transfer_completion` | A transfer from submission to completion |
| `host_cpu_execution` | CPU work on the host clock |
| `host_operation` / `host_api_only` | Host-only bookkeeping (task graphs, allocators, command pools) |
| `runtime_dispatch_phase` | Diagnostic per-phase timings (`--profile-phases`), never a performance result |

Each result keeps every sample plus the median (p50) and p95. Outputs are validated during warm-up
and after every timed sample.

## Interleave, then compare medians of ratios

```sh
python3 tools/kora-bench/run_ab.py --a before/kora-bench --b after/kora-bench \
  --out ab --pairs 10 --device 1
python3 tools/kora-bench/render_ab.py --families --before ab/a-*.json.gz --after ab/b-*.json.gz
python3 tools/kora-bench/render_ab.py --before ab/a-*.json.gz --after ab/b-*.json.gz \
  --output-dir charts --change "my change" --prefix my-change
```

`run_ab.py` alternates which build runs first (ABBA). For each case, the change is the median of the
per-pair after/before p50 ratios, reported with how many pairs the new build won; a family's ratio is
the median over all its sizes and pairs. Claims of improvement in this project need at least 10
pairs.

## Know the noise floor

Estimate it from cases whose code is the same in both builds (on Apple M4, raw-Vulkan families land
within about ±6% and whole families within about ±10% between repeated sets). A change inside that
band is not a result.

## Scope

Results are specific to the device, driver, OS, compiler, and build named in each report's
`environment` and `gpu_measurements`. Software Vulkan (lavapipe) is for correctness, not
performance. No Linux or Windows GPU, and no CUDA, comparison is claimed by this project.
