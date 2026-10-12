# Benchmark comparison

| Elements | CPU median | CPU p95 | KORA default median | KORA default p95 | KORA deferred median | KORA deferred p95 | Raw host median | Raw host p95 | KORA GPU median | KORA GPU p95 | Raw GPU median | Raw GPU p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2.083 µs | 2.125 µs | 3.238 ms | 4.705 ms | 338.042 µs | 437.625 µs | 414.417 µs | 528.750 µs | unavailable | unavailable | 20.500 µs | 22.917 µs |

Matched same-device Vulkan compute workloads:
| Workload | KORA host p50 | KORA host p95 | Raw Vulkan host p50 | Raw Vulkan host p95 | KORA GPU p50 | KORA GPU p95 | Raw Vulkan GPU p50 | Raw Vulkan GPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAXPY / 4,096 | 1.046 ms | 1.215 ms | 315.209 µs | 374.875 µs | 20.333 µs | 23.958 µs | 21.042 µs | 23.542 µs |
| Reduction / 4,096 | 659.375 µs | 800.750 µs | 295.875 µs | 395.542 µs | 23.167 µs | 26.208 µs | 23.000 µs | 25.875 µs |
| Scan / 4,096 | 688.625 µs | 850.250 µs | 306.458 µs | 358.417 µs | 24.000 µs | 27.291 µs | 23.791 µs | 26.750 µs |
| Tiled matmul / 64×64 | 654.666 µs | 803.875 µs | 318.333 µs | 348.250 µs | 33.459 µs | 124.459 µs | 32.792 µs | 35.125 µs |
| Stencil / 4,096 | 638.875 µs | 776.125 µs | 300.042 µs | 366.667 µs | unavailable | unavailable | 20.750 µs | 23.083 µs |

Host rows include submit through completion and are not kernel-only. Device-timestamp rows isolate execution; unavailable samples are not estimated. Raw Vulkan uses persistent host-visible buffers on this benchmark path, so its host latency has a different residency/readback boundary from KORA.
