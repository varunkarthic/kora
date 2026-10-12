# Benchmark comparison

| Elements | CPU median | CPU p95 | KORA default median | KORA default p95 | KORA deferred median | KORA deferred p95 | Raw host median | Raw host p95 | KORA GPU median | KORA GPU p95 | Raw GPU median | Raw GPU p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2.083 µs | 2.125 µs | 2.411 ms | 4.389 ms | 396.084 µs | 457.541 µs | 494.208 µs | 551.250 µs | unavailable | unavailable | 19.333 µs | 22.708 µs |

Matched same-device Vulkan compute workloads:
| Workload | KORA default p50 | KORA default p95 | KORA deferred p50 | KORA deferred p95 | Raw Vulkan host p50 | Raw Vulkan host p95 | KORA GPU p50 | KORA GPU p95 | Raw Vulkan GPU p50 | Raw Vulkan GPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAXPY / 4,096 | 1.379 ms | 1.498 ms | 888.875 µs | 1.036 ms | 315.125 µs | 362.875 µs | 22.167 µs | 25.958 µs | 17.792 µs | 23.958 µs |
| Reduction / 4,096 | 959.459 µs | 1.105 ms | 522.750 µs | 579.041 µs | 385.709 µs | 419.250 µs | 24.208 µs | 28.500 µs | 22.083 µs | 26.375 µs |
| Scan / 4,096 | 925.209 µs | 1.040 ms | 396.333 µs | 466.375 µs | 370.834 µs | 416.708 µs | 24.791 µs | 29.167 µs | 23.084 µs | 28.292 µs |
| Tiled matmul / 64×64 | 619.625 µs | 899.875 µs | 295.959 µs | 400.167 µs | 384.125 µs | 441.250 µs | 33.833 µs | 36.166 µs | 31.583 µs | 37.208 µs |
| Stencil / 4,096 | 643.042 µs | 754.583 µs | 387.708 µs | 420.500 µs | 366.625 µs | 410.916 µs | 21.708 µs | 23.625 µs | 20.792 µs | 25.459 µs |

Host rows include submit through completion and are not kernel-only. KORA default includes writable-output readback; deferred KORA leaves output device-resident; raw Vulkan uses persistent host-visible buffers. These columns therefore expose distinct transfer/residency boundaries. Device-timestamp rows isolate execution; unavailable samples are not estimated.
