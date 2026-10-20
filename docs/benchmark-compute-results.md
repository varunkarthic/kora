# Benchmark comparison

| Elements | CPU median | CPU p95 | KORA default median | KORA default p95 | KORA deferred median | KORA deferred p95 | Raw host median | Raw host p95 | KORA GPU median | KORA GPU p95 | Raw GPU median | Raw GPU p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2.084 µs | 2.166 µs | 749.334 µs | 1.223 ms | 344.500 µs | 398.292 µs | 457.417 µs | 547.209 µs | 22.708 µs | 26.333 µs | 22.791 µs | 29.250 µs |
| 65536 | 34.333 µs | 34.958 µs | 732.875 µs | 1.010 ms | 344.042 µs | 412.292 µs | 476.584 µs | 545.375 µs | 31.458 µs | 35.375 µs | 30.500 µs | 34.833 µs |
| 1048576 | 566.291 µs | 730.584 µs | 1.035 ms | 1.838 ms | 1.140 ms | 1.275 ms | 454.958 µs | 757.208 µs | 570.209 µs | 620.708 µs | 127.583 µs | 130.917 µs |
| 4194240 | 2.303 ms | 4.771 ms | 1.666 ms | 2.196 ms | 1.803 ms | 2.011 ms | 958.958 µs | 2.041 ms | 1.055 ms | 1.358 ms | 583.958 µs | 1.279 ms |

Matched same-device Vulkan compute workloads:
| Workload | KORA default p50 | KORA default p95 | KORA deferred p50 | KORA deferred p95 | Raw Vulkan host p50 | Raw Vulkan host p95 | Deferred/raw p50 | KORA GPU p50 | KORA GPU p95 | Raw Vulkan GPU p50 | Raw Vulkan GPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAXPY / 4,096 | 388.250 µs | 478.375 µs | 354.666 µs | 399.000 µs | 236.250 µs | 358.542 µs | 1.50× | 23.667 µs | 29.375 µs | 21.167 µs | 25.083 µs |
| SAXPY / 65,536 | 507.833 µs | 582.917 µs | 443.916 µs | 542.291 µs | 368.417 µs | 471.792 µs | 1.20× | 32.708 µs | 38.459 µs | 27.166 µs | 34.167 µs |
| SAXPY / 1,048,576 | 720.542 µs | 914.083 µs | 677.125 µs | 1.292 ms | 495.209 µs | 620.959 µs | 1.37× | 98.958 µs | 387.208 µs | 95.834 µs | 154.791 µs |
| Reduction / 4,096 | 236.584 µs | 341.750 µs | 227.667 µs | 372.208 µs | 312.167 µs | 352.417 µs | 0.73× | 24.083 µs | 28.209 µs | 22.125 µs | 27.583 µs |
| Reduction / 65,536 | 385.625 µs | 438.500 µs | 370.334 µs | 466.166 µs | 436.500 µs | 473.458 µs | 0.85× | 36.625 µs | 41.458 µs | 31.541 µs | 39.458 µs |
| Reduction / 1,048,576 | 712.250 µs | 796.417 µs | 703.416 µs | 787.375 µs | 307.375 µs | 312.292 µs | 2.29× | 249.250 µs | 257.666 µs | 77.083 µs | 143.958 µs |
| Scan / 4,096 | 379.500 µs | 475.416 µs | 433.708 µs | 510.541 µs | 332.542 µs | 373.041 µs | 1.30× | 25.208 µs | 29.000 µs | 22.333 µs | 29.750 µs |
| Scan / 65,536 | 432.625 µs | 527.875 µs | 454.000 µs | 564.917 µs | 236.250 µs | 418.042 µs | 1.92× | 43.333 µs | 49.417 µs | 39.042 µs | 46.709 µs |
| Scan / 1,048,576 | 511.209 µs | 675.875 µs | 396.458 µs | 983.417 µs | 405.708 µs | 458.500 µs | 0.98× | 374.084 µs | 456.708 µs | 89.041 µs | 94.334 µs |
| Stencil / 4,096 | 366.250 µs | 448.625 µs | 342.292 µs | 403.167 µs | 329.542 µs | 421.541 µs | 1.04× | 25.084 µs | 29.625 µs | 21.000 µs | 24.875 µs |
| Stencil / 65,536 | 421.167 µs | 522.750 µs | 397.542 µs | 489.542 µs | 231.833 µs | 387.292 µs | 1.71× | 29.541 µs | 34.875 µs | 23.583 µs | 31.792 µs |
| Stencil / 1,048,576 | 487.417 µs | 656.959 µs | 388.583 µs | 428.042 µs | 370.250 µs | 380.791 µs | 1.05× | 358.375 µs | 427.666 µs | 70.750 µs | 74.333 µs |
| Tiled matmul / 16×16 | 308.625 µs | 396.250 µs | 379.000 µs | 437.833 µs | 220.416 µs | 223.916 µs | 1.72× | 26.416 µs | 30.959 µs | 21.208 µs | 28.834 µs |
| Tiled matmul / 64×64 | 474.833 µs | 545.084 µs | 403.959 µs | 502.458 µs | 234.542 µs | 399.833 µs | 1.72× | 37.500 µs | 43.542 µs | 31.667 µs | 39.583 µs |
| Tiled matmul / 128×128 | 480.875 µs | 605.334 µs | 499.292 µs | 606.000 µs | 341.167 µs | 482.333 µs | 1.46× | 67.000 µs | 74.666 µs | 46.875 µs | 75.750 µs |

Host rows include submit through completion and are not kernel-only. KORA default includes writable-output readback; deferred KORA leaves output device-resident; raw Vulkan uses persistent host-visible buffers. These columns therefore expose distinct transfer/residency boundaries. Device-timestamp rows isolate execution; unavailable samples are not estimated.
