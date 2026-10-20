# Benchmark comparison

| Elements | CPU median | CPU p95 | KORA default median | KORA default p95 | KORA deferred median | KORA deferred p95 | Raw host median | Raw host p95 | KORA GPU median | KORA GPU p95 | Raw GPU median | Raw GPU p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2.083 µs | 2.125 µs | 1.264 ms | 1.619 ms | 268.416 µs | 303.750 µs | 260.459 µs | 301.958 µs | 12.834 µs | 15.667 µs | 12.417 µs | 13.291 µs |
| 65536 | 34.250 µs | 35.042 µs | 1.088 ms | 1.882 ms | 314.792 µs | 423.500 µs | 265.834 µs | 323.333 µs | 21.042 µs | 31.333 µs | 23.208 µs | 32.416 µs |
| 1048576 | 563.166 µs | 573.667 µs | 2.046 ms | 2.333 ms | 1.129 ms | 1.282 ms | 1.247 ms | 1.385 ms | 416.417 µs | 632.042 µs | 582.916 µs | 645.708 µs |
| 4194240 | 2.243 ms | 2.328 ms | 2.670 ms | 3.341 ms | 1.790 ms | 2.092 ms | 1.847 ms | 2.109 ms | 1.240 ms | 1.414 ms | 1.239 ms | 1.406 ms |

Matched same-device Vulkan compute workloads:
| Workload | KORA default p50 | KORA default p95 | KORA deferred p50 | KORA deferred p95 | Raw Vulkan host p50 | Raw Vulkan host p95 | Deferred/raw p50 | KORA GPU p50 | KORA GPU p95 | Raw Vulkan GPU p50 | Raw Vulkan GPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAXPY / 4,096 | 314.166 µs | 391.917 µs | 262.792 µs | 361.917 µs | 302.083 µs | 389.458 µs | 0.87× | 20.416 µs | 25.583 µs | 20.542 µs | 24.834 µs |
| SAXPY / 65,536 | 351.833 µs | 494.333 µs | 419.166 µs | 482.250 µs | 258.250 µs | 407.458 µs | 1.62× | 24.167 µs | 34.042 µs | 26.417 µs | 33.125 µs |
| SAXPY / 1,048,576 | 1.224 ms | 2.181 ms | 1.513 ms | 2.229 ms | 504.458 µs | 1.022 ms | 3.00× | 404.375 µs | 597.708 µs | 103.458 µs | 228.333 µs |
| Reduction / 4,096 | 316.917 µs | 398.917 µs | 371.750 µs | 394.334 µs | 304.875 µs | 387.917 µs | 1.22× | 23.042 µs | 25.833 µs | 21.125 µs | 26.791 µs |
| Reduction / 65,536 | 360.208 µs | 408.709 µs | 371.417 µs | 411.000 µs | 372.417 µs | 451.584 µs | 1.00× | 36.333 µs | 39.208 µs | 36.583 µs | 42.375 µs |
| Reduction / 1,048,576 | 720.708 µs | 803.167 µs | 697.833 µs | 912.667 µs | 290.958 µs | 305.250 µs | 2.40× | 247.792 µs | 251.917 µs | 244.959 µs | 255.167 µs |
| Scan / 4,096 | 387.666 µs | 417.166 µs | 376.416 µs | 414.292 µs | 330.834 µs | 393.500 µs | 1.14× | 22.167 µs | 26.667 µs | 23.292 µs | 29.208 µs |
| Scan / 65,536 | 400.959 µs | 445.917 µs | 292.667 µs | 411.625 µs | 416.958 µs | 504.542 µs | 0.70× | 41.666 µs | 44.291 µs | 39.834 µs | 47.375 µs |
| Scan / 1,048,576 | 592.125 µs | 702.500 µs | 871.167 µs | 1.215 ms | 404.208 µs | 760.791 µs | 2.16× | 392.917 µs | 481.500 µs | 119.083 µs | 396.958 µs |
| Stencil / 4,096 | 328.792 µs | 395.000 µs | 343.833 µs | 400.167 µs | 320.208 µs | 388.459 µs | 1.07× | 21.917 µs | 24.666 µs | 12.709 µs | 22.583 µs |
| Stencil / 65,536 | 386.916 µs | 425.875 µs | 375.958 µs | 403.625 µs | 335.416 µs | 386.084 µs | 1.12× | 27.625 µs | 30.209 µs | 25.958 µs | 30.417 µs |
| Stencil / 1,048,576 | 642.709 µs | 1.310 ms | 460.708 µs | 1.230 ms | 383.250 µs | 892.250 µs | 1.20× | 357.542 µs | 472.125 µs | 80.042 µs | 106.750 µs |
| Tiled matmul / 16×16 | 362.042 µs | 413.500 µs | 387.958 µs | 418.583 µs | 319.375 µs | 383.958 µs | 1.21× | 15.416 µs | 21.000 µs | 14.583 µs | 15.500 µs |
| Tiled matmul / 64×64 | 326.917 µs | 404.875 µs | 310.042 µs | 386.041 µs | 253.209 µs | 313.916 µs | 1.22× | 34.083 µs | 36.500 µs | 25.709 µs | 34.750 µs |
| Tiled matmul / 128×128 | 430.292 µs | 469.541 µs | 434.125 µs | 483.292 µs | 321.667 µs | 430.458 µs | 1.35× | 60.083 µs | 71.541 µs | 66.166 µs | 72.792 µs |

Host rows include submit through completion and are not kernel-only. KORA default includes writable-output readback; deferred KORA leaves output device-resident; raw Vulkan uses persistent host-visible buffers. These columns therefore expose distinct transfer/residency boundaries. Device-timestamp rows isolate execution; unavailable samples are not estimated.
