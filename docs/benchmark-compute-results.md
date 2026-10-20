# Benchmark comparison

| Elements | CPU median | CPU p95 | KORA default median | KORA default p95 | KORA deferred median | KORA deferred p95 | Raw host median | Raw host p95 | KORA GPU median | KORA GPU p95 | Raw GPU median | Raw GPU p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2.083 µs | 2.166 µs | 3.716 ms | 5.856 ms | 342.625 µs | 402.125 µs | 383.500 µs | 403.834 µs | unavailable | unavailable | 21.166 µs | 23.833 µs |
| 65536 | 34.583 µs | 35.375 µs | 5.517 ms | 12.743 ms | 413.875 µs | 1.284 ms | 406.625 µs | 482.209 µs | unavailable | unavailable | 29.958 µs | 32.792 µs |
| 1048576 | 570.375 µs | 821.042 µs | 6.092 ms | 11.576 ms | 1.271 ms | 2.832 ms | 1.202 ms | 1.540 ms | unavailable | unavailable | 592.292 µs | 689.541 µs |
| 4194240 | 2.717 ms | 4.651 ms | 6.865 ms | 15.021 ms | 2.130 ms | 2.458 ms | 2.111 ms | 2.394 ms | unavailable | unavailable | 1.342 ms | 1.512 ms |

Matched same-device Vulkan compute workloads:
| Workload | KORA default p50 | KORA default p95 | KORA deferred p50 | KORA deferred p95 | Raw Vulkan host p50 | Raw Vulkan host p95 | Deferred/raw p50 | KORA GPU p50 | KORA GPU p95 | Raw Vulkan GPU p50 | Raw Vulkan GPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAXPY / 4,096 | 922.416 µs | 1.095 ms | 639.042 µs | 705.792 µs | 298.084 µs | 339.334 µs | 2.14× | 17.000 µs | 23.875 µs | 19.917 µs | 21.792 µs |
| SAXPY / 65,536 | 882.334 µs | 2.001 ms | 729.542 µs | 1.381 ms | 337.875 µs | 370.209 µs | 2.16× | unavailable | unavailable | 26.583 µs | 34.209 µs |
| SAXPY / 1,048,576 | 4.266 ms | 6.172 ms | 2.992 ms | 5.120 ms | 468.792 µs | 647.333 µs | 6.38× | 483.875 µs | 634.500 µs | 117.375 µs | 147.166 µs |
| Reduction / 4,096 | 703.250 µs | 2.038 ms | 293.000 µs | 392.542 µs | 299.417 µs | 322.959 µs | 0.98× | unavailable | unavailable | 22.667 µs | 24.708 µs |
| Reduction / 65,536 | 680.459 µs | 1.095 ms | 376.500 µs | 416.458 µs | 300.750 µs | 353.916 µs | 1.25× | unavailable | unavailable | 30.417 µs | 37.333 µs |
| Reduction / 1,048,576 | 1.004 ms | 1.729 ms | 671.833 µs | 1.035 ms | 284.625 µs | 335.667 µs | 2.36× | unavailable | unavailable | 247.250 µs | 251.500 µs |
| Scan / 4,096 | 615.458 µs | 797.292 µs | 371.583 µs | 530.625 µs | 286.458 µs | 317.458 µs | 1.30× | 16.875 µs | 25.417 µs | 23.583 µs | 25.792 µs |
| Scan / 65,536 | 524.708 µs | 754.542 µs | 317.000 µs | 401.292 µs | 386.209 µs | 442.834 µs | 0.82× | unavailable | unavailable | 41.583 µs | 45.667 µs |
| Scan / 1,048,576 | 2.283 ms | 2.925 ms | 1.121 ms | 1.433 ms | 378.542 µs | 814.292 µs | 2.96× | 355.625 µs | 453.166 µs | 140.167 µs | 448.292 µs |
| Stencil / 4,096 | 626.084 µs | 1.023 ms | 370.708 µs | 464.000 µs | 191.708 µs | 241.000 µs | 1.93× | 22.625 µs | 25.042 µs | 20.917 µs | 23.250 µs |
| Stencil / 65,536 | 660.041 µs | 847.250 µs | 387.542 µs | 701.208 µs | 318.375 µs | 351.084 µs | 1.22× | unavailable | unavailable | 27.250 µs | 30.084 µs |
| Stencil / 1,048,576 | 2.375 ms | 2.759 ms | 1.124 ms | 1.827 ms | 363.542 µs | 430.709 µs | 3.09× | 220.375 µs | 398.083 µs | 228.875 µs | 458.667 µs |
| Tiled matmul / 16×16 | 974.417 µs | 2.331 ms | 540.875 µs | 1.326 ms | 304.500 µs | 335.584 µs | 1.78× | 15.833 µs | 25.542 µs | 25.459 µs | 27.541 µs |
| Tiled matmul / 64×64 | 781.958 µs | 1.396 ms | 486.375 µs | 1.588 ms | 303.000 µs | 328.625 µs | 1.61× | 25.500 µs | 33.375 µs | 34.875 µs | 37.875 µs |
| Tiled matmul / 128×128 | 845.834 µs | 2.193 ms | 419.333 µs | 671.583 µs | 420.667 µs | 468.250 µs | 1.00× | 66.292 µs | 70.250 µs | 65.791 µs | 69.166 µs |

Host rows include submit through completion and are not kernel-only. KORA default includes writable-output readback; deferred KORA leaves output device-resident; raw Vulkan uses persistent host-visible buffers. These columns therefore expose distinct transfer/residency boundaries. Device-timestamp rows isolate execution; unavailable samples are not estimated.
