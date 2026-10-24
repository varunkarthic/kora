# Performance

Every number on these pages comes from retained `kora-bench` JSON reports on a named device, and
every comparison interleaves runs. A single run on one machine can move several-fold with GPU clock
state, so single-run numbers are never compared.

<div class="grid cards" markdown>

-   __[Methodology](methodology.md)__ — timing domains, interleaving, and how to read a result.
-   __[Rust vs. C++ (1.0)](rust-vs-cpp.md)__ — KORA 1.0 against 0.3 on the same GPU, family by family.
-   __[Against Metal, OpenCL, OpenGL](competitors.md)__ — the same kernels through other APIs (0.3).
-   __[0.x optimization history](history.md)__ — the studies behind dispatch overhead, completion
    latency, unified-memory access, and on-demand timestamps.

</div>

![KORA 1.0, 0.3, and raw Vulkan dispatch latency on Apple M4 / MoltenVK](../assets/charts/readme-dispatch-moltenvk-light.svg#only-light)
![KORA 1.0, 0.3, and raw Vulkan dispatch latency on Apple M4 / MoltenVK](../assets/charts/readme-dispatch-moltenvk-dark.svg#only-dark)
