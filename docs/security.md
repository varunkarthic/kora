# Security

The authoritative reporting policy is
[`SECURITY.md`](https://github.com/varunkarthic/kora/blob/main/SECURITY.md). Do not disclose suspected
vulnerabilities in public issues or pull requests.

## Runtime guidance

- KORA is not a GPU sandbox. Do not load or execute untrusted kernel packages in a privileged or
  sensitive process; isolate the process with operating-system controls and least privilege.
- KORA performs bounded structural/reflection checks on kernel packages and can run SPIRV-Tools
  validation when that optional dependency is available. These checks catch malformed or unsupported
  modules; they do not prove that a kernel is safe, terminates, respects application policy, or cannot
  trigger a vulnerability in the Vulkan loader, driver, or GPU.
- Vulkan validation layers improve diagnostics, not security. They do not isolate a device or make
  malicious workloads safe.
- Treat imported JSON metadata, persistent files, and traces as untrusted input too. Apply your own
  size limits and filesystem permissions at application boundaries; never consume traces as trusted
  commands or executable configuration.
- A caller-side `Event` timeout can stop waiting, but it is not a GPU execution budget and does not
  guarantee cancellation of already-submitted device work. Enforce CPU, memory, device, and wall-clock
  limits outside KORA when processing untrusted workloads.
- Do not write credentials, private buffer contents, proprietary kernels, or sensitive device dumps
  to logs or traces.
- Keep the Vulkan loader, driver, MoltenVK, compiler, and operating system updated independently of
  KORA.

## Deployment boundary

KORA is a local compute runtime, not an authentication or network authorization layer. Applications
embedding KORA remain responsible for validating callers, protecting inputs, limiting resources, and
securing transport and storage.

## Input validation performed by KORA

Validation is defense in depth, not a trust grant. For packaged SPIR-V kernels, the loader checks the
module header and byte order, rejects modules larger than 16 MiB, verifies instruction word counts,
reflects a documented subset of entry points, descriptors, push constants, specialization constants,
and capabilities, then compares that interface against the package metadata. Unknown or unsupported
constructs fail closed as invalid kernels. If SPIRV-Tools is available at build time, KORA additionally
validates the module against the Vulkan environment corresponding to the selected device. SPIRV-Tools
is optional, and a structurally valid module can still contain expensive, non-terminating, or
application-inappropriate work.

Metadata JSON parsing uses default bounds of 1 MiB input, nesting depth 32, 64 KiB per string, and
100,000 values. These bounds limit parser exposure; they are not a complete memory or CPU quota for
subsequent application behavior. Vulkan allocation and dispatch remain subject to device/driver
behavior and application-provided workload limits.

The detailed parser contract, reflected subset, and test coverage are documented in
[SPIR-V validation and reflection](internals/spirv.md). Kernel authors should also consult
[Kernel format](kernel-format.md), [Configuration](configuration.md), and [Recovery](recovery.md).

## Validation reference

| Boundary | Checks performed | What the checks do not establish |
| --- | --- | --- |
| Kernel package metadata | Bounded JSON parsing; schema/type/range validation; required fields and cross-field consistency. | Trustworthiness of the author, intent, or algorithm. |
| SPIR-V module | Size/header/instruction bounds; supported-subset reflection; capability checks; metadata-interface match; optional SPIRV-Tools Vulkan validation. | Safe termination, acceptable resource consumption, freedom from compiler/driver defects, or policy compliance. |
| Runtime resources | Queried feature/limit checks, checked arithmetic, memory-budget admission, structured failures. | A hard process-wide or device-wide quota against hostile concurrent activity. |
| Duplicate execution | Optional output comparison under an explicit `VerificationPolicy`. | Which device is correct after mismatch, or whether the computation itself is safe. |
| Trace export | Trace JSON is generated from runtime timing/identity fields and parsed before return. | Safety of importing untrusted trace files into third-party viewers; exported diagnostics may still be sensitive in an application context. |

Function-level details: `detail::json::parse` enforces configured parser limits;
`detail::spirv::validate_and_reflect` checks and reflects kernel input;
`PipelineManager::get_or_create` uses the selected device and reflected metadata to build/cache a
pipeline; `Runtime::dispatch` validates dispatch/resource requirements before submission; and
`Runtime::export_trace_json` validates its generated JSON. These internal names describe the current
implementation and are not a stable public loading API.
