# Security

Report suspected vulnerabilities privately as described in
[`SECURITY.md`](https://github.com/varunkarthic/kora/blob/main/SECURITY.md), not in public issues.

## Trust boundary

- **KORA is not a GPU sandbox.** Do not run untrusted kernels in a privileged or sensitive process.
  KORA's checks (bounded SPIR-V structure and reflection, manifest agreement, 16 MiB module limit)
  catch malformed or unsupported modules; they do not prove a kernel is safe, terminates, or cannot
  trigger a driver bug.
- **Validation layers** improve diagnostics, not security.
- **Untrusted inputs**: kernel manifests (parser limits: 1 MiB, depth 32, 64 KiB strings, 100,000
  values), cache files, and traces. Apply your own size limits and file permissions.
- **Timeouts** stop waiting; they do not cancel submitted GPU work. Enforce resource limits outside
  KORA for untrusted workloads.
- **Logs and traces** never contain buffer contents; do not add credentials, proprietary kernels, or
  sensitive device dumps to them.

## Remote workers

- A worker without `KORA_WORKER_TOKEN` accepts only loopback clients.
- With a token, clients authenticate with an HMAC-SHA256 challenge over a random nonce, verified in
  constant time; an unauthenticated peer cannot displace a connected client.
- Frames are size-limited and CRC32C-checked; a malformed reply is validated completely before any
  buffer changes and drops the link.
- Traffic is **not encrypted**. Use a trusted network, an SSH tunnel, or a VPN.
- The `kora-top` status stream binds only to loopback addresses and is read-only.

Keep your Vulkan loader, drivers, MoltenVK, and OS updated independently of KORA.
