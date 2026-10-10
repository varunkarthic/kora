# Security

The authoritative reporting policy is
[`SECURITY.md`](https://github.com/varunkarthic/kora/blob/main/SECURITY.md). Do not disclose suspected
vulnerabilities in public issues or pull requests.

## Runtime guidance

- Treat untrusted SPIR-V, metadata, caches, and traces as unsafe until the dedicated validation and
  security issues are complete.
- Use operating-system sandboxing and least privilege for untrusted workloads.
- Vulkan validation layers improve diagnostics but are not a security boundary.
- Do not write credentials, private buffer contents, proprietary kernels, or sensitive device dumps
  to logs or traces.
- Keep the Vulkan loader, driver, MoltenVK, compiler, and operating system updated independently of
  KORA.

## Deployment boundary

KORA is a local compute runtime, not an authentication or network authorization layer. Applications
embedding KORA remain responsible for validating callers, protecting inputs, limiting resources, and
securing transport and storage.
