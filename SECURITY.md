# Security policy

## Supported versions

KORA is in early development. Security fixes are applied to the latest release and the current
`main` branch only. Older tags are not maintained unless a release notice explicitly says otherwise.

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| `main` | Yes |
| Older releases | No |

## Reporting a vulnerability

Do not open a public issue, pull request, discussion, or proof-of-concept repository for a suspected
vulnerability.

Use GitHub's private vulnerability reporting flow from the repository's **Security** tab. Include:

- the affected version or commit;
- the platform, compiler, Vulkan implementation, and relevant device details;
- a minimal reproduction or precise sequence of operations;
- the expected and observed behavior;
- the security impact and any known preconditions;
- suggested mitigations, if available.

Do not include credentials, private data, proprietary kernels, or sensitive device dumps. A report
may be closed if it cannot be reproduced safely or falls outside KORA's code and documented runtime
boundary.

## Response process

The maintainer will assess reproducibility, impact, affected versions, and whether a coordinated fix
is practical. Confirmed issues may be addressed through a private fix, a new release, an advisory,
and release-note attribution when requested. No response-time or embargo guarantee is offered for
this solo-maintained project.

## Scope and trust boundaries

KORA parses kernel metadata and SPIR-V and interacts with Vulkan drivers, device memory, persistent
caches, and user-provided buffers. Treat untrusted kernel packages, cache files, and trace inputs as
unsafe even when KORA's structural validation succeeds. Run untrusted workloads only in an
operating-system sandbox with least privilege; KORA is not a GPU sandbox and does not provide a
complete process, device-time, or resource-isolation boundary.

Vulkan validation layers are diagnostic tools; they are not a security boundary. TLS protects data
in transit but does not authenticate a caller by itself. Never place secrets in command-line
arguments, logs, test fixtures, cache keys, or diagnostic traces.

## External changes

This repository does not accept external pull requests or ordinary feature/support issues. Private
security reports are accepted solely to enable responsible disclosure and remediation.
