# Core implementation

The currently implemented core lives in `src/core`.

## `errors.cpp`

Maps every `ErrorCode` to a stable name and renders structured, single-line cause chains. The mapping
has unit-test coverage across all enumerators.

## `options.cpp`

Parses environment values without mutating the environment or throwing on malformed input. An
injectable lookup makes precedence and failure behavior deterministic in tests.

## `log.cpp`

Stores category thresholds atomically and protects sink replacement/output with a mutex. The sink is
process-wide. Tests cover filtering, argument evaluation, category overrides, parsing, and concurrent
writes.

## `version.cpp`

Formats the generated compile-time version. Release jobs provide the patch value during package
configuration.

## Planned subsystem boundaries

The `vulkan`, `memory`, `execution`, `scheduler`, `kernels`, `profiling`, and `recovery` directories
are reserved for later issue-scoped implementations. Their presence is structural scaffolding, not
evidence that those subsystems are complete.
