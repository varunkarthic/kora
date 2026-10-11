# Kernel package metadata (`.kora.json`)

KORA kernel packages use a small, versioned JSON manifest to describe the interface shared by
kernel loading, validation, dispatch, and CPU fallback selection. The parser is part of KORA and has
no third-party JSON dependency. Documents are limited to 1 MiB, 32 nesting levels, 64 KiB per
string, and 100,000 JSON values. JSON syntax is strict: duplicate object members, malformed UTF-16
surrogate escapes, trailing data, non-finite numeric values, and unknown schema members are errors.
Failures return `ErrorCode::InvalidKernel`; messages include a JSON byte offset or metadata field
path to make a bad package diagnosable.

## Version 1 example

```json
{
  "schema_version": 1,
  "kernel_id": "vector.add",
  "bindings": [
    { "set": 0, "binding": 0, "name": "lhs", "kind": "storage_buffer", "access": "read" },
    { "set": 0, "binding": 1, "name": "rhs", "kind": "storage_buffer", "access": "read" },
    { "set": 0, "binding": 2, "name": "output", "kind": "storage_buffer", "access": "write" }
  ],
  "push_constant_size": 4,
  "push_constants": [
    { "name": "count", "type": "u32", "offset": 0, "size": 4 }
  ],
  "specialization_constants": [
    { "id": 0, "name": "workgroup_width", "type": "u32", "default": 64 }
  ],
  "local_size": [64, 1, 1],
  "cpu_fallback_key": "vector.add.scalar",
  "semantics": ["elementwise", "pure"]
}
```

`schema_version`, `kernel_id`, `bindings`, `push_constant_size`, `push_constants`,
`specialization_constants`, and `local_size` are required. `cpu_fallback_key` and `semantics` are
optional. Empty arrays are valid. Binding `kind` is one of `storage_buffer`, `uniform_buffer`,
`storage_image`, or `sampled_image`; `access` is `read`, `write`, or `read_write`. Specialization
defaults are booleans or finite JSON numbers. Integer tokens are preserved exactly, including values
larger than 2^53, so callers can apply the declared type without a floating-point round trip.

## Invariants

- Only schema version 1 is accepted; unknown fields are rejected so misspellings cannot silently
  change meaning.
- Kernel, binding, push-field, specialization, fallback, and semantic names are bounded. Binding
  `(set, binding)` coordinates, names within a collection, specialization IDs, and semantic entries
  must be unique.
- Push-constant fields have a nonzero size, fit within `push_constant_size`, and may not overlap.
  The metadata format caps the declared block at 65,536 bytes. Checked-width arithmetic is used for
  all field-end calculations.
- Local-size dimensions must be nonzero and their product must fit in `uint32_t`. The manifest does
  not claim that a particular device supports those dimensions; reflection and pipeline creation
  apply the selected device's actual limits.
- Structurally valid access modes and resource kinds do not prove that a SPIR-V module agrees with
  the manifest. The reflection/validation stage is responsible for comparing the two.

## C++ API

`kora::KernelMetadata` and its binding, push-constant, and specialization-constant records can be
populated directly by applications. Call `validate_kernel_metadata(metadata)` before packaging or
passing a programmatically-created object to a later loading stage. `parse_kernel_metadata(text)`
performs strict JSON parsing and returns the same validated model. Both APIs return `Status` or
`Result<KernelMetadata>` and report expected input failures as `InvalidKernel`; neither throws for
invalid input.

The public declarations are in `kora/kernel_metadata.hpp`. The schema is intentionally limited to
portable metadata; it does not encode device IDs, compiled pipelines, or vendor-specific fields.

### Function reference

- `validate_kernel_metadata(const KernelMetadata&)` validates an already constructed value without
  mutating it. It returns an empty `Status` on success, or `InvalidKernel` with a field path on the
  first detected invariant violation.
- `parse_kernel_metadata(std::string_view)` parses one complete version-1 JSON document, converts
  each schema field, and runs the same model validation before returning `Result<KernelMetadata>`.
  The input is not retained. Syntax/limit failures report the JSON byte offset; schema/model failures
  report a field path.

The JSON parser's configurable internal `Limits` defaults are 1 MiB per document, depth 32, 64 KiB
per decoded string, and 100,000 values. `parse_kernel_metadata` uses those defaults. Applications
that need a smaller package limit should check the input byte length before calling it; the public
metadata API deliberately does not expose internal parser policy knobs.
