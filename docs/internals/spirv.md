# SPIR-V validation and reflection

This page describes the internal checks used when a packaged compute kernel's SPIR-V module is
matched with its `.kora.json` metadata. It documents the current internal implementation, not a
stable public module-loading API.

## Validation boundary

`detail::spirv::validate_and_reflect` is the single internal entry point. It validates the metadata
model first, applies bounded checks to the SPIR-V header and instruction stream, optionally asks
SPIRV-Tools to validate the module for the selected Vulkan API version, reflects the supported
interface subset, and compares each reflected field with metadata. Failures return
`ErrorCode::InvalidKernel` with a field- or instruction-oriented explanation.

The parser always checks the five-word header, SPIR-V magic and byte order, supported version-word
shape, nonzero ID bound, reserved schema word, and every instruction word count. Modules larger
than 4,194,304 words (16 MiB) are rejected before parsing. When SPIRV-Tools is available, KORA also
uses `spvtools::SpirvTools::Validate` with the Vulkan environment derived from the device API
version and module version. SPIRV-Tools is optional; the bounded parser and interface checks remain
available without it.

## Reflected subset and invariants

| SPIR-V construct | KORA interpretation | Invariant |
| --- | --- | --- |
| `OpEntryPoint` | Named compute entry point | The requested name must exist and use `GLCompute`. |
| `OpExecutionMode LocalSize` | Three workgroup dimensions | All dimensions must be present, nonzero, literal, and equal metadata. `LocalSizeId` is not part of metadata schema version 1. |
| `OpVariable` plus `DescriptorSet` / `Binding` | Descriptor location, kind, name, and conservative access | Both decorations are required together; locations must be unique; reflected bindings must match metadata. |
| Push-constant variable and struct member `Offset` decorations | Field type, offset, extent, and total block extent | One struct block is supported; each field needs an offset; extent is the highest member end. |
| `SpecId` on scalar specialization constants | ID, scalar type, name, and default value | Supported values are bool, 32/64-bit integer, and 32/64-bit float; version 1 does not accept `OpSpecConstantOp`. |
| `OpCapability` | Declared module requirements | Unknown requirements fail closed; integer, double, and subgroup requirements are checked against the capability snapshot. |

Descriptor variables in `StorageBuffer`, `Uniform`, and supported `UniformConstant` forms are
recognized. Runtime storage arrays are valid as buffer payloads but do not have a fixed standalone
byte size. The reflected access mode uses `NonWritable` and `NonReadable` decorations when present;
absence of such proof is treated as read/write. This is deliberately conservative and does not
attempt whole-program use/def analysis.

Push-constant extent uses explicit member offsets and scalar/vector/array/matrix sizes. Array and
matrix strides are honored when decorated. Unsupported or recursive types that cannot be sized
reliably are rejected rather than guessed. `LocalSizeId`, non-scalar specialization values,
specialization operations, sampler-only descriptors, and unsupported descriptor classes are not
silently coerced into version-1 metadata.

## Function reference

- `capabilities_from`: copies the immutable `DeviceInfo` API, scalar shader-feature, and subgroup
  snapshots needed to select the SPIRV-Tools environment and evaluate capabilities.
- `validate_and_reflect`: checks metadata and module bounds, invokes the optional validator, then
  delegates reflection and consistency checks to `ModuleParser`.
- `ModuleParser::validate_header`: enforces the fixed header and module-size bounds before reading
  instructions.
- `ModuleParser::parse_instructions`: walks instructions only after checking each declared word
  count; it collects entry points, execution modes, names, decorations, type declarations,
  variables, and scalar constants.
- `ModuleParser::reflect_bindings`: maps decorated variables to supported descriptor kinds, checks
  duplicate set/binding pairs, and sorts the result for deterministic comparison.
- `ModuleParser::reflect_push_constants`: derives field extents from layout decorations and
  computes the block end with checked arithmetic.
- `ModuleParser::reflect_specialization_constants`: converts literal scalar defaults to the
  metadata value model and rejects unsupported operations/types.
- `ModuleParser::validate_capabilities`: rejects requirements that the device snapshot cannot
  establish as supported.
- `ModuleParser::compare_metadata`: reports the first mismatch with a stable metadata field path.

## Tests and regeneration

`tests/unit/spirv_reflection.comp` is compiled to SPIR-V at build time when `glslangValidator` is
installed. The fixture exercises a compute entry point, local size, storage buffer, push constant,
and scalar specialization constant. The unit tests also cover truncated instructions, invalid
magic, and precise metadata mismatch reporting. On Linux CI, `glslang-tools` and SPIRV-Tools are
installed so both the fixture and full Vulkan-environment validation run. On other platforms the
fixture is conditional on `glslangValidator`; structural negative tests remain unconditional.
