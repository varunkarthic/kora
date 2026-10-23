//! Structural SPIR-V validation and compute-interface reflection.
//!
//! The parser walks the module once, records the declarations that describe a compute interface
//! (entry points, execution modes, decorations, types, variables, and constants), and compares the
//! reflected interface with the kernel's version-1 metadata. It is deliberately not a full SPIR-V
//! validator: it rejects malformed instruction streams and interface mismatches so a bad module
//! fails with a precise [`ErrorCode::InvalidKernel`] instead of reaching the driver.

use std::collections::{HashMap, HashSet};

use crate::device::DeviceInfo;
use crate::error::{Error, ErrorCode, Result, Status};
use crate::metadata::{
    KernelAccess, KernelBindingKind, KernelMetadata, PushConstantFieldMetadata,
    SpecializationConstantMetadata, SpecializationValue, validate_kernel_metadata,
};

const MAGIC: u32 = 0x0723_0203;
const HEADER_WORDS: usize = 5;
const MAXIMUM_MODULE_WORDS: usize = 4 * 1024 * 1024;
const MAXIMUM_TYPE_DEPTH: usize = 32;

mod op {
    pub const NAME: u32 = 5;
    pub const MEMBER_NAME: u32 = 6;
    pub const ENTRY_POINT: u32 = 15;
    pub const EXECUTION_MODE: u32 = 16;
    pub const CAPABILITY: u32 = 17;
    pub const TYPE_VOID: u32 = 19;
    pub const TYPE_BOOL: u32 = 20;
    pub const TYPE_INT: u32 = 21;
    pub const TYPE_FLOAT: u32 = 22;
    pub const TYPE_VECTOR: u32 = 23;
    pub const TYPE_MATRIX: u32 = 24;
    pub const TYPE_IMAGE: u32 = 25;
    pub const TYPE_SAMPLER: u32 = 26;
    pub const TYPE_SAMPLED_IMAGE: u32 = 27;
    pub const TYPE_ARRAY: u32 = 28;
    pub const TYPE_RUNTIME_ARRAY: u32 = 29;
    pub const TYPE_STRUCT: u32 = 30;
    pub const TYPE_POINTER: u32 = 32;
    pub const TYPE_FORWARD_POINTER: u32 = 39;
    pub const CONSTANT: u32 = 43;
    pub const CONSTANT_NULL: u32 = 46;
    pub const SPEC_CONSTANT_TRUE: u32 = 48;
    pub const SPEC_CONSTANT_FALSE: u32 = 49;
    pub const SPEC_CONSTANT: u32 = 50;
    pub const SPEC_CONSTANT_OP: u32 = 52;
    pub const VARIABLE: u32 = 59;
    pub const DECORATE: u32 = 71;
    pub const MEMBER_DECORATE: u32 = 72;
    pub const EXECUTION_MODE_ID: u32 = 331;
}

mod decoration {
    pub const SPEC_ID: u32 = 1;
    pub const BLOCK: u32 = 2;
    pub const BUFFER_BLOCK: u32 = 3;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const MATRIX_STRIDE: u32 = 7;
    pub const NON_WRITABLE: u32 = 24;
    pub const NON_READABLE: u32 = 25;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
}

mod storage {
    pub const UNIFORM_CONSTANT: u32 = 0;
    pub const UNIFORM: u32 = 2;
    pub const PUSH_CONSTANT: u32 = 9;
    pub const STORAGE_BUFFER: u32 = 12;
}

const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;
const EXECUTION_MODE_LOCAL_SIZE_ID: u32 = 38;

/// SPIR-V capability numbers this module recognizes.
pub mod capability {
    /// `Matrix`.
    pub const MATRIX: u32 = 0;
    /// `Shader`.
    pub const SHADER: u32 = 1;
    /// `Float64`.
    pub const FLOAT64: u32 = 10;
    /// `Int64`.
    pub const INT64: u32 = 11;
    /// `Int16`.
    pub const INT16: u32 = 22;
    /// `GroupNonUniform`.
    pub const GROUP_NON_UNIFORM: u32 = 61;
    /// `GroupNonUniformVote`.
    pub const GROUP_NON_UNIFORM_VOTE: u32 = 62;
    /// `GroupNonUniformArithmetic`.
    pub const GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
    /// `GroupNonUniformBallot`.
    pub const GROUP_NON_UNIFORM_BALLOT: u32 = 64;
    /// `GroupNonUniformShuffle`.
    pub const GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;
    /// `GroupNonUniformShuffleRelative`.
    pub const GROUP_NON_UNIFORM_SHUFFLE_RELATIVE: u32 = 66;
    /// `GroupNonUniformClustered`.
    pub const GROUP_NON_UNIFORM_CLUSTERED: u32 = 67;
    /// `GroupNonUniformQuad`.
    pub const GROUP_NON_UNIFORM_QUAD: u32 = 68;
}

/// Device features a module's capabilities are checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCapabilities {
    /// Vulkan API major version.
    pub api_major: u32,
    /// Vulkan API minor version.
    pub api_minor: u32,
    /// `shaderInt16`.
    pub shader_int16: bool,
    /// `shaderInt64`.
    pub shader_int64: bool,
    /// `shaderFloat64`.
    pub shader_float64: bool,
    /// Compute-stage subgroup operations are available at all.
    pub subgroup_supported: bool,
    /// Basic subgroup operations.
    pub subgroup_basic: bool,
    /// Subgroup vote.
    pub subgroup_vote: bool,
    /// Subgroup arithmetic.
    pub subgroup_arithmetic: bool,
    /// Subgroup ballot.
    pub subgroup_ballot: bool,
    /// Subgroup shuffle.
    pub subgroup_shuffle: bool,
    /// Relative subgroup shuffle.
    pub subgroup_shuffle_relative: bool,
    /// Clustered subgroup operations.
    pub subgroup_clustered: bool,
    /// Quad subgroup operations.
    pub subgroup_quad: bool,
    /// `LocalSizeId` execution modes are allowed (`maintenance4`).
    pub local_size_id: bool,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            api_major: 1,
            api_minor: 0,
            shader_int16: false,
            shader_int64: false,
            shader_float64: false,
            subgroup_supported: false,
            subgroup_basic: false,
            subgroup_vote: false,
            subgroup_arithmetic: false,
            subgroup_ballot: false,
            subgroup_shuffle: false,
            subgroup_shuffle_relative: false,
            subgroup_clustered: false,
            subgroup_quad: false,
            local_size_id: false,
        }
    }
}

impl DeviceCapabilities {
    /// Capabilities of a queried device.
    pub fn from_device(device: &DeviceInfo) -> Self {
        let f = &device.features;
        let s = &device.subgroup;
        Self {
            api_major: device.api_version.major,
            api_minor: device.api_version.minor,
            shader_int16: f.shader_int16,
            shader_int64: f.shader_int64,
            shader_float64: f.shader_float64,
            subgroup_supported: s.supported,
            subgroup_basic: s.basic,
            subgroup_vote: s.vote,
            subgroup_arithmetic: s.arithmetic,
            subgroup_ballot: s.ballot,
            subgroup_shuffle: s.shuffle,
            subgroup_shuffle_relative: s.shuffle_relative,
            subgroup_clustered: s.clustered,
            subgroup_quad: s.quad,
            local_size_id: f.maintenance4,
        }
    }

    fn allows(&self, id: u32) -> bool {
        use self::capability as c;
        match id {
            c::MATRIX | c::SHADER => true,
            c::INT16 => self.shader_int16,
            c::INT64 => self.shader_int64,
            c::FLOAT64 => self.shader_float64,
            c::GROUP_NON_UNIFORM => self.subgroup_supported && self.subgroup_basic,
            c::GROUP_NON_UNIFORM_VOTE => self.subgroup_vote,
            c::GROUP_NON_UNIFORM_ARITHMETIC => self.subgroup_arithmetic,
            c::GROUP_NON_UNIFORM_BALLOT => self.subgroup_ballot,
            c::GROUP_NON_UNIFORM_SHUFFLE => self.subgroup_shuffle,
            c::GROUP_NON_UNIFORM_SHUFFLE_RELATIVE => self.subgroup_shuffle_relative,
            c::GROUP_NON_UNIFORM_CLUSTERED => self.subgroup_clustered,
            c::GROUP_NON_UNIFORM_QUAD => self.subgroup_quad,
            _ => false,
        }
    }
}

/// One reflected descriptor binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReflectedBinding {
    /// Descriptor set.
    pub set: u32,
    /// Binding number.
    pub binding: u32,
    /// Debug name of the variable, or empty.
    pub name: String,
    /// Descriptor class.
    pub kind: KernelBindingKind,
    /// Access implied by `NonWritable`/`NonReadable` decorations.
    pub access: KernelAccess,
}

/// The reflected compute interface of one entry point.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Reflection {
    /// Entry-point name.
    pub entry_point: String,
    /// Workgroup size (the specialization defaults for `LocalSizeId`).
    pub local_size: [u32; 3],
    /// The workgroup size comes from `LocalSizeId` specialization constants.
    pub local_size_id_based: bool,
    /// Result ids named by `LocalSizeId`.
    pub local_size_ids: [u32; 3],
    /// `SpecId`s of the `LocalSizeId` constants.
    pub local_size_spec_ids: [u32; 3],
    /// Default values of the `LocalSizeId` constants.
    pub local_size_spec_defaults: [u32; 3],
    /// Descriptor bindings sorted by `(set, binding)`.
    pub bindings: Vec<ReflectedBinding>,
    /// Extent of the push-constant block in bytes.
    pub push_constant_size: u32,
    /// Push-constant block members in declaration order.
    pub push_constants: Vec<PushConstantFieldMetadata>,
    /// Scalar specialization constants sorted by `SpecId`.
    pub specialization_constants: Vec<SpecializationConstantMetadata>,
    /// Declared capabilities in declaration order.
    pub capabilities: Vec<u32>,
}

fn invalid(message: impl AsRef<str>) -> Error {
    Error::new(ErrorCode::InvalidKernel, format!("SPIR-V: {}", message.as_ref()))
}

const fn member_key(type_id: u32, member: u32) -> u64 {
    ((type_id as u64) << 32) | member as u64
}

#[derive(Debug, Default)]
struct Decorations {
    set: Option<u32>,
    binding: Option<u32>,
    spec_id: Option<u32>,
    array_stride: Option<u32>,
    block: bool,
    buffer_block: bool,
    non_writable: bool,
    non_readable: bool,
}

#[derive(Debug, Default, Clone, Copy)]
struct MemberDecorations {
    offset: Option<u32>,
    matrix_stride: Option<u32>,
    non_writable: bool,
    non_readable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypeKind {
    Bool,
    Integer,
    Float,
    Vector,
    Matrix,
    Array,
    RuntimeArray,
    Struct,
    Pointer,
    Image,
    Sampler,
    SampledImage,
}

#[derive(Debug)]
struct Type {
    kind: TypeKind,
    width: u32,
    signed: bool,
    element: u32,
    count: u32,
    storage_class: u32,
    sampled: u32,
    members: Vec<u32>,
}

impl Type {
    const fn new(kind: TypeKind) -> Self {
        Self {
            kind,
            width: 0,
            signed: false,
            element: 0,
            count: 0,
            storage_class: 0,
            sampled: 0,
            members: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct Variable {
    pointer_type: u32,
    id: u32,
    storage_class: u32,
}

#[derive(Debug)]
struct EntryPoint {
    model: u32,
    id: u32,
    name: String,
}

#[derive(Debug, Clone, Copy)]
struct ExecutionSize {
    dimensions: [u32; 3],
    id_based: bool,
}

#[derive(Debug)]
struct SpecConstant {
    type_id: u32,
    result_id: u32,
    bits: u64,
    boolean: Option<bool>,
}

/// One instruction's operands (the words after the opcode word).
#[derive(Clone, Copy)]
struct Operands<'a> {
    words: &'a [u32],
}

impl Operands<'_> {
    fn len(&self) -> usize {
        self.words.len()
    }
    fn get(&self, index: usize) -> u32 {
        self.words[index]
    }
    fn string(&self, first: usize) -> Result<String> {
        let mut bytes = Vec::new();
        for word in self.words.get(first..).unwrap_or_default() {
            for byte in word.to_le_bytes() {
                if byte == 0 {
                    return Ok(String::from_utf8_lossy(&bytes).into_owned());
                }
                bytes.push(byte);
            }
        }
        Err(invalid("instruction contains an unterminated literal string"))
    }
}

#[derive(Debug, Default)]
struct Module {
    id_bound: u32,
    types: HashMap<u32, Type>,
    decorations: HashMap<u32, Decorations>,
    member_decorations: HashMap<u64, MemberDecorations>,
    names: HashMap<u32, String>,
    member_names: HashMap<u64, String>,
    constants: HashMap<u32, u64>,
    spec_op_ids: HashSet<u32>,
    capabilities: Vec<u32>,
    variables: Vec<Variable>,
    entry_points: Vec<EntryPoint>,
    execution_sizes: HashMap<u32, ExecutionSize>,
    spec_constants: Vec<SpecConstant>,
}

fn check_bounds(words: &[u32]) -> Status {
    if words.len() < HEADER_WORDS {
        return Err(invalid("module is shorter than the five-word header"));
    }
    if words.len() > MAXIMUM_MODULE_WORDS {
        return Err(invalid("module exceeds the 16 MiB word limit"));
    }
    if words[0] != MAGIC {
        return Err(invalid("invalid magic number or byte order"));
    }
    Ok(())
}

fn validate_header(words: &[u32]) -> Status {
    check_bounds(words)?;
    let version = words[1];
    let major = (version >> 16) & 0xff;
    let minor = (version >> 8) & 0xff;
    if version & 0xff00_00ff != 0 || major != 1 || minor > 6 {
        return Err(invalid("unsupported or malformed SPIR-V version word"));
    }
    if words[3] == 0 {
        return Err(invalid("id bound must be nonzero"));
    }
    if words[4] != 0 {
        return Err(invalid("reserved schema word must be zero"));
    }
    Ok(())
}

impl Module {
    fn parse(words: &[u32]) -> Result<Self> {
        validate_header(words)?;
        let mut module = Self {
            id_bound: words[3],
            ..Self::default()
        };
        let mut seen_capabilities = HashSet::new();
        let mut cursor = HEADER_WORDS;
        while cursor < words.len() {
            let instruction = words[cursor];
            let word_count = (instruction >> 16) as usize;
            let opcode = instruction & 0xffff;
            if word_count == 0 || word_count > words.len() - cursor {
                return Err(invalid(format!(
                    "instruction at word {cursor} has an invalid word count"
                )));
            }
            let o = Operands {
                words: &words[cursor + 1..cursor + word_count],
            };
            module.instruction(opcode, o, &mut seen_capabilities)?;
            cursor += word_count;
        }
        Ok(module)
    }

    fn valid_id(&self, id: u32) -> bool {
        id != 0 && id < self.id_bound
    }

    fn instruction(&mut self, opcode: u32, o: Operands<'_>, seen: &mut HashSet<u32>) -> Status {
        let n = o.len();
        match opcode {
            op::CAPABILITY => {
                if n != 1 {
                    return Err(invalid("OpCapability has an invalid word count"));
                }
                if seen.insert(o.get(0)) {
                    self.capabilities.push(o.get(0));
                }
            }
            op::ENTRY_POINT => {
                if n < 3 {
                    return Err(invalid("OpEntryPoint is truncated"));
                }
                if !self.valid_id(o.get(1)) {
                    return Err(invalid("OpEntryPoint uses an invalid function id"));
                }
                self.entry_points.push(EntryPoint {
                    model: o.get(0),
                    id: o.get(1),
                    name: o.string(2)?,
                });
            }
            op::EXECUTION_MODE | op::EXECUTION_MODE_ID => {
                if n < 2 {
                    return Err(invalid("execution-mode instruction is truncated"));
                }
                if !self.valid_id(o.get(0)) {
                    return Err(invalid("execution mode uses an invalid entry-point id"));
                }
                if matches!(o.get(1), EXECUTION_MODE_LOCAL_SIZE | EXECUTION_MODE_LOCAL_SIZE_ID) {
                    if n != 5 {
                        return Err(invalid("LocalSize execution mode must contain three dimensions"));
                    }
                    let size = ExecutionSize {
                        dimensions: [o.get(2), o.get(3), o.get(4)],
                        id_based: opcode == op::EXECUTION_MODE_ID,
                    };
                    if self.execution_sizes.insert(o.get(0), size).is_some() {
                        return Err(invalid("entry point has duplicate LocalSize execution modes"));
                    }
                }
            }
            op::NAME => {
                if n < 2 || !self.valid_id(o.get(0)) {
                    return Err(invalid("OpName is malformed"));
                }
                self.names.insert(o.get(0), o.string(1)?);
            }
            op::MEMBER_NAME => {
                if n < 3 || !self.valid_id(o.get(0)) {
                    return Err(invalid("OpMemberName is malformed"));
                }
                self.member_names
                    .insert(member_key(o.get(0), o.get(1)), o.string(2)?);
            }
            op::DECORATE => {
                if n < 2 || !self.valid_id(o.get(0)) {
                    return Err(invalid("OpDecorate is malformed"));
                }
                self.decorate(o)?;
            }
            op::MEMBER_DECORATE => {
                if n < 3 || !self.valid_id(o.get(0)) {
                    return Err(invalid("OpMemberDecorate is malformed"));
                }
                self.decorate_member(o)?;
            }
            op::TYPE_VOID..=op::TYPE_FORWARD_POINTER => self.declare_type(opcode, o)?,
            op::VARIABLE => {
                if !(3..=4).contains(&n) || !self.valid_id(o.get(1)) {
                    return Err(invalid("OpVariable is malformed"));
                }
                self.variables.push(Variable {
                    pointer_type: o.get(0),
                    id: o.get(1),
                    storage_class: o.get(2),
                });
            }
            op::CONSTANT
            | op::CONSTANT_NULL
            | op::SPEC_CONSTANT
            | op::SPEC_CONSTANT_TRUE
            | op::SPEC_CONSTANT_FALSE => self.declare_constant(opcode, o)?,
            op::SPEC_CONSTANT_OP => {
                if n < 3 || !self.valid_id(o.get(1)) {
                    return Err(invalid("OpSpecConstantOp is malformed"));
                }
                self.spec_op_ids.insert(o.get(1));
            }
            _ => {}
        }
        Ok(())
    }

    fn decorate(&mut self, o: Operands<'_>) -> Status {
        let target = o.get(0);
        let d = self.decorations.entry(target).or_default();
        let set = |slot: &mut Option<u32>| -> Status {
            if o.len() != 3 {
                return Err(invalid("decoration requires one literal operand"));
            }
            if slot.is_some() {
                return Err(invalid(format!("duplicate decoration on id {target}")));
            }
            *slot = Some(o.get(2));
            Ok(())
        };
        match o.get(1) {
            decoration::SPEC_ID => set(&mut d.spec_id)?,
            decoration::DESCRIPTOR_SET => set(&mut d.set)?,
            decoration::BINDING => set(&mut d.binding)?,
            decoration::ARRAY_STRIDE => set(&mut d.array_stride)?,
            decoration::BLOCK => {
                if o.len() != 2 {
                    return Err(invalid("Block decoration has unexpected operands"));
                }
                d.block = true;
            }
            decoration::BUFFER_BLOCK => {
                if o.len() != 2 {
                    return Err(invalid("BufferBlock decoration has unexpected operands"));
                }
                d.buffer_block = true;
            }
            decoration::NON_WRITABLE => d.non_writable = true,
            decoration::NON_READABLE => d.non_readable = true,
            _ => {}
        }
        Ok(())
    }

    fn decorate_member(&mut self, o: Operands<'_>) -> Status {
        let d = self
            .member_decorations
            .entry(member_key(o.get(0), o.get(1)))
            .or_default();
        let set = |slot: &mut Option<u32>| -> Status {
            if o.len() != 4 {
                return Err(invalid("member layout decoration requires one literal operand"));
            }
            if slot.is_some() {
                return Err(invalid("duplicate member layout decoration"));
            }
            *slot = Some(o.get(3));
            Ok(())
        };
        match o.get(2) {
            decoration::OFFSET => set(&mut d.offset)?,
            decoration::MATRIX_STRIDE => set(&mut d.matrix_stride)?,
            decoration::NON_WRITABLE => d.non_writable = true,
            decoration::NON_READABLE => d.non_readable = true,
            // Row-major layout (and every other decoration) does not change the reflected extent.
            _ => {}
        }
        Ok(())
    }

    fn declare_type(&mut self, opcode: u32, o: Operands<'_>) -> Status {
        let n = o.len();
        if n < 1 {
            return Err(invalid("type declaration is truncated"));
        }
        let result = o.get(0);
        if !self.valid_id(result) {
            return Err(invalid("type declaration uses an invalid result id"));
        }
        let expect = |count: usize, what: &str| -> Status {
            if n == count {
                Ok(())
            } else {
                Err(invalid(format!("{what} is malformed")))
            }
        };
        let ty = match opcode {
            op::TYPE_BOOL => {
                expect(1, "OpTypeBool")?;
                Type::new(TypeKind::Bool)
            }
            op::TYPE_INT => {
                expect(3, "OpTypeInt")?;
                Type {
                    width: o.get(1),
                    signed: o.get(2) != 0,
                    ..Type::new(TypeKind::Integer)
                }
            }
            op::TYPE_FLOAT => {
                expect(2, "OpTypeFloat")?;
                Type {
                    width: o.get(1),
                    ..Type::new(TypeKind::Float)
                }
            }
            op::TYPE_VECTOR | op::TYPE_MATRIX | op::TYPE_ARRAY => {
                let (kind, what) = match opcode {
                    op::TYPE_VECTOR => (TypeKind::Vector, "OpTypeVector"),
                    op::TYPE_MATRIX => (TypeKind::Matrix, "OpTypeMatrix"),
                    _ => (TypeKind::Array, "OpTypeArray"),
                };
                expect(3, what)?;
                Type {
                    element: o.get(1),
                    count: o.get(2),
                    ..Type::new(kind)
                }
            }
            op::TYPE_RUNTIME_ARRAY => {
                expect(2, "OpTypeRuntimeArray")?;
                Type {
                    element: o.get(1),
                    ..Type::new(TypeKind::RuntimeArray)
                }
            }
            op::TYPE_STRUCT => Type {
                members: o.words[1..].to_vec(),
                ..Type::new(TypeKind::Struct)
            },
            op::TYPE_POINTER => {
                expect(3, "OpTypePointer")?;
                Type {
                    storage_class: o.get(1),
                    element: o.get(2),
                    ..Type::new(TypeKind::Pointer)
                }
            }
            op::TYPE_IMAGE => {
                if !(8..=9).contains(&n) {
                    return Err(invalid("OpTypeImage is malformed"));
                }
                Type {
                    element: o.get(1),
                    sampled: o.get(6),
                    ..Type::new(TypeKind::Image)
                }
            }
            op::TYPE_SAMPLER => {
                expect(1, "OpTypeSampler")?;
                Type::new(TypeKind::Sampler)
            }
            op::TYPE_SAMPLED_IMAGE => {
                expect(2, "OpTypeSampledImage")?;
                Type {
                    element: o.get(1),
                    ..Type::new(TypeKind::SampledImage)
                }
            }
            _ => return Ok(()),
        };
        if self.types.insert(result, ty).is_some() {
            return Err(invalid("duplicate type result id"));
        }
        Ok(())
    }

    fn declare_constant(&mut self, opcode: u32, o: Operands<'_>) -> Status {
        let n = o.len();
        if n < 2 {
            return Err(invalid("constant instruction is truncated"));
        }
        let (type_id, result_id) = (o.get(0), o.get(1));
        if !self.valid_id(type_id) || !self.valid_id(result_id) {
            return Err(invalid("constant uses an invalid id"));
        }
        let mut bits = 0_u64;
        let mut boolean = None;
        match opcode {
            op::CONSTANT_NULL => {
                if n != 2 {
                    return Err(invalid("OpConstantNull is malformed"));
                }
            }
            op::SPEC_CONSTANT_TRUE | op::SPEC_CONSTANT_FALSE => {
                if n != 2 {
                    return Err(invalid("boolean specialization constant is malformed"));
                }
                boolean = Some(opcode == op::SPEC_CONSTANT_TRUE);
            }
            _ => {
                if !(3..=4).contains(&n) {
                    return Err(invalid("scalar constant has an unsupported literal width"));
                }
                bits = u64::from(o.get(2));
                if n == 4 {
                    bits |= u64::from(o.get(3)) << 32;
                }
            }
        }
        self.constants.insert(result_id, bits);
        if matches!(
            opcode,
            op::SPEC_CONSTANT | op::SPEC_CONSTANT_TRUE | op::SPEC_CONSTANT_FALSE
        ) {
            self.spec_constants.push(SpecConstant {
                type_id,
                result_id,
                bits,
                boolean,
            });
        }
        Ok(())
    }

    fn ty(&self, id: u32) -> Result<&Type> {
        self.types
            .get(&id)
            .ok_or_else(|| invalid(format!("type id {id} is undeclared")))
    }

    fn type_size(&self, id: u32, layout: Option<MemberDecorations>, depth: usize) -> Result<u64> {
        if depth > MAXIMUM_TYPE_DEPTH {
            return Err(invalid("type nesting exceeds 32 levels"));
        }
        let ty = self.ty(id)?;
        let mul = |a: u64, b: u64| {
            a.checked_mul(b)
                .ok_or_else(|| invalid("type size multiplication overflows uint64"))
        };
        match ty.kind {
            TypeKind::Bool => Ok(4),
            TypeKind::Integer | TypeKind::Float => {
                if ty.width == 0 || ty.width % 8 != 0 || ty.width > 64 {
                    return Err(invalid(format!("unsupported scalar width in type id {id}")));
                }
                Ok(u64::from(ty.width / 8))
            }
            TypeKind::Vector => mul(self.type_size(ty.element, None, depth + 1)?, u64::from(ty.count)),
            TypeKind::Matrix => {
                let column = self.type_size(ty.element, None, depth + 1)?;
                let stride = layout.and_then(|l| l.matrix_stride).map_or(column, u64::from);
                mul(stride, u64::from(ty.count))
            }
            TypeKind::Array => {
                let length = *self
                    .constants
                    .get(&ty.count)
                    .ok_or_else(|| invalid("array length is not a literal constant"))?;
                let element = self.type_size(ty.element, None, depth + 1)?;
                let stride = self
                    .decorations
                    .get(&id)
                    .and_then(|d| d.array_stride)
                    .map_or(element, u64::from);
                mul(stride, length)
            }
            TypeKind::Struct => {
                let mut end = 0_u64;
                for (member, &member_type) in (0_u32..).zip(&ty.members) {
                    let d = self
                        .member_decorations
                        .get(&member_key(id, member))
                        .filter(|d| d.offset.is_some())
                        .ok_or_else(|| {
                            invalid("push-constant structure member lacks an Offset decoration")
                        })?;
                    let size = self.type_size(member_type, Some(*d), depth + 1)?;
                    let member_end = u64::from(d.offset.unwrap_or_default())
                        .checked_add(size)
                        .ok_or_else(|| invalid("type size addition overflows uint64"))?;
                    end = end.max(member_end);
                }
                Ok(end)
            }
            TypeKind::RuntimeArray
            | TypeKind::Pointer
            | TypeKind::Image
            | TypeKind::Sampler
            | TypeKind::SampledImage => Err(invalid("type has no fixed byte size")),
        }
    }

    fn type_name(&self, id: u32, depth: usize) -> Result<String> {
        if depth > MAXIMUM_TYPE_DEPTH {
            return Err(invalid("type nesting exceeds 32 levels"));
        }
        let ty = self.ty(id)?;
        Ok(match ty.kind {
            TypeKind::Bool => "bool".into(),
            TypeKind::Integer => format!("{}{}", if ty.signed { 'i' } else { 'u' }, ty.width),
            TypeKind::Float => format!("f{}", ty.width),
            TypeKind::Vector => format!("vec{}<{}>", ty.count, self.type_name(ty.element, depth + 1)?),
            TypeKind::Matrix => {
                let column = self
                    .types
                    .get(&ty.element)
                    .filter(|c| c.kind == TypeKind::Vector)
                    .ok_or_else(|| invalid("matrix column type is invalid"))?;
                format!(
                    "mat{}x{}<{}>",
                    ty.count,
                    column.count,
                    self.type_name(column.element, depth + 1)?
                )
            }
            TypeKind::Array => {
                let length = self
                    .constants
                    .get(&ty.count)
                    .ok_or_else(|| invalid("array length is not a literal constant"))?;
                format!("array[{length}]<{}>", self.type_name(ty.element, depth + 1)?)
            }
            TypeKind::RuntimeArray => return Err(invalid("runtime arrays are not valid metadata fields")),
            TypeKind::Struct => "struct".into(),
            TypeKind::Pointer => return self.type_name(ty.element, depth + 1),
            TypeKind::Image if ty.sampled == 2 => "storage_image".into(),
            TypeKind::Image | TypeKind::SampledImage => "sampled_image".into(),
            TypeKind::Sampler => "sampler".into(),
        })
    }

    fn pointee(&self, variable: &Variable) -> Result<u32> {
        let ty = self
            .types
            .get(&variable.pointer_type)
            .filter(|t| t.kind == TypeKind::Pointer)
            .ok_or_else(|| invalid("variable result type is not an OpTypePointer"))?;
        if ty.storage_class != variable.storage_class {
            return Err(invalid("variable storage class does not match its pointer type"));
        }
        Ok(ty.element)
    }

    fn binding_kind(&self, variable: &Variable, pointee: u32) -> Result<KernelBindingKind> {
        match variable.storage_class {
            storage::STORAGE_BUFFER => Ok(KernelBindingKind::StorageBuffer),
            storage::UNIFORM => match self.decorations.get(&pointee) {
                Some(d) if d.buffer_block => Ok(KernelBindingKind::StorageBuffer),
                Some(d) if d.block => Ok(KernelBindingKind::UniformBuffer),
                _ => Err(invalid(
                    "Uniform descriptor type lacks Block or BufferBlock decoration",
                )),
            },
            storage::UNIFORM_CONSTANT => {
                let ty = self
                    .types
                    .get(&pointee)
                    .ok_or_else(|| invalid("descriptor type is undeclared"))?;
                match ty.kind {
                    TypeKind::Image if ty.sampled == 2 => Ok(KernelBindingKind::StorageImage),
                    TypeKind::Image | TypeKind::SampledImage => Ok(KernelBindingKind::SampledImage),
                    _ => Err(invalid(
                        "sampler-only or unsupported UniformConstant descriptor type",
                    )),
                }
            }
            _ => Err(invalid("descriptor uses an unsupported storage class")),
        }
    }

    fn binding_access(&self, variable: &Variable, pointee: u32) -> KernelAccess {
        let mut read = true;
        let mut write = true;
        for d in [self.decorations.get(&variable.id), self.decorations.get(&pointee)]
            .into_iter()
            .flatten()
        {
            write &= !d.non_writable;
            read &= !d.non_readable;
        }
        if let Some(ty) = self.types.get(&pointee)
            && ty.kind == TypeKind::Struct
            && !ty.members.is_empty()
        {
            let members = || {
                (0..ty.members.len()).map(|i| {
                    self.member_decorations
                        .get(&member_key(pointee, u32::try_from(i).unwrap_or(u32::MAX)))
                })
            };
            if members().all(|d| d.is_some_and(|d| d.non_writable)) {
                write = false;
            }
            if members().all(|d| d.is_some_and(|d| d.non_readable)) {
                read = false;
            }
        }
        match (read, write) {
            (true, false) => KernelAccess::Read,
            (false, true) => KernelAccess::Write,
            _ => KernelAccess::ReadWrite,
        }
    }

    fn reflect_bindings(&self) -> Result<Vec<ReflectedBinding>> {
        let mut result = Vec::new();
        let mut locations = HashSet::new();
        for variable in &self.variables {
            let Some(d) = self.decorations.get(&variable.id) else {
                continue;
            };
            let (set, binding) = match (d.set, d.binding) {
                (None, None) => continue,
                (Some(set), Some(binding)) => (set, binding),
                _ => {
                    return Err(invalid(
                        "descriptor variable must have both DescriptorSet and Binding decorations",
                    ));
                }
            };
            let pointee = self.pointee(variable)?;
            let kind = self.binding_kind(variable, pointee)?;
            if !locations.insert((set, binding)) {
                return Err(invalid("duplicate descriptor set/binding in module"));
            }
            let access = match kind {
                KernelBindingKind::StorageBuffer | KernelBindingKind::StorageImage => {
                    self.binding_access(variable, pointee)
                }
                KernelBindingKind::UniformBuffer | KernelBindingKind::SampledImage => KernelAccess::Read,
            };
            result.push(ReflectedBinding {
                set,
                binding,
                name: self.names.get(&variable.id).cloned().unwrap_or_default(),
                kind,
                access,
            });
        }
        result.sort_by_key(|b| (b.set, b.binding));
        Ok(result)
    }

    fn reflect_push_constants(&self) -> Result<(u32, Vec<PushConstantFieldMetadata>)> {
        let mut block = None;
        for variable in &self.variables {
            if variable.storage_class != storage::PUSH_CONSTANT {
                continue;
            }
            if block.is_some() {
                return Err(invalid("module declares more than one push-constant block"));
            }
            block = Some(self.pointee(variable)?);
        }
        let Some(block) = block else {
            return Ok((0, Vec::new()));
        };
        let ty = self
            .types
            .get(&block)
            .filter(|t| t.kind == TypeKind::Struct)
            .ok_or_else(|| invalid("push-constant variable does not point to a structure"))?;
        let total = u32::try_from(self.type_size(block, None, 0)?)
            .map_err(|_| invalid("push-constant block exceeds uint32 range"))?;
        let mut fields = Vec::with_capacity(ty.members.len());
        for (member, &member_type) in (0_u32..).zip(&ty.members) {
            let key = member_key(block, member);
            let d = self
                .member_decorations
                .get(&key)
                .filter(|d| d.offset.is_some())
                .ok_or_else(|| invalid("push-constant field is missing an Offset decoration"))?;
            let size = u32::try_from(self.type_size(member_type, Some(*d), 0)?)
                .map_err(|_| invalid("push-constant field exceeds uint32 range"))?;
            fields.push(PushConstantFieldMetadata {
                name: self.member_names.get(&key).cloned().unwrap_or_default(),
                type_name: self.type_name(member_type, 0)?,
                offset: d.offset.unwrap_or_default(),
                size,
            });
        }
        Ok((total, fields))
    }

    fn reflect_specialization_constants(&self) -> Result<Vec<SpecializationConstantMetadata>> {
        if self
            .spec_op_ids
            .iter()
            .any(|id| self.decorations.get(id).is_some_and(|d| d.spec_id.is_some()))
        {
            return Err(invalid(
                "OpSpecConstantOp with SpecId is not supported by metadata version 1",
            ));
        }
        let mut result = Vec::new();
        for constant in &self.spec_constants {
            let Some(spec_id) = self.decorations.get(&constant.result_id).and_then(|d| d.spec_id) else {
                continue;
            };
            let ty = self
                .types
                .get(&constant.type_id)
                .ok_or_else(|| invalid("specialization constant type is undeclared"))?;
            let (type_name, default_value) = spec_value(constant, ty)?;
            result.push(SpecializationConstantMetadata {
                id: spec_id,
                name: self.names.get(&constant.result_id).cloned().unwrap_or_default(),
                type_name,
                default_value,
            });
        }
        result.sort_by_key(|s| s.id);
        Ok(result)
    }

    fn reflect(
        &self,
        metadata: &KernelMetadata,
        device: &DeviceCapabilities,
        entry: &str,
    ) -> Result<Reflection> {
        if let Some(&unsupported) = self.capabilities.iter().find(|&&c| !device.allows(c)) {
            return Err(invalid(format!(
                "capability {unsupported} is unsupported or cannot be verified by the selected device"
            )));
        }
        let entry = self
            .entry_points
            .iter()
            .find(|e| e.name == entry)
            .ok_or_else(|| invalid(format!("entry point '{entry}' was not found")))?;
        if entry.model != EXECUTION_MODEL_GL_COMPUTE {
            return Err(invalid(format!(
                "entry point '{}' is not a compute entry point",
                entry.name
            )));
        }
        let size = self
            .execution_sizes
            .get(&entry.id)
            .ok_or_else(|| invalid("compute entry point has no LocalSize mode"))?;
        let mut reflection = Reflection {
            entry_point: entry.name.clone(),
            local_size: size.dimensions,
            local_size_id_based: size.id_based,
            capabilities: self.capabilities.clone(),
            ..Reflection::default()
        };
        if size.id_based && !device.local_size_id {
            return Err(invalid(
                "LocalSizeId requires the maintenance4 feature, which the device does not support",
            ));
        }
        if size.id_based {
            reflection.local_size_ids = size.dimensions;
            for dimension in 0..3 {
                let id = size.dimensions[dimension];
                let constant = self
                    .spec_constants
                    .iter()
                    .find(|c| c.result_id == id)
                    .ok_or_else(|| {
                        invalid("LocalSizeId must refer directly to a scalar specialization constant")
                    })?;
                let spec_id = self.decorations.get(&id).and_then(|d| d.spec_id);
                let unsigned32 = self
                    .types
                    .get(&constant.type_id)
                    .is_some_and(|t| t.kind == TypeKind::Integer && t.width == 32 && !t.signed);
                let (Some(spec_id), true, None) = (spec_id, unsigned32, constant.boolean) else {
                    return Err(invalid(
                        "LocalSizeId specialization constants must be unsigned 32-bit integers",
                    ));
                };
                // Truncation is exact: the type was checked to be a 32-bit integer.
                let default = (constant.bits & u64::from(u32::MAX)) as u32;
                reflection.local_size_spec_ids[dimension] = spec_id;
                reflection.local_size_spec_defaults[dimension] = default;
                reflection.local_size[dimension] = default;
            }
        }
        if reflection.local_size.contains(&0) {
            return Err(invalid("LocalSize dimensions must all be nonzero"));
        }
        reflection.bindings = self.reflect_bindings()?;
        (reflection.push_constant_size, reflection.push_constants) = self.reflect_push_constants()?;
        reflection.specialization_constants = self.reflect_specialization_constants()?;
        compare_metadata(&reflection, metadata)?;
        Ok(reflection)
    }
}

fn spec_value(constant: &SpecConstant, ty: &Type) -> Result<(String, SpecializationValue)> {
    if let Some(value) = constant.boolean {
        return Ok(("bool".into(), SpecializationValue::Bool(value)));
    }
    match ty.kind {
        TypeKind::Integer => {
            let width = ty.width;
            if width == 0 || width > 64 {
                return Err(invalid("specialization integer width is unsupported"));
            }
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1_u64 << width) - 1
            };
            let bits = constant.bits & mask;
            if ty.signed {
                // Sign-extend from `width` bits, then reinterpret.
                let shift = 64 - width;
                let value = ((bits << shift).cast_signed()) >> shift;
                Ok((format!("i{width}"), SpecializationValue::I64(value)))
            } else {
                Ok((format!("u{width}"), SpecializationValue::U64(bits)))
            }
        }
        TypeKind::Float => {
            let value = match ty.width {
                32 => f64::from(f32::from_bits((constant.bits & u64::from(u32::MAX)) as u32)),
                64 => f64::from_bits(constant.bits),
                _ => return Err(invalid("specialization floating-point width is unsupported")),
            };
            Ok((format!("f{}", ty.width), SpecializationValue::F64(value)))
        }
        _ => Err(invalid(
            "specialization constant is not a scalar bool, integer, or float",
        )),
    }
}

fn same_spec_value(
    reflected: &SpecializationConstantMetadata,
    expected: &SpecializationConstantMetadata,
) -> bool {
    match (
        reflected.type_name.as_str(),
        reflected.default_value,
        expected.default_value,
    ) {
        // The module stores an f32; compare against the metadata value rounded the same way.
        ("f32", SpecializationValue::F64(r), SpecializationValue::F64(e)) => r == f64::from(e as f32),
        _ => reflected.default_value == expected.default_value,
    }
}

fn names_agree(expected: &str, reflected: &str) -> bool {
    expected.is_empty() || reflected.is_empty() || expected == reflected
}

fn compare_metadata(reflection: &Reflection, metadata: &KernelMetadata) -> Status {
    if !reflection.local_size_id_based && reflection.local_size != metadata.local_size {
        return Err(invalid("local_size does not match the selected entry point"));
    }
    if reflection.bindings.len() != metadata.bindings.len() {
        return Err(invalid("bindings count does not match the module"));
    }
    for (i, expected) in metadata.bindings.iter().enumerate() {
        let path = format!("bindings[{i}]");
        let found = reflection
            .bindings
            .iter()
            .find(|b| b.set == expected.set && b.binding == expected.binding)
            .ok_or_else(|| invalid(format!("{path} location is absent from the module")))?;
        if found.kind != expected.kind {
            return Err(invalid(format!("{path}.kind does not match the module")));
        }
        if found.access != expected.access {
            return Err(invalid(format!("{path}.access does not match the module")));
        }
        if !names_agree(&expected.name, &found.name) {
            return Err(invalid(format!("{path}.name does not match the module")));
        }
    }
    if reflection.push_constant_size != metadata.push_constant_size {
        return Err(invalid(
            "push_constant_size does not match the reflected block extent",
        ));
    }
    if reflection.push_constants.len() != metadata.push_constants.len() {
        return Err(invalid("push_constants count does not match the module"));
    }
    for (i, (expected, reflected)) in metadata
        .push_constants
        .iter()
        .zip(&reflection.push_constants)
        .enumerate()
    {
        let path = format!("push_constants[{i}]");
        if reflected.offset != expected.offset {
            return Err(invalid(format!("{path}.offset does not match the module")));
        }
        if reflected.size != expected.size {
            return Err(invalid(format!("{path}.size does not match the module")));
        }
        if reflected.type_name != expected.type_name {
            return Err(invalid(format!("{path}.type does not match the module")));
        }
        if !names_agree(&expected.name, &reflected.name) {
            return Err(invalid(format!("{path}.name does not match the module")));
        }
    }
    if reflection.specialization_constants.len() != metadata.specialization_constants.len() {
        return Err(invalid(
            "specialization_constants count does not match the module",
        ));
    }
    for (i, expected) in metadata.specialization_constants.iter().enumerate() {
        let path = format!("specialization_constants[{i}]");
        let found = reflection
            .specialization_constants
            .iter()
            .find(|s| s.id == expected.id)
            .ok_or_else(|| invalid(format!("{path}.id is absent from the module")))?;
        let is_local_size =
            reflection.local_size_id_based && reflection.local_size_spec_ids.contains(&expected.id);
        if found.type_name != expected.type_name {
            return Err(invalid(format!("{path}.type does not match the module")));
        }
        if !names_agree(&expected.name, &found.name) {
            return Err(invalid(format!("{path}.name does not match the module")));
        }
        if !is_local_size && !same_spec_value(found, expected) {
            return Err(invalid(format!("{path}.default does not match the module")));
        }
    }
    Ok(())
}

/// Structurally validates a module, reflects the compute interface of `entry_point`, checks its
/// capabilities against `device`, and compares the interface with `metadata`.
///
/// # Errors
///
/// [`ErrorCode::InvalidKernel`] for malformed modules, unsupported capabilities, and any metadata
/// mismatch; the message names the offending field.
pub fn validate_and_reflect(
    words: &[u32],
    metadata: &KernelMetadata,
    device: &DeviceCapabilities,
    entry_point: &str,
) -> Result<Reflection> {
    validate_kernel_metadata(metadata)?;
    check_bounds(words)?;
    Module::parse(words)?.reflect(metadata, device, entry_point)
}

/// Checks the header and instruction stream of a module without reflecting it.
///
/// # Errors
///
/// [`ErrorCode::InvalidKernel`] when the module is malformed.
pub fn validate_module_structure(words: &[u32]) -> Status {
    Module::parse(words).map(drop)
}

/// Converts little-endian SPIR-V bytes into words.
///
/// # Errors
///
/// [`ErrorCode::InvalidKernel`] when the length is not a multiple of four.
pub fn words_from_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(invalid("module size is not a multiple of four bytes"));
    }
    Ok(chunks.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::KernelBindingMetadata;

    pub(crate) const FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/spirv_reflection.spv");
    pub(crate) const LOCAL_SIZE_ID: &[u8] = include_bytes!("../../tests/fixtures/autotune.spv");

    pub(crate) fn fixture_words() -> Vec<u32> {
        words_from_bytes(FIXTURE).unwrap()
    }

    pub(crate) fn expected_metadata() -> KernelMetadata {
        KernelMetadata {
            kernel_id: "spirv-reflection-fixture".into(),
            local_size: [8, 2, 1],
            bindings: vec![
                KernelBindingMetadata {
                    set: 0,
                    binding: 0,
                    name: "data_buffer".into(),
                    kind: KernelBindingKind::StorageBuffer,
                    access: KernelAccess::ReadWrite,
                },
                KernelBindingMetadata {
                    set: 0,
                    binding: 1,
                    name: "second_buffer".into(),
                    kind: KernelBindingKind::StorageBuffer,
                    access: KernelAccess::ReadWrite,
                },
            ],
            push_constant_size: 4,
            push_constants: vec![PushConstantFieldMetadata {
                name: "count".into(),
                type_name: "u32".into(),
                offset: 0,
                size: 4,
            }],
            specialization_constants: vec![SpecializationConstantMetadata {
                id: 7,
                name: "SCALE".into(),
                type_name: "u32".into(),
                default_value: SpecializationValue::U64(2),
            }],
            ..KernelMetadata::default()
        }
    }

    fn reflect(words: &[u32], metadata: &KernelMetadata) -> Result<Reflection> {
        validate_and_reflect(words, metadata, &DeviceCapabilities::default(), "main")
    }

    fn message(result: Result<Reflection>) -> String {
        let error = result.unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidKernel);
        error.message().to_owned()
    }

    /// Word offsets of every instruction after the header.
    fn instructions(words: &[u32]) -> Vec<usize> {
        let mut offsets = Vec::new();
        let mut cursor = HEADER_WORDS;
        while cursor < words.len() {
            offsets.push(cursor);
            cursor += (words[cursor] >> 16) as usize;
        }
        offsets
    }

    fn find(words: &[u32], opcode: u32, predicate: impl Fn(&[u32]) -> bool) -> usize {
        instructions(words)
            .into_iter()
            .find(|&c| words[c] & 0xffff == opcode && predicate(&words[c..]))
            .expect("instruction present in fixture")
    }

    fn insert_after_capabilities(words: &mut Vec<u32>, instruction: &[u32]) {
        let at = instructions(words)
            .into_iter()
            .find(|&c| words[c] & 0xffff != op::CAPABILITY)
            .unwrap();
        words.splice(at..at, instruction.iter().copied());
    }

    #[test]
    fn rejects_truncated_and_corrupt_modules() {
        let metadata = KernelMetadata {
            kernel_id: "invalid-module".into(),
            ..KernelMetadata::default()
        };
        assert!(message(reflect(&[], &metadata)).contains("five-word header"));
        assert!(message(reflect(&[0, 0x0001_0000, 0, 1, 0], &metadata)).contains("magic"));
        let truncated = [MAGIC, 0x0001_0000, 0, 1, 0, 0x0002_000b];
        assert!(message(reflect(&truncated, &metadata)).contains("invalid word count"));
        let zero_count = [MAGIC, 0x0001_0000, 0, 1, 0, 0x0000_0011];
        assert!(validate_module_structure(&zero_count).is_err());
        assert!(validate_module_structure(&[MAGIC, 0x0001_0000, 0, 0, 0]).is_err());
        assert!(validate_module_structure(&[MAGIC, 0x0001_0000, 0, 1, 1]).is_err());
        assert!(validate_module_structure(&[MAGIC, 0x0002_0000, 0, 1, 0]).is_err());
        assert!(validate_module_structure(&[MAGIC, 0x0001_0700, 0, 1, 0]).is_err());
        assert!(validate_module_structure(&[MAGIC, 0x0001_0001, 0, 1, 0]).is_err());
        validate_module_structure(&[MAGIC, 0x0001_0600, 0, 1, 0]).unwrap();
        assert!(words_from_bytes(&[1, 2, 3]).is_err());
    }

    #[test]
    fn reflects_and_checks_compute_interface_against_metadata() {
        let words = fixture_words();
        validate_module_structure(&words).unwrap();
        let reflection = reflect(&words, &expected_metadata()).unwrap();
        assert_eq!(reflection.entry_point, "main");
        assert_eq!(reflection.local_size, [8, 2, 1]);
        assert!(!reflection.local_size_id_based);
        assert_eq!(reflection.bindings.len(), 2);
        assert_eq!(
            (reflection.bindings[0].set, reflection.bindings[0].binding),
            (0, 0)
        );
        assert_eq!(reflection.bindings[0].kind, KernelBindingKind::StorageBuffer);
        assert_eq!(reflection.bindings[0].access, KernelAccess::ReadWrite);
        assert_eq!(reflection.bindings[1].binding, 1);
        assert_eq!(reflection.push_constant_size, 4);
        assert_eq!(reflection.push_constants[0].type_name, "u32");
        assert_eq!(reflection.specialization_constants.len(), 1);
        assert_eq!(reflection.specialization_constants[0].id, 7);
        assert_eq!(
            reflection.specialization_constants[0].default_value,
            SpecializationValue::U64(2)
        );
        assert!(reflection.capabilities.contains(&capability::SHADER));
    }

    #[test]
    fn reports_precise_metadata_mismatches() {
        let words = fixture_words();
        type Mutation = fn(&mut KernelMetadata);
        let cases: [(&str, Mutation); 10] = [
            ("local_size", |m| m.local_size = [1, 1, 1]),
            ("bindings count", |m| {
                m.bindings.pop();
            }),
            ("bindings[1] location", |m| m.bindings[1].binding = 5),
            ("bindings[0].kind", |m| {
                m.bindings[0].kind = KernelBindingKind::UniformBuffer
            }),
            ("bindings[0].access", |m| {
                m.bindings[0].access = KernelAccess::Read
            }),
            ("bindings[0].name", |m| m.bindings[0].name = "other".into()),
            ("push_constant_size", |m| m.push_constant_size = 8),
            ("push_constants[0].type", |m| {
                m.push_constants[0].type_name = "i32".into()
            }),
            ("specialization_constants[0].default", |m| {
                m.specialization_constants[0].default_value = SpecializationValue::U64(3);
            }),
            ("specialization_constants[0].id", |m| {
                m.specialization_constants[0].id = 9
            }),
        ];
        for (expected, mutate) in cases {
            let mut metadata = expected_metadata();
            mutate(&mut metadata);
            let text = message(reflect(&words, &metadata));
            assert!(text.contains(expected), "{expected}: {text}");
        }
    }

    #[test]
    fn rejects_metadata_that_fails_its_own_validation() {
        let mut metadata = expected_metadata();
        metadata.kernel_id.clear();
        assert_eq!(
            reflect(&fixture_words(), &metadata).unwrap_err().code(),
            ErrorCode::InvalidKernel
        );
    }

    #[test]
    fn rejects_capabilities_not_present_in_the_device_snapshot() {
        let mut words = fixture_words();
        insert_after_capabilities(&mut words, &[0x0002_0011, capability::FLOAT64]);
        assert!(message(reflect(&words, &expected_metadata())).contains("capability 10"));
        let device = DeviceCapabilities {
            shader_float64: true,
            ..DeviceCapabilities::default()
        };
        validate_and_reflect(&words, &expected_metadata(), &device, "main").unwrap();
    }

    #[test]
    fn capability_table_matches_device_features() {
        use super::capability as c;
        let none = DeviceCapabilities::default();
        for cap in [c::MATRIX, c::SHADER] {
            assert!(none.allows(cap));
        }
        let all = [
            c::INT16,
            c::INT64,
            c::FLOAT64,
            c::GROUP_NON_UNIFORM,
            c::GROUP_NON_UNIFORM_VOTE,
            c::GROUP_NON_UNIFORM_ARITHMETIC,
            c::GROUP_NON_UNIFORM_BALLOT,
            c::GROUP_NON_UNIFORM_SHUFFLE,
            c::GROUP_NON_UNIFORM_SHUFFLE_RELATIVE,
            c::GROUP_NON_UNIFORM_CLUSTERED,
            c::GROUP_NON_UNIFORM_QUAD,
        ];
        for cap in all {
            assert!(!none.allows(cap), "{cap}");
        }
        let everything = DeviceCapabilities {
            api_major: 1,
            api_minor: 3,
            shader_int16: true,
            shader_int64: true,
            shader_float64: true,
            subgroup_supported: true,
            subgroup_basic: true,
            subgroup_vote: true,
            subgroup_arithmetic: true,
            subgroup_ballot: true,
            subgroup_shuffle: true,
            subgroup_shuffle_relative: true,
            subgroup_clustered: true,
            subgroup_quad: true,
            local_size_id: true,
        };
        for cap in all {
            assert!(everything.allows(cap), "{cap}");
        }
        assert!(!everything.allows(4_461), "unknown capabilities are rejected");
        let basic_without_support = DeviceCapabilities {
            subgroup_basic: true,
            ..DeviceCapabilities::default()
        };
        assert!(!basic_without_support.allows(c::GROUP_NON_UNIFORM));
    }

    #[test]
    fn rejects_non_compute_and_missing_entry_points() {
        let mut words = fixture_words();
        assert!(
            message(validate_and_reflect(
                &words,
                &expected_metadata(),
                &DeviceCapabilities::default(),
                "other"
            ))
            .contains("'other' was not found")
        );
        let entry = find(&words, op::ENTRY_POINT, |_| true);
        words[entry + 1] = 4; // Fragment
        assert!(message(reflect(&words, &expected_metadata())).contains("not a compute entry point"));
    }

    #[test]
    fn rejects_zero_and_missing_local_size() {
        let mut words = fixture_words();
        let mode = find(&words, op::EXECUTION_MODE, |w| w[2] == EXECUTION_MODE_LOCAL_SIZE);
        words[mode + 4] = 0;
        assert!(message(reflect(&words, &expected_metadata())).contains("nonzero"));
        words[mode + 2] = 99; // Not a LocalSize mode any more.
        assert!(message(reflect(&words, &expected_metadata())).contains("no LocalSize"));
    }

    #[test]
    fn rejects_duplicate_and_partial_decorations() {
        let words = fixture_words();
        let binding = find(&words, op::DECORATE, |w| w[2] == decoration::BINDING);
        let mut duplicate = words.clone();
        let copy: Vec<u32> = words[binding..binding + 4].to_vec();
        duplicate.splice(binding..binding, copy);
        assert!(message(reflect(&duplicate, &expected_metadata())).contains("duplicate decoration"));

        let mut partial = words.clone();
        partial[binding + 2] = 9_999; // Drops the Binding decoration for this variable.
        assert!(message(reflect(&partial, &expected_metadata())).contains("both DescriptorSet and Binding"));
    }

    #[test]
    fn reflects_local_size_id_kernels_and_ignores_local_size_defaults() {
        let words = words_from_bytes(LOCAL_SIZE_ID).unwrap();
        let module = Module::parse(&words).unwrap();
        let ids = module.execution_sizes.values().next().unwrap();
        assert!(ids.id_based);
        let mut metadata = KernelMetadata {
            kernel_id: "autotune".into(),
            local_size: [64, 1, 1],
            ..KernelMetadata::default()
        };
        // Learn the interface from the module itself, then check it round-trips.
        let reflected = module.reflect_bindings().unwrap();
        metadata.bindings = reflected
            .iter()
            .map(|b| KernelBindingMetadata {
                set: b.set,
                binding: b.binding,
                name: b.name.clone(),
                kind: b.kind,
                access: b.access,
            })
            .collect();
        (metadata.push_constant_size, metadata.push_constants) = module.reflect_push_constants().unwrap();
        metadata.specialization_constants = module.reflect_specialization_constants().unwrap();
        for spec in &mut metadata.specialization_constants {
            if spec.name.is_empty() {
                spec.name = format!("spec{}", spec.id);
            }
        }
        assert!(message(reflect(&words, &metadata)).contains("maintenance4"));
        let maintenance4 = DeviceCapabilities {
            local_size_id: true,
            ..DeviceCapabilities::default()
        };
        let with_ids = |m: &KernelMetadata| validate_and_reflect(&words, m, &maintenance4, "main");
        let reflection = with_ids(&metadata).unwrap();
        assert!(reflection.local_size_id_based);
        assert_eq!(reflection.local_size, reflection.local_size_spec_defaults);
        // LocalSizeId defaults are free to differ: the autotuner specializes them.
        for spec in &mut metadata.specialization_constants {
            if reflection.local_size_spec_ids.contains(&spec.id) {
                spec.default_value = SpecializationValue::U64(1);
            }
        }
        metadata.local_size = [1, 1, 1];
        with_ids(&metadata).unwrap();
    }

    #[test]
    fn decodes_signed_float_and_boolean_specialization_values() {
        let int = |width, signed| Type {
            width,
            signed,
            ..Type::new(TypeKind::Integer)
        };
        let float = |width| Type {
            width,
            ..Type::new(TypeKind::Float)
        };
        let constant = |bits, boolean| SpecConstant {
            type_id: 1,
            result_id: 2,
            bits,
            boolean,
        };
        assert_eq!(
            spec_value(&constant(0xffff_fffe, None), &int(32, true)).unwrap(),
            ("i32".into(), SpecializationValue::I64(-2))
        );
        assert_eq!(
            spec_value(&constant(u64::MAX, None), &int(64, true)).unwrap(),
            ("i64".into(), SpecializationValue::I64(-1))
        );
        assert_eq!(
            spec_value(&constant(0x1_0000_0005, None), &int(32, false)).unwrap(),
            ("u32".into(), SpecializationValue::U64(5))
        );
        assert_eq!(
            spec_value(&constant(u64::from(1.5_f32.to_bits()), None), &float(32)).unwrap(),
            ("f32".into(), SpecializationValue::F64(1.5))
        );
        assert_eq!(
            spec_value(&constant((-0.25_f64).to_bits(), None), &float(64)).unwrap(),
            ("f64".into(), SpecializationValue::F64(-0.25))
        );
        assert_eq!(
            spec_value(&constant(0, Some(true)), &Type::new(TypeKind::Bool)).unwrap(),
            ("bool".into(), SpecializationValue::Bool(true))
        );
        assert!(spec_value(&constant(0, None), &float(16)).is_err());
        assert!(spec_value(&constant(0, None), &int(0, false)).is_err());
        assert!(spec_value(&constant(0, None), &Type::new(TypeKind::Struct)).is_err());

        let f32_spec = |value| SpecializationConstantMetadata {
            id: 0,
            name: String::new(),
            type_name: "f32".into(),
            default_value: SpecializationValue::F64(value),
        };
        let stored = f64::from(0.1_f32);
        assert!(same_spec_value(&f32_spec(stored), &f32_spec(0.1)));
        assert!(!same_spec_value(&f32_spec(stored), &f32_spec(0.2)));
    }

    #[test]
    fn computes_layout_sizes_and_type_names() {
        let mut module = Module {
            id_bound: 100,
            ..Module::default()
        };
        let mut declare = |id, ty| {
            module.types.insert(id, ty);
        };
        declare(
            1,
            Type {
                width: 32,
                ..Type::new(TypeKind::Float)
            },
        );
        declare(
            2,
            Type {
                element: 1,
                count: 4,
                ..Type::new(TypeKind::Vector)
            },
        );
        declare(
            3,
            Type {
                element: 2,
                count: 4,
                ..Type::new(TypeKind::Matrix)
            },
        );
        declare(
            4,
            Type {
                element: 1,
                count: 50,
                ..Type::new(TypeKind::Array)
            },
        );
        declare(5, Type::new(TypeKind::Bool));
        declare(
            6,
            Type {
                width: 12,
                ..Type::new(TypeKind::Integer)
            },
        );
        declare(
            7,
            Type {
                element: 1,
                ..Type::new(TypeKind::RuntimeArray)
            },
        );
        module.constants.insert(50, 3);
        module.decorations.entry(4).or_default().array_stride = Some(16);
        assert_eq!(module.type_size(2, None, 0).unwrap(), 16);
        assert_eq!(module.type_size(3, None, 0).unwrap(), 64);
        let strided = MemberDecorations {
            matrix_stride: Some(32),
            ..MemberDecorations::default()
        };
        assert_eq!(module.type_size(3, Some(strided), 0).unwrap(), 128);
        assert_eq!(module.type_size(4, None, 0).unwrap(), 48);
        assert_eq!(module.type_size(5, None, 0).unwrap(), 4);
        assert!(module.type_size(6, None, 0).is_err());
        assert!(module.type_size(7, None, 0).is_err());
        assert!(module.type_size(99, None, 0).is_err());
        assert_eq!(module.type_name(2, 0).unwrap(), "vec4<f32>");
        assert_eq!(module.type_name(3, 0).unwrap(), "mat4x4<f32>");
        assert_eq!(module.type_name(4, 0).unwrap(), "array[3]<f32>");
        assert!(module.type_name(7, 0).is_err());

        // A self-referential chain must stop at the depth limit instead of overflowing the stack.
        module.types.insert(
            8,
            Type {
                element: 8,
                count: 1,
                ..Type::new(TypeKind::Vector)
            },
        );
        assert!(
            module
                .type_size(8, None, 0)
                .unwrap_err()
                .message()
                .contains("32 levels")
        );
        assert!(
            module
                .type_name(8, 0)
                .unwrap_err()
                .message()
                .contains("32 levels")
        );
    }

    #[test]
    fn hostile_inputs_never_panic() {
        // Deterministic mutation fuzzing of the real fixture: every result is Ok or Err.
        let base = fixture_words();
        let metadata = expected_metadata();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..4_000 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let mut words = base.clone();
            let index = HEADER_WORDS + (state >> 33) as usize % (words.len() - HEADER_WORDS);
            words[index] = match state & 3 {
                0 => (state >> 7) as u32,
                1 => words[index] ^ (1 << ((state >> 13) % 32)),
                2 => 0,
                _ => u32::MAX,
            };
            let _ = reflect(&words, &metadata);
            let _ = reflect(&words[..index], &metadata);
        }
    }
}
