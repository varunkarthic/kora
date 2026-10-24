//! Versioned, device-independent metadata for a packaged compute kernel (`.kora.json`).

use std::collections::BTreeSet;

use crate::error::{Error, ErrorCode, Result, Status};
use crate::json::{self, Value};

/// Descriptor class of a kernel binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KernelBindingKind {
    /// `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`.
    #[default]
    StorageBuffer,
    /// `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER`.
    UniformBuffer,
    /// `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE`.
    StorageImage,
    /// `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE`.
    SampledImage,
}

impl KernelBindingKind {
    /// Manifest spelling such as `"storage_buffer"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::StorageBuffer => "storage_buffer",
            Self::UniformBuffer => "uniform_buffer",
            Self::StorageImage => "storage_image",
            Self::SampledImage => "sampled_image",
        }
    }
    fn parse(text: &str) -> Option<Self> {
        [
            Self::StorageBuffer,
            Self::UniformBuffer,
            Self::StorageImage,
            Self::SampledImage,
        ]
        .into_iter()
        .find(|kind| kind.name() == text)
    }
}

/// How a kernel accesses a binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KernelAccess {
    /// Read only.
    Read,
    /// Write only.
    Write,
    /// Read and write.
    #[default]
    ReadWrite,
}

impl KernelAccess {
    /// Manifest spelling such as `"read_write"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::ReadWrite => "read_write",
        }
    }
    /// `true` for [`KernelAccess::Read`] and [`KernelAccess::ReadWrite`].
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }
    /// `true` for [`KernelAccess::Write`] and [`KernelAccess::ReadWrite`].
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
    fn parse(text: &str) -> Option<Self> {
        [Self::Read, Self::Write, Self::ReadWrite]
            .into_iter()
            .find(|access| access.name() == text)
    }
}

/// One descriptor binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct KernelBindingMetadata {
    /// Descriptor set.
    pub set: u32,
    /// Binding number within the set.
    pub binding: u32,
    /// Unique binding name.
    pub name: String,
    /// Descriptor class.
    pub kind: KernelBindingKind,
    /// Access mode.
    pub access: KernelAccess,
}

/// One push-constant field.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PushConstantFieldMetadata {
    /// Unique field name.
    pub name: String,
    /// Scalar or vector type name such as `"u32"`.
    pub type_name: String,
    /// Byte offset.
    pub offset: u32,
    /// Byte size.
    pub size: u32,
}

/// Default value of a scalar specialization constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecializationValue {
    /// `bool`.
    Bool(bool),
    /// `i32` / `i64`.
    I64(i64),
    /// `u32` / `u64`.
    U64(u64),
    /// `f32` / `f64`.
    F64(f64),
}

impl Default for SpecializationValue {
    fn default() -> Self {
        Self::U64(0)
    }
}

/// One scalar specialization constant.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpecializationConstantMetadata {
    /// SPIR-V `SpecId`.
    pub id: u32,
    /// Unique name.
    pub name: String,
    /// One of `bool`, `i32`, `i64`, `u32`, `u64`, `f32`, `f64`.
    pub type_name: String,
    /// Default value, which must match `type_name`.
    pub default_value: SpecializationValue,
}

/// Version-1 kernel metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelMetadata {
    /// Must be `1`.
    pub schema_version: u32,
    /// Stable kernel identifier (1–256 bytes).
    pub kernel_id: String,
    /// Descriptor bindings.
    pub bindings: Vec<KernelBindingMetadata>,
    /// Total push-constant block size in bytes (at most 65 536).
    pub push_constant_size: u32,
    /// Push-constant fields; ranges must not overlap.
    pub push_constants: Vec<PushConstantFieldMetadata>,
    /// Scalar specialization constants.
    pub specialization_constants: Vec<SpecializationConstantMetadata>,
    /// Declared workgroup size.
    pub local_size: [u32; 3],
    /// Semantic key of a registered CPU implementation, or empty.
    pub cpu_fallback_key: String,
    /// Free-form semantic tags (unique).
    pub semantics: Vec<String>,
}

impl Default for KernelMetadata {
    fn default() -> Self {
        Self {
            schema_version: 1,
            kernel_id: String::new(),
            bindings: Vec::new(),
            push_constant_size: 0,
            push_constants: Vec::new(),
            specialization_constants: Vec::new(),
            local_size: [1, 1, 1],
            cpu_fallback_key: String::new(),
            semantics: Vec::new(),
        }
    }
}

fn invalid(path: impl Into<String>, why: &str) -> Error {
    Error::new(ErrorCode::InvalidKernel, format!("{}: {why}", path.into()))
}

fn bounded(text: &str, max: usize) -> bool {
    !text.is_empty() && text.len() <= max
}

/// Validates caller-constructed metadata; errors are `InvalidKernel` with the failing field path.
pub fn validate_kernel_metadata(m: &KernelMetadata) -> Status {
    if m.schema_version != 1 {
        return Err(invalid("schema_version", "only version 1 is supported"));
    }
    if !bounded(&m.kernel_id, 256) {
        return Err(invalid("kernel_id", "must contain 1 to 256 bytes"));
    }
    if m.bindings.len() > 4096 {
        return Err(invalid("bindings", "too many bindings"));
    }
    let mut locations = BTreeSet::new();
    let mut names = BTreeSet::new();
    for (i, b) in m.bindings.iter().enumerate() {
        let p = format!("bindings[{i}]");
        if !bounded(&b.name, 256) {
            return Err(invalid(format!("{p}.name"), "must contain 1 to 256 bytes"));
        }
        if !names.insert(b.name.as_str()) {
            return Err(invalid(format!("{p}.name"), "duplicate binding name"));
        }
        if !locations.insert((b.set, b.binding)) {
            return Err(invalid(p, "duplicate (set, binding) location"));
        }
    }
    if m.push_constant_size > 65536 {
        return Err(invalid(
            "push_constant_size",
            "exceeds portable metadata limit of 65536 bytes",
        ));
    }
    if m.push_constants.len() > 4096 {
        return Err(invalid("push_constants", "too many fields"));
    }
    let mut ranges = Vec::with_capacity(m.push_constants.len());
    names.clear();
    for (i, f) in m.push_constants.iter().enumerate() {
        let p = format!("push_constants[{i}]");
        if !bounded(&f.name, 256) || !bounded(&f.type_name, 64) {
            return Err(invalid(p, "field name and type must be nonempty and bounded"));
        }
        if !names.insert(f.name.as_str()) {
            return Err(invalid(format!("{p}.name"), "duplicate field name"));
        }
        if f.size == 0 {
            return Err(invalid(format!("{p}.size"), "must be greater than zero"));
        }
        let end = u64::from(f.offset) + u64::from(f.size);
        if end > u64::from(m.push_constant_size) {
            return Err(invalid(p, "field range exceeds push_constant_size"));
        }
        ranges.push((u64::from(f.offset), end));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|pair| pair[1].0 < pair[0].1) {
        return Err(invalid("push_constants", "field ranges overlap"));
    }
    if let Some(i) = m.local_size.iter().position(|&n| n == 0) {
        return Err(invalid(format!("local_size[{i}]"), "must be greater than zero"));
    }
    if m.local_size
        .iter()
        .try_fold(1u32, |acc, &n| acc.checked_mul(n))
        .is_none()
    {
        return Err(invalid("local_size", "dimension product exceeds uint32 range"));
    }
    if m.cpu_fallback_key.len() > 256 {
        return Err(invalid("cpu_fallback_key", "exceeds 256 bytes"));
    }
    if m.specialization_constants.len() > 4096 {
        return Err(invalid("specialization_constants", "too many constants"));
    }
    let mut ids = BTreeSet::new();
    names.clear();
    for (i, s) in m.specialization_constants.iter().enumerate() {
        let p = format!("specialization_constants[{i}]");
        if !ids.insert(s.id) {
            return Err(invalid(format!("{p}.id"), "duplicate specialization id"));
        }
        if !bounded(&s.name, 256) || !bounded(&s.type_name, 64) {
            return Err(invalid(p, "name and type must be nonempty and bounded"));
        }
        if !names.insert(s.name.as_str()) {
            return Err(invalid(format!("{p}.name"), "duplicate specialization name"));
        }
        if let SpecializationValue::F64(d) = s.default_value
            && !d.is_finite()
        {
            return Err(invalid(format!("{p}.default"), "must be finite"));
        }
        use SpecializationValue as V;
        let matches_type = match (s.type_name.as_str(), s.default_value) {
            ("bool", V::Bool(_)) => true,
            ("i32" | "i64", V::I64(_)) => true,
            ("u32" | "u64", V::U64(_)) => true,
            ("f32" | "f64", V::F64(_)) => true,
            ("bool" | "i32" | "i64" | "u32" | "u64" | "f32" | "f64", _) => false,
            _ => {
                return Err(invalid(
                    format!("{p}.type"),
                    "unsupported specialization scalar type",
                ));
            }
        };
        if !matches_type {
            return Err(invalid(
                format!("{p}.default"),
                "default value does not match declared scalar type",
            ));
        }
        match (s.type_name.as_str(), s.default_value) {
            ("i32", V::I64(v)) if i32::try_from(v).is_err() => {
                return Err(invalid(format!("{p}.default"), "exceeds i32 range"));
            }
            ("u32", V::U64(v)) if u32::try_from(v).is_err() => {
                return Err(invalid(format!("{p}.default"), "exceeds u32 range"));
            }
            ("f32", V::F64(v)) if v.abs() > f64::from(f32::MAX) => {
                return Err(invalid(format!("{p}.default"), "exceeds finite f32 range"));
            }
            _ => {}
        }
    }
    if m.semantics.len() > 256 {
        return Err(invalid("semantics", "too many entries"));
    }
    names.clear();
    for (i, semantic) in m.semantics.iter().enumerate() {
        if !bounded(semantic, 256) {
            return Err(invalid(format!("semantics[{i}]"), "must contain 1 to 256 bytes"));
        }
        if !names.insert(semantic.as_str()) {
            return Err(invalid(format!("semantics[{i}]"), "duplicate semantic"));
        }
    }
    Ok(())
}

fn fields(value: &Value, allowed: &[&str], path: &str) -> Status {
    let members = value
        .as_object()
        .ok_or_else(|| invalid(path, "expected an object"))?;
    for (name, _) in members {
        if !allowed.contains(&name.as_str()) {
            return Err(invalid(format!("{path}.{name}"), "unknown field"));
        }
    }
    Ok(())
}

fn required<'v>(value: &'v Value, key: &str, path: &str) -> Result<&'v Value> {
    value
        .find(key)
        .ok_or_else(|| invalid(format!("{path}.{key}"), "field is required"))
}

fn unsigned(value: &Value, path: &str) -> Result<u64> {
    match value {
        Value::Number(text) if !text.starts_with('-') && !text.contains(['.', 'e', 'E']) => text
            .parse()
            .map_err(|_| invalid(path, "integer is outside the supported range")),
        _ => Err(invalid(path, "expected an unsigned integer")),
    }
}

fn uint32(value: &Value, path: &str) -> Result<u32> {
    u32::try_from(unsigned(value, path)?).map_err(|_| invalid(path, "integer exceeds uint32 range"))
}

fn string(value: &Value, path: &str) -> Result<String> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid(path, "expected a string"))
}

fn enumerated<E>(value: &Value, path: &str, parse: fn(&str) -> Option<E>) -> Result<E> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid(path, "expected a string enum value"))?;
    parse(text).ok_or_else(|| invalid(path, &format!("unrecognized enum value '{text}'")))
}

fn array<'v>(value: &'v Value, path: &str) -> Result<&'v [Value]> {
    value.as_array().ok_or_else(|| invalid(path, "expected an array"))
}

fn parse_binding(value: &Value, index: usize) -> Result<KernelBindingMetadata> {
    let path = format!("bindings[{index}]");
    fields(value, &["set", "binding", "name", "kind", "access"], &path)?;
    Ok(KernelBindingMetadata {
        set: uint32(required(value, "set", &path)?, &format!("{path}.set"))?,
        binding: uint32(required(value, "binding", &path)?, &format!("{path}.binding"))?,
        name: string(required(value, "name", &path)?, &format!("{path}.name"))?,
        kind: enumerated(
            required(value, "kind", &path)?,
            &format!("{path}.kind"),
            KernelBindingKind::parse,
        )?,
        access: enumerated(
            required(value, "access", &path)?,
            &format!("{path}.access"),
            KernelAccess::parse,
        )?,
    })
}

fn parse_push_field(value: &Value, index: usize) -> Result<PushConstantFieldMetadata> {
    let path = format!("push_constants[{index}]");
    fields(value, &["name", "type", "offset", "size"], &path)?;
    Ok(PushConstantFieldMetadata {
        name: string(required(value, "name", &path)?, &format!("{path}.name"))?,
        type_name: string(required(value, "type", &path)?, &format!("{path}.type"))?,
        offset: uint32(required(value, "offset", &path)?, &format!("{path}.offset"))?,
        size: uint32(required(value, "size", &path)?, &format!("{path}.size"))?,
    })
}

fn parse_spec(value: &Value, index: usize) -> Result<SpecializationConstantMetadata> {
    let path = format!("specialization_constants[{index}]");
    fields(value, &["id", "name", "type", "default"], &path)?;
    let id = uint32(required(value, "id", &path)?, &format!("{path}.id"))?;
    let name = string(required(value, "name", &path)?, &format!("{path}.name"))?;
    let type_name = string(required(value, "type", &path)?, &format!("{path}.type"))?;
    let default_path = format!("{path}.default");
    let default_value = match required(value, "default", &path)? {
        Value::Bool(b) => SpecializationValue::Bool(*b),
        Value::Number(text) if text.contains(['.', 'e', 'E']) => SpecializationValue::F64(
            json::parse_finite_double(text)
                .ok_or_else(|| invalid(&default_path, "expected finite number"))?,
        ),
        Value::Number(text) if text.starts_with('-') => SpecializationValue::I64(
            text.parse()
                .map_err(|_| invalid(&default_path, "integer is outside int64 range"))?,
        ),
        number @ Value::Number(_) => SpecializationValue::U64(unsigned(number, &default_path)?),
        _ => return Err(invalid(&default_path, "expected boolean or number")),
    };
    // JSON has one number type: a non-negative integer default for a signed or floating-point
    // constant is coerced to the declared type instead of being rejected as a type mismatch.
    let default_value = match (type_name.as_str(), default_value) {
        ("i32" | "i64", SpecializationValue::U64(u)) => SpecializationValue::I64(
            i64::try_from(u).map_err(|_| invalid(&default_path, "integer is outside int64 range"))?,
        ),
        ("f32" | "f64", SpecializationValue::U64(u)) => SpecializationValue::F64(u as f64),
        ("f32" | "f64", SpecializationValue::I64(i)) => SpecializationValue::F64(i as f64),
        (_, value) => value,
    };
    Ok(SpecializationConstantMetadata {
        id,
        name,
        type_name,
        default_value,
    })
}

/// Parses a strict `.kora.json` document. Unknown and duplicate fields are rejected and the result
/// is validated with [`validate_kernel_metadata`].
pub fn parse_kernel_metadata(document: &str) -> Result<KernelMetadata> {
    let root = json::parse(document.as_bytes())?;
    fields(
        &root,
        &[
            "schema_version",
            "kernel_id",
            "bindings",
            "push_constant_size",
            "push_constants",
            "specialization_constants",
            "local_size",
            "cpu_fallback_key",
            "semantics",
        ],
        "$",
    )?;
    let mut m = KernelMetadata {
        schema_version: uint32(required(&root, "schema_version", "$")?, "schema_version")?,
        kernel_id: string(required(&root, "kernel_id", "$")?, "kernel_id")?,
        ..Default::default()
    };
    for (i, b) in array(required(&root, "bindings", "$")?, "bindings")?
        .iter()
        .enumerate()
    {
        m.bindings.push(parse_binding(b, i)?);
    }
    m.push_constant_size = uint32(required(&root, "push_constant_size", "$")?, "push_constant_size")?;
    for (i, f) in array(required(&root, "push_constants", "$")?, "push_constants")?
        .iter()
        .enumerate()
    {
        m.push_constants.push(parse_push_field(f, i)?);
    }
    let specs = array(
        required(&root, "specialization_constants", "$")?,
        "specialization_constants",
    )?;
    for (i, s) in specs.iter().enumerate() {
        m.specialization_constants.push(parse_spec(s, i)?);
    }
    let local = required(&root, "local_size", "$")?
        .as_array()
        .filter(|items| items.len() == 3)
        .ok_or_else(|| invalid("local_size", "expected an array of exactly three integers"))?;
    for (i, n) in local.iter().enumerate() {
        m.local_size[i] = uint32(n, &format!("local_size[{i}]"))?;
    }
    if let Some(key) = root.find("cpu_fallback_key") {
        m.cpu_fallback_key = string(key, "cpu_fallback_key")?;
    }
    if let Some(semantics) = root.find("semantics") {
        for (i, s) in array(semantics, "semantics")?.iter().enumerate() {
            m.semantics.push(string(s, &format!("semantics[{i}]"))?);
        }
    }
    validate_kernel_metadata(&m)?;
    Ok(m)
}

impl KernelMetadata {
    /// Serializes in the documented manifest format accepted by [`parse_kernel_metadata`].
    pub fn to_json(&self) -> String {
        let spec_default = |value: SpecializationValue| match value {
            SpecializationValue::Bool(b) => Value::Bool(b),
            SpecializationValue::I64(i) => Value::int(i),
            SpecializationValue::U64(u) => Value::uint(u),
            SpecializationValue::F64(f) => Value::float(f),
        };
        let object = |members: Vec<(&str, Value)>| {
            Value::Object(members.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
        };
        object(vec![
            ("schema_version", Value::uint(u64::from(self.schema_version))),
            ("kernel_id", Value::string(&self.kernel_id)),
            (
                "bindings",
                Value::Array(
                    self.bindings
                        .iter()
                        .map(|b| {
                            object(vec![
                                ("set", Value::uint(u64::from(b.set))),
                                ("binding", Value::uint(u64::from(b.binding))),
                                ("name", Value::string(&b.name)),
                                ("kind", Value::string(b.kind.name())),
                                ("access", Value::string(b.access.name())),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "push_constant_size",
                Value::uint(u64::from(self.push_constant_size)),
            ),
            (
                "push_constants",
                Value::Array(
                    self.push_constants
                        .iter()
                        .map(|f| {
                            object(vec![
                                ("name", Value::string(&f.name)),
                                ("type", Value::string(&f.type_name)),
                                ("offset", Value::uint(u64::from(f.offset))),
                                ("size", Value::uint(u64::from(f.size))),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "specialization_constants",
                Value::Array(
                    self.specialization_constants
                        .iter()
                        .map(|c| {
                            object(vec![
                                ("id", Value::uint(u64::from(c.id))),
                                ("name", Value::string(&c.name)),
                                ("type", Value::string(&c.type_name)),
                                ("default", spec_default(c.default_value)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "local_size",
                Value::Array(
                    self.local_size
                        .iter()
                        .map(|&n| Value::uint(u64::from(n)))
                        .collect(),
                ),
            ),
            ("cpu_fallback_key", Value::string(&self.cpu_fallback_key)),
            (
                "semantics",
                Value::Array(self.semantics.iter().map(Value::string).collect()),
            ),
        ])
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"{
  "schema_version": 1,
  "kernel_id": "vector.add",
  "bindings": [{"set":0,"binding":0,"name":"a","kind":"storage_buffer","access":"read"}],
  "push_constant_size": 8,
  "push_constants": [{"name":"count","type":"u32","offset":0,"size":4},{"name":"stride","type":"u32","offset":4,"size":4}],
    "specialization_constants": [{"id":4,"name":"tile","type":"u64","default":18446744073709551615}],
  "local_size": [64,1,1],
  "cpu_fallback_key": "vector.add.cpu",
  "semantics": ["elementwise", "pure"]
}"#;

    fn minimal(extra_spec: &str) -> String {
        format!(
            r#"{{"schema_version":1,"kernel_id":"x","bindings":[],"push_constant_size":0,"push_constants":[],"specialization_constants":[{extra_spec}],"local_size":[1,1,1]}}"#
        )
    }

    #[test]
    fn parses_valid_schema_and_allows_programmatic_construction() {
        let parsed = parse_kernel_metadata(VALID).unwrap();
        assert_eq!(parsed.kernel_id, "vector.add");
        assert_eq!(parsed.bindings.len(), 1);
        assert_eq!(parsed.bindings[0].access, KernelAccess::Read);
        assert_eq!(
            parsed.specialization_constants[0].default_value,
            SpecializationValue::U64(u64::MAX)
        );
        assert_eq!(parsed.local_size, [64, 1, 1]);
        assert_eq!(parsed.semantics, ["elementwise", "pure"]);
        let constructed = KernelMetadata {
            kernel_id: "manual".into(),
            ..Default::default()
        };
        validate_kernel_metadata(&constructed).unwrap();
    }

    #[test]
    fn integer_defaults_are_coerced_to_the_declared_scalar_type() {
        let signed =
            parse_kernel_metadata(&minimal(r#"{"id":1,"name":"s","type":"i32","default":7}"#)).unwrap();
        assert_eq!(
            signed.specialization_constants[0].default_value,
            SpecializationValue::I64(7)
        );
        let float =
            parse_kernel_metadata(&minimal(r#"{"id":1,"name":"f","type":"f32","default":2}"#)).unwrap();
        assert_eq!(
            float.specialization_constants[0].default_value,
            SpecializationValue::F64(2.0)
        );
        let negative =
            parse_kernel_metadata(&minimal(r#"{"id":1,"name":"f","type":"f64","default":-3}"#)).unwrap();
        assert_eq!(
            negative.specialization_constants[0].default_value,
            SpecializationValue::F64(-3.0)
        );
        let too_large = parse_kernel_metadata(&minimal(
            r#"{"id":1,"name":"s","type":"i64","default":9223372036854775808}"#,
        ))
        .unwrap_err();
        assert!(too_large.message().contains("int64"), "{}", too_large.message());
        assert!(
            parse_kernel_metadata(&minimal(
                r#"{"id":1,"name":"s","type":"i32","default":2147483648}"#
            ))
            .is_err()
        );
        assert!(parse_kernel_metadata(&minimal(r#"{"id":1,"name":"u","type":"u32","default":-1}"#)).is_err());
    }

    #[test]
    fn rejects_unknown_fields_and_invalid_metadata_precisely() {
        let unknown = parse_kernel_metadata(
            r#"{"schema_version":1,"kernel_id":"x","bindings":[],"push_constant_size":0,"push_constants":[],"specialization_constants":[],"local_size":[1,1,1],"surprise":true}"#,
        )
        .unwrap_err();
        assert_eq!(unknown.code(), ErrorCode::InvalidKernel);
        assert!(unknown.message().contains("$.surprise"));
        let duplicate = parse_kernel_metadata(
            r#"{"schema_version":1,"kernel_id":"x","bindings":[{"set":0,"binding":0,"name":"a","kind":"storage_buffer","access":"read"},{"set":0,"binding":0,"name":"b","kind":"storage_buffer","access":"read"}],"push_constant_size":0,"push_constants":[],"specialization_constants":[],"local_size":[1,1,1]}"#,
        )
        .unwrap_err();
        assert!(duplicate.message().contains("duplicate (set, binding)"));
        let mismatch = parse_kernel_metadata(&minimal(r#"{"id":0,"name":"flag","type":"bool","default":1}"#))
            .unwrap_err();
        assert!(mismatch.message().contains("does not match declared scalar type"));
    }

    #[test]
    fn rejects_overflow_overlap_and_truncated_documents() {
        let mut m = KernelMetadata {
            kernel_id: "range".into(),
            push_constant_size: u32::MAX,
            ..Default::default()
        };
        assert!(
            validate_kernel_metadata(&m)
                .unwrap_err()
                .message()
                .contains("portable metadata limit")
        );
        m.push_constant_size = 8;
        let field = |name: &str, offset| PushConstantFieldMetadata {
            name: name.into(),
            type_name: "u32".into(),
            offset,
            size: 4,
        };
        m.push_constants = vec![field("x", 0), field("y", 2)];
        assert!(
            validate_kernel_metadata(&m)
                .unwrap_err()
                .message()
                .contains("overlap")
        );
        let truncated = parse_kernel_metadata("{\"schema_version\":1,\"kernel_id\":").unwrap_err();
        assert_eq!(truncated.code(), ErrorCode::InvalidKernel);
    }

    #[test]
    fn rejects_every_truncated_prefix_of_a_valid_manifest() {
        for size in 0..VALID.len() {
            if !VALID.is_char_boundary(size) {
                continue;
            }
            let error = parse_kernel_metadata(&VALID[..size]).expect_err(&format!("prefix {size}"));
            assert_eq!(error.code(), ErrorCode::InvalidKernel);
        }
    }

    #[test]
    fn parses_finite_floating_specialization_defaults() {
        let parsed = parse_kernel_metadata(&minimal(
            r#"{"id":0,"name":"scale","type":"f32","default":1.25e1}"#,
        ))
        .unwrap();
        assert_eq!(
            parsed.specialization_constants[0].default_value,
            SpecializationValue::F64(12.5)
        );
    }

    #[test]
    fn scalar_ranges_are_enforced() {
        for (spec, why) in [
            (
                r#"{"id":0,"name":"a","type":"i32","default":-2147483649}"#,
                "exceeds i32 range",
            ),
            (
                r#"{"id":0,"name":"a","type":"u32","default":4294967296}"#,
                "exceeds u32 range",
            ),
            (
                r#"{"id":0,"name":"a","type":"f32","default":1e39}"#,
                "exceeds finite f32 range",
            ),
            (
                r#"{"id":0,"name":"a","type":"half","default":1}"#,
                "unsupported specialization scalar type",
            ),
            (
                r#"{"id":0,"name":"a","type":"i64","default":"1"}"#,
                "expected boolean or number",
            ),
            (
                r#"{"id":0,"name":"a","type":"i64","default":-9223372036854775809}"#,
                "outside int64 range",
            ),
        ] {
            let error = parse_kernel_metadata(&minimal(spec)).unwrap_err();
            assert!(error.message().contains(why), "{spec}: {}", error.message());
        }
        parse_kernel_metadata(&minimal(
            r#"{"id":0,"name":"a","type":"i32","default":-2147483648}"#,
        ))
        .unwrap();
    }

    #[test]
    fn duplicate_names_ids_and_semantics_are_rejected() {
        let mut m = KernelMetadata {
            kernel_id: "k".into(),
            ..Default::default()
        };
        m.semantics = vec!["a".into(), "a".into()];
        assert!(
            validate_kernel_metadata(&m)
                .unwrap_err()
                .message()
                .contains("duplicate semantic")
        );
        m.semantics.clear();
        let spec = |id, name: &str| SpecializationConstantMetadata {
            id,
            name: name.into(),
            type_name: "u32".into(),
            default_value: SpecializationValue::U64(1),
        };
        m.specialization_constants = vec![spec(1, "a"), spec(1, "b")];
        assert!(
            validate_kernel_metadata(&m)
                .unwrap_err()
                .message()
                .contains("duplicate specialization id")
        );
        m.specialization_constants = vec![spec(1, "a"), spec(2, "a")];
        assert!(
            validate_kernel_metadata(&m)
                .unwrap_err()
                .message()
                .contains("duplicate specialization name")
        );
    }

    #[test]
    fn local_size_must_be_positive_and_bounded() {
        let mut m = KernelMetadata {
            kernel_id: "k".into(),
            local_size: [64, 0, 1],
            ..Default::default()
        };
        assert!(
            validate_kernel_metadata(&m)
                .unwrap_err()
                .message()
                .starts_with("local_size[1]")
        );
        m.local_size = [65536, 65536, 2];
        assert!(
            validate_kernel_metadata(&m)
                .unwrap_err()
                .message()
                .contains("uint32 range")
        );
    }

    #[test]
    fn serialization_round_trips() {
        let parsed = parse_kernel_metadata(VALID).unwrap();
        let reparsed = parse_kernel_metadata(&parsed.to_json()).unwrap();
        assert_eq!(parsed, reparsed);
        let mut with_float = parsed.clone();
        with_float
            .specialization_constants
            .push(SpecializationConstantMetadata {
                id: 9,
                name: "scale".into(),
                type_name: "f64".into(),
                default_value: SpecializationValue::F64(2.0),
            });
        with_float
            .specialization_constants
            .push(SpecializationConstantMetadata {
                id: 10,
                name: "neg".into(),
                type_name: "i64".into(),
                default_value: SpecializationValue::I64(-5),
            });
        assert_eq!(parse_kernel_metadata(&with_float.to_json()).unwrap(), with_float);
    }

    #[test]
    fn access_helpers() {
        assert!(KernelAccess::Read.reads() && !KernelAccess::Read.writes());
        assert!(KernelAccess::Write.writes() && !KernelAccess::Write.reads());
        assert!(KernelAccess::ReadWrite.reads() && KernelAccess::ReadWrite.writes());
    }
}
