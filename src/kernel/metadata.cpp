#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <set>
#include <sstream>

#include "core/json.hpp"
#include "kora/kernel_metadata.hpp"

namespace kora {
namespace {

Error invalid(std::string path, std::string why) {
    return Error(ErrorCode::InvalidKernel, std::move(path) + ": " + std::move(why));
}

const detail::json::Value::Object* object(const detail::json::Value& value) {
    return std::get_if<detail::json::Value::Object>(&value.storage);
}
const detail::json::Value::Array* array(const detail::json::Value& value) {
    return std::get_if<detail::json::Value::Array>(&value.storage);
}
const std::string* string(const detail::json::Value& value) {
    return std::get_if<std::string>(&value.storage);
}
const detail::json::Value::Number* number(const detail::json::Value& value) {
    return std::get_if<detail::json::Value::Number>(&value.storage);
}

bool parse_finite_double(std::string_view token, double& value) {
    std::istringstream stream{std::string(token)};
    stream.imbue(std::locale::classic());
    stream >> std::noskipws >> value;
    if (stream.fail() || stream.peek() != std::char_traits<char>::eof() || !std::isfinite(value))
        return false;
    const auto exponent = token.find_first_of("eE");
    const auto mantissa = token.substr(0, exponent);
    const bool nonzero_mantissa =
        std::any_of(mantissa.begin(), mantissa.end(), [](char c) { return c >= '1' && c <= '9'; });
    return value != 0 || !nonzero_mantissa;
}

Status fields(const detail::json::Value& value, std::initializer_list<std::string_view> allowed,
              std::string_view path) {
    const auto* members = object(value);
    if (!members) return invalid(std::string(path), "expected an object");
    for (const auto& member : *members) {
        if (std::find(allowed.begin(), allowed.end(), member.first) == allowed.end())
            return invalid(std::string(path) + "." + member.first, "unknown field");
    }
    return {};
}

const detail::json::Value* required(const detail::json::Value& value, std::string_view key,
                                    std::string_view path, Error& error) {
    const auto* result = value.find(std::string(key));
    if (!result) error = invalid(std::string(path) + "." + std::string(key), "field is required");
    return result;
}

Result<std::uint64_t> unsigned_integer(const detail::json::Value& value, std::string path) {
    const auto* n = number(value);
    if (!n || n->text.empty() || n->text[0] == '-' || n->text.find_first_of(".eE") != std::string::npos)
        return invalid(std::move(path), "expected an unsigned integer");
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(n->text.data(), n->text.data() + n->text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != n->text.data() + n->text.size())
        return invalid(std::move(path), "integer is outside the supported range");
    return result;
}

Result<std::uint32_t> uint32_field(const detail::json::Value& value, std::string path) {
    auto n = unsigned_integer(value, path);
    if (!n) return std::move(n).error();
    if (n.value() > std::numeric_limits<std::uint32_t>::max())
        return invalid(std::move(path), "integer exceeds uint32 range");
    return static_cast<std::uint32_t>(n.value());
}

Result<std::string> string_field(const detail::json::Value& value, std::string path) {
    const auto* s = string(value);
    if (!s) return invalid(std::move(path), "expected a string");
    return *s;
}

template <class E>
Result<E> enum_field(const detail::json::Value& value, std::string path,
                     std::initializer_list<std::pair<std::string_view, E>> values) {
    const auto* s = string(value);
    if (!s) return invalid(std::move(path), "expected a string enum value");
    for (const auto& [name, result] : values)
        if (*s == name) return result;
    return invalid(std::move(path), "unrecognized enum value '" + *s + "'");
}

Result<KernelBindingMetadata> parse_binding(const detail::json::Value& value, std::size_t index) {
    const auto path = "bindings[" + std::to_string(index) + "]";
    auto status = fields(value, {"set", "binding", "name", "kind", "access"}, path);
    if (!status) return std::move(status).error();
    KernelBindingMetadata result;
    Error err;
    auto* v = required(value, "set", path, err);
    if (!v) return err;
    auto set = uint32_field(*v, path + ".set");
    if (!set) return std::move(set).error();
    result.set = set.value();
    v = required(value, "binding", path, err);
    if (!v) return err;
    auto binding = uint32_field(*v, path + ".binding");
    if (!binding) return std::move(binding).error();
    result.binding = binding.value();
    v = required(value, "name", path, err);
    if (!v) return err;
    auto name = string_field(*v, path + ".name");
    if (!name) return std::move(name).error();
    result.name = std::move(name).value();
    v = required(value, "kind", path, err);
    if (!v) return err;
    auto kind = enum_field<KernelBindingKind>(*v, path + ".kind",
                                              {{"storage_buffer", KernelBindingKind::StorageBuffer},
                                               {"uniform_buffer", KernelBindingKind::UniformBuffer},
                                               {"storage_image", KernelBindingKind::StorageImage},
                                               {"sampled_image", KernelBindingKind::SampledImage}});
    if (!kind) return std::move(kind).error();
    result.kind = kind.value();
    v = required(value, "access", path, err);
    if (!v) return err;
    auto access = enum_field<KernelAccess>(*v, path + ".access",
                                           {{"read", KernelAccess::Read},
                                            {"write", KernelAccess::Write},
                                            {"read_write", KernelAccess::ReadWrite}});
    if (!access) return std::move(access).error();
    result.access = access.value();
    return result;
}

Result<PushConstantFieldMetadata> parse_push_field(const detail::json::Value& value, std::size_t index) {
    const auto path = "push_constants[" + std::to_string(index) + "]";
    auto status = fields(value, {"name", "type", "offset", "size"}, path);
    if (!status) return std::move(status).error();
    PushConstantFieldMetadata result;
    Error err;
    auto* v = required(value, "name", path, err);
    if (!v) return err;
    auto name = string_field(*v, path + ".name");
    if (!name) return std::move(name).error();
    result.name = std::move(name).value();
    v = required(value, "type", path, err);
    if (!v) return err;
    auto type = string_field(*v, path + ".type");
    if (!type) return std::move(type).error();
    result.type = std::move(type).value();
    v = required(value, "offset", path, err);
    if (!v) return err;
    auto offset = uint32_field(*v, path + ".offset");
    if (!offset) return std::move(offset).error();
    result.offset = offset.value();
    v = required(value, "size", path, err);
    if (!v) return err;
    auto size = uint32_field(*v, path + ".size");
    if (!size) return std::move(size).error();
    result.size = size.value();
    return result;
}

Result<SpecializationConstantMetadata> parse_spec(const detail::json::Value& value, std::size_t index) {
    const auto path = "specialization_constants[" + std::to_string(index) + "]";
    auto status = fields(value, {"id", "name", "type", "default"}, path);
    if (!status) return std::move(status).error();
    SpecializationConstantMetadata result;
    Error err;
    auto* v = required(value, "id", path, err);
    if (!v) return err;
    auto id = uint32_field(*v, path + ".id");
    if (!id) return std::move(id).error();
    result.id = id.value();
    v = required(value, "name", path, err);
    if (!v) return err;
    auto name = string_field(*v, path + ".name");
    if (!name) return std::move(name).error();
    result.name = std::move(name).value();
    v = required(value, "type", path, err);
    if (!v) return err;
    auto type = string_field(*v, path + ".type");
    if (!type) return std::move(type).error();
    result.type = std::move(type).value();
    v = required(value, "default", path, err);
    if (!v) return err;
    if (const auto* b = std::get_if<bool>(&v->storage))
        result.default_value = *b;
    else if (const auto* n = number(*v)) {
        if (n->text.find_first_of(".eE") != std::string::npos) {
            double d = 0;
            if (!parse_finite_double(n->text, d)) return invalid(path + ".default", "expected finite number");
            result.default_value = d;
        } else if (!n->text.empty() && n->text[0] == '-') {
            std::int64_t i = 0;
            const auto parsed = std::from_chars(n->text.data(), n->text.data() + n->text.size(), i);
            if (parsed.ec != std::errc{}) return invalid(path + ".default", "integer is outside int64 range");
            result.default_value = i;
        } else {
            auto u = unsigned_integer(*v, path + ".default");
            if (!u) return std::move(u).error();
            result.default_value = u.value();
        }
    } else
        return invalid(path + ".default", "expected boolean or number");
    return result;
}

}  // namespace

Status validate_kernel_metadata(const KernelMetadata& m) {
    if (m.schema_version != 1) return invalid("schema_version", "only version 1 is supported");
    if (m.kernel_id.empty() || m.kernel_id.size() > 256)
        return invalid("kernel_id", "must contain 1 to 256 bytes");
    if (m.bindings.size() > 4096) return invalid("bindings", "too many bindings");
    std::set<std::pair<std::uint32_t, std::uint32_t>> locations;
    std::set<std::string> names;
    for (std::size_t i = 0; i < m.bindings.size(); ++i) {
        const auto& b = m.bindings[i];
        const auto p = "bindings[" + std::to_string(i) + "]";
        if (b.name.empty() || b.name.size() > 256) return invalid(p + ".name", "must contain 1 to 256 bytes");
        if (!names.insert(b.name).second) return invalid(p + ".name", "duplicate binding name");
        if (!locations.emplace(b.set, b.binding).second)
            return invalid(p, "duplicate (set, binding) location");
    }
    if (m.push_constant_size > 65536)
        return invalid("push_constant_size", "exceeds portable metadata limit of 65536 bytes");
    if (m.push_constants.size() > 4096) return invalid("push_constants", "too many fields");
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    names.clear();
    for (std::size_t i = 0; i < m.push_constants.size(); ++i) {
        const auto& f = m.push_constants[i];
        const auto p = "push_constants[" + std::to_string(i) + "]";
        if (f.name.empty() || f.name.size() > 256 || f.type.empty() || f.type.size() > 64)
            return invalid(p, "field name and type must be nonempty and bounded");
        if (!names.insert(f.name).second) return invalid(p + ".name", "duplicate field name");
        if (f.size == 0) return invalid(p + ".size", "must be greater than zero");
        const std::uint64_t end = static_cast<std::uint64_t>(f.offset) + f.size;
        if (end > m.push_constant_size) return invalid(p, "field range exceeds push_constant_size");
        ranges.emplace_back(f.offset, end);
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i)
        if (ranges[i].first < ranges[i - 1].second) return invalid("push_constants", "field ranges overlap");
    for (std::size_t i = 0; i < m.local_size.size(); ++i)
        if (m.local_size[i] == 0)
            return invalid("local_size[" + std::to_string(i) + "]", "must be greater than zero");
    std::uint64_t invocations = 1;
    for (auto n : m.local_size) {
        if (invocations > std::numeric_limits<std::uint32_t>::max() / n)
            return invalid("local_size", "dimension product exceeds uint32 range");
        invocations *= n;
    }
    if (m.cpu_fallback_key.size() > 256) return invalid("cpu_fallback_key", "exceeds 256 bytes");
    if (m.specialization_constants.size() > 4096)
        return invalid("specialization_constants", "too many constants");
    std::set<std::uint32_t> ids;
    names.clear();
    for (std::size_t i = 0; i < m.specialization_constants.size(); ++i) {
        const auto& s = m.specialization_constants[i];
        const auto p = "specialization_constants[" + std::to_string(i) + "]";
        if (!ids.insert(s.id).second) return invalid(p + ".id", "duplicate specialization id");
        if (s.name.empty() || s.name.size() > 256 || s.type.empty() || s.type.size() > 64)
            return invalid(p, "name and type must be nonempty and bounded");
        if (!names.insert(s.name).second) return invalid(p + ".name", "duplicate specialization name");
        if (const auto* d = std::get_if<double>(&s.default_value); d && !std::isfinite(*d))
            return invalid(p + ".default", "must be finite");
        const auto is_boolean = std::holds_alternative<bool>(s.default_value);
        const auto is_signed = std::holds_alternative<std::int64_t>(s.default_value);
        const auto is_unsigned = std::holds_alternative<std::uint64_t>(s.default_value);
        const auto is_float = std::holds_alternative<double>(s.default_value);
        if ((s.type == "bool" && !is_boolean) || ((s.type == "i32" || s.type == "i64") && !is_signed) ||
            ((s.type == "u32" || s.type == "u64") && !is_unsigned) ||
            ((s.type == "f32" || s.type == "f64") && !is_float))
            return invalid(p + ".default", "default value does not match declared scalar type");
        if (s.type != "bool" && s.type != "i32" && s.type != "i64" && s.type != "u32" && s.type != "u64" &&
            s.type != "f32" && s.type != "f64")
            return invalid(p + ".type", "unsupported specialization scalar type");
        if (s.type == "i32") {
            const auto value = std::get<std::int64_t>(s.default_value);
            if (value < INT32_MIN || value > INT32_MAX) return invalid(p + ".default", "exceeds i32 range");
        }
        if (s.type == "u32" && std::get<std::uint64_t>(s.default_value) > UINT32_MAX)
            return invalid(p + ".default", "exceeds u32 range");
        if (s.type == "f32" &&
            std::abs(std::get<double>(s.default_value)) > std::numeric_limits<float>::max())
            return invalid(p + ".default", "exceeds finite f32 range");
    }
    if (m.semantics.size() > 256) return invalid("semantics", "too many entries");
    names.clear();
    for (std::size_t i = 0; i < m.semantics.size(); ++i) {
        if (m.semantics[i].empty() || m.semantics[i].size() > 256)
            return invalid("semantics[" + std::to_string(i) + "]", "must contain 1 to 256 bytes");
        if (!names.insert(m.semantics[i]).second)
            return invalid("semantics[" + std::to_string(i) + "]", "duplicate semantic");
    }
    return {};
}

Result<KernelMetadata> parse_kernel_metadata(std::string_view document) {
    auto parsed = detail::json::parse(document);
    if (!parsed) return std::move(parsed).error();
    const auto& root = parsed.value();
    auto status = fields(root,
                         {"schema_version", "kernel_id", "bindings", "push_constant_size", "push_constants",
                          "specialization_constants", "local_size", "cpu_fallback_key", "semantics"},
                         "$");
    if (!status) return std::move(status).error();
    KernelMetadata m;
    Error err;
    auto* v = required(root, "schema_version", "$", err);
    if (!v) return err;
    auto version = uint32_field(*v, "schema_version");
    if (!version) return std::move(version).error();
    m.schema_version = version.value();
    v = required(root, "kernel_id", "$", err);
    if (!v) return err;
    auto id = string_field(*v, "kernel_id");
    if (!id) return std::move(id).error();
    m.kernel_id = std::move(id).value();
    v = required(root, "bindings", "$", err);
    if (!v) return err;
    const auto* bindings = array(*v);
    if (!bindings) return invalid("bindings", "expected an array");
    for (std::size_t i = 0; i < bindings->size(); ++i) {
        auto b = parse_binding((*bindings)[i], i);
        if (!b) return std::move(b).error();
        m.bindings.push_back(std::move(b).value());
    }
    v = required(root, "push_constant_size", "$", err);
    if (!v) return err;
    auto pcs = uint32_field(*v, "push_constant_size");
    if (!pcs) return std::move(pcs).error();
    m.push_constant_size = pcs.value();
    v = required(root, "push_constants", "$", err);
    if (!v) return err;
    const auto* push = array(*v);
    if (!push) return invalid("push_constants", "expected an array");
    for (std::size_t i = 0; i < push->size(); ++i) {
        auto f = parse_push_field((*push)[i], i);
        if (!f) return std::move(f).error();
        m.push_constants.push_back(std::move(f).value());
    }
    v = required(root, "specialization_constants", "$", err);
    if (!v) return err;
    const auto* specs = array(*v);
    if (!specs) return invalid("specialization_constants", "expected an array");
    for (std::size_t i = 0; i < specs->size(); ++i) {
        auto s = parse_spec((*specs)[i], i);
        if (!s) return std::move(s).error();
        m.specialization_constants.push_back(std::move(s).value());
    }
    v = required(root, "local_size", "$", err);
    if (!v) return err;
    const auto* local = array(*v);
    if (!local || local->size() != 3)
        return invalid("local_size", "expected an array of exactly three integers");
    for (std::size_t i = 0; i < 3; ++i) {
        auto n = uint32_field((*local)[i], "local_size[" + std::to_string(i) + "]");
        if (!n) return std::move(n).error();
        m.local_size[i] = n.value();
    }
    if (const auto* cpu = root.find("cpu_fallback_key")) {
        auto key = string_field(*cpu, "cpu_fallback_key");
        if (!key) return std::move(key).error();
        m.cpu_fallback_key = std::move(key).value();
    }
    if (const auto* value = root.find("semantics")) {
        const auto* semantics = array(*value);
        if (!semantics) return invalid("semantics", "expected an array");
        for (std::size_t i = 0; i < semantics->size(); ++i) {
            auto s = string_field((*semantics)[i], "semantics[" + std::to_string(i) + "]");
            if (!s) return std::move(s).error();
            m.semantics.push_back(std::move(s).value());
        }
    }
    status = validate_kernel_metadata(m);
    if (!status) return std::move(status).error();
    return m;
}

}  // namespace kora
