#include "kernel/spirv.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <limits>
#include <optional>
#include <spirv/unified1/spirv.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef KORA_HAVE_SPIRV_TOOLS
#include <spirv-tools/libspirv.h>

#include <spirv-tools/libspirv.hpp>
#endif

namespace kora::detail::spirv {
namespace {

constexpr std::uint32_t kMagic = 0x07230203U;
constexpr std::size_t kHeaderWords = 5;
constexpr std::size_t kMaximumModuleWords = 4U * 1024U * 1024U;

Error invalid(std::string message) {
    return Error(ErrorCode::InvalidKernel, "SPIR-V: " + std::move(message));
}

std::uint64_t member_key(std::uint32_t type, std::uint32_t member) {
    return (static_cast<std::uint64_t>(type) << 32U) | member;
}

struct Decorations {
    std::optional<std::uint32_t> set;
    std::optional<std::uint32_t> binding;
    std::optional<std::uint32_t> spec_id;
    std::optional<std::uint32_t> array_stride;
    bool block = false;
    bool buffer_block = false;
    bool non_writable = false;
    bool non_readable = false;
};

struct MemberDecorations {
    std::optional<std::uint32_t> offset;
    std::optional<std::uint32_t> matrix_stride;
    bool row_major = false;
    bool non_writable = false;
    bool non_readable = false;
};

enum class TypeKind {
    Unknown,
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
    SampledImage
};

struct Type {
    TypeKind kind = TypeKind::Unknown;
    std::uint32_t width = 0;
    bool is_signed = false;
    std::uint32_t element = 0;
    std::uint32_t count = 0;
    std::uint32_t storage_class = 0;
    std::uint32_t sampled = 0;
    std::vector<std::uint32_t> members;
};

struct Variable {
    std::uint32_t pointer_type = 0;
    std::uint32_t id = 0;
    std::uint32_t storage_class = 0;
};

struct EntryPoint {
    std::uint32_t model = 0;
    std::uint32_t id = 0;
    std::string name;
};

struct ExecutionSize {
    std::array<std::uint32_t, 3> dimensions{};
    bool id_based = false;
};

struct SpecConstant {
    std::uint32_t type_id = 0;
    std::uint32_t result_id = 0;
    std::uint64_t bits = 0;
    bool boolean = false;
    bool value = false;
};

class ModuleParser {
public:
    explicit ModuleParser(std::span<const std::uint32_t> words) : words_(words) {}

    Result<Reflection> run(const KernelMetadata& metadata, const DeviceCapabilities& device,
                           std::string_view selected_entry_point) {
        auto header = validate_header();
        if (!header) return std::move(header).error();
        auto instructions = parse_instructions();
        if (!instructions) return std::move(instructions).error();

        Reflection reflection;
        reflection.capabilities.assign(capabilities_.begin(), capabilities_.end());
        auto capability_status = validate_capabilities(reflection.capabilities, device);
        if (!capability_status) return std::move(capability_status).error();

        const auto entry = std::find_if(entry_points_.begin(), entry_points_.end(),
                                        [&](const auto& item) { return item.name == selected_entry_point; });
        if (entry == entry_points_.end())
            return invalid("entry point '" + std::string(selected_entry_point) + "' was not found");
        if (entry->model != static_cast<std::uint32_t>(spv::ExecutionModelGLCompute))
            return invalid("entry point '" + entry->name + "' is not a compute entry point");
        reflection.entry_point = entry->name;

        const auto local_size = execution_sizes_.find(entry->id);
        if (local_size == execution_sizes_.end()) return invalid("compute entry point has no LocalSize mode");
        if (local_size->second.id_based)
            return invalid("LocalSizeId execution modes are not supported by metadata version 1");
        reflection.local_size = local_size->second.dimensions;
        if (std::any_of(reflection.local_size.begin(), reflection.local_size.end(),
                        [](std::uint32_t dimension) { return dimension == 0; }))
            return invalid("LocalSize dimensions must all be nonzero");

        auto bindings = reflect_bindings();
        if (!bindings) return std::move(bindings).error();
        reflection.bindings = std::move(bindings).value();
        auto push = reflect_push_constants();
        if (!push) return std::move(push).error();
        reflection.push_constant_size = push.value().first;
        reflection.push_constants = std::move(push).value().second;
        auto specs = reflect_specialization_constants();
        if (!specs) return std::move(specs).error();
        reflection.specialization_constants = std::move(specs).value();

        auto metadata_status = compare_metadata(reflection, metadata);
        if (!metadata_status) return std::move(metadata_status).error();
        return reflection;
    }

private:
    Status validate_header() const {
        if (words_.size() < kHeaderWords) return invalid("module is shorter than the five-word header");
        if (words_.size() > kMaximumModuleWords) return invalid("module exceeds the 16 MiB word limit");
        if (words_[0] != kMagic) return invalid("invalid magic number or byte order");
        const auto version = words_[1];
        const auto major = (version >> 16U) & 0xffU;
        const auto minor = (version >> 8U) & 0xffU;
        if ((version & 0xff0000ffU) != 0 || major != 1 || minor > 6)
            return invalid("unsupported or malformed SPIR-V version word");
        if (words_[3] == 0) return invalid("id bound must be nonzero");
        if (words_[4] != 0) return invalid("reserved schema word must be zero");
        return {};
    }

    bool valid_id(std::uint32_t id) const { return id != 0 && id < words_[3]; }

    Result<std::pair<std::string, std::size_t>> read_string(std::size_t first, std::size_t end) const {
        std::string value;
        for (std::size_t i = first; i < end; ++i) {
            const auto word = words_[i];
            for (unsigned byte = 0; byte < 4; ++byte) {
                const char c = static_cast<char>((word >> (byte * 8U)) & 0xffU);
                if (c == '\0') return std::pair{std::move(value), i + 1};
                value.push_back(c);
            }
        }
        return invalid("instruction contains an unterminated literal string");
    }

    Status parse_instructions() {
        std::size_t cursor = kHeaderWords;
        while (cursor < words_.size()) {
            const auto instruction = words_[cursor];
            const auto word_count = static_cast<std::size_t>(instruction >> 16U);
            const auto opcode = static_cast<std::uint32_t>(instruction & 0xffffU);
            if (word_count == 0 || word_count > words_.size() - cursor)
                return invalid("instruction at word " + std::to_string(cursor) +
                               " has an invalid word count");
            const auto* operands = words_.data() + cursor;
            const auto operand_count = word_count - 1;
            const auto operand = [&](std::size_t index) -> std::uint32_t { return operands[index + 1]; };
            const auto op = [&](spv::Op value) { return opcode == static_cast<std::uint32_t>(value); };

            if (op(spv::OpCapability)) {
                if (operand_count != 1) return invalid("OpCapability has an invalid word count");
                capabilities_.insert(operand(0));
            } else if (op(spv::OpEntryPoint)) {
                if (operand_count < 3) return invalid("OpEntryPoint is truncated");
                const auto id = operand(1);
                if (!valid_id(id)) return invalid("OpEntryPoint uses an invalid function id");
                auto name = read_string(cursor + 3, cursor + word_count);
                if (!name) return std::move(name).error();
                entry_points_.push_back({operand(0), id, std::move(name).value().first});
            } else if (op(spv::OpExecutionMode) || op(spv::OpExecutionModeId)) {
                if (operand_count < 2) return invalid("execution-mode instruction is truncated");
                const auto id = operand(0);
                if (!valid_id(id)) return invalid("execution mode uses an invalid entry-point id");
                const auto local_mode = static_cast<std::uint32_t>(spv::ExecutionModeLocalSize);
                const auto local_mode_id = static_cast<std::uint32_t>(spv::ExecutionModeLocalSizeId);
                if (operand(1) == local_mode || operand(1) == local_mode_id) {
                    if (operand_count != 5)
                        return invalid("LocalSize execution mode must contain three dimensions");
                    ExecutionSize size{{operand(2), operand(3), operand(4)}, op(spv::OpExecutionModeId)};
                    if (!execution_sizes_.emplace(id, size).second)
                        return invalid("entry point has duplicate LocalSize execution modes");
                }
            } else if (op(spv::OpName)) {
                if (operand_count < 2 || !valid_id(operand(0))) return invalid("OpName is malformed");
                auto name = read_string(cursor + 2, cursor + word_count);
                if (!name) return std::move(name).error();
                names_[operand(0)] = std::move(name).value().first;
            } else if (op(spv::OpMemberName)) {
                if (operand_count < 3 || !valid_id(operand(0))) return invalid("OpMemberName is malformed");
                auto name = read_string(cursor + 3, cursor + word_count);
                if (!name) return std::move(name).error();
                member_names_[member_key(operand(0), operand(1))] = std::move(name).value().first;
            } else if (op(spv::OpDecorate)) {
                if (operand_count < 2 || !valid_id(operand(0))) return invalid("OpDecorate is malformed");
                auto status = parse_decoration(operand(0), operand(1), operand_count, operand);
                if (!status) return std::move(status).error();
            } else if (op(spv::OpMemberDecorate)) {
                if (operand_count < 3 || !valid_id(operand(0)))
                    return invalid("OpMemberDecorate is malformed");
                auto status =
                    parse_member_decoration(operand(0), operand(1), operand(2), operand_count, operand);
                if (!status) return std::move(status).error();
            } else if (is_type_instruction(opcode)) {
                auto status = parse_type(opcode, operand_count, operand);
                if (!status) return std::move(status).error();
            } else if (op(spv::OpVariable)) {
                if (operand_count < 3 || operand_count > 4 || !valid_id(operand(1)))
                    return invalid("OpVariable is malformed");
                variables_.push_back({operand(0), operand(1), operand(2)});
            } else if (op(spv::OpConstant) || op(spv::OpConstantNull) || op(spv::OpSpecConstant) ||
                       op(spv::OpSpecConstantTrue) || op(spv::OpSpecConstantFalse)) {
                auto status = parse_constant(opcode, operand_count, operand);
                if (!status) return std::move(status).error();
            } else if (op(spv::OpSpecConstantOp)) {
                if (operand_count < 3 || !valid_id(operand(1)))
                    return invalid("OpSpecConstantOp is malformed");
                spec_op_ids_.insert(operand(1));
            }
            cursor += word_count;
        }
        return {};
    }

    static bool is_type_instruction(std::uint32_t opcode) {
        return opcode >= static_cast<std::uint32_t>(spv::OpTypeVoid) &&
               opcode <= static_cast<std::uint32_t>(spv::OpTypeForwardPointer);
    }

    template <class Operand>
    Status parse_decoration(std::uint32_t target, std::uint32_t decoration, std::size_t count,
                            Operand operand) {
        auto& d = decorations_[target];
        const auto set_value = [&](std::optional<std::uint32_t>& out) -> Status {
            if (count != 3) return invalid("decoration requires one literal operand");
            if (out) return invalid("duplicate decoration on id " + std::to_string(target));
            out = operand(2);
            return {};
        };
        switch (decoration) {
            case spv::DecorationSpecId:
                return set_value(d.spec_id);
            case spv::DecorationDescriptorSet:
                return set_value(d.set);
            case spv::DecorationBinding:
                return set_value(d.binding);
            case spv::DecorationArrayStride:
                return set_value(d.array_stride);
            case spv::DecorationBlock:
                if (count != 2) return invalid("Block decoration has unexpected operands");
                d.block = true;
                break;
            case spv::DecorationBufferBlock:
                if (count != 2) return invalid("BufferBlock decoration has unexpected operands");
                d.buffer_block = true;
                break;
            case spv::DecorationNonWritable:
                d.non_writable = true;
                break;
            case spv::DecorationNonReadable:
                d.non_readable = true;
                break;
            default:
                break;
        }
        return {};
    }

    template <class Operand>
    Status parse_member_decoration(std::uint32_t target, std::uint32_t member, std::uint32_t decoration,
                                   std::size_t count, Operand operand) {
        auto& d = member_decorations_[member_key(target, member)];
        const auto set_value = [&](std::optional<std::uint32_t>& out) -> Status {
            if (count != 4) return invalid("member layout decoration requires one literal operand");
            if (out) return invalid("duplicate member layout decoration");
            out = operand(3);
            return {};
        };
        switch (decoration) {
            case spv::DecorationOffset:
                return set_value(d.offset);
            case spv::DecorationMatrixStride:
                return set_value(d.matrix_stride);
            case spv::DecorationRowMajor:
                d.row_major = true;
                break;
            case spv::DecorationNonWritable:
                d.non_writable = true;
                break;
            case spv::DecorationNonReadable:
                d.non_readable = true;
                break;
            default:
                break;
        }
        return {};
    }

    template <class Operand>
    Status parse_type(std::uint32_t opcode, std::size_t count, Operand operand) {
        if (count < 1) return invalid("type declaration is truncated");
        const auto result_id = operand(0);
        if (!valid_id(result_id)) return invalid("type declaration uses an invalid result id");
        Type type;
        const auto op = static_cast<spv::Op>(opcode);
        switch (op) {
            case spv::OpTypeBool:
                if (count != 1) return invalid("OpTypeBool is malformed");
                type.kind = TypeKind::Bool;
                break;
            case spv::OpTypeInt:
                if (count != 3) return invalid("OpTypeInt is malformed");
                type.kind = TypeKind::Integer;
                type.width = operand(1);
                type.is_signed = operand(2) != 0;
                break;
            case spv::OpTypeFloat:
                if (count != 2) return invalid("OpTypeFloat is malformed");
                type.kind = TypeKind::Float;
                type.width = operand(1);
                break;
            case spv::OpTypeVector:
            case spv::OpTypeMatrix:
                if (count != 3) return invalid("vector or matrix type is malformed");
                type.kind = op == spv::OpTypeVector ? TypeKind::Vector : TypeKind::Matrix;
                type.element = operand(1);
                type.count = operand(2);
                break;
            case spv::OpTypeArray:
                if (count != 3) return invalid("OpTypeArray is malformed");
                type.kind = TypeKind::Array;
                type.element = operand(1);
                type.count = operand(2);
                break;
            case spv::OpTypeRuntimeArray:
                if (count != 2) return invalid("OpTypeRuntimeArray is malformed");
                type.kind = TypeKind::RuntimeArray;
                type.element = operand(1);
                break;
            case spv::OpTypeStruct:
                type.kind = TypeKind::Struct;
                for (std::size_t i = 1; i < count; ++i) type.members.push_back(operand(i));
                break;
            case spv::OpTypePointer:
                if (count != 3) return invalid("OpTypePointer is malformed");
                type.kind = TypeKind::Pointer;
                type.storage_class = operand(1);
                type.element = operand(2);
                break;
            case spv::OpTypeImage:
                if (count < 8 || count > 9) return invalid("OpTypeImage is malformed");
                type.kind = TypeKind::Image;
                type.element = operand(1);
                type.sampled = operand(6);
                break;
            case spv::OpTypeSampler:
                if (count != 1) return invalid("OpTypeSampler is malformed");
                type.kind = TypeKind::Sampler;
                break;
            case spv::OpTypeSampledImage:
                if (count != 2) return invalid("OpTypeSampledImage is malformed");
                type.kind = TypeKind::SampledImage;
                type.element = operand(1);
                break;
            default:
                return {};
        }
        if (types_.contains(result_id)) return invalid("duplicate type result id");
        types_.emplace(result_id, std::move(type));
        return {};
    }

    template <class Operand>
    Status parse_constant(std::uint32_t opcode, std::size_t count, Operand operand) {
        if (count < 2) return invalid("constant instruction is truncated");
        const auto type_id = operand(0);
        const auto result_id = operand(1);
        if (!valid_id(type_id) || !valid_id(result_id)) return invalid("constant uses an invalid id");
        std::uint64_t bits = 0;
        bool is_boolean = false;
        bool boolean_value = false;
        const auto op = static_cast<spv::Op>(opcode);
        if (op == spv::OpConstantNull) {
            if (count != 2) return invalid("OpConstantNull is malformed");
        } else if (op == spv::OpSpecConstantTrue || op == spv::OpSpecConstantFalse) {
            if (count != 2) return invalid("boolean specialization constant is malformed");
            is_boolean = true;
            boolean_value = op == spv::OpSpecConstantTrue;
        } else {
            if (count < 3 || count > 4) return invalid("scalar constant has an unsupported literal width");
            bits = operand(2);
            if (count == 4) bits |= static_cast<std::uint64_t>(operand(3)) << 32U;
        }
        constants_[result_id] = bits;
        if (op == spv::OpSpecConstant || op == spv::OpSpecConstantTrue || op == spv::OpSpecConstantFalse)
            spec_constants_.push_back({type_id, result_id, bits, is_boolean, boolean_value});
        return {};
    }

    Result<std::uint64_t> type_size(std::uint32_t id, std::optional<MemberDecorations> layout = {},
                                    std::size_t depth = 0) const {
        if (depth > 32) return invalid("type nesting exceeds 32 levels");
        const auto found = types_.find(id);
        if (found == types_.end()) return invalid("type id " + std::to_string(id) + " is undeclared");
        const auto& type = found->second;
        switch (type.kind) {
            case TypeKind::Bool:
                return std::uint64_t{4};
            case TypeKind::Integer:
            case TypeKind::Float:
                if (type.width == 0 || (type.width % 8) != 0 || type.width > 64)
                    return invalid("unsupported scalar width in type id " + std::to_string(id));
                return static_cast<std::uint64_t>(type.width / 8);
            case TypeKind::Vector: {
                auto component = type_size(type.element, {}, depth + 1);
                if (!component) return std::move(component).error();
                return checked_multiply(component.value(), type.count);
            }
            case TypeKind::Matrix: {
                auto column = type_size(type.element, {}, depth + 1);
                if (!column) return std::move(column).error();
                const auto stride = layout && layout->matrix_stride ? *layout->matrix_stride : column.value();
                return checked_multiply(stride, type.count);
            }
            case TypeKind::Array: {
                const auto length = constants_.find(type.count);
                if (length == constants_.end()) return invalid("array length is not a literal constant");
                auto element = type_size(type.element, {}, depth + 1);
                if (!element) return std::move(element).error();
                const auto decoration = decorations_.find(id);
                const auto stride = decoration != decorations_.end() && decoration->second.array_stride
                                        ? *decoration->second.array_stride
                                        : element.value();
                return checked_multiply(stride, length->second);
            }
            case TypeKind::Struct: {
                std::uint64_t end = 0;
                for (std::size_t member = 0; member < type.members.size(); ++member) {
                    const auto d =
                        member_decorations_.find(member_key(id, static_cast<std::uint32_t>(member)));
                    if (d == member_decorations_.end() || !d->second.offset)
                        return invalid("push-constant structure member lacks an Offset decoration");
                    auto size = type_size(
                        type.members[member],
                        d == member_decorations_.end() ? std::optional<MemberDecorations>{} : d->second,
                        depth + 1);
                    if (!size) return std::move(size).error();
                    auto member_end = checked_add(*d->second.offset, size.value());
                    if (!member_end) return std::move(member_end).error();
                    end = std::max(end, member_end.value());
                }
                return end;
            }
            case TypeKind::RuntimeArray:
            case TypeKind::Pointer:
            case TypeKind::Image:
            case TypeKind::Sampler:
            case TypeKind::SampledImage:
            case TypeKind::Unknown:
                return invalid("type has no fixed byte size");
        }
        return invalid("unknown type kind");
    }

    static Result<std::uint64_t> checked_multiply(std::uint64_t a, std::uint64_t b) {
        if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b)
            return invalid("type size multiplication overflows uint64");
        return a * b;
    }

    static Result<std::uint64_t> checked_add(std::uint64_t a, std::uint64_t b) {
        if (a > std::numeric_limits<std::uint64_t>::max() - b)
            return invalid("type size addition overflows uint64");
        return a + b;
    }

    Result<std::string> type_name(std::uint32_t id, std::size_t depth = 0) const {
        if (depth > 32) return invalid("type nesting exceeds 32 levels");
        const auto found = types_.find(id);
        if (found == types_.end()) return invalid("type id " + std::to_string(id) + " is undeclared");
        const auto& type = found->second;
        switch (type.kind) {
            case TypeKind::Bool:
                return std::string("bool");
            case TypeKind::Integer:
                return std::string(type.is_signed ? "i" : "u") + std::to_string(type.width);
            case TypeKind::Float:
                return std::string("f") + std::to_string(type.width);
            case TypeKind::Vector: {
                auto element = type_name(type.element, depth + 1);
                if (!element) return std::move(element).error();
                return "vec" + std::to_string(type.count) + "<" + element.value() + ">";
            }
            case TypeKind::Matrix: {
                const auto column = types_.find(type.element);
                if (column == types_.end() || column->second.kind != TypeKind::Vector)
                    return invalid("matrix column type is invalid");
                auto element = type_name(column->second.element, depth + 1);
                if (!element) return std::move(element).error();
                return "mat" + std::to_string(type.count) + "x" + std::to_string(column->second.count) + "<" +
                       element.value() + ">";
            }
            case TypeKind::Array: {
                const auto length = constants_.find(type.count);
                if (length == constants_.end()) return invalid("array length is not a literal constant");
                auto element = type_name(type.element, depth + 1);
                if (!element) return std::move(element).error();
                return "array[" + std::to_string(length->second) + "]<" + element.value() + ">";
            }
            case TypeKind::RuntimeArray:
                return invalid("runtime arrays are not valid metadata fields");
            case TypeKind::Struct:
                return std::string("struct");
            case TypeKind::Pointer:
                return type_name(type.element, depth + 1);
            case TypeKind::Image:
                return std::string(type.sampled == 2 ? "storage_image" : "sampled_image");
            case TypeKind::Sampler:
                return std::string("sampler");
            case TypeKind::SampledImage:
                return std::string("sampled_image");
            case TypeKind::Unknown:
                return invalid("unsupported type in metadata interface");
        }
        return invalid("unknown type kind");
    }

    Result<std::uint32_t> pointee_type(std::uint32_t pointer_id, std::uint32_t storage_class) const {
        const auto found = types_.find(pointer_id);
        if (found == types_.end() || found->second.kind != TypeKind::Pointer)
            return invalid("variable result type is not an OpTypePointer");
        if (found->second.storage_class != storage_class)
            return invalid("variable storage class does not match its pointer type");
        return found->second.element;
    }

    Result<KernelBindingKind> binding_kind(const Variable& variable, std::uint32_t pointee) const {
        const auto type = types_.find(pointee);
        if (variable.storage_class == static_cast<std::uint32_t>(spv::StorageClassStorageBuffer))
            return KernelBindingKind::StorageBuffer;
        if (variable.storage_class == static_cast<std::uint32_t>(spv::StorageClassUniform)) {
            const auto decoration = decorations_.find(pointee);
            if (decoration != decorations_.end() && decoration->second.buffer_block)
                return KernelBindingKind::StorageBuffer;
            if (decoration != decorations_.end() && decoration->second.block)
                return KernelBindingKind::UniformBuffer;
            return invalid("Uniform descriptor type lacks Block or BufferBlock decoration");
        }
        if (variable.storage_class == static_cast<std::uint32_t>(spv::StorageClassUniformConstant)) {
            if (type == types_.end()) return invalid("descriptor type is undeclared");
            if (type->second.kind == TypeKind::Image)
                return type->second.sampled == 2 ? KernelBindingKind::StorageImage
                                                 : KernelBindingKind::SampledImage;
            if (type->second.kind == TypeKind::SampledImage) return KernelBindingKind::SampledImage;
            return invalid("sampler-only or unsupported UniformConstant descriptor type");
        }
        return invalid("descriptor uses an unsupported storage class");
    }

    KernelAccess binding_access(const Variable& variable, std::uint32_t pointee) const {
        bool any_read = true;
        bool any_write = true;
        const auto variable_decoration = decorations_.find(variable.id);
        const auto type_decoration = decorations_.find(pointee);
        if (variable_decoration != decorations_.end()) {
            if (variable_decoration->second.non_writable) any_write = false;
            if (variable_decoration->second.non_readable) any_read = false;
        }
        if (type_decoration != decorations_.end()) {
            if (type_decoration->second.non_writable) any_write = false;
            if (type_decoration->second.non_readable) any_read = false;
        }
        const auto type = types_.find(pointee);
        if (type != types_.end() && type->second.kind == TypeKind::Struct && !type->second.members.empty()) {
            bool every_member_non_writable = true;
            bool every_member_non_readable = true;
            for (std::size_t i = 0; i < type->second.members.size(); ++i) {
                const auto d = member_decorations_.find(member_key(pointee, static_cast<std::uint32_t>(i)));
                every_member_non_writable &= d != member_decorations_.end() && d->second.non_writable;
                every_member_non_readable &= d != member_decorations_.end() && d->second.non_readable;
            }
            if (every_member_non_writable) any_write = false;
            if (every_member_non_readable) any_read = false;
        }
        if (any_read && any_write) return KernelAccess::ReadWrite;
        if (any_read) return KernelAccess::Read;
        if (any_write) return KernelAccess::Write;
        return KernelAccess::ReadWrite;
    }

    Result<std::vector<ReflectedBinding>> reflect_bindings() const {
        std::vector<ReflectedBinding> result;
        std::unordered_set<std::uint64_t> locations;
        for (const auto& variable : variables_) {
            const auto decoration = decorations_.find(variable.id);
            if (decoration == decorations_.end()) continue;
            const bool has_location =
                decoration->second.set.has_value() || decoration->second.binding.has_value();
            if (!has_location) continue;
            if (!decoration->second.set || !decoration->second.binding)
                return invalid("descriptor variable must have both DescriptorSet and Binding decorations");
            auto pointee = pointee_type(variable.pointer_type, variable.storage_class);
            if (!pointee) return std::move(pointee).error();
            auto kind = binding_kind(variable, pointee.value());
            if (!kind) return std::move(kind).error();
            const auto location =
                (static_cast<std::uint64_t>(*decoration->second.set) << 32U) | *decoration->second.binding;
            if (!locations.insert(location).second)
                return invalid("duplicate descriptor set/binding in module");
            KernelAccess access = binding_access(variable, pointee.value());
            if (kind.value() != KernelBindingKind::StorageBuffer &&
                kind.value() != KernelBindingKind::StorageImage)
                access = KernelAccess::Read;
            const auto name = names_.find(variable.id);
            result.push_back({*decoration->second.set, *decoration->second.binding,
                              name == names_.end() ? std::string{} : name->second, kind.value(), access});
        }
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            return std::pair{a.set, a.binding} < std::pair{b.set, b.binding};
        });
        return result;
    }

    Result<std::pair<std::uint32_t, std::vector<PushConstantFieldMetadata>>> reflect_push_constants() const {
        std::optional<std::uint32_t> block_type;
        for (const auto& variable : variables_) {
            if (variable.storage_class != static_cast<std::uint32_t>(spv::StorageClassPushConstant)) continue;
            if (block_type) return invalid("module declares more than one push-constant block");
            auto pointee = pointee_type(variable.pointer_type, variable.storage_class);
            if (!pointee) return std::move(pointee).error();
            block_type = pointee.value();
        }
        if (!block_type) return std::pair<std::uint32_t, std::vector<PushConstantFieldMetadata>>{0, {}};
        const auto type = types_.find(*block_type);
        if (type == types_.end() || type->second.kind != TypeKind::Struct)
            return invalid("push-constant variable does not point to a structure");
        auto total_size = type_size(*block_type);
        if (!total_size) return std::move(total_size).error();
        if (total_size.value() > std::numeric_limits<std::uint32_t>::max())
            return invalid("push-constant block exceeds uint32 range");
        std::vector<PushConstantFieldMetadata> fields;
        for (std::size_t i = 0; i < type->second.members.size(); ++i) {
            const auto key = member_key(*block_type, static_cast<std::uint32_t>(i));
            const auto member_decoration = member_decorations_.find(key);
            if (member_decoration == member_decorations_.end() || !member_decoration->second.offset)
                return invalid("push-constant field is missing an Offset decoration");
            const auto member_type = type->second.members[i];
            auto size = type_size(member_type, member_decoration->second);
            if (!size) return std::move(size).error();
            if (size.value() > std::numeric_limits<std::uint32_t>::max())
                return invalid("push-constant field exceeds uint32 range");
            auto name = member_names_.find(key);
            auto type_label = type_name(member_type);
            if (!type_label) return std::move(type_label).error();
            fields.push_back({name == member_names_.end() ? std::string{} : name->second,
                              std::move(type_label).value(), *member_decoration->second.offset,
                              static_cast<std::uint32_t>(size.value())});
        }
        return std::pair{static_cast<std::uint32_t>(total_size.value()), std::move(fields)};
    }

    Result<std::vector<SpecializationConstantMetadata>> reflect_specialization_constants() const {
        std::vector<SpecializationConstantMetadata> result;
        for (const auto result_id : spec_op_ids_) {
            const auto decoration = decorations_.find(result_id);
            if (decoration != decorations_.end() && decoration->second.spec_id)
                return invalid("OpSpecConstantOp with SpecId is not supported by metadata version 1");
        }
        for (const auto& constant : spec_constants_) {
            const auto decoration = decorations_.find(constant.result_id);
            if (decoration == decorations_.end() || !decoration->second.spec_id) continue;
            auto type = types_.find(constant.type_id);
            if (type == types_.end()) return invalid("specialization constant type is undeclared");
            SpecializationValue value;
            std::string type_label;
            if (constant.boolean) {
                value = constant.value;
                type_label = "bool";
            } else if (type->second.kind == TypeKind::Integer) {
                const auto width = type->second.width;
                if (width == 0 || width > 64) return invalid("specialization integer width is unsupported");
                if (type->second.is_signed) {
                    std::int64_t signed_value = 0;
                    if (width == 64)
                        signed_value = std::bit_cast<std::int64_t>(constant.bits);
                    else {
                        const auto mask = (std::uint64_t{1} << width) - 1U;
                        auto bits = constant.bits & mask;
                        if ((bits & (std::uint64_t{1} << (width - 1U))) != 0) bits |= ~mask;
                        signed_value = std::bit_cast<std::int64_t>(bits);
                    }
                    value = signed_value;
                    type_label = "i" + std::to_string(width);
                } else {
                    const auto mask = width == 64 ? UINT64_MAX : ((std::uint64_t{1} << width) - 1U);
                    value = constant.bits & mask;
                    type_label = "u" + std::to_string(width);
                }
            } else if (type->second.kind == TypeKind::Float) {
                if (type->second.width == 32) {
                    value =
                        static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(constant.bits)));
                } else if (type->second.width == 64) {
                    value = std::bit_cast<double>(constant.bits);
                } else {
                    return invalid("specialization floating-point width is unsupported");
                }
                type_label = "f" + std::to_string(type->second.width);
            } else {
                return invalid("specialization constant is not a scalar bool, integer, or float");
            }
            const auto name = names_.find(constant.result_id);
            result.push_back({*decoration->second.spec_id,
                              name == names_.end() ? std::string{} : name->second, std::move(type_label),
                              std::move(value)});
        }
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
        return result;
    }

    static bool same_spec_value(const SpecializationConstantMetadata& reflected,
                                const SpecializationConstantMetadata& metadata) {
        if (reflected.type == "f32" && std::holds_alternative<double>(metadata.default_value)) {
            return std::get<double>(reflected.default_value) ==
                   static_cast<double>(static_cast<float>(std::get<double>(metadata.default_value)));
        }
        return reflected.default_value == metadata.default_value;
    }

    Status compare_metadata(const Reflection& reflection, const KernelMetadata& metadata) const {
        if (reflection.local_size != metadata.local_size)
            return invalid("local_size does not match the selected entry point");
        if (reflection.bindings.size() != metadata.bindings.size())
            return invalid("bindings count does not match the module");
        for (std::size_t i = 0; i < metadata.bindings.size(); ++i) {
            const auto& expected = metadata.bindings[i];
            const auto found = std::find_if(
                reflection.bindings.begin(), reflection.bindings.end(), [&](const auto& binding) {
                    return binding.set == expected.set && binding.binding == expected.binding;
                });
            const auto path = "bindings[" + std::to_string(i) + "]";
            if (found == reflection.bindings.end())
                return invalid(path + " location is absent from the module");
            if (found->kind != expected.kind) return invalid(path + ".kind does not match the module");
            if (found->access != expected.access) return invalid(path + ".access does not match the module");
            if (!expected.name.empty() && !found->name.empty() && found->name != expected.name)
                return invalid(path + ".name does not match the module");
        }
        if (reflection.push_constant_size != metadata.push_constant_size)
            return invalid("push_constant_size does not match the reflected block extent");
        if (reflection.push_constants.size() != metadata.push_constants.size())
            return invalid("push_constants count does not match the module");
        for (std::size_t i = 0; i < metadata.push_constants.size(); ++i) {
            const auto& expected = metadata.push_constants[i];
            const auto& reflected = reflection.push_constants[i];
            const auto path = "push_constants[" + std::to_string(i) + "]";
            if (reflected.offset != expected.offset)
                return invalid(path + ".offset does not match the module");
            if (reflected.size != expected.size) return invalid(path + ".size does not match the module");
            if (reflected.type != expected.type) return invalid(path + ".type does not match the module");
            if (!expected.name.empty() && !reflected.name.empty() && reflected.name != expected.name)
                return invalid(path + ".name does not match the module");
        }
        if (reflection.specialization_constants.size() != metadata.specialization_constants.size())
            return invalid("specialization_constants count does not match the module");
        for (std::size_t i = 0; i < metadata.specialization_constants.size(); ++i) {
            const auto& expected = metadata.specialization_constants[i];
            const auto found = std::find_if(reflection.specialization_constants.begin(),
                                            reflection.specialization_constants.end(),
                                            [&](const auto& spec) { return spec.id == expected.id; });
            const auto path = "specialization_constants[" + std::to_string(i) + "]";
            if (found == reflection.specialization_constants.end())
                return invalid(path + ".id is absent from the module");
            if (found->type != expected.type) return invalid(path + ".type does not match the module");
            if (!expected.name.empty() && !found->name.empty() && found->name != expected.name)
                return invalid(path + ".name does not match the module");
            if (!same_spec_value(*found, expected))
                return invalid(path + ".default does not match the module");
        }
        return {};
    }

    Status validate_capabilities(const std::vector<std::uint32_t>& capabilities,
                                 const DeviceCapabilities& device) const {
        for (const auto capability : capabilities) {
            if (capability == static_cast<std::uint32_t>(spv::CapabilityMatrix) ||
                capability == static_cast<std::uint32_t>(spv::CapabilityShader))
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityInt16) && device.shader_int16)
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityInt64) && device.shader_int64)
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityFloat64) && device.shader_float64)
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityGroupNonUniform) &&
                device.subgroup_supported && device.subgroup_basic)
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformVote) &&
                device.subgroup_vote)
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformArithmetic) &&
                device.subgroup_arithmetic)
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformBallot) &&
                device.subgroup_ballot)
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformShuffle) &&
                device.subgroup_shuffle)
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformShuffleRelative) &&
                device.subgroup_shuffle_relative)
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformClustered) &&
                device.subgroup_clustered)
                continue;
            if (capability == static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformQuad) &&
                device.subgroup_quad)
                continue;
            return invalid("capability " + std::to_string(capability) +
                           " is unsupported or cannot be verified by the selected device");
        }
        return {};
    }

    std::span<const std::uint32_t> words_;
    std::unordered_map<std::uint32_t, Type> types_;
    std::unordered_map<std::uint32_t, Decorations> decorations_;
    std::unordered_map<std::uint64_t, MemberDecorations> member_decorations_;
    std::unordered_map<std::uint32_t, std::string> names_;
    std::unordered_map<std::uint64_t, std::string> member_names_;
    std::unordered_map<std::uint32_t, std::uint64_t> constants_;
    std::unordered_set<std::uint32_t> spec_op_ids_;
    std::unordered_set<std::uint32_t> capabilities_;
    std::vector<Variable> variables_;
    std::vector<EntryPoint> entry_points_;
    std::unordered_map<std::uint32_t, ExecutionSize> execution_sizes_;
    std::vector<SpecConstant> spec_constants_;
};

#ifdef KORA_HAVE_SPIRV_TOOLS
Status validate_with_spirv_tools(std::span<const std::uint32_t> words, const DeviceCapabilities& device) {
    const auto vk_version = (device.api_major << 22U) | (device.api_minor << 12U);
    spv_target_env environment = SPV_ENV_UNIVERSAL_1_0;
    if (!spvParseVulkanEnv(vk_version, words[1], &environment))
        return invalid("SPIRV-Tools has no Vulkan target environment for the device and module versions");
    spvtools::SpirvTools tools(environment);
    std::string diagnostic;
    tools.SetMessageConsumer(
        [&](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
            if (diagnostic.empty()) {
                diagnostic = "word " + std::to_string(position.index) + ": " +
                             (message ? std::string(message) : std::string("validation failed"));
            }
        });
    if (!tools.Validate(words.data(), words.size()))
        return invalid("SPIRV-Tools validation failed" +
                       (diagnostic.empty() ? std::string{} : ": " + diagnostic));
    return {};
}
#endif

}  // namespace

DeviceCapabilities capabilities_from(const DeviceInfo& device) noexcept {
    const auto version = device.api_version();
    const auto& features = device.features();
    const auto& subgroup = device.subgroup();
    return {version.major,           version.minor,      features.shader_int16, features.shader_int64,
            features.shader_float64, subgroup.supported, subgroup.basic,        subgroup.vote,
            subgroup.arithmetic,     subgroup.ballot,    subgroup.shuffle,      subgroup.shuffle_relative,
            subgroup.clustered,      subgroup.quad};
}

Result<Reflection> validate_and_reflect(std::span<const std::uint32_t> words, const KernelMetadata& metadata,
                                        const DeviceCapabilities& device, std::string_view entry_point) {
    auto metadata_status = validate_kernel_metadata(metadata);
    if (!metadata_status) return std::move(metadata_status).error();
    if (words.size() < kHeaderWords) return invalid("module is shorter than the five-word header");
    if (words.size() > kMaximumModuleWords) return invalid("module exceeds the 16 MiB word limit");
    if (words[0] != kMagic) return invalid("invalid magic number or byte order");
#ifdef KORA_HAVE_SPIRV_TOOLS
    auto tools_status = validate_with_spirv_tools(words, device);
    if (!tools_status) return std::move(tools_status).error();
#endif
    return ModuleParser(words).run(metadata, device, entry_point);
}

}  // namespace kora::detail::spirv
