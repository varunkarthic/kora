#pragma once
/// @file kernel_metadata.hpp
/// Versioned, device-independent metadata for a packaged compute kernel.
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "kora/errors.hpp"

namespace kora {

enum class KernelBindingKind { StorageBuffer, UniformBuffer, StorageImage, SampledImage };
enum class KernelAccess { Read, Write, ReadWrite };

struct KernelBindingMetadata {
    std::uint32_t set = 0;
    std::uint32_t binding = 0;
    std::string name;
    KernelBindingKind kind = KernelBindingKind::StorageBuffer;
    KernelAccess access = KernelAccess::ReadWrite;
};

struct PushConstantFieldMetadata {
    std::string name;
    std::string type;
    std::uint32_t offset = 0;
    std::uint32_t size = 0;
};

using SpecializationValue = std::variant<bool, std::int64_t, std::uint64_t, double>;
struct SpecializationConstantMetadata {
    std::uint32_t id = 0;
    std::string name;
    std::string type;
    SpecializationValue default_value = std::uint64_t{0};
};

struct KernelMetadata {
    std::uint32_t schema_version = 1;
    std::string kernel_id;
    std::vector<KernelBindingMetadata> bindings;
    std::uint32_t push_constant_size = 0;
    std::vector<PushConstantFieldMetadata> push_constants;
    std::vector<SpecializationConstantMetadata> specialization_constants;
    std::array<std::uint32_t, 3> local_size{1, 1, 1};
    std::string cpu_fallback_key;
    std::vector<std::string> semantics;
};

/// Validate metadata constructed by a caller; returns InvalidKernel with the failing field path.
Status validate_kernel_metadata(const KernelMetadata& metadata);

/// Parse a strict .kora.json document. Unknown and duplicate fields are rejected.
Result<KernelMetadata> parse_kernel_metadata(std::string_view document);

}  // namespace kora
