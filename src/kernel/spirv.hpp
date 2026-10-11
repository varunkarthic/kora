#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kora/device_info.hpp"
#include "kora/kernel_metadata.hpp"

namespace kora::detail::spirv {

struct DeviceCapabilities {
    std::uint32_t api_major = 1;
    std::uint32_t api_minor = 0;
    bool shader_int16 = false;
    bool shader_int64 = false;
    bool shader_float64 = false;
    bool subgroup_supported = false;
    bool subgroup_basic = false;
    bool subgroup_vote = false;
    bool subgroup_arithmetic = false;
    bool subgroup_ballot = false;
    bool subgroup_shuffle = false;
    bool subgroup_shuffle_relative = false;
    bool subgroup_clustered = false;
    bool subgroup_quad = false;
};

struct ReflectedBinding {
    std::uint32_t set = 0;
    std::uint32_t binding = 0;
    std::string name;
    KernelBindingKind kind = KernelBindingKind::StorageBuffer;
    KernelAccess access = KernelAccess::Read;
};

struct Reflection {
    std::string entry_point;
    std::array<std::uint32_t, 3> local_size{};
    std::vector<ReflectedBinding> bindings;
    std::uint32_t push_constant_size = 0;
    std::vector<PushConstantFieldMetadata> push_constants;
    std::vector<SpecializationConstantMetadata> specialization_constants;
    std::vector<std::uint32_t> capabilities;
};

DeviceCapabilities capabilities_from(const DeviceInfo& device) noexcept;

/// Structurally validates, reflects, and compares a module with its version-1 package metadata.
/// Full Vulkan-environment SPIRV-Tools validation is added when that optional dependency is found.
Result<Reflection> validate_and_reflect(std::span<const std::uint32_t> words, const KernelMetadata& metadata,
                                        const DeviceCapabilities& device,
                                        std::string_view entry_point = "main");

}  // namespace kora::detail::spirv
