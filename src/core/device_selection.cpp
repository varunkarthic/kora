#include "kora/device_selection.hpp"

#include <algorithm>
#include <cctype>
#include <tuple>

namespace kora {
namespace {

bool valid_uuid(const std::string& uuid) {
    return uuid.size() == 32 &&
           std::all_of(uuid.begin(), uuid.end(), [](unsigned char c) { return std::isxdigit(c); });
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool eligible(const DeviceCandidate& device) {
    return device.satisfies_requirements && device.compute_queue_count != 0;
}

std::uint64_t capability_score(const DeviceCandidate& device) {
    const auto tier = static_cast<std::uint64_t>(device.tier);
    const auto memory_gib = std::min<std::uint64_t>(device.device_local_memory_bytes >> 30U, 4095U);
    const auto queues = std::min<std::uint64_t>(device.compute_queue_count, 255U);
    return (tier << 48U) | (memory_gib << 16U) | queues;
}

int kind_preference(DevicePolicy policy, DeviceKind kind) {
    if (policy == DevicePolicy::PreferDiscrete) return kind == DeviceKind::Discrete ? 1 : 0;
    if (policy == DevicePolicy::PreferIntegrated) return kind == DeviceKind::Integrated ? 1 : 0;
    return 0;
}

}  // namespace

Result<std::vector<DeviceId>> select_devices(const ResolvedOptions& options,
                                             const std::vector<DeviceCandidate>& devices) {
    if (options.device_policy == DevicePolicy::ByIndex) {
        const auto found = std::find_if(devices.begin(), devices.end(), [&](const auto& device) {
            return device.id.index == options.device_index;
        });
        if (found == devices.end()) {
            return Error(ErrorCode::NoEligibleDevice, "Requested device index was not found");
        }
        if (!eligible(*found)) {
            return Error(ErrorCode::NoEligibleDevice, "Requested device index is not eligible");
        }
        return std::vector<DeviceId>{found->id};
    }
    if (options.device_policy == DevicePolicy::ByUuid) {
        if (!valid_uuid(options.device_uuid)) {
            return Error(ErrorCode::InvalidArgument,
                         "Device UUID must contain exactly 32 hexadecimal characters");
        }
        const auto requested = lower(options.device_uuid);
        const auto found = std::find_if(devices.begin(), devices.end(), [&](const auto& device) {
            return valid_uuid(device.id.uuid) && lower(device.id.uuid) == requested;
        });
        if (found == devices.end()) {
            return Error(ErrorCode::NoEligibleDevice, "Requested device UUID was not found");
        }
        if (!eligible(*found)) {
            return Error(ErrorCode::NoEligibleDevice, "Requested device UUID is not eligible");
        }
        return std::vector<DeviceId>{found->id};
    }

    std::vector<const DeviceCandidate*> ranked;
    for (const auto& device : devices) {
        if (eligible(device)) ranked.push_back(&device);
    }
    if (ranked.empty()) return Error(ErrorCode::NoEligibleDevice, "No eligible compute device was found");

    std::sort(ranked.begin(), ranked.end(), [&](const auto* left, const auto* right) {
        const auto left_key =
            std::tuple(kind_preference(options.device_policy, left->kind), capability_score(*left));
        const auto right_key =
            std::tuple(kind_preference(options.device_policy, right->kind), capability_score(*right));
        if (left_key != right_key) return left_key > right_key;
        if (left->id.index != right->id.index) return left->id.index < right->id.index;
        return lower(left->id.uuid) < lower(right->id.uuid);
    });

    const std::size_t count = options.device_policy == DevicePolicy::AllEligible ? ranked.size() : 1;
    std::vector<DeviceId> selected;
    selected.reserve(count);
    for (std::size_t i = 0; i < count; ++i) selected.push_back(ranked[i]->id);
    return selected;
}

}  // namespace kora
