#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "kernel/pipeline_manager.hpp"

namespace kora::detail::kernel {
namespace {

KernelMetadata metadata() {
    KernelMetadata value;
    value.kernel_id = "pipeline-key-test";
    value.local_size = {8, 2, 1};
    value.bindings.push_back({0, 1, "output", KernelBindingKind::StorageBuffer, KernelAccess::Write});
    value.push_constant_size = 4;
    value.push_constants.push_back({"count", "u32", 0, 4});
    value.specialization_constants.push_back({9, "factor", "u32", std::uint64_t{3}});
    return value;
}

TEST(PipelineKey, StableAndIncludesModuleEntryMetadataAndDeviceIdentity) {
    const std::array<std::uint32_t, 4> module{0x07230203, 0x00010000, 0, 1};
    PipelineCacheIdentity identity;
    identity.vendor_id = 42;
    identity.device_id = 7;
    identity.driver_version = 11;
    identity.device_uuid[0] = 1;
    const auto original_metadata = metadata();
    const auto first = make_pipeline_key(module, original_metadata, "main", identity);
    const auto repeat = make_pipeline_key(module, original_metadata, "main", identity);
    EXPECT_EQ(first, repeat);
    EXPECT_NE(first.hash, 0U);

    auto changed_module = module;
    changed_module[3] = 2;
    EXPECT_NE(first.serialized,
              make_pipeline_key(changed_module, original_metadata, "main", identity).serialized);
    EXPECT_NE(first.serialized, make_pipeline_key(module, original_metadata, "other", identity).serialized);

    auto changed_metadata = original_metadata;
    changed_metadata.local_size[0] = 4;
    EXPECT_NE(first.serialized, make_pipeline_key(module, changed_metadata, "main", identity).serialized);
    changed_metadata = original_metadata;
    changed_metadata.specialization_constants[0].default_value = std::uint64_t{4};
    EXPECT_NE(first.serialized, make_pipeline_key(module, changed_metadata, "main", identity).serialized);

    auto changed_identity = identity;
    changed_identity.device_uuid[0] = 2;
    EXPECT_NE(first.serialized,
              make_pipeline_key(module, original_metadata, "main", changed_identity).serialized);
}

}  // namespace
}  // namespace kora::detail::kernel
