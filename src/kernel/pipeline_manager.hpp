#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernel/spirv.hpp"
#include "vulkan/api.hpp"

namespace kora::detail::kernel {

struct PipelineCacheIdentity {
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t driver_version = 0;
    std::array<std::uint8_t, VK_UUID_SIZE> device_uuid{};
    std::array<std::uint8_t, VK_UUID_SIZE> driver_uuid{};
};

struct PipelineKey {
    std::string serialized;
    std::uint64_t hash = 0;
    friend bool operator==(const PipelineKey&, const PipelineKey&) = default;
};

PipelineKey make_pipeline_key(std::span<const std::uint32_t> module, const KernelMetadata& metadata,
                              std::string_view entry_point, const PipelineCacheIdentity& identity);

struct PipelineManagerOptions {
    std::size_t max_cached_pipelines = 128;
    std::uint32_t descriptor_sets_per_pool = 32;
    std::uint32_t max_descriptor_pools_per_layout = 8;
};

struct PipelineManagerStats {
    std::uint64_t pipeline_cache_hits = 0;
    std::uint64_t pipeline_cache_misses = 0;
    std::uint64_t pipeline_evictions = 0;
    std::uint64_t descriptor_layout_cache_hits = 0;
    std::uint64_t pipeline_layout_cache_hits = 0;
    std::uint64_t descriptor_pool_growths = 0;
    std::uint64_t descriptor_sets_allocated = 0;
    bool persisted_pipeline_cache_loaded = false;
};

struct ComputePipeline {
    VkPipeline handle = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
    std::vector<KernelBindingMetadata> bindings;
    std::uint64_t key_hash = 0;

private:
    friend class PipelineManager;
    const void* owner = nullptr;
};

/// Per-device compute pipeline, descriptor layout, descriptor pool, and VkPipelineCache owner.
/// Returned pipeline records are non-owning views and must not outlive this manager's device.
class PipelineManager final {
public:
    using PipelineRef = std::shared_ptr<const ComputePipeline>;

    static Result<std::unique_ptr<PipelineManager>> create(vk::Api api, VkDevice device,
                                                           spirv::DeviceCapabilities capabilities,
                                                           PipelineCacheIdentity identity,
                                                           std::filesystem::path cache_file,
                                                           PipelineManagerOptions options = {});

    ~PipelineManager();
    PipelineManager(const PipelineManager&) = delete;
    PipelineManager& operator=(const PipelineManager&) = delete;

    Result<PipelineRef> get_or_create(std::span<const std::uint32_t> module, const KernelMetadata& metadata,
                                      std::string_view entry_point = "main");
    Result<PipelineRef> get_or_create(std::span<const std::uint32_t> module, const KernelMetadata& metadata,
                                      std::span<const SpecializationConstantMetadata> overrides,
                                      std::string_view entry_point = "main");
    Result<VkDescriptorSet> allocate_descriptor_set(const PipelineRef& pipeline, std::uint32_t set_index);
    Status free_descriptor_set(VkDescriptorSet descriptor_set);
    Status save_pipeline_cache();
    PipelineManagerStats stats() const;

private:
    struct CacheEntry {
        std::shared_ptr<ComputePipeline> pipeline;
        std::list<std::string>::iterator lru_position;
    };

    PipelineManager(vk::Api api, VkDevice device, spirv::DeviceCapabilities capabilities,
                    PipelineCacheIdentity identity, std::filesystem::path cache_file,
                    PipelineManagerOptions options);

    Result<VkPipelineCache> ensure_pipeline_cache();
    Result<std::vector<VkDescriptorSetLayout>> descriptor_layouts(const KernelMetadata& metadata);
    Result<VkPipelineLayout> pipeline_layout(const KernelMetadata& metadata,
                                             const std::vector<VkDescriptorSetLayout>& set_layouts);
    Result<PipelineRef> create_pipeline(std::span<const std::uint32_t> module, const KernelMetadata& metadata,
                                        std::span<const SpecializationConstantMetadata> specialization,
                                        std::string_view entry_point, std::string key,
                                        std::uint64_t key_hash);
    Status grow_descriptor_pools(VkDescriptorSetLayout layout);
    void evict_if_needed();
    void collect_retired();
    void destroy_pipeline_objects() noexcept;

    vk::Api api_;
    VkDevice device_ = VK_NULL_HANDLE;
    spirv::DeviceCapabilities capabilities_;
    PipelineCacheIdentity identity_;
    std::filesystem::path cache_file_;
    PipelineManagerOptions options_;
    mutable std::mutex mutex_;
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    bool cache_initialized_ = false;
    PipelineManagerStats stats_;
    std::unordered_map<std::string, CacheEntry> pipelines_;
    std::list<std::string> lru_;
    std::vector<std::shared_ptr<ComputePipeline>> retired_;
    std::unordered_map<std::string, VkDescriptorSetLayout> descriptor_layouts_;
    std::unordered_map<std::string, VkPipelineLayout> pipeline_layouts_;
    std::unordered_map<VkDescriptorSetLayout, std::vector<VkDescriptorPoolSize>> pool_sizes_per_set_;
    std::unordered_map<VkDescriptorSetLayout, std::vector<VkDescriptorPool>> descriptor_pools_;
    std::unordered_map<VkDescriptorSetLayout, std::vector<VkDescriptorSet>> free_sets_;
    std::unordered_map<VkDescriptorSet, VkDescriptorSetLayout> allocated_sets_;
};

}  // namespace kora::detail::kernel
