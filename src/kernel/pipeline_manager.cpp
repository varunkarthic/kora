#include "kernel/pipeline_manager.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <system_error>
#include <type_traits>

#include "kora/log.hpp"

namespace kora::detail::kernel {
namespace {

constexpr std::array<std::uint8_t, 8> kCacheMagic{'K', 'O', 'R', 'A', 'P', 'C', 'H', '1'};
constexpr std::uint32_t kCacheSchema = 1;
constexpr std::size_t kCacheHeaderSize = 72;
constexpr std::uint64_t kMaximumCacheFileSize = 64U * 1024U * 1024U;
constexpr std::uint32_t kMaximumDescriptorSetIndex = 31;

Error pipeline_error(std::string message, ErrorCode code = ErrorCode::PipelineCreationFailed) {
    return Error(code, "Pipeline manager: " + std::move(message));
}

std::uint64_t fnv1a(std::string_view bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void append_bytes(std::string& output, const void* source, std::size_t size) {
    output.append(static_cast<const char*>(source), size);
}

template <class T>
void append_integer(std::string& output, T value) {
    static_assert(std::is_integral_v<T>);
    using Unsigned = std::make_unsigned_t<T>;
    auto bits = static_cast<std::uint64_t>(static_cast<Unsigned>(value));
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        output.push_back(static_cast<char>(bits & 0xffU));
        bits >>= 8U;
    }
}

void append_string(std::string& output, std::string_view value) {
    append_integer(output, static_cast<std::uint64_t>(value.size()));
    output.append(value);
}

void append_identity(std::string& output, const PipelineCacheIdentity& identity) {
    append_integer(output, identity.vendor_id);
    append_integer(output, identity.device_id);
    append_integer(output, identity.driver_version);
    append_bytes(output, identity.device_uuid.data(), identity.device_uuid.size());
    append_bytes(output, identity.driver_uuid.data(), identity.driver_uuid.size());
}

void append_specialization(std::string& output, const SpecializationConstantMetadata& item) {
    append_integer(output, item.id);
    append_string(output, item.type);
    if (const auto* boolean_value = std::get_if<bool>(&item.default_value)) {
        append_integer(output, std::uint8_t{0});
        append_integer(output, static_cast<std::uint8_t>(*boolean_value));
    } else if (const auto* signed_value = std::get_if<std::int64_t>(&item.default_value)) {
        append_integer(output, std::uint8_t{1});
        append_integer(output, *signed_value);
    } else if (const auto* unsigned_value = std::get_if<std::uint64_t>(&item.default_value)) {
        append_integer(output, std::uint8_t{2});
        append_integer(output, *unsigned_value);
    } else {
        append_integer(output, std::uint8_t{3});
        append_integer(output, std::bit_cast<std::uint64_t>(std::get<double>(item.default_value)));
    }
}

std::string serialize_pipeline_key(std::span<const std::uint32_t> module, const KernelMetadata& metadata,
                                   std::string_view entry_point, const PipelineCacheIdentity& identity) {
    std::string key;
    key.reserve(module.size_bytes() + metadata.bindings.size() * 24U + 256U);
    append_integer(key, kCacheSchema);
    append_identity(key, identity);
    append_integer(key, static_cast<std::uint64_t>(module.size_bytes()));
    append_bytes(key, module.data(), module.size_bytes());
    append_string(key, entry_point);
    for (const auto dimension : metadata.local_size) append_integer(key, dimension);

    auto bindings = metadata.bindings;
    std::sort(bindings.begin(), bindings.end(), [](const auto& left, const auto& right) {
        return std::pair{left.set, left.binding} < std::pair{right.set, right.binding};
    });
    append_integer(key, static_cast<std::uint64_t>(bindings.size()));
    for (const auto& binding : bindings) {
        append_integer(key, binding.set);
        append_integer(key, binding.binding);
        append_integer(key, static_cast<std::uint32_t>(binding.kind));
        append_integer(key, static_cast<std::uint32_t>(binding.access));
    }
    append_integer(key, metadata.push_constant_size);
    append_integer(key, static_cast<std::uint64_t>(metadata.specialization_constants.size()));
    auto constants = metadata.specialization_constants;
    std::sort(constants.begin(), constants.end(),
              [](const auto& left, const auto& right) { return left.id < right.id; });
    for (const auto& item : constants) append_specialization(key, item);
    return key;
}

VkDescriptorType descriptor_type(KernelBindingKind kind) {
    switch (kind) {
        case KernelBindingKind::StorageBuffer:
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case KernelBindingKind::UniformBuffer:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case KernelBindingKind::StorageImage:
            return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case KernelBindingKind::SampledImage:
            return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

std::string make_descriptor_layout_key(std::span<const KernelBindingMetadata> bindings) {
    std::string key;
    append_integer(key, static_cast<std::uint64_t>(bindings.size()));
    for (const auto& binding : bindings) {
        append_integer(key, binding.binding);
        append_integer(key, static_cast<std::uint32_t>(descriptor_type(binding.kind)));
        append_integer(key, std::uint32_t{1});
        append_integer(key, static_cast<std::uint32_t>(VK_SHADER_STAGE_COMPUTE_BIT));
    }
    return key;
}

template <class Handle>
std::uint64_t handle_value(Handle handle) {
    if constexpr (std::is_pointer_v<Handle>) {
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(handle));
    } else {
        return static_cast<std::uint64_t>(handle);
    }
}

std::uint64_t cache_checksum(std::span<const std::uint8_t> bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

template <class T>
void write_little_endian(std::vector<std::uint8_t>& output, T value) {
    static_assert(std::is_integral_v<T>);
    using Unsigned = std::make_unsigned_t<T>;
    auto bits = static_cast<std::uint64_t>(static_cast<Unsigned>(value));
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        output.push_back(static_cast<std::uint8_t>(bits & 0xffU));
        bits >>= 8U;
    }
}

template <class T>
bool read_little_endian(std::span<const std::uint8_t> input, std::size_t offset, T& value) {
    static_assert(std::is_integral_v<T>);
    if (offset > input.size() || sizeof(T) > input.size() - offset) return false;
    using Unsigned = std::make_unsigned_t<T>;
    Unsigned bits = 0;
    for (std::size_t index = 0; index < sizeof(T); ++index)
        bits |= static_cast<Unsigned>(input[offset + index]) << (index * 8U);
    value = static_cast<T>(bits);
    return true;
}

std::vector<std::uint8_t> make_cache_file(const PipelineCacheIdentity& identity,
                                          std::span<const std::uint8_t> payload) {
    std::vector<std::uint8_t> output;
    output.reserve(kCacheHeaderSize + payload.size());
    output.insert(output.end(), kCacheMagic.begin(), kCacheMagic.end());
    write_little_endian(output, kCacheSchema);
    write_little_endian(output, identity.vendor_id);
    write_little_endian(output, identity.device_id);
    write_little_endian(output, identity.driver_version);
    output.insert(output.end(), identity.device_uuid.begin(), identity.device_uuid.end());
    output.insert(output.end(), identity.driver_uuid.begin(), identity.driver_uuid.end());
    write_little_endian(output, static_cast<std::uint64_t>(payload.size()));
    write_little_endian(output, cache_checksum(payload));
    output.insert(output.end(), payload.begin(), payload.end());
    return output;
}

bool read_valid_cache_file(const std::filesystem::path& path, const PipelineCacheIdentity& identity,
                           std::vector<std::uint8_t>& payload) {
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error || file_size < kCacheHeaderSize || file_size > kMaximumCacheFileSize) return false;
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input || bytes.size() < kCacheHeaderSize ||
        !std::equal(kCacheMagic.begin(), kCacheMagic.end(), bytes.begin()))
        return false;

    const auto all = std::span<const std::uint8_t>(bytes);
    std::uint32_t schema = 0;
    std::uint32_t vendor = 0;
    std::uint32_t device = 0;
    std::uint32_t driver = 0;
    std::uint64_t payload_size = 0;
    std::uint64_t checksum = 0;
    if (!read_little_endian(all, 8, schema) || !read_little_endian(all, 12, vendor) ||
        !read_little_endian(all, 16, device) || !read_little_endian(all, 20, driver) ||
        !read_little_endian(all, 56, payload_size) || !read_little_endian(all, 64, checksum))
        return false;
    if (schema != kCacheSchema || vendor != identity.vendor_id || device != identity.device_id ||
        driver != identity.driver_version ||
        !std::equal(identity.device_uuid.begin(), identity.device_uuid.end(), bytes.begin() + 24) ||
        !std::equal(identity.driver_uuid.begin(), identity.driver_uuid.end(), bytes.begin() + 40) ||
        payload_size != bytes.size() - kCacheHeaderSize)
        return false;

    const auto data = all.subspan(kCacheHeaderSize);
    if (cache_checksum(data) != checksum) return false;
    payload.assign(data.begin(), data.end());
    return !payload.empty();
}

Status write_cache_file(const std::filesystem::path& path, std::span<const std::uint8_t> bytes) {
    if (path.empty()) return {};
    std::error_code error;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error)
            return pipeline_error("cannot create cache directory: " + error.message(), ErrorCode::IoError);
    }

    static std::atomic<std::uint64_t> sequence{1};
    auto temporary = path;
    temporary += ".tmp." + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return pipeline_error("cannot open temporary pipeline cache", ErrorCode::IoError);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        output.flush();
        if (!output) {
            output.close();
            std::filesystem::remove(temporary, error);
            return pipeline_error("cannot write temporary pipeline cache", ErrorCode::IoError);
        }
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        // Windows does not replace an existing destination with rename; the cache is rebuildable.
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
    }
    if (error) {
        std::error_code remove_error;
        std::filesystem::remove(temporary, remove_error);
        return pipeline_error("cannot atomically publish pipeline cache: " + error.message(),
                              ErrorCode::IoError);
    }
    return {};
}

Status vk_status(VkResult result, std::string operation) {
    if (result == VK_SUCCESS) return {};
    return pipeline_error(std::move(operation) + " failed with VkResult " + std::to_string(result));
}

void append_specialization_value(std::vector<std::byte>& data, std::vector<VkSpecializationMapEntry>& entries,
                                 const SpecializationConstantMetadata& item) {
    VkSpecializationMapEntry entry{};
    entry.constantID = item.id;
    entry.offset = static_cast<std::uint32_t>(data.size());

    auto append = [&](const auto& value) {
        const auto* first = reinterpret_cast<const std::byte*>(&value);
        data.insert(data.end(), first, first + sizeof(value));
        entry.size = sizeof(value);
    };
    if (item.type == "bool") {
        const VkBool32 value = std::get<bool>(item.default_value) ? VK_TRUE : VK_FALSE;
        append(value);
    } else if (item.type == "i32") {
        const auto value = static_cast<std::int32_t>(std::get<std::int64_t>(item.default_value));
        append(value);
    } else if (item.type == "u32") {
        const auto value = static_cast<std::uint32_t>(std::get<std::uint64_t>(item.default_value));
        append(value);
    } else if (item.type == "i64") {
        const auto value = std::get<std::int64_t>(item.default_value);
        append(value);
    } else if (item.type == "u64") {
        const auto value = std::get<std::uint64_t>(item.default_value);
        append(value);
    } else if (item.type == "f32") {
        const auto value = static_cast<float>(std::get<double>(item.default_value));
        append(value);
    } else {
        const auto value = std::get<double>(item.default_value);
        append(value);
    }
    entries.push_back(entry);
}

}  // namespace

PipelineKey make_pipeline_key(std::span<const std::uint32_t> module, const KernelMetadata& metadata,
                              std::string_view entry_point, const PipelineCacheIdentity& identity) {
    PipelineKey key;
    key.serialized = serialize_pipeline_key(module, metadata, entry_point, identity);
    key.hash = fnv1a(key.serialized);
    return key;
}

PipelineManager::PipelineManager(vk::Api api, VkDevice device, spirv::DeviceCapabilities capabilities,
                                 PipelineCacheIdentity identity, std::filesystem::path cache_file,
                                 PipelineManagerOptions options)
    : api_(std::move(api)),
      device_(device),
      capabilities_(capabilities),
      identity_(identity),
      cache_file_(std::move(cache_file)),
      options_(options) {}

Result<std::unique_ptr<PipelineManager>> PipelineManager::create(vk::Api api, VkDevice device,
                                                                 spirv::DeviceCapabilities capabilities,
                                                                 PipelineCacheIdentity identity,
                                                                 std::filesystem::path cache_file,
                                                                 PipelineManagerOptions options) {
    if (device == VK_NULL_HANDLE)
        return pipeline_error("a live Vulkan device is required", ErrorCode::InvalidArgument);
    if (options.max_cached_pipelines == 0 || options.descriptor_sets_per_pool == 0 ||
        options.max_descriptor_pools_per_layout == 0)
        return pipeline_error("cache and descriptor-pool limits must be greater than zero",
                              ErrorCode::InvalidArgument);
    if (api.create_shader_module == nullptr || api.destroy_shader_module == nullptr ||
        api.create_descriptor_set_layout == nullptr || api.destroy_descriptor_set_layout == nullptr ||
        api.create_pipeline_layout == nullptr || api.destroy_pipeline_layout == nullptr ||
        api.create_descriptor_pool == nullptr || api.destroy_descriptor_pool == nullptr ||
        api.allocate_descriptor_sets == nullptr || api.free_descriptor_sets == nullptr ||
        api.create_compute_pipelines == nullptr || api.destroy_pipeline == nullptr ||
        api.create_pipeline_cache == nullptr || api.destroy_pipeline_cache == nullptr ||
        api.get_pipeline_cache_data == nullptr)
        return pipeline_error("Vulkan pipeline or descriptor entry points are unavailable",
                              ErrorCode::UnsupportedFeature);
    return std::unique_ptr<PipelineManager>(
        new PipelineManager(std::move(api), device, capabilities, identity, std::move(cache_file), options));
}

PipelineManager::~PipelineManager() {
    auto status = save_pipeline_cache();
    if (!status) log::write(log::Level::Warn, log::Category::Device, status.error().message);
    std::lock_guard lock(mutex_);
    destroy_pipeline_objects();
}

Result<VkPipelineCache> PipelineManager::ensure_pipeline_cache() {
    if (cache_initialized_) return pipeline_cache_;

    std::vector<std::uint8_t> initial_data;
    if (!cache_file_.empty() && read_valid_cache_file(cache_file_, identity_, initial_data)) {
        stats_.persisted_pipeline_cache_loaded = true;
    }
    VkPipelineCacheCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    create_info.initialDataSize = initial_data.size();
    create_info.pInitialData = initial_data.empty() ? nullptr : initial_data.data();
    auto result = api_.create_pipeline_cache(device_, &create_info, nullptr, &pipeline_cache_);
    if (result != VK_SUCCESS && !initial_data.empty()) {
        if (pipeline_cache_ != VK_NULL_HANDLE) api_.destroy_pipeline_cache(device_, pipeline_cache_, nullptr);
        pipeline_cache_ = VK_NULL_HANDLE;
        create_info.initialDataSize = 0;
        create_info.pInitialData = nullptr;
        result = api_.create_pipeline_cache(device_, &create_info, nullptr, &pipeline_cache_);
        stats_.persisted_pipeline_cache_loaded = false;
    }
    auto status = vk_status(result, "vkCreatePipelineCache");
    if (!status) return std::move(status).error();
    cache_initialized_ = true;
    return pipeline_cache_;
}

Result<std::vector<VkDescriptorSetLayout>> PipelineManager::descriptor_layouts(
    const KernelMetadata& metadata) {
    std::uint32_t max_set = 0;
    bool has_bindings = false;
    for (const auto& binding : metadata.bindings) {
        if (binding.set > kMaximumDescriptorSetIndex)
            return pipeline_error("descriptor set index exceeds the supported limit of 31",
                                  ErrorCode::InvalidKernel);
        max_set = std::max(max_set, binding.set);
        has_bindings = true;
    }
    if (!has_bindings) return std::vector<VkDescriptorSetLayout>{};

    std::vector<VkDescriptorSetLayout> result;
    result.reserve(static_cast<std::size_t>(max_set) + 1U);
    for (std::uint32_t set = 0; set <= max_set; ++set) {
        std::vector<KernelBindingMetadata> bindings;
        for (const auto& item : metadata.bindings)
            if (item.set == set) bindings.push_back(item);
        std::sort(bindings.begin(), bindings.end(),
                  [](const auto& left, const auto& right) { return left.binding < right.binding; });

        const auto key = make_descriptor_layout_key(bindings);
        const auto cached = descriptor_layouts_.find(key);
        if (cached != descriptor_layouts_.end()) {
            ++stats_.descriptor_layout_cache_hits;
            result.push_back(cached->second);
            continue;
        }

        std::vector<VkDescriptorSetLayoutBinding> vk_bindings;
        std::unordered_map<VkDescriptorType, std::uint32_t> descriptor_counts;
        vk_bindings.reserve(bindings.size());
        for (const auto& binding : bindings) {
            const auto type = descriptor_type(binding.kind);
            if (type == VK_DESCRIPTOR_TYPE_MAX_ENUM)
                return pipeline_error("metadata contains an unsupported descriptor kind",
                                      ErrorCode::InvalidKernel);
            vk_bindings.push_back({binding.binding, type, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
            ++descriptor_counts[type];
        }
        VkDescriptorSetLayoutCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create_info.bindingCount = static_cast<std::uint32_t>(vk_bindings.size());
        create_info.pBindings = vk_bindings.empty() ? nullptr : vk_bindings.data();
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        const auto result_code = api_.create_descriptor_set_layout(device_, &create_info, nullptr, &layout);
        auto status = vk_status(result_code, "vkCreateDescriptorSetLayout");
        if (!status) return std::move(status).error();
        if (layout == VK_NULL_HANDLE)
            return pipeline_error("vkCreateDescriptorSetLayout returned a null handle");

        std::vector<VkDescriptorPoolSize> pool_sizes;
        pool_sizes.reserve(descriptor_counts.size());
        for (const auto& [type, count] : descriptor_counts) pool_sizes.push_back({type, count});
        std::sort(pool_sizes.begin(), pool_sizes.end(),
                  [](const auto& left, const auto& right) { return left.type < right.type; });
        descriptor_layouts_.emplace(key, layout);
        pool_sizes_per_set_.emplace(layout, std::move(pool_sizes));
        result.push_back(layout);
    }
    return result;
}

Result<VkPipelineLayout> PipelineManager::pipeline_layout(
    const KernelMetadata& metadata, const std::vector<VkDescriptorSetLayout>& set_layouts) {
    std::string key;
    append_integer(key, static_cast<std::uint64_t>(set_layouts.size()));
    for (const auto layout : set_layouts) append_integer(key, handle_value(layout));
    append_integer(key, metadata.push_constant_size);
    const auto cached = pipeline_layouts_.find(key);
    if (cached != pipeline_layouts_.end()) {
        ++stats_.pipeline_layout_cache_hits;
        return cached->second;
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = metadata.push_constant_size;
    VkPipelineLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    create_info.setLayoutCount = static_cast<std::uint32_t>(set_layouts.size());
    create_info.pSetLayouts = set_layouts.empty() ? nullptr : set_layouts.data();
    create_info.pushConstantRangeCount = metadata.push_constant_size == 0 ? 0U : 1U;
    create_info.pPushConstantRanges = metadata.push_constant_size == 0 ? nullptr : &push_range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    const auto result_code = api_.create_pipeline_layout(device_, &create_info, nullptr, &layout);
    auto status = vk_status(result_code, "vkCreatePipelineLayout");
    if (!status) return std::move(status).error();
    if (layout == VK_NULL_HANDLE) return pipeline_error("vkCreatePipelineLayout returned a null handle");
    pipeline_layouts_.emplace(std::move(key), layout);
    return layout;
}

Result<PipelineManager::PipelineRef> PipelineManager::create_pipeline(
    std::span<const std::uint32_t> module, const KernelMetadata& metadata,
    std::span<const SpecializationConstantMetadata> specialization_values, std::string_view entry_point,
    std::string key, std::uint64_t key_hash) {
    auto layouts = descriptor_layouts(metadata);
    if (!layouts) return std::move(layouts).error();
    auto layout = pipeline_layout(metadata, layouts.value());
    if (!layout) return std::move(layout).error();
    auto cache = ensure_pipeline_cache();
    if (!cache) return std::move(cache).error();

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = module.size_bytes();
    shader_info.pCode = module.data();
    VkShaderModule shader = VK_NULL_HANDLE;
    const auto shader_result = api_.create_shader_module(device_, &shader_info, nullptr, &shader);
    auto status = vk_status(shader_result, "vkCreateShaderModule");
    if (!status) return std::move(status).error();
    if (shader == VK_NULL_HANDLE) return pipeline_error("vkCreateShaderModule returned a null handle");

    std::vector<VkSpecializationMapEntry> map_entries;
    std::vector<std::byte> specialization_data;
    map_entries.reserve(specialization_values.size());
    for (const auto& item : specialization_values)
        append_specialization_value(specialization_data, map_entries, item);
    VkSpecializationInfo specialization{};
    specialization.mapEntryCount = static_cast<std::uint32_t>(map_entries.size());
    specialization.pMapEntries = map_entries.empty() ? nullptr : map_entries.data();
    specialization.dataSize = specialization_data.size();
    specialization.pData = specialization_data.empty() ? nullptr : specialization_data.data();

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    const std::string entry_name(entry_point);
    stage.pName = entry_name.c_str();
    stage.pSpecializationInfo = map_entries.empty() ? nullptr : &specialization;
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = layout.value();
    VkPipeline pipeline_handle = VK_NULL_HANDLE;
    const auto create_result =
        api_.create_compute_pipelines(device_, cache.value(), 1, &pipeline_info, nullptr, &pipeline_handle);
    api_.destroy_shader_module(device_, shader, nullptr);
    status = vk_status(create_result, "vkCreateComputePipelines");
    if (!status) {
        if (pipeline_handle != VK_NULL_HANDLE) api_.destroy_pipeline(device_, pipeline_handle, nullptr);
        return std::move(status).error();
    }
    if (pipeline_handle == VK_NULL_HANDLE)
        return pipeline_error("vkCreateComputePipelines returned a null handle");

    auto pipeline = std::make_shared<ComputePipeline>();
    pipeline->handle = pipeline_handle;
    pipeline->layout = layout.value();
    pipeline->descriptor_set_layouts = std::move(layouts).value();
    pipeline->bindings = metadata.bindings;
    pipeline->key_hash = key_hash;
    pipeline->owner = this;
    lru_.push_front(std::move(key));
    pipelines_.emplace(lru_.front(), CacheEntry{pipeline, lru_.begin()});
    evict_if_needed();
    return PipelineRef(std::move(pipeline));
}

Result<PipelineManager::PipelineRef> PipelineManager::get_or_create(std::span<const std::uint32_t> module,
                                                                    const KernelMetadata& metadata,
                                                                    std::string_view entry_point) {
    return get_or_create(module, metadata, metadata.specialization_constants, entry_point);
}

Result<PipelineManager::PipelineRef> PipelineManager::get_or_create(
    std::span<const std::uint32_t> module, const KernelMetadata& metadata,
    std::span<const SpecializationConstantMetadata> overrides, std::string_view entry_point) {
    std::lock_guard lock(mutex_);
    collect_retired();
    auto reflected = spirv::validate_and_reflect(module, metadata, capabilities_, entry_point);
    if (!reflected) return std::move(reflected).error();
    auto resolved_metadata = metadata;
    for (const auto& override : overrides) {
        const auto original =
            std::find_if(metadata.specialization_constants.begin(), metadata.specialization_constants.end(),
                         [&](const auto& item) { return item.id == override.id; });
        if (original == metadata.specialization_constants.end() || original->type != override.type)
            return pipeline_error("specialization override has an unknown ID or mismatched type",
                                  ErrorCode::InvalidArgument);
    }
    for (auto& value : resolved_metadata.specialization_constants) {
        const auto override = std::find_if(overrides.begin(), overrides.end(),
                                           [&](const auto& item) { return item.id == value.id; });
        if (override != overrides.end()) value.default_value = override->default_value;
    }
    if (reflected.value().local_size_id_based) {
        KernelMetadata reflected_metadata = metadata;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
            reflected_metadata.local_size[dimension] = reflected.value().local_size[dimension];
        reflected = spirv::validate_and_reflect(module, reflected_metadata, capabilities_, entry_point);
        if (!reflected) return std::move(reflected).error();
        for (std::size_t dimension = 0; dimension < 3; ++dimension) {
            const auto found =
                std::find_if(resolved_metadata.specialization_constants.begin(),
                             resolved_metadata.specialization_constants.end(), [&](const auto& item) {
                                 return item.id == reflected.value().local_size_spec_ids[dimension];
                             });
            if (found == resolved_metadata.specialization_constants.end() || found->type != "u32" ||
                !std::holds_alternative<std::uint64_t>(found->default_value) ||
                std::get<std::uint64_t>(found->default_value) == 0 ||
                std::get<std::uint64_t>(found->default_value) > UINT32_MAX)
                return pipeline_error("LocalSizeId override must be a nonzero u32 specialization value",
                                      ErrorCode::InvalidArgument);
            resolved_metadata.local_size[dimension] =
                static_cast<std::uint32_t>(std::get<std::uint64_t>(found->default_value));
        }
    }
    auto pipeline_key = make_pipeline_key(module, resolved_metadata, entry_point, identity_);
    auto key = std::move(pipeline_key.serialized);
    const auto key_hash = pipeline_key.hash;
    const auto found = pipelines_.find(key);
    if (found != pipelines_.end()) {
        lru_.splice(lru_.begin(), lru_, found->second.lru_position);
        found->second.lru_position = lru_.begin();
        ++stats_.pipeline_cache_hits;
        return PipelineRef(found->second.pipeline);
    }
    ++stats_.pipeline_cache_misses;
    return create_pipeline(module, resolved_metadata, resolved_metadata.specialization_constants, entry_point,
                           std::move(key), key_hash);
}

std::optional<PipelineManager::PipelineRef> PipelineManager::find_cached(
    std::span<const std::uint32_t> module, const KernelMetadata& metadata,
    std::span<const SpecializationConstantMetadata> overrides, std::string_view entry_point) {
    std::lock_guard lock(mutex_);
    collect_retired();
    auto resolved_metadata = metadata;
    for (const auto& override : overrides) {
        const auto original =
            std::find_if(metadata.specialization_constants.begin(), metadata.specialization_constants.end(),
                         [&](const auto& item) { return item.id == override.id; });
        if (original == metadata.specialization_constants.end() || original->type != override.type)
            return std::nullopt;
    }
    for (auto& value : resolved_metadata.specialization_constants) {
        const auto specialization_override = std::find_if(
            overrides.begin(), overrides.end(), [&](const auto& item) { return item.id == value.id; });
        if (specialization_override != overrides.end())
            value.default_value = specialization_override->default_value;
    }
    auto pipeline_key = make_pipeline_key(module, resolved_metadata, entry_point, identity_);
    const auto found = pipelines_.find(pipeline_key.serialized);
    if (found == pipelines_.end()) return std::nullopt;
    lru_.splice(lru_.begin(), lru_, found->second.lru_position);
    found->second.lru_position = lru_.begin();
    ++stats_.pipeline_cache_hits;
    return PipelineRef(found->second.pipeline);
}

void PipelineManager::evict_if_needed() {
    while (pipelines_.size() > options_.max_cached_pipelines) {
        const auto last = std::prev(lru_.end());
        const auto found = pipelines_.find(*last);
        if (found != pipelines_.end()) {
            auto pipeline = std::move(found->second.pipeline);
            pipelines_.erase(found);
            if (pipeline.use_count() > 1) {
                retired_.push_back(std::move(pipeline));
            } else if (pipeline->handle != VK_NULL_HANDLE) {
                api_.destroy_pipeline(device_, pipeline->handle, nullptr);
            }
            ++stats_.pipeline_evictions;
        }
        lru_.erase(last);
    }
}

void PipelineManager::collect_retired() {
    auto iterator = retired_.begin();
    while (iterator != retired_.end()) {
        if (iterator->use_count() == 1) {
            if ((*iterator)->handle != VK_NULL_HANDLE)
                api_.destroy_pipeline(device_, (*iterator)->handle, nullptr);
            iterator = retired_.erase(iterator);
        } else {
            ++iterator;
        }
    }
}

Status PipelineManager::grow_descriptor_pools(VkDescriptorSetLayout layout) {
    auto& pools = descriptor_pools_[layout];
    if (pools.size() >= options_.max_descriptor_pools_per_layout)
        return pipeline_error("descriptor pool growth limit reached", ErrorCode::AllocationFailed);
    std::vector<VkDescriptorPoolSize> sizes;
    for (const auto& size : pool_sizes_per_set_[layout]) {
        if (size.descriptorCount >
            std::numeric_limits<std::uint32_t>::max() / options_.descriptor_sets_per_pool)
            return pipeline_error("descriptor pool size overflows uint32", ErrorCode::Overflow);
        sizes.push_back({size.type, size.descriptorCount * options_.descriptor_sets_per_pool});
    }
    if (sizes.empty())
        sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, options_.descriptor_sets_per_pool});
    VkDescriptorPoolCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    create_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    create_info.maxSets = options_.descriptor_sets_per_pool;
    create_info.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    create_info.pPoolSizes = sizes.data();
    VkDescriptorPool pool = VK_NULL_HANDLE;
    const auto result_code = api_.create_descriptor_pool(device_, &create_info, nullptr, &pool);
    auto status = vk_status(result_code, "vkCreateDescriptorPool");
    if (!status) return status;
    if (pool == VK_NULL_HANDLE) return pipeline_error("vkCreateDescriptorPool returned a null handle");
    pools.push_back(pool);
    ++stats_.descriptor_pool_growths;
    return {};
}

Result<VkDescriptorSet> PipelineManager::allocate_descriptor_set(const PipelineRef& pipeline,
                                                                 std::uint32_t set_index) {
    std::lock_guard lock(mutex_);
    if (!pipeline || pipeline->owner != this)
        return pipeline_error("pipeline handle does not belong to this manager", ErrorCode::InvalidArgument);
    if (set_index >= pipeline->descriptor_set_layouts.size())
        return pipeline_error("descriptor set index is out of range", ErrorCode::InvalidArgument);
    const auto layout = pipeline->descriptor_set_layouts[set_index];
    auto& free = free_sets_[layout];
    if (!free.empty()) {
        const auto descriptor_set = free.back();
        free.pop_back();
        allocated_sets_[descriptor_set] = layout;
        ++stats_.descriptor_sets_allocated;
        return descriptor_set;
    }

    auto& pools = descriptor_pools_[layout];
    for (const auto pool : pools) {
        VkDescriptorSetAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate_info.descriptorPool = pool;
        allocate_info.descriptorSetCount = 1;
        allocate_info.pSetLayouts = &layout;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        const auto result_code = api_.allocate_descriptor_sets(device_, &allocate_info, &descriptor_set);
        if (result_code == VK_ERROR_OUT_OF_POOL_MEMORY || result_code == VK_ERROR_FRAGMENTED_POOL) continue;
        auto status = vk_status(result_code, "vkAllocateDescriptorSets");
        if (!status) return std::move(status).error();
        if (descriptor_set == VK_NULL_HANDLE) return pipeline_error("vkAllocateDescriptorSets returned null");
        allocated_sets_[descriptor_set] = layout;
        ++stats_.descriptor_sets_allocated;
        return descriptor_set;
    }

    auto status = grow_descriptor_pools(layout);
    if (!status) return std::move(status).error();
    const auto pool = descriptor_pools_[layout].back();
    VkDescriptorSetAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = pool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    const auto result_code = api_.allocate_descriptor_sets(device_, &allocate_info, &descriptor_set);
    status = vk_status(result_code, "vkAllocateDescriptorSets after pool growth");
    if (!status) return std::move(status).error();
    if (descriptor_set == VK_NULL_HANDLE) return pipeline_error("vkAllocateDescriptorSets returned null");
    allocated_sets_[descriptor_set] = layout;
    ++stats_.descriptor_sets_allocated;
    return descriptor_set;
}

Status PipelineManager::free_descriptor_set(VkDescriptorSet descriptor_set) {
    std::lock_guard lock(mutex_);
    const auto found = allocated_sets_.find(descriptor_set);
    if (found == allocated_sets_.end())
        return pipeline_error("descriptor set is unknown or already free", ErrorCode::InvalidArgument);
    const auto layout = found->second;
    // Keep the Vulkan allocation and recycle it at the KORA layer. Calling vkFreeDescriptorSets
    // would invalidate the handle, so the free list owns the still-allocated set until reuse/pool
    // destruction.
    allocated_sets_.erase(found);
    free_sets_[layout].push_back(descriptor_set);
    return {};
}

Status PipelineManager::save_pipeline_cache() {
    std::lock_guard lock(mutex_);
    if (!cache_initialized_ || pipeline_cache_ == VK_NULL_HANDLE || cache_file_.empty()) return {};
    std::size_t size = 0;
    auto result_code = api_.get_pipeline_cache_data(device_, pipeline_cache_, &size, nullptr);
    auto status = vk_status(result_code, "vkGetPipelineCacheData(size)");
    if (!status) return status;
    if (size > kMaximumCacheFileSize - kCacheHeaderSize)
        return pipeline_error("driver pipeline cache exceeds the 64 MiB persistence limit",
                              ErrorCode::IoError);
    std::vector<std::uint8_t> payload(size);
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
        std::size_t written = payload.size();
        result_code = api_.get_pipeline_cache_data(device_, pipeline_cache_, &written,
                                                   payload.empty() ? nullptr : payload.data());
        if (result_code == VK_SUCCESS) {
            payload.resize(written);
            break;
        }
        if (result_code != VK_INCOMPLETE || attempt == 3)
            return vk_status(result_code, "vkGetPipelineCacheData(data)");
        if (written > kMaximumCacheFileSize - kCacheHeaderSize)
            return pipeline_error("driver pipeline cache exceeds the 64 MiB persistence limit",
                                  ErrorCode::IoError);
        payload.resize(std::max(written, payload.size() * 2U));
    }
    const auto file = make_cache_file(identity_, payload);
    return write_cache_file(cache_file_, file);
}

PipelineManagerStats PipelineManager::stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}

void PipelineManager::destroy_pipeline_objects() noexcept {
    for (auto& [key, entry] : pipelines_) {
        (void)key;
        if (entry.pipeline && entry.pipeline->handle != VK_NULL_HANDLE)
            api_.destroy_pipeline(device_, entry.pipeline->handle, nullptr);
    }
    pipelines_.clear();
    for (auto& pipeline : retired_)
        if (pipeline && pipeline->handle != VK_NULL_HANDLE)
            api_.destroy_pipeline(device_, pipeline->handle, nullptr);
    retired_.clear();
    for (const auto& [key, layout] : pipeline_layouts_) {
        (void)key;
        if (layout != VK_NULL_HANDLE) api_.destroy_pipeline_layout(device_, layout, nullptr);
    }
    pipeline_layouts_.clear();
    for (const auto& [layout, pools] : descriptor_pools_) {
        (void)layout;
        for (const auto pool : pools)
            if (pool != VK_NULL_HANDLE) api_.destroy_descriptor_pool(device_, pool, nullptr);
    }
    descriptor_pools_.clear();
    for (const auto& [key, layout] : descriptor_layouts_) {
        (void)key;
        if (layout != VK_NULL_HANDLE) api_.destroy_descriptor_set_layout(device_, layout, nullptr);
    }
    descriptor_layouts_.clear();
    if (pipeline_cache_ != VK_NULL_HANDLE) {
        api_.destroy_pipeline_cache(device_, pipeline_cache_, nullptr);
        pipeline_cache_ = VK_NULL_HANDLE;
    }
}

}  // namespace kora::detail::kernel
