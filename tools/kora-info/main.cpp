#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

#include "core/device_caps.hpp"
#include "kora/version.hpp"
#include "vulkan/api.hpp"

namespace {

std::string version_string(uint32_t version) {
    return std::to_string(VK_VERSION_MAJOR(version)) + "." + std::to_string(VK_VERSION_MINOR(version)) + "." +
           std::to_string(VK_VERSION_PATCH(version));
}

const char* topology_name(kora::MemoryTopology topology) {
    switch (topology) {
        case kora::MemoryTopology::Discrete:
            return "discrete";
        case kora::MemoryTopology::Uma:
            return "uma";
        case kora::MemoryTopology::DiscreteWithBar:
            return "discrete_with_bar";
        case kora::MemoryTopology::Unknown:
            return "unknown";
    }
    return "unknown";
}

void json_string(std::ostream& out, std::string_view value) {
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (c < 0x20U) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << '"';
}

bool has_extension(const kora::vk::PhysicalDeviceInfo& device, const char* name) {
    return std::any_of(device.extensions.begin(), device.extensions.end(), [name](const auto& extension) {
        return std::string_view(extension.extensionName) == name;
    });
}

bool is_moltenvk(const kora::vk::PhysicalDeviceInfo& device) {
    return std::string_view(device.vulkan12_properties.driverName).find("MoltenVK") != std::string_view::npos;
}

void print_human_device(std::size_t index, const kora::vk::PhysicalDeviceInfo& raw,
                        const kora::DeviceInfo& device) {
    std::cout << "\nDevice [" << index << "]: " << device.name() << '\n'
              << "  Vulkan API: " << version_string(raw.properties.apiVersion) << '\n'
              << "  Driver: "
              << (raw.vulkan12_properties.driverName[0] == '\0' ? "unreported"
                                                                : raw.vulkan12_properties.driverName)
              << '\n'
              << "  Portability subset: " << (device.features().portability_subset ? "yes" : "no")
              << " (MoltenVK: " << (is_moltenvk(raw) ? "yes" : "no") << ")\n"
              << "  Memory topology: " << topology_name(device.memory_topology())
              << " (benchmark needed: " << (device.memory_topology_needs_benchmark() ? "yes" : "no") << ")\n";
    const auto& features = device.features();
    std::cout << "  Features: timeline=" << features.timeline_semaphore
              << " sync2=" << features.synchronization2 << " bda=" << features.buffer_device_address
              << " descriptor_indexing=" << features.descriptor_indexing
              << " runtime_descriptor_array=" << features.runtime_descriptor_array << '\n'
              << "  Subgroup: supported=" << device.subgroup().supported << " size=" << device.subgroup().size
              << " basic=" << device.subgroup().basic << " arithmetic=" << device.subgroup().arithmetic
              << '\n'
              << "  Queue families: " << device.queue_families().size() << '\n';
    for (std::size_t i = 0; i < device.queue_families().size(); ++i) {
        const auto& queue = device.queue_families()[i];
        std::cout << "    [" << i << "] count=" << queue.queue_count << " graphics=" << queue.graphics
                  << " compute=" << queue.compute << " transfer=" << queue.transfer
                  << " dedicated_compute=" << queue.dedicated_compute
                  << " dedicated_transfer=" << queue.dedicated_transfer << '\n';
    }
    const bool budgets = has_extension(raw, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    std::cout << "  Memory heaps: " << device.memory_heaps().size()
              << " (budget extension: " << (budgets ? "yes" : "no") << ")\n";
    for (std::size_t i = 0; i < device.memory_heaps().size(); ++i) {
        const auto& heap = device.memory_heaps()[i];
        std::cout << "    [" << i << "] size=" << heap.size_bytes << " device_local=" << heap.device_local;
        if (budgets)
            std::cout << " budget=" << raw.memory_budget_properties.heapBudget[i]
                      << " usage=" << raw.memory_budget_properties.heapUsage[i];
        std::cout << '\n';
    }
    std::cout << "  Memory types: " << device.memory_types().size() << '\n';
    for (std::size_t i = 0; i < device.memory_types().size(); ++i) {
        const auto& type = device.memory_types()[i];
        std::cout << "    [" << i << "] heap=" << type.heap_index << " device_local=" << type.device_local
                  << " host_visible=" << type.host_visible << " coherent=" << type.host_coherent
                  << " cached=" << type.host_cached << '\n';
    }
}

void print_json_device(std::size_t index, const kora::vk::PhysicalDeviceInfo& raw,
                       const kora::DeviceInfo& device) {
    const auto& f = device.features();
    const auto& subgroup = device.subgroup();
    const bool budgets = has_extension(raw, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    std::cout << "{\"index\":" << index << ",\"name\":";
    json_string(std::cout, device.name());
    std::cout << ",\"api_version\":";
    json_string(std::cout, version_string(raw.properties.apiVersion));
    std::cout << ",\"driver_name\":";
    json_string(std::cout, raw.vulkan12_properties.driverName);
    std::cout << ",\"portability_subset\":" << f.portability_subset << ",\"moltenvk\":" << is_moltenvk(raw)
              << ",\"memory_topology\":";
    json_string(std::cout, topology_name(device.memory_topology()));
    std::cout << ",\"memory_topology_needs_benchmark\":" << device.memory_topology_needs_benchmark()
              << ",\"features\":{\"timeline_semaphore\":" << f.timeline_semaphore
              << ",\"synchronization2\":" << f.synchronization2
              << ",\"buffer_device_address\":" << f.buffer_device_address
              << ",\"descriptor_indexing\":" << f.descriptor_indexing
              << ",\"runtime_descriptor_array\":" << f.runtime_descriptor_array << "}"
              << ",\"subgroup\":{\"supported\":" << subgroup.supported << ",\"size\":" << subgroup.size
              << ",\"basic\":" << subgroup.basic << ",\"vote\":" << subgroup.vote
              << ",\"arithmetic\":" << subgroup.arithmetic << ",\"ballot\":" << subgroup.ballot
              << ",\"shuffle\":" << subgroup.shuffle << ",\"shuffle_relative\":" << subgroup.shuffle_relative
              << ",\"clustered\":" << subgroup.clustered << ",\"quad\":" << subgroup.quad
              << "},\"queue_families\":[";
    for (std::size_t i = 0; i < device.queue_families().size(); ++i) {
        if (i != 0) std::cout << ',';
        const auto& q = device.queue_families()[i];
        std::cout << "{\"index\":" << i << ",\"count\":" << q.queue_count
                  << ",\"timestamp_valid_bits\":" << q.timestamp_valid_bits << ",\"graphics\":" << q.graphics
                  << ",\"compute\":" << q.compute << ",\"transfer\":" << q.transfer
                  << ",\"sparse_binding\":" << q.sparse_binding
                  << ",\"dedicated_compute\":" << q.dedicated_compute
                  << ",\"dedicated_transfer\":" << q.dedicated_transfer << '}';
    }
    std::cout << "],\"memory\":{\"budget_available\":" << budgets << ",\"heaps\":[";
    for (std::size_t i = 0; i < device.memory_heaps().size(); ++i) {
        if (i != 0) std::cout << ',';
        const auto& heap = device.memory_heaps()[i];
        std::cout << "{\"index\":" << i << ",\"size_bytes\":" << heap.size_bytes
                  << ",\"device_local\":" << heap.device_local
                  << ",\"multi_instance\":" << heap.multi_instance;
        if (budgets)
            std::cout << ",\"budget_bytes\":" << raw.memory_budget_properties.heapBudget[i]
                      << ",\"usage_bytes\":" << raw.memory_budget_properties.heapUsage[i];
        std::cout << '}';
    }
    std::cout << "],\"types\":[";
    for (std::size_t i = 0; i < device.memory_types().size(); ++i) {
        if (i != 0) std::cout << ',';
        const auto& type = device.memory_types()[i];
        std::cout << "{\"index\":" << i << ",\"heap_index\":" << type.heap_index
                  << ",\"device_local\":" << type.device_local << ",\"host_visible\":" << type.host_visible
                  << ",\"host_coherent\":" << type.host_coherent << ",\"host_cached\":" << type.host_cached
                  << ",\"lazily_allocated\":" << type.lazily_allocated
                  << ",\"protected\":" << type.protected_memory << '}';
    }
    std::cout << "]}}";
}

int report_error(const char* operation, const kora::Error& error) {
    std::cerr << "kora-info: " << operation << ": " << error.describe() << '\n';
    return EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    const bool json = argc == 2 && std::string_view(argv[1]) == "--json";
    if (argc != 1 && !json) {
        std::cerr << "Usage: kora-info [--json]\n";
        return EXIT_FAILURE;
    }
    std::cout << std::boolalpha;
    auto loader = kora::vk::Loader::load();
    if (!loader) return report_error("unable to load Vulkan", loader.error());
    auto instance = loader.value().create_instance();
    if (!instance) return report_error("unable to create Vulkan instance", instance.error());
    auto devices = instance.value().physical_devices();
    if (!devices) return report_error("unable to enumerate physical devices", devices.error());

    if (json) {
        std::cout << "{\"schema_version\":2,\"kora_version\":";
        json_string(std::cout, kora::version_string());
        std::cout << ",\"loader_api_version\":";
        json_string(std::cout, version_string(loader.value().supported_api_version()));
        std::cout << ",\"instance_api_version\":";
        json_string(std::cout, version_string(instance.value().api_version()));
        std::cout << ",\"pipeline_cache_state\":\"not_queried\",\"devices\":[";
    } else {
        std::cout << "KORA version: " << kora::version_string() << '\n'
                  << "Vulkan loader API version: " << version_string(loader.value().supported_api_version())
                  << '\n'
                  << "Vulkan instance API version: " << version_string(instance.value().api_version()) << '\n'
                  << "Physical devices: " << devices.value().size() << '\n'
                  << "Pipeline cache state: not queried (kora-info does not initialize a Runtime)\n";
    }
    for (std::size_t index = 0; index < devices.value().size(); ++index) {
        const auto& raw = devices.value()[index];
        const auto device = kora::detail::DeviceCaps::from_vulkan(raw).to_device_info();
        if (json) {
            if (index != 0) std::cout << ',';
            print_json_device(index, raw, device);
        } else {
            print_human_device(index, raw, device);
        }
    }
    if (json) std::cout << "]}\n";
    return EXIT_SUCCESS;
}
