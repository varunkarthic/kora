#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "kora/version.hpp"
#include "vulkan/api.hpp"

namespace {

void print_vulkan_version(uint32_t version) {
    std::cout << VK_VERSION_MAJOR(version) << '.' << VK_VERSION_MINOR(version) << '.'
              << VK_VERSION_PATCH(version);
}

int report_error(const char* operation, const kora::Error& error) {
    std::cerr << "kora-info: " << operation << ": " << error.describe() << '\n';
    return EXIT_FAILURE;
}

}  // namespace

int main() {
    auto loader = kora::vk::Loader::load();
    if (!loader) return report_error("unable to load Vulkan", loader.error());

    auto instance = loader.value().create_instance();
    if (!instance) return report_error("unable to create Vulkan instance", instance.error());

    auto devices = instance.value().physical_devices();
    if (!devices) return report_error("unable to enumerate physical devices", devices.error());

    std::cout << "KORA version: " << kora::version_string() << '\n';
    std::cout << "Vulkan loader API version: ";
    print_vulkan_version(loader.value().supported_api_version());
    std::cout << '\n';
    std::cout << "Vulkan instance API version: ";
    print_vulkan_version(instance.value().api_version());
    std::cout << '\n';
    std::cout << "Physical devices: " << devices.value().size() << '\n';

    for (std::size_t index = 0; index < devices.value().size(); ++index) {
        const auto& device = devices.value()[index];
        std::cout << "  [" << index << "] " << device.properties.deviceName << " (Vulkan ";
        print_vulkan_version(device.properties.apiVersion);
        if (device.vulkan12_properties.driverName[0] != '\0') {
            std::cout << ", driver " << device.vulkan12_properties.driverName;
        }
        std::cout << ")\n";
    }

    return EXIT_SUCCESS;
}
