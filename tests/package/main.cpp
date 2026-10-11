#include <cstdint>
#include <iostream>
#include <kora/buffer.hpp>
#include <kora/device_info.hpp>
#include <kora/kernel_metadata.hpp>
#include <kora/runtime.hpp>
#include <kora/version.hpp>
#include <type_traits>
#include <utility>

static_assert(!std::is_default_constructible_v<kora::DeviceInfo>);
static_assert(!std::is_copy_assignable_v<kora::DeviceInfo>);
static_assert(!std::is_copy_constructible_v<kora::Runtime>);
static_assert(std::is_move_constructible_v<kora::Runtime>);

int main() {
    kora::KernelMetadata metadata;
    metadata.kernel_id = "package.consumer";
    if (!kora::validate_kernel_metadata(metadata)) return 1;
    auto parsed_metadata = kora::parse_kernel_metadata(
        R"({"schema_version":1,"kernel_id":"package.consumer","bindings":[],"push_constant_size":0,"push_constants":[],"specialization_constants":[],"local_size":[1,1,1]})");
    if (!parsed_metadata) return 1;

    kora::Runtime empty_runtime;
    if (!empty_runtime.devices().empty()) return 1;

    kora::BufferManager buffers;
    auto created = buffers.create<std::uint32_t>(2, kora::BufferUsage::Storage);
    if (!created) return 1;
    auto values = std::move(created).value();
    {
        auto mapped = values.map(kora::MapAccess::Write);
        if (!mapped) return 1;
        mapped.value()[0] = 42;
    }
    if (!values.mark_used(1)) return 1;
    const auto id = values.id();
    values = {};
    auto reclaimed = buffers.advance_completion(1);
    if (!reclaimed || reclaimed.value() != 1 || buffers.inspect(id)) return 1;

    std::cout << kora::version_string() << '\n';
    return 0;
}
