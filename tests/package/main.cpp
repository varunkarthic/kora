#include <iostream>
#include <kora/device_info.hpp>
#include <kora/version.hpp>
#include <type_traits>

static_assert(!std::is_default_constructible_v<kora::DeviceInfo>);
static_assert(!std::is_copy_assignable_v<kora::DeviceInfo>);

int main() {
    std::cout << kora::version_string() << '\n';
    return 0;
}
