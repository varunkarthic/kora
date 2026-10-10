#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "support/vulkan_test.hpp"
#include "vulkan/api.hpp"

namespace {

class IcdEnvironment {
public:
    explicit IcdEnvironment(const char* value) {
        if (const char* previous = std::getenv("VK_ICD_FILENAMES")) {
            existed_ = true;
            previous_ = previous;
        }
#if defined(_WIN32)
        setenv_result_ = _putenv_s("VK_ICD_FILENAMES", value);
#else
        setenv_result_ = setenv("VK_ICD_FILENAMES", value, 1);
#endif
    }

    ~IcdEnvironment() {
        if (setenv_result_ != 0) return;
#if defined(_WIN32)
        _putenv_s("VK_ICD_FILENAMES", existed_ ? previous_.c_str() : "");
#else
        if (existed_)
            setenv("VK_ICD_FILENAMES", previous_.c_str(), 1);
        else
            unsetenv("VK_ICD_FILENAMES");
#endif
    }

    bool configured() const noexcept { return setenv_result_ == 0; }

private:
    bool existed_ = false;
    int setenv_result_ = -1;
    std::string previous_;
};

}  // namespace

TEST(VulkanInstance, MissingIcdReturnsNoImplementation) {
    IcdEnvironment no_icd("kora-no-such-vulkan-icd.json");
    ASSERT_TRUE(no_icd.configured());

    auto loader = kora::vk::Loader::load();
    if (!loader) {
        EXPECT_EQ(loader.error().code, kora::ErrorCode::NoVulkanImplementation);
        return;
    }
    auto instance = loader.value().create_instance();
    ASSERT_FALSE(instance);
    EXPECT_EQ(instance.error().code, kora::ErrorCode::NoVulkanImplementation);
}

TEST(VulkanInstance, RepeatedCreationAndDestruction) {
    auto loader = kora::vk::Loader::load();
    if (!loader) KORA_REQUIRE_VULKAN(loader);

    kora::vk::Loader::InstanceOptions options;
    options.enable_validation = true;
    for (int i = 0; i < 100; ++i) {
        auto instance = loader.value().create_instance(options);
        if (!instance && instance.error().code == kora::ErrorCode::NoVulkanImplementation)
            KORA_REQUIRE_VULKAN(false);
        ASSERT_TRUE(instance) << instance.error().describe();
        EXPECT_NE(instance.value().handle(), VK_NULL_HANDLE);
        EXPECT_LE(instance.value().api_version(), VK_API_VERSION_1_3);
        EXPECT_TRUE(instance.value().validation_enabled() || !options.enable_validation);
    }
}

TEST(VulkanInstance, RequestedApiVersionIsClamped) {
    auto loader = kora::vk::Loader::load();
    if (!loader) KORA_REQUIRE_VULKAN(loader);

    kora::vk::Loader::InstanceOptions options;
    options.requested_api_version = VK_API_VERSION_1_1;
    auto instance = loader.value().create_instance(options);
    if (!instance && instance.error().code == kora::ErrorCode::NoVulkanImplementation)
        KORA_REQUIRE_VULKAN(false);
    ASSERT_TRUE(instance) << instance.error().describe();
    EXPECT_LE(instance.value().api_version(), VK_API_VERSION_1_1);
}
