kora_add_test(kora_integration_tests integration
  USES_VULKAN
  SOURCES
    test_physical_device.cpp
    test_vulkan_instance.cpp
    test_vulkan_skip.cpp
)
target_link_libraries(kora_integration_tests PRIVATE ${KORA_VULKAN_HEADERS})
