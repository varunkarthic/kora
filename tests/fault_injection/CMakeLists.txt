kora_add_test(kora_fault_injection_tests fault_injection
  SOURCES
    test_vulkan_api.cpp
)
target_link_libraries(kora_fault_injection_tests PRIVATE ${KORA_VULKAN_HEADERS})
