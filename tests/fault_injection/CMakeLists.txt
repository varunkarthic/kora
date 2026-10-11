kora_add_test(kora_fault_injection_tests fault_injection
  SOURCES
    test_vulkan_api.cpp
    test_command_pool_set.cpp
    test_queue_context.cpp
    test_staging_ring.cpp
)
target_link_libraries(kora_fault_injection_tests PRIVATE ${KORA_VULKAN_HEADERS})
