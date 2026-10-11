kora_add_test(kora_unit_tests unit
  SOURCES
    test_errors.cpp
    test_event.cpp
    test_checked.cpp
    test_device_caps.cpp
    test_device_selection.cpp
    test_options.cpp
    test_log.cpp
    test_suballocator.cpp
    test_memory_manager.cpp
    test_buffer_manager.cpp
    test_dirty_ranges.cpp
    test_residency.cpp
)
target_link_libraries(kora_unit_tests PRIVATE ${KORA_VULKAN_HEADERS})
