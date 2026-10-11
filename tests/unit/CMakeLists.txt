kora_add_test(kora_unit_tests unit
  SOURCES
    test_errors.cpp
    test_event.cpp
    test_json.cpp
    test_kernel_metadata.cpp
    test_spirv.cpp
    test_pipeline_cache.cpp
    test_checked.cpp
    test_device_caps.cpp
    test_device_selection.cpp
    test_options.cpp
    test_calibration.cpp
    test_log.cpp
    test_suballocator.cpp
    test_memory_manager.cpp
    test_buffer_manager.cpp
    test_dirty_ranges.cpp
    test_residency.cpp
)
target_link_libraries(kora_unit_tests PRIVATE ${KORA_VULKAN_HEADERS})

if(KORA_GLSLANG_VALIDATOR)
  add_dependencies(kora_unit_tests kora_spirv_fixture)
  target_compile_definitions(kora_unit_tests PRIVATE
    KORA_SPIRV_FIXTURE_PATH="${KORA_SPIRV_FIXTURE}")
endif()
