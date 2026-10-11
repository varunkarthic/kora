kora_add_test(kora_unit_tests unit
  SOURCES
    test_errors.cpp
    test_event.cpp
    test_json.cpp
    test_kernel_metadata.cpp
    test_spirv.cpp
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

find_program(KORA_GLSLANG_VALIDATOR NAMES glslangValidator)
if(KORA_GLSLANG_VALIDATOR)
  set(KORA_SPIRV_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/spirv_reflection.comp.spv)
  add_custom_command(
    OUTPUT ${KORA_SPIRV_FIXTURE}
    COMMAND ${KORA_GLSLANG_VALIDATOR} -V --target-env vulkan1.0
            -o ${KORA_SPIRV_FIXTURE}
            ${CMAKE_CURRENT_SOURCE_DIR}/spirv_reflection.comp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/spirv_reflection.comp
    VERBATIM)
  add_custom_target(kora_spirv_fixture DEPENDS ${KORA_SPIRV_FIXTURE})
  add_dependencies(kora_unit_tests kora_spirv_fixture)
  target_compile_definitions(kora_unit_tests PRIVATE
    KORA_SPIRV_FIXTURE_PATH="${KORA_SPIRV_FIXTURE}")
endif()
