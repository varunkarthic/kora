kora_add_test(kora_integration_tests integration
  USES_VULKAN
  SOURCES
    test_physical_device.cpp
    test_runtime.cpp
    test_cluster.cpp
    test_pipeline_manager.cpp
    test_command_pool_set.cpp
    test_queue_context.cpp
    test_staging_ring.cpp
    test_transfer_engine.cpp
    test_vulkan_instance.cpp
    test_vulkan_skip.cpp
)
target_link_libraries(kora_integration_tests PRIVATE ${KORA_VULKAN_HEADERS})
target_include_directories(kora_integration_tests PRIVATE ${KORA_SPIRV_HEADERS_INCLUDE_DIR})
if(KORA_GLSLANG_VALIDATOR)
  set(KORA_VECTOR_ADD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/vector_add.comp.spv)
  set(KORA_SAXPY_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/saxpy.comp.spv)
  set(KORA_PARTITION_ADD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/partition_add.comp.spv)
  add_custom_command(
    OUTPUT ${KORA_VECTOR_ADD_FIXTURE}
    COMMAND ${KORA_GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -o ${KORA_VECTOR_ADD_FIXTURE}
            ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.comp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.comp
    VERBATIM)
  add_custom_command(
    OUTPUT ${KORA_SAXPY_FIXTURE}
    COMMAND ${KORA_GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -o ${KORA_SAXPY_FIXTURE}
            ${CMAKE_CURRENT_SOURCE_DIR}/saxpy.comp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/saxpy.comp
    VERBATIM)
  add_custom_command(
    OUTPUT ${KORA_PARTITION_ADD_FIXTURE}
    COMMAND ${KORA_GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -o ${KORA_PARTITION_ADD_FIXTURE}
            ${CMAKE_CURRENT_SOURCE_DIR}/partition_add.comp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/partition_add.comp
    VERBATIM)
  add_custom_target(kora_dispatch_fixtures DEPENDS ${KORA_VECTOR_ADD_FIXTURE} ${KORA_SAXPY_FIXTURE}
    ${KORA_PARTITION_ADD_FIXTURE})
  add_dependencies(kora_integration_tests kora_spirv_fixture)
  add_dependencies(kora_integration_tests kora_dispatch_fixtures)
  target_compile_definitions(kora_integration_tests PRIVATE
    KORA_SPIRV_FIXTURE_PATH="${KORA_SPIRV_FIXTURE}"
    KORA_VECTOR_ADD_FIXTURE_PATH="${KORA_VECTOR_ADD_FIXTURE}"
    KORA_SAXPY_FIXTURE_PATH="${KORA_SAXPY_FIXTURE}")
  target_compile_definitions(kora_integration_tests PRIVATE
    KORA_PARTITION_ADD_FIXTURE_PATH="${KORA_PARTITION_ADD_FIXTURE}")
endif()

find_program(KORA_SPIRV_ASSEMBLER NAMES spirv-as)
if(KORA_SPIRV_ASSEMBLER)
  set(KORA_AUTOTUNE_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/autotune.spv)
  add_custom_command(
    OUTPUT ${KORA_AUTOTUNE_FIXTURE}
    COMMAND ${KORA_SPIRV_ASSEMBLER} --target-env vulkan1.2 -o ${KORA_AUTOTUNE_FIXTURE}
            ${CMAKE_CURRENT_SOURCE_DIR}/autotune.spvasm
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/autotune.spvasm
    VERBATIM)
  add_custom_target(kora_autotune_fixture DEPENDS ${KORA_AUTOTUNE_FIXTURE})
  add_dependencies(kora_integration_tests kora_autotune_fixture)
  target_compile_definitions(kora_integration_tests PRIVATE
    KORA_AUTOTUNE_FIXTURE_PATH="${KORA_AUTOTUNE_FIXTURE}")
endif()

if(TARGET kora-info)
  find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
  add_test(NAME KoraInfo.ListsDevices COMMAND $<TARGET_FILE:kora-info>)
  set_tests_properties(KoraInfo.ListsDevices PROPERTIES
    LABELS integration
    PASS_REGULAR_EXPRESSION "Physical devices: [0-9]+"
    SKIP_REGULAR_EXPRESSION "NoVulkanImplementation")

  add_test(NAME KoraInfo.MissingVulkan
    COMMAND ${CMAKE_COMMAND}
      "-DKORA_INFO_EXECUTABLE=$<TARGET_FILE:kora-info>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_kora_info_failure.cmake")
  set_tests_properties(KoraInfo.MissingVulkan PROPERTIES LABELS integration)

  add_test(NAME KoraInfo.Json
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_kora_info_json.py"
      $<TARGET_FILE:kora-info>)
  set_tests_properties(KoraInfo.Json PROPERTIES LABELS integration SKIP_RETURN_CODE 77)
  if(KORA_TEST_VULKAN_ICD)
    set_tests_properties(KoraInfo.Json PROPERTIES
      ENVIRONMENT "VK_ICD_FILENAMES=${KORA_TEST_VULKAN_ICD}")
  endif()
endif()
