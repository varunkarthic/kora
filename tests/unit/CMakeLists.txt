kora_add_test(kora_unit_tests unit
  SOURCES
    test_assert.cpp
    test_errors.cpp
    test_event.cpp
    test_json.cpp
    test_kernel_metadata.cpp
    test_autotuner.cpp
    test_spirv.cpp
    test_pipeline_cache.cpp
    test_checked.cpp
    test_device_caps.cpp
    test_device_selection.cpp
    test_device_initialization.cpp
    test_options.cpp
    test_calibration.cpp
    test_performance_db.cpp
    test_transfer_model.cpp
    test_scheduler_cost.cpp
    test_scheduler_policy.cpp
    test_task_partition.cpp
    test_log.cpp
    test_suballocator.cpp
    test_memory_manager.cpp
    test_eviction.cpp
    test_recovery.cpp
    test_verification.cpp
    test_timestamp.cpp
    test_buffer_manager.cpp
    test_cpu_executor.cpp
    test_cpu_fallback.cpp
    test_dirty_ranges.cpp
    test_residency.cpp
    test_hazard_tracker.cpp
    test_queue_family_transfer.cpp
    test_task_graph.cpp
    test_execution_plan.cpp
    test_temporary_buffer_aliasing.cpp
)
target_link_libraries(kora_unit_tests PRIVATE ${KORA_VULKAN_HEADERS})
target_include_directories(kora_unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

if(KORA_GLSLANG_VALIDATOR)
  add_dependencies(kora_unit_tests kora_spirv_fixture)
  target_compile_definitions(kora_unit_tests PRIVATE
    KORA_SPIRV_FIXTURE_PATH="${KORA_SPIRV_FIXTURE}")
endif()
