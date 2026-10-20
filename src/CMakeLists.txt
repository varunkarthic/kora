add_library(kora
  core/errors.cpp
  core/event.cpp
  core/device_caps.cpp
  core/buffer_manager.cpp
  core/cpu_executor.cpp
  core/calibration.cpp
  core/calibration_cost.cpp
  core/scheduler_placement.cpp
  core/device_selection.cpp
  core/execution_plan.cpp
  core/log.cpp
  core/json.cpp
  core/options.cpp
  core/performance_db.cpp
  core/transfer_model.cpp
  core/scheduler_cost.cpp
  core/scheduler_policy.cpp
  core/task_partition.cpp
  core/runtime.cpp
  profiling/timestamp.cpp
  core/task_graph.cpp
  core/temporary_buffer_aliasing.cpp
  core/version.cpp
  memory/suballocator.cpp
  memory/memory_pool.cpp
  memory/staging_ring.cpp
  memory/transfer_engine.cpp
  memory/memory_manager.cpp
  memory/eviction.cpp
  memory/recovery.cpp
  memory/dirty_ranges.cpp
  memory/residency.cpp
  kernel/metadata.cpp
  kernel/autotuner.cpp
  kernel/pipeline_manager.cpp
  kernel/spirv.cpp
  vulkan/loader.cpp
  vulkan/command_pool_set.cpp
  vulkan/hazard_tracker.cpp
  vulkan/queue_family_transfer.cpp
  vulkan/queue_context.cpp
  remote/sha256.cpp
  remote/socket.cpp
  remote/protocol.cpp
  remote/worker_server.cpp
  remote/cluster.cpp
)
add_library(kora::kora ALIAS kora)
target_include_directories(kora PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(kora PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kora PUBLIC Threads::Threads
  PRIVATE $<BUILD_INTERFACE:${KORA_VULKAN_HEADERS}> ${CMAKE_DL_LIBS})
if(WIN32)
  target_link_libraries(kora PRIVATE ws2_32)
endif()
target_include_directories(kora PRIVATE ${KORA_SPIRV_HEADERS_INCLUDE_DIR})
target_compile_definitions(kora PRIVATE VK_NO_PROTOTYPES)
if(KORA_HAVE_SPIRV_TOOLS)
  target_link_libraries(kora PRIVATE ${KORA_SPIRV_TOOLS_TARGET})
  target_compile_definitions(kora PRIVATE KORA_HAVE_SPIRV_TOOLS=1)
endif()
kora_configure_target(kora)
