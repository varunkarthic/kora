add_library(kora
  core/errors.cpp
  core/device_caps.cpp
  core/device_selection.cpp
  core/log.cpp
  core/options.cpp
  core/version.cpp
  memory/suballocator.cpp
  memory/memory_pool.cpp
  vulkan/loader.cpp
)
add_library(kora::kora ALIAS kora)
target_include_directories(kora PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(kora PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kora PUBLIC Threads::Threads
  PRIVATE $<BUILD_INTERFACE:${KORA_VULKAN_HEADERS}> ${CMAKE_DL_LIBS})
target_compile_definitions(kora PRIVATE VK_NO_PROTOTYPES)
if(KORA_HAVE_SPIRV_TOOLS)
  target_link_libraries(kora PRIVATE ${KORA_SPIRV_TOOLS_TARGET})
  target_compile_definitions(kora PRIVATE KORA_HAVE_SPIRV_TOOLS=1)
endif()
kora_configure_target(kora)
