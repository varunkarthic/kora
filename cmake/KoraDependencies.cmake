# Build-time dependencies. The Vulkan *loader* is never linked: KORA dlopen()s it at runtime so a
# machine without Vulkan gets ErrorCode::NoVulkanImplementation instead of a load failure.
include(FetchContent)
find_package(Threads REQUIRED)

# Vulkan headers: system package first, pinned Khronos release as fallback.
find_package(VulkanHeaders CONFIG QUIET)
if(TARGET Vulkan::Headers)
  set(KORA_VULKAN_HEADERS Vulkan::Headers)
else()
  find_package(Vulkan QUIET)
  if(TARGET Vulkan::Headers)
    set(KORA_VULKAN_HEADERS Vulkan::Headers)
  else()
    message(STATUS "KORA: fetching Vulkan-Headers")
    FetchContent_Declare(vulkan_headers
      URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.3.296.tar.gz)
    FetchContent_MakeAvailable(vulkan_headers)
    set(KORA_VULKAN_HEADERS Vulkan::Headers)
  endif()
endif()

# SPIRV-Tools (optional): full SPIR-V validation. Built-in structural checks always run.
set(KORA_HAVE_SPIRV_TOOLS OFF)
if(KORA_ENABLE_SPIRV_TOOLS)
  find_package(SPIRV-Tools CONFIG QUIET)
  if(TARGET SPIRV-Tools-static)
    set(KORA_SPIRV_TOOLS_TARGET SPIRV-Tools-static)
    set(KORA_HAVE_SPIRV_TOOLS ON)
  elseif(TARGET SPIRV-Tools)
    set(KORA_SPIRV_TOOLS_TARGET SPIRV-Tools)
    set(KORA_HAVE_SPIRV_TOOLS ON)
  endif()
endif()
message(STATUS "KORA: SPIRV-Tools validation: ${KORA_HAVE_SPIRV_TOOLS}")

# Python and pybind11 are required only by the opt-in bindings build. Keeping this conditional avoids
# adding Python to the native runtime's build or installation contract.
if(KORA_BUILD_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    message(STATUS "KORA: fetching pybind11")
    FetchContent_Declare(pybind11
      URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz)
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()
