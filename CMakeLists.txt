cmake_minimum_required(VERSION 3.21)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# MAJOR.MINOR live in VERSION; PATCH is injected by the release pipeline (-DKORA_VERSION_PATCH=N).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _kora_ver)
string(STRIP "${_kora_ver}" _kora_ver)
if(NOT _kora_ver MATCHES "^([0-9]+)\\.([0-9]+)$")
  message(FATAL_ERROR "VERSION must contain MAJOR.MINOR, got '${_kora_ver}'")
endif()
set(KORA_VERSION_MAJOR ${CMAKE_MATCH_1})
set(KORA_VERSION_MINOR ${CMAKE_MATCH_2})
set(KORA_VERSION_PATCH 0 CACHE STRING "Patch version (set by the release pipeline)")

project(kora VERSION ${KORA_VERSION_MAJOR}.${KORA_VERSION_MINOR}.${KORA_VERSION_PATCH} LANGUAGES CXX
        DESCRIPTION "Kernel Offload Runtime Architecture")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(KORA_BUILD_TESTS        "Build unit/integration tests" ON)
option(KORA_BUILD_TOOLS        "Build kora-info / kora-bench" ON)
option(KORA_BUILD_BENCHMARKS   "Build benchmarks" OFF)
option(KORA_BUILD_COMPETITOR_BENCH "Build the optional macOS Metal/OpenCL/OpenGL/Accelerate comparison tool" OFF)
option(KORA_BUILD_PYTHON       "Build Python bindings" OFF)
option(KORA_ENABLE_SPIRV_TOOLS "Use SPIRV-Tools for SPIR-V validation when available" ON)
option(KORA_SANITIZE_ADDRESS   "AddressSanitizer" OFF)
option(KORA_SANITIZE_UNDEFINED "UndefinedBehaviorSanitizer" OFF)
option(KORA_SANITIZE_THREAD    "ThreadSanitizer" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(KoraWarnings)
include(KoraDependencies)
include(GNUInstallDirs)

configure_file(include/kora/version.hpp.in ${PROJECT_BINARY_DIR}/include/kora/version.hpp @ONLY)

add_subdirectory(src)
if(KORA_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
if(KORA_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  enable_testing()
  add_subdirectory(tests)
endif()
if(KORA_BUILD_BENCHMARKS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt)
  add_subdirectory(benchmarks)
endif()
if(KORA_BUILD_PYTHON AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/CMakeLists.txt)
  add_subdirectory(bindings/python)
endif()

install(TARGETS kora EXPORT koraTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/kora DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} PATTERN "*.in" EXCLUDE)
install(FILES ${PROJECT_BINARY_DIR}/include/kora/version.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kora)
install(EXPORT koraTargets NAMESPACE kora:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kora)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/koraConfig.cmake.in ${PROJECT_BINARY_DIR}/koraConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kora)
write_basic_package_version_file(${PROJECT_BINARY_DIR}/koraConfigVersion.cmake COMPATIBILITY SameMinorVersion)
install(FILES ${PROJECT_BINARY_DIR}/koraConfig.cmake ${PROJECT_BINARY_DIR}/koraConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kora)
