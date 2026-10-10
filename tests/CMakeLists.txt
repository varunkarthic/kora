find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

set(KORA_TEST_VULKAN_ICD "" CACHE FILEPATH
  "Software Vulkan ICD JSON used by tests declared with USES_VULKAN")

# kora_add_test(<exe> <label> [USES_VULKAN] SOURCES <sources...>): create and discover a
# GoogleTest executable. Vulkan tests inherit the requested software ICD without affecting other tests.
function(kora_add_test exe label)
  cmake_parse_arguments(KORA_TEST "USES_VULKAN" "" "SOURCES" ${ARGN})
  if(KORA_TEST_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "kora_add_test(${exe}): unknown arguments: ${KORA_TEST_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT KORA_TEST_SOURCES)
    message(FATAL_ERROR "kora_add_test(${exe}): SOURCES must not be empty")
  endif()

  add_executable(${exe} ${KORA_TEST_SOURCES})
  target_link_libraries(${exe} PRIVATE kora::kora GTest::gtest_main)
  target_include_directories(${exe} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tests)
  kora_configure_target(${exe})

  set(_properties LABELS ${label})
  if(KORA_TEST_USES_VULKAN AND KORA_TEST_VULKAN_ICD)
    list(APPEND _properties ENVIRONMENT "VK_ICD_FILENAMES=${KORA_TEST_VULKAN_ICD}")
  endif()
  gtest_discover_tests(${exe} PROPERTIES ${_properties} DISCOVERY_TIMEOUT 60)
endfunction()

add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(stress)
add_subdirectory(fault_injection)
