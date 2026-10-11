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
  cmake_parse_arguments(KORA_TEST "USES_VULKAN" "TIMEOUT" "SOURCES" ${ARGN})
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
  if(KORA_TEST_TIMEOUT)
    list(APPEND _properties TIMEOUT ${KORA_TEST_TIMEOUT})
  endif()
  if(KORA_TEST_USES_VULKAN AND KORA_TEST_VULKAN_ICD)
    list(APPEND _properties ENVIRONMENT "VK_ICD_FILENAMES=${KORA_TEST_VULKAN_ICD}")
  endif()
  gtest_discover_tests(${exe} PROPERTIES ${_properties} DISCOVERY_TIMEOUT 60)
endfunction()

find_program(KORA_GLSLANG_VALIDATOR NAMES glslangValidator)
if(KORA_GLSLANG_VALIDATOR)
  set(KORA_SPIRV_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/spirv_reflection.comp.spv)
  add_custom_command(
    OUTPUT ${KORA_SPIRV_FIXTURE}
    COMMAND ${KORA_GLSLANG_VALIDATOR} -V --target-env vulkan1.0
            -o ${KORA_SPIRV_FIXTURE}
            ${CMAKE_CURRENT_SOURCE_DIR}/unit/spirv_reflection.comp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unit/spirv_reflection.comp
    VERBATIM)
  add_custom_target(kora_spirv_fixture DEPENDS ${KORA_SPIRV_FIXTURE})
endif()

add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(stress)
add_subdirectory(fault_injection)

if(TARGET kora-bench)
  find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
  add_test(NAME KoraBench.Smoke
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_kora_bench_json.py"
      $<TARGET_FILE:kora-bench>)
  set_tests_properties(KoraBench.Smoke PROPERTIES LABELS integration TIMEOUT 30)
  # TSan's instrumentation around host queue waits and external Vulkan drivers makes the GPU
  # microbenchmark exceed a practical test timeout; the CPU JSON smoke still runs under TSan.
  if(NOT KORA_SANITIZE_THREAD)
    add_test(NAME KoraBench.GpuSmoke
      COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/check_kora_bench_json.py"
        $<TARGET_FILE:kora-bench> --gpu --smoke)
    set_tests_properties(KoraBench.GpuSmoke PROPERTIES LABELS integration TIMEOUT 60 SKIP_RETURN_CODE 77)
  endif()
endif()
