find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

# kora_add_tests(<exe> <label> sources...): one executable per category, discovered into ctest with a label.
function(kora_add_tests exe label)
  add_executable(${exe} ${ARGN})
  target_link_libraries(${exe} PRIVATE kora::kora GTest::gtest_main)
  target_include_directories(${exe} PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  kora_configure_target(${exe})
  gtest_discover_tests(${exe} PROPERTIES LABELS ${label} DISCOVERY_TIMEOUT 60)
endfunction()

add_subdirectory(unit)
