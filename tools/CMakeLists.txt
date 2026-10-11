add_executable(kora-info kora-info/main.cpp)
target_include_directories(kora-info PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(kora-info PRIVATE kora::kora ${KORA_VULKAN_HEADERS})
kora_configure_target(kora-info)
install(TARGETS kora-info RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(kora-bench kora-bench/main.cpp)
target_include_directories(kora-bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(kora-bench PRIVATE kora::kora)
kora_configure_target(kora-bench)
find_program(KORA_BENCH_GLSLANG_VALIDATOR NAMES glslangValidator)
if(KORA_BENCH_GLSLANG_VALIDATOR)
  set(KORA_BENCH_VECTOR_ADD ${CMAKE_CURRENT_BINARY_DIR}/kora_bench_vector_add.spv)
  set(KORA_BENCH_SAXPY ${CMAKE_CURRENT_BINARY_DIR}/kora_bench_saxpy.spv)
  add_custom_command(
    OUTPUT ${KORA_BENCH_VECTOR_ADD}
    COMMAND ${KORA_BENCH_GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -o ${KORA_BENCH_VECTOR_ADD}
            ${PROJECT_SOURCE_DIR}/tests/integration/vector_add.comp
    DEPENDS ${PROJECT_SOURCE_DIR}/tests/integration/vector_add.comp
    VERBATIM)
  add_custom_command(
    OUTPUT ${KORA_BENCH_SAXPY}
    COMMAND ${KORA_BENCH_GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -o ${KORA_BENCH_SAXPY}
            ${PROJECT_SOURCE_DIR}/tests/integration/saxpy.comp
    DEPENDS ${PROJECT_SOURCE_DIR}/tests/integration/saxpy.comp
    VERBATIM)
  add_custom_target(kora_bench_shader_fixtures DEPENDS ${KORA_BENCH_VECTOR_ADD} ${KORA_BENCH_SAXPY})
  add_dependencies(kora-bench kora_bench_shader_fixtures)
  target_compile_definitions(kora-bench PRIVATE KORA_BENCH_VECTOR_ADD_FIXTURE_NAME="kora_bench_vector_add.spv")
  target_compile_definitions(kora-bench PRIVATE KORA_BENCH_SAXPY_FIXTURE_NAME="kora_bench_saxpy.spv")
  install(FILES ${KORA_BENCH_VECTOR_ADD} DESTINATION ${CMAKE_INSTALL_DATADIR}/kora/benchmarks)
  install(FILES ${KORA_BENCH_SAXPY} DESTINATION ${CMAKE_INSTALL_DATADIR}/kora/benchmarks)
endif()
install(TARGETS kora-bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
