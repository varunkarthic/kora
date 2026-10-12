add_executable(kora-info kora-info/main.cpp)
target_include_directories(kora-info PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(kora-info PRIVATE kora::kora ${KORA_VULKAN_HEADERS})
kora_configure_target(kora-info)
install(TARGETS kora-info RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(kora-bench kora-bench/main.cpp)
target_include_directories(kora-bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(kora-bench PRIVATE kora::kora ${KORA_VULKAN_HEADERS})
target_compile_definitions(kora-bench PRIVATE KORA_BENCH_BUILD_TYPE="$<CONFIG>")
kora_configure_target(kora-bench)
find_program(KORA_BENCH_GLSLANG_VALIDATOR NAMES glslangValidator)
if(KORA_BENCH_GLSLANG_VALIDATOR)
  set(KORA_BENCH_VECTOR_ADD ${CMAKE_CURRENT_BINARY_DIR}/kora_bench_vector_add.spv)
  set(KORA_BENCH_SAXPY ${CMAKE_CURRENT_BINARY_DIR}/kora_bench_saxpy.spv)
  set(KORA_BENCH_REDUCTION ${CMAKE_CURRENT_BINARY_DIR}/kora_bench_reduction.spv)
  set(KORA_BENCH_SCAN ${CMAKE_CURRENT_BINARY_DIR}/kora_bench_scan.spv)
  set(KORA_BENCH_MATMUL ${CMAKE_CURRENT_BINARY_DIR}/kora_bench_tiled_matmul.spv)
  set(KORA_BENCH_STENCIL ${CMAKE_CURRENT_BINARY_DIR}/kora_bench_stencil.spv)
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
  add_custom_command(
    OUTPUT ${KORA_BENCH_REDUCTION}
    COMMAND ${KORA_BENCH_GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -o ${KORA_BENCH_REDUCTION}
            ${PROJECT_SOURCE_DIR}/tests/integration/reduction.comp
    DEPENDS ${PROJECT_SOURCE_DIR}/tests/integration/reduction.comp
    VERBATIM)
  add_custom_command(
    OUTPUT ${KORA_BENCH_SCAN}
    COMMAND ${KORA_BENCH_GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -o ${KORA_BENCH_SCAN}
            ${PROJECT_SOURCE_DIR}/tests/integration/scan.comp
    DEPENDS ${PROJECT_SOURCE_DIR}/tests/integration/scan.comp
    VERBATIM)
  add_custom_command(
    OUTPUT ${KORA_BENCH_MATMUL}
    COMMAND ${KORA_BENCH_GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -o ${KORA_BENCH_MATMUL}
            ${PROJECT_SOURCE_DIR}/tests/integration/tiled_matmul.comp
    DEPENDS ${PROJECT_SOURCE_DIR}/tests/integration/tiled_matmul.comp
    VERBATIM)
  add_custom_command(
    OUTPUT ${KORA_BENCH_STENCIL}
    COMMAND ${KORA_BENCH_GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -o ${KORA_BENCH_STENCIL}
            ${PROJECT_SOURCE_DIR}/tests/integration/stencil.comp
    DEPENDS ${PROJECT_SOURCE_DIR}/tests/integration/stencil.comp
    VERBATIM)
  add_custom_target(kora_bench_shader_fixtures DEPENDS ${KORA_BENCH_VECTOR_ADD} ${KORA_BENCH_SAXPY}
                    ${KORA_BENCH_REDUCTION} ${KORA_BENCH_SCAN} ${KORA_BENCH_MATMUL}
                    ${KORA_BENCH_STENCIL})
  add_dependencies(kora-bench kora_bench_shader_fixtures)
  target_compile_definitions(kora-bench PRIVATE KORA_BENCH_VECTOR_ADD_FIXTURE_NAME="kora_bench_vector_add.spv")
  target_compile_definitions(kora-bench PRIVATE KORA_BENCH_SAXPY_FIXTURE_NAME="kora_bench_saxpy.spv")
  target_compile_definitions(kora-bench PRIVATE KORA_BENCH_REDUCTION_FIXTURE_NAME="kora_bench_reduction.spv")
  target_compile_definitions(kora-bench PRIVATE KORA_BENCH_SCAN_FIXTURE_NAME="kora_bench_scan.spv")
  target_compile_definitions(kora-bench PRIVATE KORA_BENCH_MATMUL_FIXTURE_NAME="kora_bench_tiled_matmul.spv")
  target_compile_definitions(kora-bench PRIVATE KORA_BENCH_STENCIL_FIXTURE_NAME="kora_bench_stencil.spv")
  install(FILES ${KORA_BENCH_VECTOR_ADD} DESTINATION ${CMAKE_INSTALL_DATADIR}/kora/benchmarks)
  install(FILES ${KORA_BENCH_SAXPY} DESTINATION ${CMAKE_INSTALL_DATADIR}/kora/benchmarks)
  install(FILES ${KORA_BENCH_REDUCTION} DESTINATION ${CMAKE_INSTALL_DATADIR}/kora/benchmarks)
  install(FILES ${KORA_BENCH_SCAN} DESTINATION ${CMAKE_INSTALL_DATADIR}/kora/benchmarks)
  install(FILES ${KORA_BENCH_MATMUL} DESTINATION ${CMAKE_INSTALL_DATADIR}/kora/benchmarks)
  install(FILES ${KORA_BENCH_STENCIL} DESTINATION ${CMAKE_INSTALL_DATADIR}/kora/benchmarks)
endif()
install(TARGETS kora-bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
