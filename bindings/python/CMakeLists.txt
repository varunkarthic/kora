# The extension is binding glue; optimize runtime behavior in libkora and avoid LTO on this template-heavy TU.
pybind11_add_module(_core MODULE NO_EXTRAS module.cpp)
target_link_libraries(_core PRIVATE kora::kora)
target_compile_features(_core PRIVATE cxx_std_20)
target_compile_options(_core PRIVATE $<$<CONFIG:Release>:-O1>)
install(TARGETS _core LIBRARY DESTINATION kora RUNTIME DESTINATION kora)

if(KORA_BUILD_TESTS)
  find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
  add_test(NAME KoraBench.PythonSmoke
    COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/check_kora_python_bench.py"
      "${PROJECT_SOURCE_DIR}/tools/kora-bench/python_overhead.py" $<TARGET_FILE:_core>)
  set_tests_properties(KoraBench.PythonSmoke PROPERTIES LABELS integration TIMEOUT 30)
endif()
