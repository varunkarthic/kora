pybind11_add_module(_core MODULE module.cpp)
target_link_libraries(_core PRIVATE kora::kora)
target_compile_features(_core PRIVATE cxx_std_20)
install(TARGETS _core LIBRARY DESTINATION kora RUNTIME DESTINATION kora)
