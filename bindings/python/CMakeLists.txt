# The extension is binding glue; optimize runtime behavior in libkora and avoid LTO on this template-heavy TU.
pybind11_add_module(_core MODULE NO_EXTRAS module.cpp)
target_link_libraries(_core PRIVATE kora::kora)
target_compile_features(_core PRIVATE cxx_std_20)
target_compile_options(_core PRIVATE $<$<CONFIG:Release>:-O1>)
install(TARGETS _core LIBRARY DESTINATION kora RUNTIME DESTINATION kora)
