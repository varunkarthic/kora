#include <pybind11/pybind11.h>

#include "kora/version.hpp"

PYBIND11_MODULE(_core, module) {
    module.doc() = "Native Python bindings for the KORA C++ runtime.";
    module.def("version", &kora::version_string, "Return the linked KORA runtime version.");
    module.def(
        "schema_version", [] { return kora::kSchemaVersion; },
        "Return the persistence schema version used by this KORA build.");
}
