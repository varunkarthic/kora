# Python package

KORA's Python package is a thin pybind11 extension linked to the C++ runtime. The `kora` Python
package contains no execution policy or duplicate runtime implementation; public behavior belongs in
the C++ API. The initial binding surface exposes runtime identity only and is intentionally a package
skeleton for the Python runtime, buffer, kernel, and event bindings tracked by the following issues.

## Install from a source checkout

Python 3.9 or newer, a C++20 compiler, CMake, and a supported build backend are required. From the
repository root, create and activate a virtual environment, then install the local package:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -c "import kora; print(kora.version())"
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. A successful import prints the linked KORA
version, currently `0.1.0` for the default CMake version configuration. The wheel includes the
compiled native extension and the small Python import facade. The build uses scikit-build-core and
pybind11 from PEP 517 build isolation; Python dependencies are not required for the default native
CMake build.

To install the package and its smoke-test dependency, use `python -m pip install '.[test]'`, then run
`python -m pytest`. The test verifies that package import reaches the compiled extension and that the
exported version and persistence schema are valid.

## Native CMake build

The bindings are optional in native builds. Enable them with the `python` CMake preset or configure
with `-DKORA_BUILD_PYTHON=ON`; CMake finds Python development headers and pybind11, fetching the
project's pinned pybind11 release only when no package configuration is available. The preset builds
the extension alongside the native library. Standard Debug, Release, sanitizer, and installed C++
consumer builds remain independent of Python.

## Initial API

| Function | Contract |
| --- | --- |
| `kora.version()` | Returns the version string from the linked C++ runtime. |
| `kora.schema_version()` | Returns the C++ persistence schema version. |

These functions are provided by `kora._core`; the public package re-exports them. Runtime and buffer
objects are not yet exposed. The package version in `pyproject.toml` must be kept in step with the
repository's release version when publishing Python distributions.
