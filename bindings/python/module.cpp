#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kora/execution_plan.hpp"
#include "kora/runtime.hpp"
#include "kora/version.hpp"

namespace py = pybind11;

namespace {

PyObject* g_kora_error_type = nullptr;

class PythonKoraError final : public std::runtime_error {
public:
    explicit PythonKoraError(kora::Error error)
        : std::runtime_error(error.describe()), error_(std::move(error)) {}
    const kora::Error& error() const noexcept { return error_; }

private:
    kora::Error error_;
};

template <class T>
T value_or_throw(kora::Result<T> result) {
    if (!result) throw PythonKoraError(std::move(result).error());
    return std::move(result).value();
}

void status_or_throw(kora::Status status) {
    if (!status) throw PythonKoraError(std::move(status).error());
}

py::object make_python_error(PyObject* error_type, const kora::Error& error);

void translate_kora_exception(std::exception_ptr error) {
    if (!error) return;
    try {
        std::rethrow_exception(error);
    } catch (const PythonKoraError& exception) {
        auto instance = make_python_error(g_kora_error_type, exception.error());
        PyErr_SetObject(g_kora_error_type, instance.ptr());
    } catch (...) {
        throw;
    }
}

py::object make_python_error(PyObject* error_type, const kora::Error& error) {
    auto instance =
        py::reinterpret_steal<py::object>(PyObject_CallFunction(error_type, "s", error.describe().c_str()));
    if (!instance) throw py::error_already_set();
    instance.attr("code") = py::cast(error.code);
    instance.attr("code_name") = kora::to_string(error.code);
    instance.attr("vk_result") = error.vk_result;
    instance.attr("kernel_id") = error.kernel_id;
    instance.attr("device") = error.device;
    instance.attr("task_id") = error.task_id;
    instance.attr("recovery_action") = error.recovery_action;
    return instance;
}

struct PyRuntime {
    std::shared_ptr<kora::Runtime> value;
};

struct PyBuffer {
    kora::BufferHandle handle;
    std::shared_ptr<PyRuntime> owner;

    std::shared_ptr<struct PyMappedView> map(kora::MapAccess access) const;
    py::bytes read() const {
        std::optional<kora::Result<std::vector<std::byte>>> result;
        {
            py::gil_scoped_release release;
            result.emplace(handle.read_bytes());
        }
        const auto bytes = value_or_throw(std::move(*result));
        return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    void write(py::buffer source) const {
        const auto view = source.request();
        if (view.ndim != 1 || view.itemsize <= 0 || view.strides[0] != view.itemsize || view.size < 0)
            throw PythonKoraError(kora::Error(kora::ErrorCode::InvalidArgument,
                                              "buffer write requires a contiguous one-dimensional exporter"));
        const auto byte_count = static_cast<std::size_t>(view.size) * static_cast<std::size_t>(view.itemsize);
        const auto* data = static_cast<const std::byte*>(view.ptr);
        kora::Status status;
        {
            py::gil_scoped_release release;
            status = handle.write_bytes(std::span<const std::byte>(data, byte_count));
        }
        status_or_throw(std::move(status));
    }
};

struct PyMappedView {
    kora::MappedBytes mapping;
    bool readonly = false;

    PyMappedView(kora::MappedBytes mapped, bool is_readonly)
        : mapping(std::move(mapped)), readonly(is_readonly) {}
};

std::shared_ptr<PyMappedView> PyBuffer::map(kora::MapAccess access) const {
    auto mapped = value_or_throw(handle.map(access));
    return std::make_shared<PyMappedView>(std::move(mapped), access == kora::MapAccess::Read);
}

struct PyKernel {
    kora::Kernel value;
    std::shared_ptr<PyRuntime> owner;
};

struct PyExecutionPlan {
    kora::ExecutionPlan value;
};

struct PythonTaskExecutor {
    py::function callback;

    static kora::Status invoke(const kora::TaskNode& task, void* opaque) {
        auto& self = *static_cast<PythonTaskExecutor*>(opaque);
        try {
            py::object result = self.callback(task);
            if (result.is_none()) return {};
            if (py::isinstance<py::bool_>(result)) {
                if (result.cast<bool>()) return {};
                return kora::Error(kora::ErrorCode::PythonBindingError,
                                   "Python TaskGraph executor returned False");
            }
            return kora::Error(kora::ErrorCode::PythonBindingError,
                               "Python TaskGraph executor must return None or True");
        } catch (py::error_already_set& error) {
            return kora::Error(kora::ErrorCode::PythonBindingError, error.what());
        }
    }
};

struct AwaitGuard {
    py::object loop;
    py::object future;
    py::object resolver;
    py::object error_type;
};

struct PyEvent {
    kora::Event value;
    std::shared_ptr<PyRuntime> owner;
    std::vector<std::shared_ptr<PyBuffer>> keepalive;

    void wait() const {
        kora::Status status;
        {
            py::gil_scoped_release release;
            status = value.wait();
        }
        status_or_throw(std::move(status));
    }

    void wait_for(double seconds) const {
        if (seconds < 0.0)
            throw PythonKoraError(
                kora::Error(kora::ErrorCode::InvalidArgument, "event wait timeout must be nonnegative"));
        kora::Status status;
        {
            py::gil_scoped_release release;
            status = value.wait_for(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(seconds)));
        }
        status_or_throw(std::move(status));
    }

    std::shared_ptr<AwaitGuard> register_awaitable(py::object loop, py::object future, py::object resolver,
                                                   py::object error_type) const {
        auto guard = std::make_shared<AwaitGuard>(
            AwaitGuard{std::move(loop), std::move(future), std::move(resolver), std::move(error_type)});
        const std::weak_ptr<AwaitGuard> weak_guard = guard;
        auto subscribed = value.on_complete([weak_guard](kora::Status completion) {
            py::gil_scoped_acquire acquire;
            const auto state = weak_guard.lock();
            if (!state) return kora::Status{};
            try {
                if (completion) {
                    state->loop.attr("call_soon_threadsafe")(state->resolver, state->future, py::none(),
                                                             py::none());
                } else {
                    const auto error = std::move(completion).error();
                    auto exception = make_python_error(state->error_type.ptr(), error);
                    state->loop.attr("call_soon_threadsafe")(state->resolver, state->future, py::none(),
                                                             std::move(exception));
                }
            } catch (py::error_already_set& error) {
                error.discard_as_unraisable("kora.Event completion callback");
            }
            return kora::Status{};
        });
        status_or_throw(std::move(subscribed));
        return guard;
    }
};

struct PyDispatchBuilder {
    kora::DispatchBuilder value;
    std::shared_ptr<PyRuntime> owner;
    std::vector<std::shared_ptr<PyBuffer>> keepalive;

    PyDispatchBuilder(std::shared_ptr<PyRuntime> runtime, const kora::Kernel& kernel)
        : value(runtime->value->dispatch(kernel)), owner(std::move(runtime)) {}

    PyDispatchBuilder& read(std::shared_ptr<PyBuffer> buffer) {
        if (buffer->owner != owner)
            throw PythonKoraError(
                kora::Error(kora::ErrorCode::InvalidArgument, "buffer belongs to a different Runtime"));
        value.read(buffer->handle);
        keepalive.push_back(std::move(buffer));
        return *this;
    }
    PyDispatchBuilder& write(std::shared_ptr<PyBuffer> buffer) {
        if (buffer->owner != owner)
            throw PythonKoraError(
                kora::Error(kora::ErrorCode::InvalidArgument, "buffer belongs to a different Runtime"));
        value.write(buffer->handle);
        keepalive.push_back(std::move(buffer));
        return *this;
    }
    PyDispatchBuilder& read_write(std::shared_ptr<PyBuffer> buffer) {
        if (buffer->owner != owner)
            throw PythonKoraError(
                kora::Error(kora::ErrorCode::InvalidArgument, "buffer belongs to a different Runtime"));
        value.read_write(buffer->handle);
        keepalive.push_back(std::move(buffer));
        return *this;
    }
    PyDispatchBuilder& push(py::buffer source) {
        const auto view = source.request();
        if (view.ndim != 1 || view.itemsize <= 0 || view.strides[0] != view.itemsize || view.size < 0)
            throw PythonKoraError(
                kora::Error(kora::ErrorCode::InvalidArgument,
                            "push constants require a contiguous one-dimensional exporter"));
        const auto byte_count = static_cast<std::size_t>(view.size) * static_cast<std::size_t>(view.itemsize);
        value.push(std::span<const std::byte>(static_cast<const std::byte*>(view.ptr), byte_count));
        return *this;
    }
    std::shared_ptr<PyEvent> submit() {
        std::optional<kora::Result<kora::Event>> result;
        {
            py::gil_scoped_release release;
            result.emplace(value.submit());
        }
        auto event = value_or_throw(std::move(*result));
        return std::make_shared<PyEvent>(PyEvent{std::move(event), owner, std::move(keepalive)});
    }
};

std::shared_ptr<PyRuntime> create_runtime(kora::RuntimeOptions options) {
    std::optional<kora::Result<kora::Runtime>> result;
    {
        py::gil_scoped_release release;
        result.emplace(kora::Runtime::create(options));
    }
    auto runtime = value_or_throw(std::move(*result));
    return std::make_shared<PyRuntime>(PyRuntime{std::make_shared<kora::Runtime>(std::move(runtime))});
}

std::shared_ptr<PyBuffer> create_buffer(const std::shared_ptr<PyRuntime>& runtime, std::uint64_t size_bytes,
                                        kora::BufferUsage usage) {
    std::optional<kora::Result<kora::BufferHandle>> result;
    {
        py::gil_scoped_release release;
        result.emplace(runtime->value->create_buffer(size_bytes, usage));
    }
    return std::make_shared<PyBuffer>(PyBuffer{value_or_throw(std::move(*result)), runtime});
}

std::shared_ptr<PyKernel> load_kernel(const std::shared_ptr<PyRuntime>& runtime,
                                      const std::filesystem::path& path, std::string entry_point,
                                      kora::KernelMetadata metadata) {
    std::optional<kora::Result<kora::Kernel>> result;
    {
        py::gil_scoped_release release;
        result.emplace(runtime->value->load_kernel(path, std::move(entry_point), std::move(metadata)));
    }
    return std::make_shared<PyKernel>(PyKernel{value_or_throw(std::move(*result)), runtime});
}

std::shared_ptr<PyKernel> load_kernel_for_device(const std::shared_ptr<PyRuntime>& runtime,
                                                 const std::filesystem::path& path, std::string entry_point,
                                                 kora::KernelMetadata metadata,
                                                 const kora::DeviceId& device) {
    std::optional<kora::Result<kora::Kernel>> result;
    {
        py::gil_scoped_release release;
        result.emplace(
            runtime->value->load_kernel(path, std::move(entry_point), std::move(metadata), device));
    }
    return std::make_shared<PyKernel>(PyKernel{value_or_throw(std::move(*result)), runtime});
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "Native Python bindings for the KORA C++ runtime.";
    module.def("version", &kora::version_string, "Return the linked KORA runtime version.");
    module.def(
        "schema_version", [] { return kora::kSchemaVersion; },
        "Return the persistence schema version used by this KORA build.");

    auto error_type =
        py::reinterpret_steal<py::object>(PyErr_NewException("kora.KoraError", PyExc_RuntimeError, nullptr));
    error_type.attr("__module__") = "kora";
    module.attr("KoraError") = error_type;
    g_kora_error_type = error_type.ptr();
    py::register_exception_translator(&translate_kora_exception);

    py::enum_<kora::ErrorCode>(module, "ErrorCode")
        .value("OK", kora::ErrorCode::Ok)
        .value("NO_VULKAN_IMPLEMENTATION", kora::ErrorCode::NoVulkanImplementation)
        .value("NO_ELIGIBLE_DEVICE", kora::ErrorCode::NoEligibleDevice)
        .value("NO_COMPUTE_QUEUE", kora::ErrorCode::NoComputeQueue)
        .value("UNSUPPORTED_FEATURE", kora::ErrorCode::UnsupportedFeature)
        .value("INVALID_KERNEL", kora::ErrorCode::InvalidKernel)
        .value("INVALID_KERNEL_BINDING", kora::ErrorCode::InvalidKernelBinding)
        .value("INVALID_DISPATCH", kora::ErrorCode::InvalidDispatch)
        .value("ALLOCATION_FAILED", kora::ErrorCode::AllocationFailed)
        .value("OUT_OF_MEMORY_BUDGET", kora::ErrorCode::OutOfMemoryBudget)
        .value("TRANSFER_FAILED", kora::ErrorCode::TransferFailed)
        .value("TIMEOUT", kora::ErrorCode::Timeout)
        .value("DEVICE_LOST", kora::ErrorCode::DeviceLost)
        .value("PIPELINE_CREATION_FAILED", kora::ErrorCode::PipelineCreationFailed)
        .value("SYNCHRONIZATION_ERROR", kora::ErrorCode::SynchronizationError)
        .value("VALIDATION_ERROR", kora::ErrorCode::ValidationError)
        .value("PYTHON_BINDING_ERROR", kora::ErrorCode::PythonBindingError)
        .value("INTERNAL_INVARIANT_VIOLATION", kora::ErrorCode::InternalInvariantViolation)
        .value("INVALID_ARGUMENT", kora::ErrorCode::InvalidArgument)
        .value("OVERFLOW", kora::ErrorCode::Overflow)
        .value("GRAPH_CYCLE", kora::ErrorCode::GraphCycle)
        .value("CANCELLED", kora::ErrorCode::Cancelled)
        .value("STALE_HANDLE", kora::ErrorCode::StaleHandle)
        .value("DATA_LOSS", kora::ErrorCode::DataLoss)
        .value("IO_ERROR", kora::ErrorCode::IoError);

    py::enum_<kora::DevicePolicy>(module, "DevicePolicy")
        .value("AUTO", kora::DevicePolicy::Auto)
        .value("PREFER_DISCRETE", kora::DevicePolicy::PreferDiscrete)
        .value("PREFER_INTEGRATED", kora::DevicePolicy::PreferIntegrated)
        .value("BY_INDEX", kora::DevicePolicy::ByIndex)
        .value("BY_UUID", kora::DevicePolicy::ByUuid)
        .value("ALL_ELIGIBLE", kora::DevicePolicy::AllEligible);
    py::enum_<kora::BufferUsage>(module, "BufferUsage")
        .value("GENERIC", kora::BufferUsage::Generic)
        .value("STORAGE", kora::BufferUsage::Storage)
        .value("UNIFORM", kora::BufferUsage::Uniform)
        .value("TRANSFER_SOURCE", kora::BufferUsage::TransferSource)
        .value("TRANSFER_DESTINATION", kora::BufferUsage::TransferDestination);
    py::enum_<kora::MapAccess>(module, "MapAccess")
        .value("READ", kora::MapAccess::Read)
        .value("WRITE", kora::MapAccess::Write)
        .value("READ_WRITE", kora::MapAccess::ReadWrite);
    py::enum_<kora::KernelBindingKind>(module, "KernelBindingKind")
        .value("STORAGE_BUFFER", kora::KernelBindingKind::StorageBuffer)
        .value("UNIFORM_BUFFER", kora::KernelBindingKind::UniformBuffer)
        .value("STORAGE_IMAGE", kora::KernelBindingKind::StorageImage)
        .value("SAMPLED_IMAGE", kora::KernelBindingKind::SampledImage);
    py::enum_<kora::KernelAccess>(module, "KernelAccess")
        .value("READ", kora::KernelAccess::Read)
        .value("WRITE", kora::KernelAccess::Write)
        .value("READ_WRITE", kora::KernelAccess::ReadWrite);
    py::class_<kora::DeviceId>(module, "DeviceId")
        .def(py::init<>())
        .def(py::init<std::uint32_t, std::string>(), py::arg("index"), py::arg("uuid") = "")
        .def_readwrite("index", &kora::DeviceId::index)
        .def_readwrite("uuid", &kora::DeviceId::uuid);
    py::enum_<kora::TaskType>(module, "TaskType")
        .value("UPLOAD", kora::TaskType::Upload)
        .value("DOWNLOAD", kora::TaskType::Download)
        .value("COPY", kora::TaskType::Copy)
        .value("FILL", kora::TaskType::Fill)
        .value("DISPATCH", kora::TaskType::Dispatch)
        .value("CPU_TASK", kora::TaskType::CpuTask)
        .value("HOST_CALLBACK", kora::TaskType::HostCallback);
    py::enum_<kora::AccessMode>(module, "AccessMode")
        .value("READ", kora::AccessMode::Read)
        .value("WRITE", kora::AccessMode::Write)
        .value("READ_WRITE", kora::AccessMode::ReadWrite);
    py::class_<kora::BufferId>(module, "BufferId")
        .def(py::init<>())
        .def(py::init<std::uint32_t, std::uint64_t>(), py::arg("slot"), py::arg("generation"))
        .def_readwrite("slot", &kora::BufferId::slot)
        .def_readwrite("generation", &kora::BufferId::generation);
    py::class_<kora::BufferAccess>(module, "BufferAccess")
        .def(py::init<>())
        .def(py::init<kora::BufferId, kora::AccessMode>(), py::arg("buffer"), py::arg("mode"))
        .def_readwrite("buffer", &kora::BufferAccess::buffer)
        .def_readwrite("mode", &kora::BufferAccess::mode);
    py::class_<kora::TaskMetadata>(module, "TaskMetadata")
        .def(py::init<>())
        .def_readwrite("type", &kora::TaskMetadata::type)
        .def_readwrite("kernel_id", &kora::TaskMetadata::kernel_id)
        .def_readwrite("cpu_function_id", &kora::TaskMetadata::cpu_function_id)
        .def_readwrite("accesses", &kora::TaskMetadata::accesses)
        .def_readwrite("problem_dimensions", &kora::TaskMetadata::problem_dimensions)
        .def_readwrite("priority", &kora::TaskMetadata::priority)
        .def_readwrite("deadline_hint_ns", &kora::TaskMetadata::deadline_hint_ns)
        .def_readwrite("allowed_devices", &kora::TaskMetadata::allowed_devices)
        .def_readwrite("preferred_device", &kora::TaskMetadata::preferred_device)
        .def_readwrite("cpu_fallback_available", &kora::TaskMetadata::cpu_fallback_available)
        .def_readwrite("estimated_bytes_read", &kora::TaskMetadata::estimated_bytes_read)
        .def_readwrite("estimated_bytes_written", &kora::TaskMetadata::estimated_bytes_written)
        .def_readwrite("estimated_work_units", &kora::TaskMetadata::estimated_work_units);
    py::class_<kora::TaskNode>(module, "TaskNode")
        .def_readonly("id", &kora::TaskNode::id)
        .def_readonly("metadata", &kora::TaskNode::metadata)
        .def_readonly("dependencies", &kora::TaskNode::dependencies);
    py::class_<kora::TaskGraphPlan>(module, "TaskGraphPlan")
        .def_readonly("tasks", &kora::TaskGraphPlan::tasks)
        .def_readonly("topological_order", &kora::TaskGraphPlan::topological_order);
    py::class_<kora::TaskGraph>(module, "TaskGraph")
        .def(py::init<>())
        .def("add_task",
             [](kora::TaskGraph& graph, kora::TaskMetadata metadata) {
                 return value_or_throw(graph.add_task(std::move(metadata)));
             })
        .def("add_dependency",
             [](kora::TaskGraph& graph, kora::TaskId prerequisite, kora::TaskId dependent) {
                 status_or_throw(graph.add_dependency(prerequisite, dependent));
             })
        .def("finalize", [](const kora::TaskGraph& graph) { return value_or_throw(graph.finalize()); })
        .def_property_readonly("size", &kora::TaskGraph::size)
        .def_property_readonly("empty", &kora::TaskGraph::empty);
    py::class_<PyExecutionPlan>(module, "ExecutionPlan")
        .def_static("create",
                    [](const kora::TaskGraph& graph) {
                        return PyExecutionPlan{value_or_throw(kora::ExecutionPlan::create(graph))};
                    })
        .def_property_readonly("size", [](const PyExecutionPlan& plan) { return plan.value.size(); })
        .def_property_readonly(
            "graph_plan",
            [](const PyExecutionPlan& plan) -> const kora::TaskGraphPlan& { return plan.value.graph_plan(); },
            py::return_value_policy::reference_internal)
        .def(
            "replay",
            [](const PyExecutionPlan& plan, py::function callback) {
                PythonTaskExecutor executor{std::move(callback)};
                status_or_throw(plan.value.replay(&PythonTaskExecutor::invoke, &executor));
            },
            py::arg("executor"));

    py::class_<kora::RuntimeOptions>(module, "RuntimeOptions")
        .def(py::init<>())
        .def_readwrite("device_policy", &kora::RuntimeOptions::device_policy)
        .def_readwrite("device_index", &kora::RuntimeOptions::device_index)
        .def_readwrite("device_uuid", &kora::RuntimeOptions::device_uuid)
        .def_readwrite("cache_dir", &kora::RuntimeOptions::cache_dir)
        .def_readwrite("disable_autotune", &kora::RuntimeOptions::disable_autotune)
        .def_readwrite("force_legacy_sync", &kora::RuntimeOptions::force_legacy_sync);

    py::class_<kora::KernelBindingMetadata>(module, "KernelBindingMetadata")
        .def(py::init<>())
        .def(py::init<std::uint32_t, std::uint32_t, std::string, kora::KernelBindingKind,
                      kora::KernelAccess>(),
             py::arg("set"), py::arg("binding"), py::arg("name"), py::arg("kind"), py::arg("access"))
        .def_readwrite("set", &kora::KernelBindingMetadata::set)
        .def_readwrite("binding", &kora::KernelBindingMetadata::binding)
        .def_readwrite("name", &kora::KernelBindingMetadata::name)
        .def_readwrite("kind", &kora::KernelBindingMetadata::kind)
        .def_readwrite("access", &kora::KernelBindingMetadata::access);
    py::class_<kora::PushConstantFieldMetadata>(module, "PushConstantFieldMetadata")
        .def(py::init<>())
        .def(py::init<std::string, std::string, std::uint32_t, std::uint32_t>(), py::arg("name"),
             py::arg("type"), py::arg("offset"), py::arg("size"))
        .def_readwrite("name", &kora::PushConstantFieldMetadata::name)
        .def_readwrite("type", &kora::PushConstantFieldMetadata::type)
        .def_readwrite("offset", &kora::PushConstantFieldMetadata::offset)
        .def_readwrite("size", &kora::PushConstantFieldMetadata::size);
    py::class_<kora::SpecializationConstantMetadata>(module, "SpecializationConstantMetadata")
        .def(py::init<>())
        .def(py::init<std::uint32_t, std::string, std::string, kora::SpecializationValue>(), py::arg("id"),
             py::arg("name"), py::arg("type"), py::arg("default_value"))
        .def_readwrite("id", &kora::SpecializationConstantMetadata::id)
        .def_readwrite("name", &kora::SpecializationConstantMetadata::name)
        .def_readwrite("type", &kora::SpecializationConstantMetadata::type)
        .def_readwrite("default_value", &kora::SpecializationConstantMetadata::default_value);
    py::class_<kora::KernelMetadata>(module, "KernelMetadata")
        .def(py::init<>())
        .def_readwrite("schema_version", &kora::KernelMetadata::schema_version)
        .def_readwrite("kernel_id", &kora::KernelMetadata::kernel_id)
        .def_readwrite("bindings", &kora::KernelMetadata::bindings)
        .def_readwrite("push_constant_size", &kora::KernelMetadata::push_constant_size)
        .def_readwrite("push_constants", &kora::KernelMetadata::push_constants)
        .def_readwrite("specialization_constants", &kora::KernelMetadata::specialization_constants)
        .def_readwrite("local_size", &kora::KernelMetadata::local_size)
        .def_readwrite("cpu_fallback_key", &kora::KernelMetadata::cpu_fallback_key)
        .def_readwrite("semantics", &kora::KernelMetadata::semantics);

    py::class_<PyMappedView, std::shared_ptr<PyMappedView>>(module, "MappedView", py::buffer_protocol())
        .def_buffer([](PyMappedView& view) {
            const auto size = static_cast<py::ssize_t>(view.mapping.size());
            return py::buffer_info(view.mapping.data(), 1, py::format_descriptor<unsigned char>::format(), 1,
                                   {size}, {1}, view.readonly);
        })
        .def_property_readonly("readonly", [](const PyMappedView& view) { return view.readonly; })
        .def("__len__", [](const PyMappedView& view) { return view.mapping.size(); });
    py::class_<PyBuffer, std::shared_ptr<PyBuffer>>(module, "Buffer")
        .def_property_readonly("size_bytes",
                               [](const PyBuffer& buffer) { return buffer.handle.size_bytes(); })
        .def("__len__", [](const PyBuffer& buffer) { return buffer.handle.size_bytes(); })
        .def("read", &PyBuffer::read, "Copy the current host-shadow contents to Python bytes.")
        .def("write", &PyBuffer::write, py::arg("data"),
             "Copy a contiguous Python buffer exporter into the KORA host shadow.")
        .def("map", &PyBuffer::map, py::arg("access") = kora::MapAccess::ReadWrite,
             "Create a pinned mapped view; memoryview(view) exposes writable bytes unless access is READ.");
    py::class_<PyKernel, std::shared_ptr<PyKernel>>(module, "Kernel")
        .def_property_readonly("entry_point",
                               [](const PyKernel& kernel) { return kernel.value.entry_point(); })
        .def_property_readonly(
            "metadata",
            [](const PyKernel& kernel) -> const kora::KernelMetadata& { return kernel.value.metadata(); },
            py::return_value_policy::reference_internal);
    py::class_<PyEvent, std::shared_ptr<PyEvent>>(module, "Event")
        .def_property_readonly("done", [](const PyEvent& event) { return event.value.done(); })
        .def("wait", &PyEvent::wait, "Block until completion, releasing the Python GIL while waiting.")
        .def("wait_for", &PyEvent::wait_for, py::arg("timeout_seconds"),
             "Wait up to a timeout without cancelling pending work.")
        .def("_register_awaitable", &PyEvent::register_awaitable);
    py::class_<AwaitGuard, std::shared_ptr<AwaitGuard>>(module, "_EventAwaitGuard");
    py::class_<PyDispatchBuilder>(module, "DispatchBuilder")
        .def(
            "global",
            [](PyDispatchBuilder& builder, std::uint64_t x) -> PyDispatchBuilder& {
                builder.value.global(x);
                return builder;
            },
            py::return_value_policy::reference_internal)
        .def(
            "global_size",
            [](PyDispatchBuilder& builder, std::uint64_t x, std::uint64_t y,
               std::uint64_t z) -> PyDispatchBuilder& {
                builder.value.global(x, y, z);
                return builder;
            },
            py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference_internal)
        .def(
            "global_size",
            [](PyDispatchBuilder& builder, std::uint64_t x) -> PyDispatchBuilder& {
                builder.value.global(x);
                return builder;
            },
            py::arg("x"), py::return_value_policy::reference_internal)
        .def(
            "global",
            [](PyDispatchBuilder& builder, std::uint64_t x, std::uint64_t y,
               std::uint64_t z) -> PyDispatchBuilder& {
                builder.value.global(x, y, z);
                return builder;
            },
            py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference_internal)
        .def("read", &PyDispatchBuilder::read, py::return_value_policy::reference_internal)
        .def("write", &PyDispatchBuilder::write, py::return_value_policy::reference_internal)
        .def("read_write", &PyDispatchBuilder::read_write, py::return_value_policy::reference_internal)
        .def("push", &PyDispatchBuilder::push, py::return_value_policy::reference_internal)
        .def(
            "defer_readback",
            [](PyDispatchBuilder& builder) -> PyDispatchBuilder& {
                builder.value.defer_readback();
                return builder;
            },
            py::return_value_policy::reference_internal)
        .def("submit", &PyDispatchBuilder::submit);
    py::class_<PyRuntime, std::shared_ptr<PyRuntime>>(module, "Runtime")
        .def_static("create", &create_runtime, py::arg("options") = kora::RuntimeOptions{},
                    "Create the native runtime, releasing the Python GIL during initialization.")
        .def_property_readonly("devices", [](const PyRuntime& runtime) { return runtime.value->devices(); })
        .def("create_buffer", &create_buffer, py::arg("size_bytes"),
             py::arg("usage") = kora::BufferUsage::Storage)
        .def("load_kernel", &load_kernel, py::arg("spirv_path"), py::arg("entry_point"), py::arg("metadata"))
        .def("load_kernel", &load_kernel_for_device, py::arg("spirv_path"), py::arg("entry_point"),
             py::arg("metadata"), py::arg("device"))
        .def("dispatch",
             [](const std::shared_ptr<PyRuntime>& runtime, const PyKernel& kernel) {
                 if (kernel.owner != runtime)
                     throw PythonKoraError(kora::Error(kora::ErrorCode::InvalidArgument,
                                                       "kernel belongs to a different Runtime"));
                 return PyDispatchBuilder(runtime, kernel.value);
             })
        .def("readback",
             [](const std::shared_ptr<PyRuntime>& runtime, const std::shared_ptr<PyBuffer>& buffer) {
                 if (buffer->owner != runtime)
                     throw PythonKoraError(kora::Error(kora::ErrorCode::InvalidArgument,
                                                       "buffer belongs to a different Runtime"));
                 std::optional<kora::Result<kora::Event>> result;
                 {
                     py::gil_scoped_release release;
                     result.emplace(runtime->value->readback(buffer->handle));
                 }
                 auto event = value_or_throw(std::move(*result));
                 return std::make_shared<PyEvent>(PyEvent{std::move(event), runtime, {buffer}});
             })
        .def(
            "readback",
            [](const std::shared_ptr<PyRuntime>& runtime, const std::shared_ptr<PyBuffer>& buffer,
               const kora::DeviceId& device) {
                if (buffer->owner != runtime)
                    throw PythonKoraError(kora::Error(kora::ErrorCode::InvalidArgument,
                                                      "buffer belongs to a different Runtime"));
                std::optional<kora::Result<kora::Event>> result;
                {
                    py::gil_scoped_release release;
                    result.emplace(runtime->value->readback(buffer->handle, device));
                }
                auto event = value_or_throw(std::move(*result));
                return std::make_shared<PyEvent>(PyEvent{std::move(event), runtime, {buffer}});
            },
            py::arg("buffer"), py::arg("device"));

    auto scope = module.attr("__dict__");
    scope["asyncio"] = py::module_::import("asyncio");
    scope["KoraError"] = error_type;
    py::exec(R"(
def _resolve_event_future(future, value, error):
    if future.done():
        return
    if error is None:
        future.set_result(value)
    else:
        future.set_exception(error)

async def _await_event(event):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    guard = event._register_awaitable(loop, future, _resolve_event_future, KoraError)
    try:
        return await future
    finally:
        del guard

def _event_await(event):
    return _await_event(event).__await__()
)",
             scope);
    module.attr("Event").attr("__await__") = scope["_event_await"];
}
