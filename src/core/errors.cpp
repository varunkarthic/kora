#include "kora/errors.hpp"

namespace kora {

const char* to_string(ErrorCode c) noexcept {
    switch (c) {
#define K(x)           \
    case ErrorCode::x: \
        return #x;
        K(Ok)
        K(NoVulkanImplementation)
        K(NoEligibleDevice)
        K(NoComputeQueue)
        K(UnsupportedFeature)
        K(InvalidKernel)
        K(InvalidKernelBinding)
        K(InvalidDispatch)
        K(AllocationFailed)
        K(OutOfMemoryBudget)
        K(TransferFailed)
        K(Timeout)
        K(DeviceLost)
        K(PipelineCreationFailed)
        K(SynchronizationError)
        K(ValidationError)
        K(PythonBindingError)
        K(InternalInvariantViolation)
        K(InvalidArgument)
        K(Overflow)
        K(GraphCycle)
        K(Cancelled)
        K(StaleHandle)
        K(DataLoss)
        K(IoError)
#undef K
        case ErrorCode::kCount_:
            break;
    }
    return "Unknown";
}

std::string Error::describe() const {
    std::string s = to_string(code);
    s += ": " + message;
    if (vk_result != 0) s += " [vk=" + std::to_string(vk_result) + "]";
    if (!device.empty()) s += " [device=" + device + "]";
    if (task_id != 0) s += " [task=" + std::to_string(task_id) + "]";
    if (!kernel_id.empty()) s += " [kernel=" + kernel_id + "]";
    if (!recovery_action.empty()) s += " [recovery=" + recovery_action + "]";
    if (cause) s += " <- " + cause->describe();
    return s;
}

}  // namespace kora
