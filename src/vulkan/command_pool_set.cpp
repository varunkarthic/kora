#include "vulkan/command_pool_set.hpp"

#include <limits>
#include <utility>

#include "kora/detail/assert.hpp"

namespace kora::vk {

CommandPoolSet::CommandPoolSet(Api api, VkDevice device, const Options& options)
    : api_(std::move(api)),
      device_(device),
      owner_(std::this_thread::get_id()),
      debug_thread_checks_(options.debug_thread_checks) {
    if (device_ == VK_NULL_HANDLE || options.ring_size == 0 ||
        options.ring_size > std::numeric_limits<uint32_t>::max() || !api_.create_command_pool ||
        !api_.destroy_command_pool || !api_.reset_command_pool || !api_.allocate_command_buffers ||
        !api_.begin_command_buffer || !api_.end_command_buffer || !api_.reset_command_buffer) {
        return;
    }
    VkCommandPoolCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    create_info.flags =
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    create_info.queueFamilyIndex = options.queue_family;
    if (api_.create_command_pool(device_, &create_info, nullptr, &pool_) != VK_SUCCESS) {
        pool_ = VK_NULL_HANDLE;
        return;
    }
    slots_.resize(options.ring_size);
    std::vector<VkCommandBuffer> buffers(options.ring_size, VK_NULL_HANDLE);
    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.commandPool = pool_;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = static_cast<uint32_t>(options.ring_size);
    if (api_.allocate_command_buffers(device_, &allocate_info, buffers.data()) != VK_SUCCESS) {
        api_.destroy_command_pool(device_, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
        slots_.clear();
        return;
    }
    for (size_t i = 0; i < slots_.size(); ++i) slots_[i].buffer = buffers[i];
    valid_ = true;
}

CommandPoolSet::~CommandPoolSet() {
    if (pool_ != VK_NULL_HANDLE) api_.destroy_command_pool(device_, pool_, nullptr);
}
CommandPoolSet::CommandPoolSet(CommandPoolSet&& other) noexcept {
    *this = std::move(other);
}
CommandPoolSet& CommandPoolSet::operator=(CommandPoolSet&& other) noexcept {
    if (this == &other) return *this;
    if (pool_ != VK_NULL_HANDLE) api_.destroy_command_pool(device_, pool_, nullptr);
    api_ = std::move(other.api_);
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    pool_ = std::exchange(other.pool_, VK_NULL_HANDLE);
    owner_ = other.owner_;
    debug_thread_checks_ = other.debug_thread_checks_;
    cursor_ = std::exchange(other.cursor_, 0);
    last_signal_value_ = std::exchange(other.last_signal_value_, 0);
    completed_batch_value_ = std::exchange(other.completed_batch_value_, 0);
    slots_ = std::move(other.slots_);
    valid_ = std::exchange(other.valid_, false);
    return *this;
}

bool CommandPoolSet::on_owner_thread() const noexcept {
    return owner_ == std::this_thread::get_id();
}
bool CommandPoolSet::valid() const noexcept {
    return valid_;
}

Status CommandPoolSet::check_thread() const {
    if (debug_thread_checks_ && !on_owner_thread()) {
        KORA_ASSERT(false && "CommandPoolSet used from a non-owning thread");
        return Error(ErrorCode::InternalInvariantViolation, "CommandPoolSet used from a non-owning thread");
    }
    return ok_status();
}

Result<VkCommandBuffer> CommandPoolSet::acquire(uint64_t completed_value) {
    KORA_TRY_STATUS(check_thread());
    if (pool_ == VK_NULL_HANDLE || !valid_)
        return Error(ErrorCode::NoVulkanImplementation, "Command pool is unavailable");
    bool has_submitted = false;
    bool all_complete = true;
    for (const auto& slot : slots_) {
        has_submitted = has_submitted || slot.completion_value != 0;
        all_complete = all_complete && !slot.acquired && slot.completion_value <= completed_value;
    }
    if (has_submitted && all_complete && completed_value > completed_batch_value_) {
        if (api_.reset_command_pool(device_, pool_, 0) != VK_SUCCESS) {
            return Error(ErrorCode::NoVulkanImplementation,
                         "Failed to batch-reset the completed command pool");
        }
        for (auto& slot : slots_) slot.completion_value = 0;
        completed_batch_value_ = completed_value;
    }
    for (size_t offset = 0; offset < slots_.size(); ++offset) {
        const size_t index = (cursor_ + offset) % slots_.size();
        auto& slot = slots_[index];
        if (slot.acquired || slot.completion_value != 0) continue;
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (api_.begin_command_buffer(slot.buffer, &begin_info) != VK_SUCCESS) {
            return Error(ErrorCode::NoVulkanImplementation, "Failed to begin a recycled command buffer");
        }
        slot.acquired = true;
        slot.ended = false;
        cursor_ = (index + 1) % slots_.size();
        return slot.buffer;
    }
    return Error(ErrorCode::OutOfMemoryBudget,
                 "Command buffer ring is full; submitted work has not completed");
}

Status CommandPoolSet::end_recording(VkCommandBuffer buffer) {
    KORA_TRY_STATUS(check_thread());
    for (auto& slot : slots_) {
        if (slot.buffer != buffer || !slot.acquired || slot.ended) continue;
        if (api_.end_command_buffer(buffer) != VK_SUCCESS) {
            slot.acquired = false;
            return Error(ErrorCode::NoVulkanImplementation, "Failed to end a command buffer");
        }
        slot.ended = true;
        return ok_status();
    }
    return Error(ErrorCode::InvalidArgument, "Command buffer was not recording in this pool set");
}

Status CommandPoolSet::discard(VkCommandBuffer buffer) {
    KORA_TRY_STATUS(check_thread());
    for (auto& slot : slots_) {
        if (slot.buffer != buffer || !slot.acquired) continue;
        const auto result = api_.reset_command_buffer(buffer, 0);
        if (result != VK_SUCCESS)
            return Error(ErrorCode::NoVulkanImplementation, "Failed to reset a discarded command buffer")
                .with_vk(result);
        slot.acquired = false;
        slot.ended = false;
        slot.completion_value = 0;
        return ok_status();
    }
    return Error(ErrorCode::InvalidArgument, "Command buffer was not acquired from this pool set");
}

Status CommandPoolSet::submitted(VkCommandBuffer buffer, uint64_t signal_value) {
    KORA_TRY_STATUS(check_thread());
    if (signal_value == 0 || signal_value < last_signal_value_) {
        return Error(ErrorCode::InvalidArgument,
                     "Submission timeline values must be nonzero and nondecreasing");
    }
    for (auto& slot : slots_) {
        if (slot.buffer != buffer || !slot.acquired || !slot.ended) continue;
        slot.completion_value = signal_value;
        slot.acquired = false;
        slot.ended = false;
        last_signal_value_ = signal_value;
        return ok_status();
    }
    return Error(ErrorCode::InvalidArgument, "Command buffer was not acquired from this pool set");
}

size_t CommandPoolSet::capacity() const noexcept {
    return slots_.size();
}
size_t CommandPoolSet::allocation_count() const noexcept {
    return slots_.size();
}

uint64_t CommandPoolSet::oldest_pending_value() const noexcept {
    uint64_t oldest = 0;
    for (const auto& slot : slots_) {
        if (slot.completion_value != 0 && (oldest == 0 || slot.completion_value < oldest))
            oldest = slot.completion_value;
    }
    return oldest;
}

}  // namespace kora::vk
