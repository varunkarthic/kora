#pragma once

#include <string>

#include "kora/kernel_metadata.hpp"

namespace kora::detail {

/// Writes `metadata` in the documented kernel-metadata JSON format accepted by parse_kernel_metadata().
std::string serialize_kernel_metadata(const KernelMetadata& metadata);

}  // namespace kora::detail
