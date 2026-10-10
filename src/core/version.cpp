#include "kora/version.hpp"

namespace kora {
std::string version_string() {
    return std::to_string(KORA_VERSION_MAJOR) + "." + std::to_string(KORA_VERSION_MINOR) + "." +
           std::to_string(KORA_VERSION_PATCH);
}
}  // namespace kora
