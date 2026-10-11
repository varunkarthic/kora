#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "kora/errors.hpp"

namespace kora::detail::json {

struct Value {
    using Array = std::vector<Value>;
    using Object = std::vector<std::pair<std::string, Value>>;
    struct Number {
        std::string text;
    };
    using Storage = std::variant<std::nullptr_t, bool, Number, std::string, Array, Object>;

    Storage storage = nullptr;
    const Value* find(const std::string& key) const noexcept;
};

struct Limits {
    std::size_t maximum_bytes = 1024U * 1024U;
    std::size_t maximum_depth = 32;
    std::size_t maximum_string_bytes = 64U * 1024U;
    std::size_t maximum_values = 100'000;
};

Result<Value> parse(std::string_view input, Limits limits = {});

}  // namespace kora::detail::json
