#include "core/json.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <string_view>

namespace kora::detail::json {
namespace {

bool parse_finite_double(std::string_view token, double& value) {
    std::istringstream stream{std::string(token)};
    stream.imbue(std::locale::classic());
    stream >> std::noskipws >> value;
    if (stream.fail() || stream.peek() != std::char_traits<char>::eof() || !std::isfinite(value))
        return false;
    const auto exponent = token.find_first_of("eE");
    const auto mantissa = token.substr(0, exponent);
    const bool nonzero_mantissa =
        std::any_of(mantissa.begin(), mantissa.end(), [](char c) { return c >= '1' && c <= '9'; });
    return value != 0 || !nonzero_mantissa;
}

class Parser {
public:
    Parser(std::string_view input, Limits limits) : input_(input), limits_(limits) {}

    Result<Value> run() {
        if (input_.size() > limits_.maximum_bytes) return error("document exceeds the configured byte limit");
        skip_space();
        auto value = parse_value(0);
        if (!value) return std::move(value).error();
        skip_space();
        if (position_ != input_.size()) return error("trailing data after the JSON value");
        return value;
    }

private:
    Error error(std::string message) const {
        return Error(ErrorCode::InvalidKernel,
                     "JSON byte " + std::to_string(position_) + ": " + std::move(message));
    }

    void skip_space() {
        while (position_ < input_.size() && (input_[position_] == ' ' || input_[position_] == '\t' ||
                                             input_[position_] == '\r' || input_[position_] == '\n'))
            ++position_;
    }

    Result<Value> parse_value(std::size_t depth) {
        if (depth > limits_.maximum_depth) return error("maximum nesting depth exceeded");
        if (++values_ > limits_.maximum_values) return error("maximum value count exceeded");
        skip_space();
        if (position_ >= input_.size()) return error("unexpected end of input");
        const char c = input_[position_];
        if (c == '{') return parse_object(depth + 1);
        if (c == '[') return parse_array(depth + 1);
        if (c == '"') {
            auto string = parse_string();
            if (!string) return std::move(string).error();
            return Value{std::move(string).value()};
        }
        if (c == 't' && consume("true")) return Value{true};
        if (c == 'f' && consume("false")) return Value{false};
        if (c == 'n' && consume("null")) return Value{nullptr};
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        return error("expected a JSON value");
    }

    Result<Value> parse_object(std::size_t depth) {
        ++position_;
        Value::Object object;
        skip_space();
        if (consume("}")) return Value{std::move(object)};
        for (;;) {
            skip_space();
            if (position_ >= input_.size() || input_[position_] != '"')
                return error("object member name must be a string");
            auto key = parse_string();
            if (!key) return std::move(key).error();
            for (const auto& member : object)
                if (member.first == key.value())
                    return error("duplicate object member '" + key.value() + "'");
            skip_space();
            if (!consume(":")) return error("expected ':' after object member name");
            auto value = parse_value(depth);
            if (!value) return std::move(value).error();
            object.emplace_back(std::move(key).value(), std::move(value).value());
            skip_space();
            if (consume("}")) return Value{std::move(object)};
            if (!consume(",")) return error("expected ',' or '}' in object");
        }
    }

    Result<Value> parse_array(std::size_t depth) {
        ++position_;
        Value::Array array;
        skip_space();
        if (consume("]")) return Value{std::move(array)};
        for (;;) {
            auto value = parse_value(depth);
            if (!value) return std::move(value).error();
            array.push_back(std::move(value).value());
            skip_space();
            if (consume("]")) return Value{std::move(array)};
            if (!consume(",")) return error("expected ',' or ']' in array");
        }
    }

    Result<std::string> parse_string() {
        if (!consume("\"")) return error("expected string");
        std::string result;
        while (position_ < input_.size()) {
            const unsigned char c = static_cast<unsigned char>(input_[position_++]);
            if (c == '"') {
                if (result.size() > limits_.maximum_string_bytes) return error("string exceeds byte limit");
                return result;
            }
            if (c < 0x20) return error("unescaped control byte in string");
            if (c != '\\') {
                if (c >= 0x80) {
                    const std::size_t continuation_count = c >= 0xc2 && c <= 0xdf   ? 1
                                                           : c >= 0xe0 && c <= 0xef ? 2
                                                           : c >= 0xf0 && c <= 0xf4 ? 3
                                                                                    : 99;
                    if (continuation_count == 99 || input_.size() - position_ < continuation_count)
                        return error("invalid UTF-8 sequence in string");
                    const auto first_continuation = static_cast<unsigned char>(input_[position_]);
                    if ((c == 0xe0 && first_continuation < 0xa0) ||
                        (c == 0xed && first_continuation > 0x9f) ||
                        (c == 0xf0 && first_continuation < 0x90) || (c == 0xf4 && first_continuation > 0x8f))
                        return error("invalid UTF-8 sequence in string");
                    for (std::size_t i = 0; i < continuation_count; ++i) {
                        const auto continuation = static_cast<unsigned char>(input_[position_ + i]);
                        if (continuation < 0x80 || continuation > 0xbf)
                            return error("invalid UTF-8 sequence in string");
                        result.push_back(input_[position_ + i]);
                    }
                    position_ += continuation_count;
                    if (result.size() > limits_.maximum_string_bytes)
                        return error("string exceeds byte limit");
                    continue;
                }
                result.push_back(static_cast<char>(c));
            } else {
                if (position_ >= input_.size()) return error("incomplete escape sequence");
                const char escape = input_[position_++];
                switch (escape) {
                    case '"':
                        result.push_back('"');
                        break;
                    case '\\':
                        result.push_back('\\');
                        break;
                    case '/':
                        result.push_back('/');
                        break;
                    case 'b':
                        result.push_back('\b');
                        break;
                    case 'f':
                        result.push_back('\f');
                        break;
                    case 'n':
                        result.push_back('\n');
                        break;
                    case 'r':
                        result.push_back('\r');
                        break;
                    case 't':
                        result.push_back('\t');
                        break;
                    case 'u': {
                        auto first = parse_hex_quad();
                        if (!first) return std::move(first).error();
                        std::uint32_t codepoint = first.value();
                        if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                            if (input_.size() - position_ < 6 || input_[position_] != '\\' ||
                                input_[position_ + 1] != 'u')
                                return error("high surrogate is missing a low surrogate");
                            position_ += 2;
                            auto second = parse_hex_quad();
                            if (!second) return std::move(second).error();
                            if (second.value() < 0xdc00 || second.value() > 0xdfff)
                                return error("invalid low surrogate");
                            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (second.value() - 0xdc00);
                        } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                            return error("unexpected low surrogate");
                        }
                        append_utf8(result, codepoint);
                        break;
                    }
                    default:
                        return error("invalid escape sequence");
                }
            }
            if (result.size() > limits_.maximum_string_bytes) return error("string exceeds byte limit");
        }
        return error("unterminated string");
    }

    Result<std::uint32_t> parse_hex_quad() {
        if (input_.size() - position_ < 4) return error("incomplete Unicode escape");
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = input_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9')
                value |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f')
                value |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                value |= static_cast<std::uint32_t>(c - 'A' + 10);
            else
                return error("invalid hexadecimal digit in Unicode escape");
        }
        return value;
    }

    static void append_utf8(std::string& output, std::uint32_t codepoint) {
        if (codepoint <= 0x7f)
            output.push_back(static_cast<char>(codepoint));
        else if (codepoint <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    Result<Value> parse_number() {
        const auto start = position_;
        if (input_[position_] == '-') ++position_;
        if (position_ >= input_.size()) return error("incomplete number");
        if (input_[position_] == '0') {
            ++position_;
            if (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9')
                return error("leading zero in number");
        } else {
            if (input_[position_] < '1' || input_[position_] > '9') return error("invalid number");
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9')
                ++position_;
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const auto fraction = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9')
                ++position_;
            if (fraction == position_) return error("fraction requires at least one digit");
        }
        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-'))
                ++position_;
            const auto exponent = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9')
                ++position_;
            if (exponent == position_) return error("exponent requires at least one digit");
        }
        const auto token = input_.substr(start, position_ - start);
        if (token.find_first_of(".eE") != std::string_view::npos) {
            double number = 0;
            if (!parse_finite_double(token, number))
                return error("number is outside the supported finite range");
        }
        return Value{Value::Number{std::string(token)}};
    }

    bool consume(std::string_view token) {
        if (input_.substr(position_, token.size()) != token) return false;
        position_ += token.size();
        return true;
    }

    std::string_view input_;
    Limits limits_;
    std::size_t position_ = 0;
    std::size_t values_ = 0;
};

}  // namespace

const Value* Value::find(const std::string& key) const noexcept {
    const auto* object = std::get_if<Object>(&storage);
    if (!object) return nullptr;
    for (const auto& member : *object)
        if (member.first == key) return &member.second;
    return nullptr;
}

Result<Value> parse(std::string_view input, Limits limits) {
    if (limits.maximum_bytes == 0 || limits.maximum_depth == 0 || limits.maximum_values == 0 ||
        limits.maximum_string_bytes == 0)
        return Error(ErrorCode::InvalidKernel, "JSON parser limits must be nonzero");
    return Parser(input, limits).run();
}

}  // namespace kora::detail::json
