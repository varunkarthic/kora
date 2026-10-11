#pragma once
/// @file assert.hpp
/// Internal invariant assertions. Expressions must have no side effects; in Release they are not evaluated.

#include <cassert>

/// Check an internal programming invariant in assertion-enabled builds.
#define KORA_ASSERT(expression) assert(expression)
