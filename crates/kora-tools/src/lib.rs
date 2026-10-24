//! Shared helpers for the KORA command-line tools.

use kora::json::Value;

/// Builds a JSON object from `(name, value)` pairs, preserving their order.
pub fn json_object(members: Vec<(&str, Value)>) -> Value {
    Value::Object(members.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}
