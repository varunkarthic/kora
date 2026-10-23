//! Bounded, strict JSON.
//!
//! Kernel manifests, persisted caches, and status streams are untrusted input, so the parser rejects
//! duplicate object members, invalid UTF-8, unpaired surrogates, leading zeros, non-finite numbers,
//! and trailing data, and enforces byte, depth, string, and value-count limits. Numbers keep their
//! exact source text so 64-bit integers never lose precision. Errors report the byte offset.

use std::fmt::{self, Write as _};

use crate::error::{Error, ErrorCode, Result};

/// A parsed JSON value. Objects preserve member order.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number, stored as its validated source text.
    Number(String),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Value>),
    /// An object with unique member names, in source order.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// Member lookup on an object; `None` for other kinds or a missing key.
    pub fn find(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(members) => members
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }
    /// Borrowed object members.
    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(members) => Some(members),
            _ => None,
        }
    }
    /// Borrowed array elements.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }
    /// Borrowed string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(text) => Some(text),
            _ => None,
        }
    }
    /// Boolean value.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(value) => Some(*value),
            _ => None,
        }
    }
    /// Unsigned integer value (no sign, fraction, or exponent; within `u64`).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(text) if !text.contains(['-', '.', 'e', 'E']) => text.parse().ok(),
            _ => None,
        }
    }
    /// Signed integer value (no fraction or exponent; within `i64`).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Number(text) if !text.contains(['.', 'e', 'E']) => text.parse().ok(),
            _ => None,
        }
    }
    /// Finite floating-point value of any number.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(text) => text.parse::<f64>().ok().filter(|v| v.is_finite()),
            _ => None,
        }
    }
    /// A number value from an unsigned integer.
    pub fn uint(value: u64) -> Value {
        Value::Number(value.to_string())
    }
    /// A number value from a signed integer.
    pub fn int(value: i64) -> Value {
        Value::Number(value.to_string())
    }
    /// A number value from a finite float; non-finite values become `null`.
    pub fn float(value: f64) -> Value {
        if value.is_finite() {
            let mut text = format!("{value:?}");
            if !text.contains(['.', 'e', 'E']) {
                text.push_str(".0");
            }
            Value::Number(text)
        } else {
            Value::Null
        }
    }
    /// A string value.
    pub fn string(value: impl Into<String>) -> Value {
        Value::String(value.into())
    }
}

impl fmt::Display for Value {
    /// Compact JSON serialization.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = String::new();
        write_value(&mut out, self);
        f.write_str(&out)
    }
}

/// Appends `value` as compact JSON.
pub fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(text) => out.push_str(text),
        Value::String(text) => write_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(members) => {
            out.push('{');
            for (index, (name, item)) in members.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_string(out, name);
                out.push(':');
                write_value(out, item);
            }
            out.push('}');
        }
    }
}

/// Appends `text` as a JSON string literal, escaping quotes, backslashes, and control bytes.
pub fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parser resource limits. Every limit must be non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Maximum document size in bytes.
    pub maximum_bytes: usize,
    /// Maximum container nesting depth.
    pub maximum_depth: usize,
    /// Maximum decoded string size in bytes.
    pub maximum_string_bytes: usize,
    /// Maximum number of values in the document.
    pub maximum_values: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            maximum_bytes: 1024 * 1024,
            maximum_depth: 32,
            maximum_string_bytes: 64 * 1024,
            maximum_values: 100_000,
        }
    }
}

/// Parses `input` with default limits.
pub fn parse(input: &[u8]) -> Result<Value> {
    parse_with(input, Limits::default())
}

/// Parses `input` with explicit limits.
pub fn parse_with(input: &[u8], limits: Limits) -> Result<Value> {
    if limits.maximum_bytes == 0
        || limits.maximum_depth == 0
        || limits.maximum_values == 0
        || limits.maximum_string_bytes == 0
    {
        return Err(Error::new(
            ErrorCode::InvalidKernel,
            "JSON parser limits must be nonzero",
        ));
    }
    Parser {
        input,
        limits,
        position: 0,
        values: 0,
    }
    .run()
}

struct Parser<'a> {
    input: &'a [u8],
    limits: Limits,
    position: usize,
    values: usize,
}

impl Parser<'_> {
    fn run(mut self) -> Result<Value> {
        if self.input.len() > self.limits.maximum_bytes {
            return Err(self.error("document exceeds the configured byte limit"));
        }
        self.skip_space();
        let value = self.parse_value(0)?;
        self.skip_space();
        if self.position != self.input.len() {
            return Err(self.error("trailing data after the JSON value"));
        }
        Ok(value)
    }

    fn error(&self, message: &str) -> Error {
        Error::new(
            ErrorCode::InvalidKernel,
            format!("JSON byte {}: {message}", self.position),
        )
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.position).copied()
    }

    fn skip_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.position += 1;
        }
    }

    fn consume(&mut self, token: &[u8]) -> bool {
        if self.input[self.position..].starts_with(token) {
            self.position += token.len();
            true
        } else {
            false
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<Value> {
        if depth > self.limits.maximum_depth {
            return Err(self.error("maximum nesting depth exceeded"));
        }
        self.values += 1;
        if self.values > self.limits.maximum_values {
            return Err(self.error("maximum value count exceeded"));
        }
        self.skip_space();
        match self.peek() {
            None => Err(self.error("unexpected end of input")),
            Some(b'{') => self.parse_object(depth + 1),
            Some(b'[') => self.parse_array(depth + 1),
            Some(b'"') => self.parse_string().map(Value::String),
            Some(b't') if self.consume(b"true") => Ok(Value::Bool(true)),
            Some(b'f') if self.consume(b"false") => Ok(Value::Bool(false)),
            Some(b'n') if self.consume(b"null") => Ok(Value::Null),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(self.error("expected a JSON value")),
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<Value> {
        self.position += 1;
        let mut members: Vec<(String, Value)> = Vec::new();
        self.skip_space();
        if self.consume(b"}") {
            return Ok(Value::Object(members));
        }
        loop {
            self.skip_space();
            if self.peek() != Some(b'"') {
                return Err(self.error("object member name must be a string"));
            }
            let key = self.parse_string()?;
            if members.iter().any(|(name, _)| *name == key) {
                return Err(self.error(&format!("duplicate object member '{key}'")));
            }
            self.skip_space();
            if !self.consume(b":") {
                return Err(self.error("expected ':' after object member name"));
            }
            let value = self.parse_value(depth)?;
            members.push((key, value));
            self.skip_space();
            if self.consume(b"}") {
                return Ok(Value::Object(members));
            }
            if !self.consume(b",") {
                return Err(self.error("expected ',' or '}' in object"));
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<Value> {
        self.position += 1;
        let mut items = Vec::new();
        self.skip_space();
        if self.consume(b"]") {
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.parse_value(depth)?);
            self.skip_space();
            if self.consume(b"]") {
                return Ok(Value::Array(items));
            }
            if !self.consume(b",") {
                return Err(self.error("expected ',' or ']' in array"));
            }
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        if !self.consume(b"\"") {
            return Err(self.error("expected string"));
        }
        let mut result: Vec<u8> = Vec::new();
        while let Some(c) = self.peek() {
            self.position += 1;
            match c {
                b'"' => {
                    if result.len() > self.limits.maximum_string_bytes {
                        return Err(self.error("string exceeds byte limit"));
                    }
                    // Every byte appended was validated UTF-8 or an encoded scalar value.
                    return String::from_utf8(result)
                        .map_err(|_| self.error("invalid UTF-8 sequence in string"));
                }
                0x00..=0x1f => return Err(self.error("unescaped control byte in string")),
                b'\\' => self.parse_escape(&mut result)?,
                0x80..=0xff => self.parse_utf8_sequence(c, &mut result)?,
                _ => result.push(c),
            }
            if result.len() > self.limits.maximum_string_bytes {
                return Err(self.error("string exceeds byte limit"));
            }
        }
        Err(self.error("unterminated string"))
    }

    fn parse_utf8_sequence(&mut self, lead: u8, out: &mut Vec<u8>) -> Result<()> {
        let continuation_count = match lead {
            0xc2..=0xdf => 1,
            0xe0..=0xef => 2,
            0xf0..=0xf4 => 3,
            _ => return Err(self.error("invalid UTF-8 sequence in string")),
        };
        if self.input.len() - self.position < continuation_count {
            return Err(self.error("invalid UTF-8 sequence in string"));
        }
        let first = self.input[self.position];
        let overlong_or_surrogate = (lead == 0xe0 && first < 0xa0)
            || (lead == 0xed && first > 0x9f)
            || (lead == 0xf0 && first < 0x90)
            || (lead == 0xf4 && first > 0x8f);
        if overlong_or_surrogate {
            return Err(self.error("invalid UTF-8 sequence in string"));
        }
        out.push(lead);
        for offset in 0..continuation_count {
            let continuation = self.input[self.position + offset];
            if !(0x80..=0xbf).contains(&continuation) {
                return Err(self.error("invalid UTF-8 sequence in string"));
            }
            out.push(continuation);
        }
        self.position += continuation_count;
        Ok(())
    }

    fn parse_escape(&mut self, out: &mut Vec<u8>) -> Result<()> {
        let Some(escape) = self.peek() else {
            return Err(self.error("incomplete escape sequence"));
        };
        self.position += 1;
        let simple = match escape {
            b'"' => b'"',
            b'\\' => b'\\',
            b'/' => b'/',
            b'b' => 0x08,
            b'f' => 0x0c,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'u' => {
                let mut codepoint = self.parse_hex_quad()?;
                if (0xd800..=0xdbff).contains(&codepoint) {
                    if self.input.len() - self.position < 6
                        || !self.input[self.position..].starts_with(b"\\u")
                    {
                        return Err(self.error("high surrogate is missing a low surrogate"));
                    }
                    self.position += 2;
                    let low = self.parse_hex_quad()?;
                    if !(0xdc00..=0xdfff).contains(&low) {
                        return Err(self.error("invalid low surrogate"));
                    }
                    codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                } else if (0xdc00..=0xdfff).contains(&codepoint) {
                    return Err(self.error("unexpected low surrogate"));
                }
                let c = char::from_u32(codepoint).ok_or_else(|| self.error("invalid Unicode escape"))?;
                let mut buffer = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buffer).as_bytes());
                return Ok(());
            }
            _ => return Err(self.error("invalid escape sequence")),
        };
        out.push(simple);
        Ok(())
    }

    fn parse_hex_quad(&mut self) -> Result<u32> {
        if self.input.len() - self.position < 4 {
            return Err(self.error("incomplete Unicode escape"));
        }
        let mut value = 0u32;
        for _ in 0..4 {
            let digit = match self.input[self.position] {
                c @ b'0'..=b'9' => c - b'0',
                c @ b'a'..=b'f' => c - b'a' + 10,
                c @ b'A'..=b'F' => c - b'A' + 10,
                _ => {
                    self.position += 1;
                    return Err(self.error("invalid hexadecimal digit in Unicode escape"));
                }
            };
            self.position += 1;
            value = (value << 4) | u32::from(digit);
        }
        Ok(value)
    }

    fn skip_digits(&mut self) -> usize {
        let start = self.position;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.position += 1;
        }
        self.position - start
    }

    fn parse_number(&mut self) -> Result<Value> {
        let start = self.position;
        if self.peek() == Some(b'-') {
            self.position += 1;
        }
        match self.peek() {
            None => return Err(self.error("incomplete number")),
            Some(b'0') => {
                self.position += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error("leading zero in number"));
                }
            }
            Some(b'1'..=b'9') => {
                self.skip_digits();
            }
            Some(_) => return Err(self.error("invalid number")),
        }
        if self.peek() == Some(b'.') {
            self.position += 1;
            if self.skip_digits() == 0 {
                return Err(self.error("fraction requires at least one digit"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            if self.skip_digits() == 0 {
                return Err(self.error("exponent requires at least one digit"));
            }
        }
        // The scanned bytes are ASCII digits, signs, '.', and exponent markers.
        let token = std::str::from_utf8(&self.input[start..self.position])
            .unwrap_or_default()
            .to_owned();
        if token.contains(['.', 'e', 'E']) && parse_finite_double(&token).is_none() {
            return Err(self.error("number is outside the supported finite range"));
        }
        Ok(Value::Number(token))
    }
}

/// Parses a finite `f64`, rejecting overflow to infinity and underflow of a non-zero mantissa to zero.
pub(crate) fn parse_finite_double(token: &str) -> Option<f64> {
    let value: f64 = token.parse().ok()?;
    if !value.is_finite() {
        return None;
    }
    let mantissa = token.split(['e', 'E']).next().unwrap_or_default();
    let nonzero_mantissa = mantissa.bytes().any(|c| (b'1'..=b'9').contains(&c));
    if value == 0.0 && nonzero_mantissa {
        return None;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strict_values_and_preserves_integer_tokens() {
        let parsed =
            parse(br#"{"large":18446744073709551615,"unicode":"\ud83d\ude80","items":[true,null,-2.5e1]}"#)
                .unwrap();
        assert_eq!(
            parsed.find("large"),
            Some(&Value::Number("18446744073709551615".into()))
        );
        assert_eq!(parsed.find("large").unwrap().as_u64(), Some(u64::MAX));
        assert_eq!(parsed.find("unicode").unwrap().as_str(), Some("\u{1F680}"));
        let items = parsed.find("items").unwrap().as_array().unwrap();
        assert_eq!(items[0], Value::Bool(true));
        assert_eq!(items[1], Value::Null);
        assert_eq!(items[2].as_f64(), Some(-25.0));
    }

    #[test]
    fn accepts_finite_decimal_and_exponent_tokens_only() {
        let parsed = parse(b"[1.25,-2.5e1]").unwrap();
        let values = parsed.as_array().unwrap();
        assert_eq!(values[0], Value::Number("1.25".into()));
        assert_eq!(values[1], Value::Number("-2.5e1".into()));
        assert!(parse(b"1e9999").is_err());
        assert!(parse(b"1e-9999").is_err());
        assert!(parse(b"0e-9999").is_ok());
    }

    #[test]
    fn rejects_malformed_duplicate_and_truncated_input() {
        for input in [
            &br#"{"a":1,"a":2}"#[..],
            b"[1,",
            b"01",
            br#""\ud800""#,
            b"true false",
            br#""\udc00""#,
            br#""\ud800\u0041""#,
            br#""\x""#,
            b"-",
            b"1.",
            b"1e",
            b"+1",
            b"{1:2}",
            b"[1 2]",
            b"nul",
            b"\"\x01\"",
            b"",
            b"\"abc",
        ] {
            let error = parse(input).expect_err(&String::from_utf8_lossy(input));
            assert_eq!(error.code(), ErrorCode::InvalidKernel);
            assert!(error.message().contains("JSON byte"), "{}", error.message());
        }
    }

    #[test]
    fn rejects_invalid_raw_utf8() {
        for input in [
            &b"\"\xc0\x80\""[..],
            b"\"\xed\xa0\x80\"",
            b"\"\xe0\x80\x80\"",
            b"\"\xf4\x90\x80\x80\"",
            b"\"\xc3\"",
        ] {
            let error = parse(input).unwrap_err();
            assert!(error.message().contains("UTF-8"), "{}", error.message());
        }
        assert_eq!(parse("\"é漢😀\"".as_bytes()).unwrap().as_str(), Some("é漢😀"));
    }

    #[test]
    fn enforces_configured_limits() {
        let limited = |f: fn(&mut Limits)| {
            let mut limits = Limits::default();
            f(&mut limits);
            limits
        };
        assert!(parse_with(b"[]", limited(|l| l.maximum_bytes = 1)).is_err());
        assert!(parse_with(b"[[[]]]", limited(|l| l.maximum_depth = 1)).is_err());
        assert!(parse_with(b"\"abcd\"", limited(|l| l.maximum_string_bytes = 3)).is_err());
        assert!(parse_with(b"[1,2]", limited(|l| l.maximum_values = 2)).is_err());
        assert!(parse_with(b"[1,2]", limited(|l| l.maximum_values = 3)).is_ok());
        assert!(parse_with(b"[]", limited(|l| l.maximum_depth = 0)).is_err());
    }

    #[test]
    fn deep_nesting_is_rejected_without_stack_overflow() {
        let deep = "[".repeat(100_000);
        assert!(parse(deep.as_bytes()).is_err());
    }

    #[test]
    fn writer_round_trips_through_parser() {
        let value = Value::Object(vec![
            ("a".into(), Value::uint(u64::MAX)),
            ("b".into(), Value::string("quote\" slash\\ nl\n tab\t ctl\u{1}")),
            (
                "c".into(),
                Value::Array(vec![
                    Value::Bool(false),
                    Value::Null,
                    Value::float(1.5),
                    Value::int(-3),
                ]),
            ),
            ("d".into(), Value::float(2.0)),
        ]);
        let text = value.to_string();
        assert_eq!(parse(text.as_bytes()).unwrap(), value);
        assert!(text.contains("\\u0001"));
        assert_eq!(Value::float(f64::NAN), Value::Null);
        assert_eq!(Value::float(2.0), Value::Number("2.0".into()));
    }

    #[test]
    fn accessors_reject_mismatched_kinds() {
        let v = parse(br#"{"n":-1,"f":1.5,"s":"x"}"#).unwrap();
        assert_eq!(v.find("n").unwrap().as_u64(), None);
        assert_eq!(v.find("n").unwrap().as_i64(), Some(-1));
        assert_eq!(v.find("f").unwrap().as_i64(), None);
        assert_eq!(v.find("s").unwrap().as_f64(), None);
        assert_eq!(v.find("missing"), None);
        assert_eq!(Value::Null.find("x"), None);
    }

    proptest::proptest! {
        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..256)) {
            let _ = parse(&bytes);
        }

        #[test]
        fn arbitrary_strings_round_trip(text in ".{0,64}") {
            let mut out = String::new();
            write_string(&mut out, &text);
            let parsed = parse(out.as_bytes()).unwrap();
            proptest::prop_assert_eq!(parsed.as_str(), Some(text.as_str()));
        }
    }
}
