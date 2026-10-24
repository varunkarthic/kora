//! Version information derived from `VERSION` and the release pipeline's patch number.

/// Major version (from `VERSION`).
pub const MAJOR: u32 = parse(env!("KORA_VERSION_MAJOR"));
/// Minor version (from `VERSION`).
pub const MINOR: u32 = parse(env!("KORA_VERSION_MINOR"));
/// Patch version (injected by the release pipeline through `KORA_VERSION_PATCH`).
pub const PATCH: u32 = parse(env!("KORA_VERSION_PATCH"));

/// Schema version for persisted data (calibration, performance database, tuning, pipeline cache).
/// Bump whenever a persisted format changes incompatibly; stale files are then discarded.
pub const SCHEMA_VERSION: u32 = 1;

/// Returns `"major.minor.patch"`.
pub fn version_string() -> String {
    format!("{MAJOR}.{MINOR}.{PATCH}")
}

const fn parse(text: &str) -> u32 {
    let bytes = text.as_bytes();
    let mut value = 0u32;
    let mut index = 0;
    while index < bytes.len() {
        value = value * 10 + (bytes[index] - b'0') as u32;
        index += 1;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_string_matches_components() {
        assert_eq!(version_string(), format!("{MAJOR}.{MINOR}.{PATCH}"));
    }

    #[test]
    fn version_file_major_minor_are_embedded() {
        let base = include_str!("../../../VERSION").trim();
        assert_eq!(base, format!("{MAJOR}.{MINOR}"));
    }
}
