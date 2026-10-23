//! Bounded little-endian fields inside remote protocol frames.

use crate::error::{Error, ErrorCode, Result};

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::TransferFailed, message)
}

/// Builds a payload using canonical little-endian integer and length-prefixed byte encodings.
#[derive(Debug, Default, Clone)]
pub struct PayloadWriter {
    bytes: Vec<u8>,
}

impl PayloadWriter {
    /// Creates an empty payload writer.
    pub fn new() -> Self {
        Self::default()
    }
    /// Appends an unsigned byte.
    pub fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    /// Appends an unsigned 32-bit integer.
    pub fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    /// Appends an unsigned 64-bit integer.
    pub fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    /// Appends a `u64` length followed by raw bytes.
    pub fn bytes(&mut self, value: &[u8]) -> Result<()> {
        self.u64(u64::try_from(value.len()).map_err(|_| malformed("payload field length overflow"))?);
        self.bytes
            .try_reserve(value.len())
            .map_err(|_| malformed("payload allocation failed"))?;
        self.bytes.extend_from_slice(value);
        Ok(())
    }
    /// Appends validated UTF-8 bytes as a length-prefixed string.
    pub fn string(&mut self, value: &str) -> Result<()> {
        self.bytes(value.as_bytes())
    }
    /// Returns the complete encoded payload.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Bounds-checked reader for a single frame payload. All variable-length fields are capped by the
/// negotiated payload limit before they are sliced or allocated.
#[derive(Debug, Clone, Copy)]
pub struct PayloadReader<'a> {
    bytes: &'a [u8],
    offset: usize,
    max_field_bytes: u64,
}

impl<'a> PayloadReader<'a> {
    /// Reads `bytes`, rejecting individual variable-length fields beyond `max_field_bytes`.
    pub fn new(bytes: &'a [u8], max_field_bytes: u64) -> Self {
        Self {
            bytes,
            offset: 0,
            max_field_bytes,
        }
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| malformed("payload offset overflow"))?;
        let field = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| malformed("truncated payload"))?;
        self.offset = end;
        Ok(field)
    }
    /// Reads an unsigned byte.
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    /// Reads an unsigned 32-bit integer.
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("fixed field")))
    }
    /// Reads an unsigned 64-bit integer.
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("fixed field")))
    }
    /// Reads a bounded `u64` length-prefixed byte field without copying it.
    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let count = self.u64()?;
        if count > self.max_field_bytes {
            return Err(malformed("payload field exceeds negotiated limit"));
        }
        let count =
            usize::try_from(count).map_err(|_| malformed("payload field does not fit this platform"))?;
        self.take(count)
    }
    /// Reads a bounded UTF-8 string without allocating it.
    pub fn string(&mut self) -> Result<&'a str> {
        std::str::from_utf8(self.bytes()?).map_err(|_| malformed("payload string is not valid UTF-8"))
    }
    /// `true` only when every payload byte has been consumed.
    pub fn is_done(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trips_canonical_fields() {
        let mut writer = PayloadWriter::new();
        writer.u8(1);
        writer.u32(2);
        writer.u64(3);
        writer.string("KORA").unwrap();
        let bytes = writer.into_bytes();
        let mut reader = PayloadReader::new(&bytes, 64);
        assert_eq!(reader.u8().unwrap(), 1);
        assert_eq!(reader.u32().unwrap(), 2);
        assert_eq!(reader.u64().unwrap(), 3);
        assert_eq!(reader.string().unwrap(), "KORA");
        assert!(reader.is_done());
    }
    #[test]
    fn rejects_truncation_oversize_and_invalid_utf8() {
        let mut reader = PayloadReader::new(&[1, 2, 3], 8);
        assert!(reader.u64().is_err());
        let mut too_large = Vec::new();
        too_large.extend_from_slice(&9_u64.to_le_bytes());
        let mut reader = PayloadReader::new(&too_large, 8);
        assert!(reader.bytes().is_err());
        let mut writer = PayloadWriter::new();
        writer.bytes(&[0xff]).unwrap();
        let payload = writer.into_bytes();
        let mut reader = PayloadReader::new(&payload, 8);
        assert!(reader.string().is_err());
    }
}
