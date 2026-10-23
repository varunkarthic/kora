//! Versioned CRC32C framing for the KORA worker protocol.
//!
//! The fixed 32-byte header is little endian:
//! `magic | version | type | request_id | payload_size | crc32c | reserved`.
//! Payload lengths are checked before allocation, and callers provide the negotiated upper bound.

use std::io::{Read, Write};

use crate::error::{Error, ErrorCode, Result};

/// ASCII `KORA`, encoded as a little-endian `u32`.
const MAGIC: u32 = 0x4152_4f4b;
/// Current remote wire version.
pub const PROTOCOL_VERSION: u16 = 1;
/// Bytes in every fixed protocol header.
const HEADER_SIZE: usize = 32;

/// A protocol message type. Unknown values are rejected rather than being ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum MessageType {
    Challenge = 1,
    Auth = 2,
    Welcome = 3,
    Reject = 4,
    LoadKernel = 10,
    KernelLoaded = 11,
    Failure = 12,
    Dispatch = 20,
    Outputs = 21,
    NeedData = 22,
    Release = 30,
    Ping = 40,
    Pong = 41,
}

impl TryFrom<u16> for MessageType {
    type Error = Error;

    fn try_from(value: u16) -> Result<Self> {
        match value {
            1 => Ok(Self::Challenge),
            2 => Ok(Self::Auth),
            3 => Ok(Self::Welcome),
            4 => Ok(Self::Reject),
            10 => Ok(Self::LoadKernel),
            11 => Ok(Self::KernelLoaded),
            12 => Ok(Self::Failure),
            20 => Ok(Self::Dispatch),
            21 => Ok(Self::Outputs),
            22 => Ok(Self::NeedData),
            30 => Ok(Self::Release),
            40 => Ok(Self::Ping),
            41 => Ok(Self::Pong),
            _ => Err(protocol_error("unknown message type")),
        }
    }
}

/// One complete, validated wire frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Kind of message.
    pub message_type: MessageType,
    /// Request/session-local correlation identifier.
    pub request_id: u64,
    /// Opaque message bytes interpreted by the session layer.
    pub payload: Vec<u8>,
}

fn protocol_error(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::TransferFailed, message)
}

fn read_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}
fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("fixed header slice"))
}
fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("fixed header slice"))
}

fn io_error(action: &str, error: std::io::Error) -> Error {
    let code = if matches!(
        error.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    ) {
        ErrorCode::Timeout
    } else {
        ErrorCode::TransferFailed
    };
    Error::new(code, format!("remote protocol {action}: {error}"))
}

/// Encodes `frame` into a complete frame. Payloads above `max_payload` are refused.
pub fn encode_frame(frame: &Frame, max_payload: u64) -> Result<Vec<u8>> {
    let size = u64::try_from(frame.payload.len())
        .map_err(|_| protocol_error("payload length does not fit the wire format"))?;
    if size > max_payload {
        return Err(protocol_error("payload exceeds negotiated message limit"));
    }
    let capacity = HEADER_SIZE
        .checked_add(frame.payload.len())
        .ok_or_else(|| protocol_error("frame length overflow"))?;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(&MAGIC.to_le_bytes());
    bytes.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(frame.message_type as u16).to_le_bytes());
    bytes.extend_from_slice(&frame.request_id.to_le_bytes());
    bytes.extend_from_slice(&size.to_le_bytes());
    bytes.extend_from_slice(&crc32c::crc32c(&frame.payload).to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&frame.payload);
    Ok(bytes)
}

/// Decodes one complete frame. Extra trailing bytes are rejected so callers cannot accidentally
/// accept frame smuggling on a datagram or test boundary.
pub fn decode_frame(bytes: &[u8], max_payload: u64) -> Result<Frame> {
    if bytes.len() < HEADER_SIZE {
        return Err(protocol_error("truncated frame header"));
    }
    if read_u32(&bytes[0..4]) != MAGIC {
        return Err(protocol_error("invalid frame magic"));
    }
    if read_u16(&bytes[4..6]) != PROTOCOL_VERSION {
        return Err(protocol_error("unsupported protocol version"));
    }
    let message_type = MessageType::try_from(read_u16(&bytes[6..8]))?;
    let request_id = read_u64(&bytes[8..16]);
    let payload_size = read_u64(&bytes[16..24]);
    let expected_crc = read_u32(&bytes[24..28]);
    if read_u32(&bytes[28..32]) != 0 {
        return Err(protocol_error("reserved header bytes are non-zero"));
    }
    if payload_size > max_payload {
        return Err(protocol_error("payload exceeds negotiated message limit"));
    }
    let payload_len =
        usize::try_from(payload_size).map_err(|_| protocol_error("payload does not fit this platform"))?;
    let expected_len = HEADER_SIZE
        .checked_add(payload_len)
        .ok_or_else(|| protocol_error("frame length overflow"))?;
    if bytes.len() != expected_len {
        return Err(protocol_error("truncated or trailing frame payload"));
    }
    let payload = bytes[HEADER_SIZE..].to_vec();
    if crc32c::crc32c(&payload) != expected_crc {
        return Err(protocol_error("frame CRC32C mismatch"));
    }
    Ok(Frame {
        message_type,
        request_id,
        payload,
    })
}

/// Writes exactly one bounded frame to a byte stream.
pub fn write_frame(writer: &mut impl Write, frame: &Frame, max_payload: u64) -> Result<()> {
    let bytes = encode_frame(frame, max_payload)?;
    writer
        .write_all(&bytes)
        .map_err(|error| io_error("write failed", error))
}

/// Reads exactly one bounded frame from a byte stream. The announced length is checked before a
/// payload allocation, preventing a peer from turning a header into an allocation attack.
pub fn read_frame(reader: &mut impl Read, max_payload: u64) -> Result<Frame> {
    let mut header = [0_u8; HEADER_SIZE];
    reader
        .read_exact(&mut header)
        .map_err(|error| io_error("header read failed", error))?;
    if read_u32(&header[0..4]) != MAGIC {
        return Err(protocol_error("invalid frame magic"));
    }
    if read_u16(&header[4..6]) != PROTOCOL_VERSION {
        return Err(protocol_error("unsupported protocol version"));
    }
    let payload_size = read_u64(&header[16..24]);
    if payload_size > max_payload {
        return Err(protocol_error("payload exceeds negotiated message limit"));
    }
    let payload_len =
        usize::try_from(payload_size).map_err(|_| protocol_error("payload does not fit this platform"))?;
    let mut bytes = Vec::with_capacity(
        HEADER_SIZE
            .checked_add(payload_len)
            .ok_or_else(|| protocol_error("frame length overflow"))?,
    );
    bytes.extend_from_slice(&header);
    bytes.resize(HEADER_SIZE + payload_len, 0);
    reader
        .read_exact(&mut bytes[HEADER_SIZE..])
        .map_err(|error| io_error("payload read failed", error))?;
    decode_frame(&bytes, max_payload)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn frame() -> Frame {
        Frame {
            message_type: MessageType::Ping,
            request_id: 42,
            payload: vec![1, 2, 3, 4],
        }
    }

    #[test]
    fn round_trips_through_bytes_and_stream() {
        let original = frame();
        let bytes = encode_frame(&original, 1024).unwrap();
        assert_eq!(decode_frame(&bytes, 1024).unwrap(), original);
        let mut stream = Cursor::new(Vec::new());
        write_frame(&mut stream, &original, 1024).unwrap();
        stream.set_position(0);
        assert_eq!(read_frame(&mut stream, 1024).unwrap(), original);
    }

    #[test]
    fn rejects_corruption_unknown_fields_and_size_attacks() {
        let mut bytes = encode_frame(&frame(), 1024).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        assert!(decode_frame(&bytes, 1024).is_err());
        let mut bytes = encode_frame(&frame(), 1024).unwrap();
        bytes[6..8].copy_from_slice(&999_u16.to_le_bytes());
        assert!(decode_frame(&bytes, 1024).is_err());
        let mut bytes = encode_frame(&frame(), 1024).unwrap();
        bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode_frame(&bytes, 1024).is_err());
        let mut bytes = encode_frame(&frame(), 1024).unwrap();
        bytes[28] = 1;
        assert!(decode_frame(&bytes, 1024).is_err());
    }

    #[test]
    fn refuses_truncation_trailing_data_and_outbound_oversize() {
        let bytes = encode_frame(&frame(), 1024).unwrap();
        assert!(decode_frame(&bytes[..31], 1024).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode_frame(&trailing, 1024).is_err());
        assert!(encode_frame(&frame(), 3).is_err());
        let mut stream = Cursor::new(bytes[..HEADER_SIZE + 2].to_vec());
        assert!(read_frame(&mut stream, 1024).is_err());
    }
}
