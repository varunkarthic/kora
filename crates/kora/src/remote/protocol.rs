//! Worker wire format.
//!
//! Every frame is a 32-byte little-endian header followed by `payload_size` bytes:
//!
//! ```text
//! u32 magic 'KORA' | u16 version | u16 type | u64 request_id | u64 payload_size | u32 crc32c | u32 reserved
//! ```
//!
//! A bad magic, version, size, or CRC32C is a link failure: the connection is dropped and the
//! client re-executes outstanding work locally. Payload fields use the [`Writer`]/[`Reader`]
//! encodings (little-endian integers; `bytes` and strings are a `u64` length plus contents).
//!
//! Version 2 (KORA 1.0) keeps the version 1 layout; peers of different versions reject each other
//! at the first frame.

use std::net::TcpStream;
use std::time::Duration;

use hmac::{Hmac, Mac};
use sha2::Sha256;

use super::socket::{receive_all, send_all};
use crate::error::{Error, ErrorCode, Result, Status};

pub(crate) const MAGIC: u32 = 0x4152_4F4B;
pub(crate) const PROTOCOL_VERSION: u16 = 2;
pub(crate) const HEADER_SIZE: usize = 32;
const HANDSHAKE_CONTEXT: &[u8] = b"kora-worker/2";

/// Frame types. Payload layouts are documented per variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub(crate) enum MessageType {
    /// worker -> client: bytes nonce.
    Challenge = 1,
    /// client -> worker: bytes hmac.
    Auth = 2,
    /// worker -> client: u64 session, string name, string device, u64 max_message.
    Welcome = 3,
    /// Either direction: string reason.
    Reject = 4,
    /// client -> worker: u64 key, string metadata_json, string entry, bytes words.
    LoadKernel = 10,
    /// worker -> client: empty.
    KernelLoaded = 11,
    /// worker -> client: string message.
    Failure = 12,
    /// client -> worker: see `cluster::remote_dispatch`.
    Dispatch = 20,
    /// worker -> client: u32 count, (u32 binding, bytes data)*.
    Outputs = 21,
    /// worker -> client: u32 count, u32 binding*.
    NeedData = 22,
    /// client -> worker: u32 count, u64 buffer_key*.
    Release = 30,
    /// Heartbeat request.
    Ping = 40,
    /// Heartbeat reply.
    Pong = 41,
}

impl MessageType {
    fn from_u16(value: u16) -> Option<Self> {
        Some(match value {
            1 => Self::Challenge,
            2 => Self::Auth,
            3 => Self::Welcome,
            4 => Self::Reject,
            10 => Self::LoadKernel,
            11 => Self::KernelLoaded,
            12 => Self::Failure,
            20 => Self::Dispatch,
            21 => Self::Outputs,
            22 => Self::NeedData,
            30 => Self::Release,
            40 => Self::Ping,
            41 => Self::Pong,
            _ => return None,
        })
    }
}

/// One received frame.
#[derive(Debug)]
pub(crate) struct Message {
    pub(crate) kind: MessageType,
    pub(crate) request_id: u64,
    pub(crate) payload: Vec<u8>,
}

fn protocol_error(what: &str) -> Error {
    Error::new(ErrorCode::IoError, format!("worker protocol: {what}"))
}

fn header(kind: MessageType, request_id: u64, payload: &[u8]) -> [u8; HEADER_SIZE] {
    let mut header = [0_u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4..6].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    header[6..8].copy_from_slice(&(kind as u16).to_le_bytes());
    header[8..16].copy_from_slice(&request_id.to_le_bytes());
    header[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    header[24..28].copy_from_slice(&crc32c::crc32c(payload).to_le_bytes());
    header
}

/// Sends one frame; `timeout` bounds each of the header and payload writes.
pub(crate) fn send_message(
    stream: &mut TcpStream,
    kind: MessageType,
    request_id: u64,
    payload: &[u8],
    timeout: Duration,
) -> Status {
    send_all(stream, &header(kind, request_id, payload), timeout)?;
    send_all(stream, payload, timeout)
}

/// Receives one frame of at most `max_payload` bytes.
pub(crate) fn receive_message(
    stream: &mut TcpStream,
    max_payload: u64,
    timeout: Duration,
) -> Result<Message> {
    let mut header = [0_u8; HEADER_SIZE];
    receive_all(stream, &mut header, timeout)?;
    let u16_at = |at: usize| u16::from_le_bytes([header[at], header[at + 1]]);
    let u32_at = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().expect("4 bytes"));
    let u64_at = |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().expect("8 bytes"));
    if u32_at(0) != MAGIC {
        return Err(protocol_error("bad frame magic"));
    }
    if u16_at(4) != PROTOCOL_VERSION {
        return Err(protocol_error(&format!(
            "peer speaks protocol version {}, this build speaks {PROTOCOL_VERSION}",
            u16_at(4)
        )));
    }
    let kind = MessageType::from_u16(u16_at(6)).ok_or_else(|| protocol_error("unknown frame type"))?;
    let size = u64_at(16);
    if size > max_payload {
        return Err(protocol_error("frame exceeds the configured size limit"));
    }
    let mut payload = vec![0_u8; usize::try_from(size).map_err(|_| protocol_error("frame too large"))?];
    receive_all(stream, &mut payload, timeout)?;
    if crc32c::crc32c(&payload) != u32_at(24) {
        return Err(protocol_error("frame checksum mismatch"));
    }
    Ok(Message {
        kind,
        request_id: u64_at(8),
        payload,
    })
}

/// HMAC-SHA256 of `nonce || "kora-worker/2"` keyed by the shared token.
pub(crate) fn handshake_mac(token: &str, nonce: &[u8]) -> [u8; 32] {
    let mut mac = Hmac::<Sha256>::new_from_slice(token.as_bytes()).expect("HMAC accepts any key length");
    mac.update(nonce);
    mac.update(HANDSHAKE_CONTEXT);
    mac.finalize().into_bytes().into()
}

/// Constant-time check of a received handshake MAC.
pub(crate) fn verify_handshake_mac(token: &str, nonce: &[u8], received: &[u8]) -> bool {
    let mut mac = Hmac::<Sha256>::new_from_slice(token.as_bytes()).expect("HMAC accepts any key length");
    mac.update(nonce);
    mac.update(HANDSHAKE_CONTEXT);
    mac.verify_slice(received).is_ok()
}

/// Payload encoder.
#[derive(Debug, Default)]
pub(crate) struct Writer {
    pub(crate) bytes: Vec<u8>,
}

impl Writer {
    pub(crate) fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    pub(crate) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    pub(crate) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    pub(crate) fn bytes(&mut self, value: &[u8]) {
        self.u64(value.len() as u64);
        self.bytes.extend_from_slice(value);
    }
    pub(crate) fn string(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }
}

/// Bounds-checked payload decoder; every accessor fails instead of reading past the end.
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if count > self.bytes.len() {
            return Err(protocol_error("truncated payload"));
        }
        let (head, tail) = self.bytes.split_at(count);
        self.bytes = tail;
        Ok(head)
    }
    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }
    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }
    pub(crate) fn bytes(&mut self) -> Result<&'a [u8]> {
        let size = self.u64()?;
        self.take(usize::try_from(size).map_err(|_| protocol_error("truncated payload"))?)
    }
    pub(crate) fn string(&mut self) -> Result<String> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| protocol_error("string is not UTF-8"))
    }
    pub(crate) fn is_done(&self) -> bool {
        self.bytes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::remote::socket::{Endpoint, accept, connect, interrupt, listen};

    fn pair() -> (TcpStream, TcpStream) {
        let listener = listen(&Endpoint::parse("127.0.0.1:0").unwrap()).unwrap();
        let port = listener.local_addr().unwrap().port();
        let client = connect(
            &Endpoint::parse(&format!("127.0.0.1:{port}")).unwrap(),
            Duration::from_secs(2),
        )
        .unwrap();
        let server = accept(&listener, Duration::from_secs(2)).unwrap().unwrap();
        (client, server)
    }

    #[test]
    fn crc32c_matches_the_standard_check_value() {
        assert_eq!(crc32c::crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c::crc32c(b""), 0);
    }

    #[test]
    fn hmac_matches_rfc4231_and_binds_the_context() {
        let mut mac = Hmac::<Sha256>::new_from_slice(b"Jefe").unwrap();
        mac.update(b"what do ya want for nothing?");
        let digest: [u8; 32] = mac.finalize().into_bytes().into();
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        let nonce = [7_u8; 32];
        let mac = handshake_mac("token", &nonce);
        assert!(verify_handshake_mac("token", &nonce, &mac));
        assert!(!verify_handshake_mac("other", &nonce, &mac));
        assert!(!verify_handshake_mac("token", &[8; 32], &mac));
        assert!(!verify_handshake_mac("token", &nonce, &mac[..31]));
    }

    #[test]
    fn reader_rejects_truncated_payloads() {
        let mut writer = Writer::default();
        writer.u8(9);
        writer.u32(7);
        writer.u64(u64::MAX);
        writer.string("hello");
        writer.bytes(&[1, 2]);
        let mut reader = Reader::new(&writer.bytes);
        assert_eq!(reader.u8().unwrap(), 9);
        assert_eq!(reader.u32().unwrap(), 7);
        assert_eq!(reader.u64().unwrap(), u64::MAX);
        assert_eq!(reader.string().unwrap(), "hello");
        assert_eq!(reader.bytes().unwrap(), [1, 2]);
        assert!(reader.is_done());
        assert!(reader.u8().is_err());

        let mut lying = Writer::default();
        lying.u64(1 << 40); // Length far beyond the payload.
        assert!(Reader::new(&lying.bytes).bytes().is_err());
        assert!(Reader::new(&[1, 2, 3]).u32().is_err());
        let mut invalid = Writer::default();
        invalid.bytes(&[0xFF, 0xFE]);
        assert!(Reader::new(&invalid.bytes).string().is_err());
    }

    #[test]
    fn frames_round_trip_and_corruption_is_detected() {
        let (mut client, mut server) = pair();
        let timeout = Duration::from_secs(2);
        send_message(&mut client, MessageType::Ping, 42, &[1, 2, 3], timeout).unwrap();
        let received = receive_message(&mut server, 1024, timeout).unwrap();
        assert_eq!(received.kind, MessageType::Ping);
        assert_eq!(received.request_id, 42);
        assert_eq!(received.payload, [1, 2, 3]);

        // Flip one payload byte after the header was computed: the CRC must reject it.
        let mut frame = header(MessageType::Ping, 1, &[1, 2, 3]).to_vec();
        frame.extend_from_slice(&[1, 99, 3]);
        send_all(&mut client, &frame, timeout).unwrap();
        let error = receive_message(&mut server, 1024, timeout).unwrap_err();
        assert!(error.message().contains("checksum"), "{error}");

        // A peer on another protocol version is rejected at its first frame.
        let mut old = header(MessageType::Ping, 1, &[]);
        old[4..6].copy_from_slice(&1_u16.to_le_bytes());
        send_all(&mut client, &old, timeout).unwrap();
        let error = receive_message(&mut server, 1024, timeout).unwrap_err();
        assert!(error.message().contains("version 1"), "{error}");

        let mut unknown = header(MessageType::Ping, 1, &[]);
        unknown[6..8].copy_from_slice(&999_u16.to_le_bytes());
        send_all(&mut client, &unknown, timeout).unwrap();
        assert!(receive_message(&mut server, 1024, timeout).is_err());
    }

    #[test]
    fn oversized_and_truncated_frames_fail_within_the_deadline() {
        {
            let (mut client, mut server) = pair();
            send_message(
                &mut client,
                MessageType::Ping,
                1,
                &[0; 64],
                Duration::from_secs(2),
            )
            .unwrap();
            assert!(receive_message(&mut server, 16, Duration::from_secs(2)).is_err());
        }
        let (mut client, mut server) = pair();
        send_all(&mut client, &[0; 10], Duration::from_secs(2)).unwrap();
        let started = Instant::now();
        assert!(receive_message(&mut server, 1024, Duration::from_millis(200)).is_err());
        assert!(started.elapsed() < Duration::from_secs(2));
        interrupt(&client);
        drop(client);
        assert!(receive_message(&mut server, 1024, Duration::from_secs(2)).is_err());
    }
}
