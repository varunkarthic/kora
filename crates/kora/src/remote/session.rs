//! Authenticated, bounded TCP sessions for KORA workers.

use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::error::{Error, ErrorCode, Result};

use super::{Frame, MessageType, challenge, read_frame, response, verify_response, write_frame};

/// A TCP stream that completed the KORA challenge-response handshake.
///
/// A session is intentionally single-reader/single-writer. Concurrent callers should serialize
/// request/response pairs above this type so a response can never be delivered to the wrong job.
#[derive(Debug)]
pub struct WorkerSession {
    stream: TcpStream,
    max_message_bytes: u64,
    next_request_id: AtomicU64,
}

impl WorkerSession {
    fn new(stream: TcpStream, max_message_bytes: u64) -> Self {
        Self {
            stream,
            max_message_bytes,
            next_request_id: AtomicU64::new(1),
        }
    }

    /// Allocates a nonzero request identifier; wraparound skips zero (reserved for handshake).
    pub fn next_request_id(&self) -> u64 {
        loop {
            let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
            if id != 0 {
                return id;
            }
        }
    }

    /// Sends one application message after the handshake.
    pub fn send(&mut self, message_type: MessageType, request_id: u64, payload: &[u8]) -> Result<()> {
        if request_id == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "request id zero is reserved for the handshake",
            ));
        }
        write_frame(
            &mut self.stream,
            &Frame {
                message_type,
                request_id,
                payload: payload.to_vec(),
            },
            self.max_message_bytes,
        )
    }

    /// Receives one application message after the handshake. Handshake message types and request id
    /// zero are rejected so a peer cannot renegotiate mid-session.
    pub fn receive(&mut self) -> Result<Frame> {
        let frame = read_frame(&mut self.stream, self.max_message_bytes)?;
        if frame.request_id == 0
            || matches!(
                frame.message_type,
                MessageType::Challenge | MessageType::Auth | MessageType::Welcome | MessageType::Reject
            )
        {
            return Err(Error::new(
                ErrorCode::TransferFailed,
                "unexpected handshake frame after session setup",
            ));
        }
        Ok(frame)
    }

    /// Peer address as reported by TCP.
    pub fn peer_addr(&self) -> Result<SocketAddr> {
        self.stream.peer_addr().map_err(|error| {
            Error::new(
                ErrorCode::TransferFailed,
                format!("could not get worker peer address: {error}"),
            )
        })
    }
}

fn configure(stream: &TcpStream, timeout: Duration) -> Result<()> {
    stream.set_nodelay(true).map_err(|error| {
        Error::new(
            ErrorCode::TransferFailed,
            format!("could not configure worker TCP_NODELAY: {error}"),
        )
    })?;
    stream.set_read_timeout(Some(timeout)).map_err(|error| {
        Error::new(
            ErrorCode::TransferFailed,
            format!("could not set worker read timeout: {error}"),
        )
    })?;
    stream.set_write_timeout(Some(timeout)).map_err(|error| {
        Error::new(
            ErrorCode::TransferFailed,
            format!("could not set worker write timeout: {error}"),
        )
    })
}

fn reject(stream: &mut TcpStream, limit: u64, reason: &str) {
    let _ = write_frame(
        stream,
        &Frame {
            message_type: MessageType::Reject,
            request_id: 0,
            payload: reason.as_bytes().to_vec(),
        },
        limit,
    );
}

/// Completes the server side of the protocol handshake on an accepted TCP stream.
pub fn accept_session(
    mut stream: TcpStream,
    token: &[u8],
    max_message_bytes: u64,
    timeout: Duration,
) -> Result<WorkerSession> {
    if token.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "authenticated worker sessions require a non-empty token",
        ));
    }
    configure(&stream, timeout)?;
    let nonce = challenge()?;
    write_frame(
        &mut stream,
        &Frame {
            message_type: MessageType::Challenge,
            request_id: 0,
            payload: nonce.to_vec(),
        },
        max_message_bytes,
    )?;
    let auth = read_frame(&mut stream, max_message_bytes)?;
    let valid = auth.message_type == MessageType::Auth
        && auth.request_id == 0
        && verify_response(token, &nonce, &auth.payload)?;
    if !valid {
        reject(&mut stream, max_message_bytes, "worker authentication failed");
        return Err(Error::new(
            ErrorCode::TransferFailed,
            "worker authentication failed",
        ));
    }
    write_frame(
        &mut stream,
        &Frame {
            message_type: MessageType::Welcome,
            request_id: 0,
            payload: Vec::new(),
        },
        max_message_bytes,
    )?;
    Ok(WorkerSession::new(stream, max_message_bytes))
}

/// Connects and completes the client side of the protocol handshake.
pub fn connect_session(
    address: SocketAddr,
    token: &[u8],
    max_message_bytes: u64,
    timeout: Duration,
) -> Result<WorkerSession> {
    if token.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "authenticated worker sessions require a non-empty token",
        ));
    }
    let mut stream = TcpStream::connect_timeout(&address, timeout).map_err(|error| {
        Error::new(
            ErrorCode::TransferFailed,
            format!("could not connect to worker: {error}"),
        )
    })?;
    configure(&stream, timeout)?;
    let challenge_frame = read_frame(&mut stream, max_message_bytes)?;
    if challenge_frame.message_type != MessageType::Challenge
        || challenge_frame.request_id != 0
        || challenge_frame.payload.len() != 32
    {
        return Err(Error::new(
            ErrorCode::TransferFailed,
            "worker sent an invalid authentication challenge",
        ));
    }
    let mac = response(token, &challenge_frame.payload)?;
    write_frame(
        &mut stream,
        &Frame {
            message_type: MessageType::Auth,
            request_id: 0,
            payload: mac.to_vec(),
        },
        max_message_bytes,
    )?;
    let welcome = read_frame(&mut stream, max_message_bytes)?;
    if welcome.message_type == MessageType::Reject {
        return Err(Error::new(
            ErrorCode::TransferFailed,
            "worker rejected authentication",
        ));
    }
    if welcome.message_type != MessageType::Welcome || welcome.request_id != 0 {
        return Err(Error::new(
            ErrorCode::TransferFailed,
            "worker sent an invalid authentication response",
        ));
    }
    Ok(WorkerSession::new(stream, max_message_bytes))
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::thread;

    use super::*;

    const TOKEN: &[u8] = b"test token";
    const LIMIT: u64 = 1024;
    const TIMEOUT: Duration = Duration::from_secs(2);

    #[test]
    fn loopback_sessions_authenticate_and_exchange_messages() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut session = accept_session(stream, TOKEN, LIMIT, TIMEOUT).unwrap();
            let request = session.receive().unwrap();
            assert_eq!(request.message_type, MessageType::Ping);
            session
                .send(MessageType::Pong, request.request_id, b"ok")
                .unwrap();
        });
        let mut client = connect_session(address, TOKEN, LIMIT, TIMEOUT).unwrap();
        let id = client.next_request_id();
        client.send(MessageType::Ping, id, b"hello").unwrap();
        let reply = client.receive().unwrap();
        assert_eq!(reply.message_type, MessageType::Pong);
        assert_eq!(reply.request_id, id);
        assert_eq!(reply.payload, b"ok");
        server.join().unwrap();
    }

    #[test]
    fn wrong_token_is_rejected() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            assert!(accept_session(stream, TOKEN, LIMIT, TIMEOUT).is_err());
        });
        assert!(connect_session(address, b"wrong", LIMIT, TIMEOUT).is_err());
        server.join().unwrap();
    }
}
