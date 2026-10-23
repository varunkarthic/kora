//! A bounded, authenticated remote-worker listener.
//!
//! This transport-stage server deliberately handles health traffic only. Dispatch execution is
//! added only once its buffer/version protocol is wired to [`Runtime`](crate::Runtime); rejecting
//! an unsupported dispatch is safer than accepting work that cannot be completed correctly.

use std::io;
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::error::{Error, ErrorCode, Result};

use super::{MessageType, accept_session};

/// Configuration for a [`WorkerServer`].
#[derive(Debug, Clone)]
pub struct WorkerOptions {
    /// Loopback or explicitly selected TCP endpoint. The default uses an ephemeral loopback port.
    pub listen: SocketAddr,
    /// Shared secret used by challenge-response authentication. It must not be empty.
    pub token: Vec<u8>,
    /// Largest accepted wire payload.
    pub max_message_bytes: u64,
    /// Deadline for each socket operation and graceful shutdown observation.
    pub io_timeout: Duration,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:0".parse().expect("constant loopback address"),
            token: Vec::new(),
            max_message_bytes: 1 << 30,
            io_timeout: Duration::from_secs(30),
        }
    }
}

/// Non-sensitive operational counters for a worker listener.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkerStats {
    /// Authenticated sessions accepted since startup.
    pub sessions: u64,
    /// Connections that failed framing or authentication.
    pub rejected_connections: u64,
    /// Health pings answered.
    pub pings: u64,
    /// Frames rejected because Runtime dispatch is not connected yet.
    pub unsupported_messages: u64,
}

#[derive(Debug, Default)]
struct Counters {
    sessions: AtomicU64,
    rejected_connections: AtomicU64,
    pings: AtomicU64,
    unsupported_messages: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> WorkerStats {
        WorkerStats {
            sessions: self.sessions.load(Ordering::Relaxed),
            rejected_connections: self.rejected_connections.load(Ordering::Relaxed),
            pings: self.pings.load(Ordering::Relaxed),
            unsupported_messages: self.unsupported_messages.load(Ordering::Relaxed),
        }
    }
}

/// One authenticated TCP worker listener. It has no device-specific behavior until the runtime
/// request executor is installed, and never binds a public endpoint by default.
#[derive(Debug)]
pub struct WorkerServer {
    port: u16,
    stopping: Arc<AtomicBool>,
    counters: Arc<Counters>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl WorkerServer {
    /// Starts a loopback-safe worker listener.
    pub fn start(options: WorkerOptions) -> Result<Self> {
        if options.token.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "worker token must not be empty",
            ));
        }
        if options.max_message_bytes == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "worker message limit must not be zero",
            ));
        }
        let listener = TcpListener::bind(options.listen).map_err(|error| {
            Error::new(
                ErrorCode::TransferFailed,
                format!("could not bind worker listener: {error}"),
            )
        })?;
        let port = listener
            .local_addr()
            .map_err(|error| {
                Error::new(
                    ErrorCode::TransferFailed,
                    format!("could not obtain worker port: {error}"),
                )
            })?
            .port();
        listener.set_nonblocking(true).map_err(|error| {
            Error::new(
                ErrorCode::TransferFailed,
                format!("could not configure worker listener: {error}"),
            )
        })?;
        let stopping = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(Counters::default());
        let thread_stopping = Arc::clone(&stopping);
        let thread_counters = Arc::clone(&counters);
        let thread = thread::Builder::new()
            .name("kora-worker-listener".into())
            .spawn(move || {
                while !thread_stopping.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            // Accepted sockets inherit nonblocking mode on macOS. Session framing
                            // uses blocking reads with explicit socket timeouts, so restore that
                            // contract before the handshake.
                            if stream.set_nonblocking(false).is_ok() {
                                serve_one(stream, &options, &thread_stopping, &thread_counters);
                            } else {
                                thread_counters
                                    .rejected_connections
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(5))
                        }
                        Err(_) => break,
                    }
                }
            })
            .map_err(|error| {
                Error::new(
                    ErrorCode::IoError,
                    format!("could not start worker listener: {error}"),
                )
            })?;
        Ok(Self {
            port,
            stopping,
            counters,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// The actual port selected by the OS.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Returns a snapshot of worker activity counters.
    pub fn stats(&self) -> WorkerStats {
        self.counters.snapshot()
    }

    /// Stops accepting new sessions and joins the listener thread. Safe to call repeatedly.
    pub fn stop(&self) {
        self.stopping.store(true, Ordering::Release);
        if let Some(thread) = self
            .thread
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            let _ = thread.join();
        }
    }
}

impl Drop for WorkerServer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn serve_one(
    stream: std::net::TcpStream,
    options: &WorkerOptions,
    stopping: &AtomicBool,
    counters: &Counters,
) {
    let mut session = match accept_session(
        stream,
        &options.token,
        options.max_message_bytes,
        options.io_timeout,
    ) {
        Ok(session) => {
            counters.sessions.fetch_add(1, Ordering::Relaxed);
            session
        }
        Err(_) => {
            counters.rejected_connections.fetch_add(1, Ordering::Relaxed);
            return;
        }
    };
    while !stopping.load(Ordering::Acquire) {
        let frame = match session.receive() {
            Ok(frame) => frame,
            Err(_) => return,
        };
        match frame.message_type {
            MessageType::Ping => {
                if session.send(MessageType::Pong, frame.request_id, &[]).is_err() {
                    return;
                }
                counters.pings.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                let _ = session.send(
                    MessageType::Failure,
                    frame.request_id,
                    b"worker dispatch execution is not available yet",
                );
                counters.unsupported_messages.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::connect_session;

    const TOKEN: &[u8] = b"worker test token";
    const LIMIT: u64 = 4096;
    const TIMEOUT: Duration = Duration::from_secs(2);

    fn server() -> WorkerServer {
        WorkerServer::start(WorkerOptions {
            token: TOKEN.to_vec(),
            max_message_bytes: LIMIT,
            io_timeout: TIMEOUT,
            ..WorkerOptions::default()
        })
        .unwrap()
    }

    #[test]
    fn accepts_authenticated_pings_and_rejects_unsupported_work() {
        let server = server();
        let address: SocketAddr = format!("127.0.0.1:{}", server.port()).parse().unwrap();
        let mut client = connect_session(address, TOKEN, LIMIT, TIMEOUT).unwrap();
        let ping = client.next_request_id();
        client.send(MessageType::Ping, ping, &[]).unwrap();
        assert_eq!(client.receive().unwrap().message_type, MessageType::Pong);
        let dispatch = client.next_request_id();
        client.send(MessageType::Dispatch, dispatch, &[]).unwrap();
        let rejection = client.receive().unwrap();
        assert_eq!(rejection.message_type, MessageType::Failure);
        assert_eq!(rejection.request_id, dispatch);
        std::thread::sleep(Duration::from_millis(10));
        let stats = server.stats();
        assert_eq!(stats.sessions, 1);
        assert_eq!(stats.pings, 1);
        assert_eq!(stats.unsupported_messages, 1);
    }

    #[test]
    fn rejects_empty_tokens() {
        assert!(WorkerServer::start(WorkerOptions::default()).is_err());
    }
}
