//! TCP helpers with deadlines, so a vanished peer never hangs a thread.

use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use crate::error::{Error, ErrorCode, Result, Status};

/// A parsed `host:port` or `[ipv6]:port`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Endpoint {
    pub(crate) host: String,
    pub(crate) port: u16,
}

impl Endpoint {
    /// Parses `host:port` or `[ipv6]:port`.
    pub(crate) fn parse(text: &str) -> Result<Endpoint> {
        let invalid = || {
            Error::new(
                ErrorCode::InvalidArgument,
                format!("invalid endpoint '{text}'; expected host:port or [ipv6]:port"),
            )
        };
        let (host, port) = if let Some(rest) = text.strip_prefix('[') {
            let (host, port) = rest.split_once("]:").ok_or_else(invalid)?;
            (host, port)
        } else {
            let (host, port) = text.rsplit_once(':').ok_or_else(invalid)?;
            if host.contains(':') {
                return Err(invalid());
            }
            (host, port)
        };
        if host.is_empty() || port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid());
        }
        let port = port.parse().map_err(|_| invalid())?;
        Ok(Endpoint {
            host: host.to_string(),
            port,
        })
    }

    /// `true` for literal loopback hosts and `localhost`.
    pub(crate) fn is_loopback(&self) -> bool {
        self.host == "localhost" || self.host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
    }

    fn addresses(&self) -> Result<Vec<SocketAddr>> {
        let addresses: Vec<SocketAddr> = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|e| io_error(&format!("cannot resolve {}", self.host), &e))?
            .collect();
        if addresses.is_empty() {
            return Err(Error::new(
                ErrorCode::IoError,
                format!("{} resolved to no address", self.host),
            ));
        }
        Ok(addresses)
    }
}

pub(crate) fn io_error(context: &str, error: &std::io::Error) -> Error {
    let code = if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) {
        ErrorCode::Timeout
    } else {
        ErrorCode::IoError
    };
    Error::new(code, format!("{context}: {error}"))
}

/// Connects to the first address of `endpoint` that answers within `timeout`.
pub(crate) fn connect(endpoint: &Endpoint, timeout: Duration) -> Result<TcpStream> {
    let mut last = None;
    for address in endpoint.addresses()? {
        match TcpStream::connect_timeout(&address, timeout.max(Duration::from_millis(1))) {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                return Ok(stream);
            }
            Err(error) => last = Some(error),
        }
    }
    let error = last.expect("addresses() is never empty");
    Err(io_error(
        &format!("cannot connect to {}:{}", endpoint.host, endpoint.port),
        &error,
    ))
}

/// Binds a listener whose `accept` polls, so callers can check a stop flag between waits.
pub(crate) fn listen(endpoint: &Endpoint) -> Result<TcpListener> {
    let mut last = None;
    for address in endpoint.addresses()? {
        match TcpListener::bind(address) {
            Ok(listener) => {
                listener
                    .set_nonblocking(true)
                    .map_err(|e| io_error("cannot configure listener", &e))?;
                return Ok(listener);
            }
            Err(error) => last = Some(error),
        }
    }
    let error = last.expect("addresses() is never empty");
    Err(io_error(
        &format!("cannot listen on {}:{}", endpoint.host, endpoint.port),
        &error,
    ))
}

/// Waits up to `timeout` for a connection; `None` when none arrived.
pub(crate) fn accept(listener: &TcpListener, timeout: Duration) -> Result<Option<TcpStream>> {
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream
                    .set_nonblocking(false)
                    .map_err(|e| io_error("cannot configure connection", &e))?;
                let _ = stream.set_nodelay(true);
                return Ok(Some(stream));
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {}
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(io_error("accept failed", &error)),
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(None);
        }
        std::thread::sleep((deadline - now).min(Duration::from_millis(5)));
    }
}

/// `true` when the peer connected over a loopback address.
pub(crate) fn peer_is_loopback(stream: &TcpStream) -> bool {
    stream.peer_addr().is_ok_and(|address| match address.ip() {
        IpAddr::V4(ip) => ip.is_loopback(),
        IpAddr::V6(ip) => ip.is_loopback() || ip.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback()),
    })
}

fn remaining(deadline: Instant) -> Result<Duration> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(Error::new(ErrorCode::Timeout, "socket deadline expired"));
    }
    Ok(left)
}

/// Writes every byte before `timeout` elapses.
pub(crate) fn send_all(stream: &mut TcpStream, mut bytes: &[u8], timeout: Duration) -> Status {
    let deadline = Instant::now() + timeout;
    while !bytes.is_empty() {
        stream
            .set_write_timeout(Some(remaining(deadline)?))
            .map_err(|e| io_error("cannot set write timeout", &e))?;
        match stream.write(bytes) {
            Ok(0) => return Err(Error::new(ErrorCode::IoError, "connection closed")),
            Ok(written) => bytes = &bytes[written..],
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(io_error("send failed", &error)),
        }
    }
    Ok(())
}

/// Fills `bytes` before `timeout` elapses; an orderly close before that is an error.
pub(crate) fn receive_all(stream: &mut TcpStream, mut bytes: &mut [u8], timeout: Duration) -> Status {
    let deadline = Instant::now() + timeout;
    while !bytes.is_empty() {
        stream
            .set_read_timeout(Some(remaining(deadline)?))
            .map_err(|e| io_error("cannot set read timeout", &e))?;
        match stream.read(bytes) {
            Ok(0) => return Err(Error::new(ErrorCode::IoError, "connection closed by peer")),
            Ok(read) => bytes = &mut bytes[read..],
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(io_error("receive failed", &error)),
        }
    }
    Ok(())
}

/// Wakes any thread blocked on `stream` (its I/O then fails). Safe from any thread.
pub(crate) fn interrupt(stream: &TcpStream) {
    let _ = stream.shutdown(Shutdown::Both);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_endpoints_and_rejects_malformed_ones() {
        assert_eq!(
            Endpoint::parse("127.0.0.1:7400").unwrap(),
            Endpoint {
                host: "127.0.0.1".into(),
                port: 7400
            }
        );
        assert_eq!(Endpoint::parse("[::1]:80").unwrap().host, "::1");
        assert_eq!(Endpoint::parse("gpu-box.local:0").unwrap().port, 0);
        for bad in [
            "",
            "host",
            "host:",
            ":80",
            "host:99999",
            "host:8a",
            "::1:80",
            "[::1]80",
            "[]:80",
        ] {
            assert!(Endpoint::parse(bad).is_err(), "{bad}");
        }
        assert!(Endpoint::parse("localhost:1").unwrap().is_loopback());
        assert!(Endpoint::parse("[::1]:1").unwrap().is_loopback());
        assert!(!Endpoint::parse("0.0.0.0:1").unwrap().is_loopback());
    }

    #[test]
    fn accept_times_out_and_receive_honours_its_deadline() {
        let listener = listen(&Endpoint::parse("127.0.0.1:0").unwrap()).unwrap();
        assert!(accept(&listener, Duration::from_millis(20)).unwrap().is_none());
        let port = listener.local_addr().unwrap().port();
        let mut client = connect(
            &Endpoint::parse(&format!("127.0.0.1:{port}")).unwrap(),
            Duration::from_secs(2),
        )
        .unwrap();
        let mut server = accept(&listener, Duration::from_secs(2)).unwrap().unwrap();
        assert!(peer_is_loopback(&server));
        send_all(&mut client, &[1, 2, 3], Duration::from_secs(1)).unwrap();
        let mut buffer = [0_u8; 4];
        let started = Instant::now();
        let error = receive_all(&mut server, &mut buffer, Duration::from_millis(100)).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Timeout);
        assert!(started.elapsed() < Duration::from_secs(2));
        interrupt(&client);
        assert!(receive_all(&mut server, &mut buffer[..1], Duration::from_secs(1)).is_err());
    }
}
