//! Challenge-response authentication for worker links.
//!
//! The transport sends an unpredictable challenge and the peer proves knowledge of the shared
//! secret with HMAC-SHA256 over a domain-separated payload. Empty secrets are intentionally not
//! accepted here: callers that want loopback-only unauthenticated development must enforce that
//! policy before invoking this module.

use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::error::{Error, ErrorCode, Result};

/// Domain separation prevents a KORA worker token from being reused as a valid MAC elsewhere.
pub const HANDSHAKE_CONTEXT: &[u8] = b"kora-worker/1";
const CHALLENGE_SIZE: usize = 32;

type HmacSha256 = Hmac<Sha256>;

/// Produces a fresh, 256-bit worker challenge from the operating-system random source.
pub fn challenge() -> Result<[u8; CHALLENGE_SIZE]> {
    let mut nonce = [0_u8; CHALLENGE_SIZE];
    getrandom::fill(&mut nonce).map_err(|error| {
        Error::new(
            ErrorCode::IoError,
            format!("could not obtain worker challenge randomness: {error}"),
        )
    })?;
    Ok(nonce)
}

/// Computes the response to `nonce`. Empty tokens are rejected to avoid accidentally presenting
/// an unauthenticated session as authenticated.
pub fn response(token: &[u8], nonce: &[u8]) -> Result<[u8; 32]> {
    if token.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "worker handshake token must not be empty",
        ));
    }
    let mut mac = HmacSha256::new_from_slice(token)
        .map_err(|_| Error::new(ErrorCode::InvalidArgument, "worker handshake token is invalid"))?;
    mac.update(HANDSHAKE_CONTEXT);
    mac.update(nonce);
    Ok(mac.finalize().into_bytes().into())
}

/// Verifies one response in constant time.
pub fn verify_response(token: &[u8], nonce: &[u8], received: &[u8]) -> Result<bool> {
    if received.len() != 32 {
        return Ok(false);
    }
    if token.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "worker handshake token must not be empty",
        ));
    }
    let mut mac = HmacSha256::new_from_slice(token)
        .map_err(|_| Error::new(ErrorCode::InvalidArgument, "worker handshake token is invalid"))?;
    mac.update(HANDSHAKE_CONTEXT);
    mac.update(nonce);
    Ok(mac.verify_slice(received).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authenticates_only_the_matching_token_and_nonce() {
        let nonce = [7_u8; 32];
        let mac = response(b"correct horse battery staple", &nonce).unwrap();
        assert!(verify_response(b"correct horse battery staple", &nonce, &mac).unwrap());
        assert!(!verify_response(b"wrong", &nonce, &mac).unwrap());
        assert!(!verify_response(b"correct horse battery staple", &[8_u8; 32], &mac).unwrap());
        assert!(!verify_response(b"correct horse battery staple", &nonce, &mac[..31]).unwrap());
        assert!(response(b"", &nonce).is_err());
    }

    #[test]
    fn challenges_are_populated() {
        let first = challenge().unwrap();
        let second = challenge().unwrap();
        assert_ne!(first, [0_u8; 32]);
        assert_ne!(first, second);
    }
}
