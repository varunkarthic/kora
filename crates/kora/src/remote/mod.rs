//! Remote-worker transport primitives.
//!
//! This module deliberately owns only the bounded, authenticated-transport-neutral wire format.
//! Higher layers perform the handshake before accepting kernel or buffer messages.  A malformed
//! frame is an [`ErrorCode::TransferFailed`](crate::ErrorCode::TransferFailed), never a panic.

mod auth;
mod protocol;

pub use auth::{HANDSHAKE_CONTEXT, challenge, response, verify_response};
pub use protocol::{
    Frame, MessageType, PROTOCOL_VERSION, decode_frame, encode_frame, read_frame, write_frame,
};
