//! Remote execution: the worker wire protocol, TCP helpers, the [`WorkerServer`] that runs
//! dispatches for a remote [`Cluster`](crate::cluster::Cluster), and the client side used by the
//! Cluster.
//!
//! Traffic is authenticated (HMAC-SHA256 challenge with a shared token) and integrity-checked
//! (CRC32C per frame) but not encrypted: use a trusted link or an SSH/VPN tunnel.

pub(crate) mod protocol;
pub(crate) mod socket;
mod worker;

pub use worker::{WorkerOptions, WorkerServer, WorkerStats};

/// Key a worker caches a buffer under: the buffer slot plus its generation.
pub(crate) fn buffer_key(id: crate::ids::BufferId) -> u64 {
    u64::from(id.slot) | (id.generation << 32)
}
