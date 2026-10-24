//! GPU timestamp profiling.

mod timestamp;

pub use timestamp::{
    GpuTimestampSample, TimestampBackend, TimestampRing, timestamp_delta, timestamp_nanoseconds,
};
