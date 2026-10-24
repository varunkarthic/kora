//! Status snapshots streamed to `kora-top`, one JSON object per line.

use std::net::{TcpListener, TcpStream};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::{OffloadPolicy, Shared, lock};
use crate::json::{self, Value};
use crate::remote::socket::{accept, peer_is_loopback, send_all};

const MAX_VIEWERS: usize = 8;

fn number(value: impl ToString) -> Value {
    Value::Number(value.to_string())
}

fn object(members: Vec<(&str, Value)>) -> Value {
    Value::Object(members.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

impl Shared {
    /// One status snapshot as a single-line JSON object; see the kora-top documentation for the
    /// fields.
    pub(super) fn status_json(&self) -> String {
        let policy = match self.options.policy {
            OffloadPolicy::Auto => "auto",
            OffloadPolicy::LocalOnly => "local-only",
            OffloadPolicy::PreferRemote => "prefer-remote",
        };
        let devices = self
            .local_status()
            .into_iter()
            .map(|d| {
                object(vec![
                    ("index", number(d.id.index)),
                    ("name", Value::String(d.name)),
                    ("dispatches", number(d.dispatches)),
                    ("slices", number(d.slices)),
                    ("inflight", number(d.inflight)),
                    ("allocated_bytes", number(d.allocated_bytes)),
                    ("budget_bytes", number(d.budget_bytes)),
                ])
            })
            .collect();
        let workers = self
            .worker_status()
            .into_iter()
            .map(|w| {
                object(vec![
                    ("endpoint", Value::String(w.endpoint)),
                    ("connected", Value::Bool(w.connected)),
                    ("name", Value::String(w.name)),
                    ("device", Value::String(w.device)),
                    ("inflight", number(w.inflight)),
                    ("dispatches", number(w.dispatches)),
                    ("failovers", number(w.failovers)),
                    ("connects", number(w.connects)),
                    ("disconnects", number(w.disconnects)),
                    ("cached_bytes", number(w.cached_bytes)),
                ])
            })
            .collect();
        let stats = self.stats();
        let events = lock(&self.events)
            .iter()
            .map(|e| {
                object(vec![
                    ("at_ms", number(e.at_ms)),
                    ("message", Value::String(e.message.clone())),
                    ("repeats", number(e.repeats)),
                ])
            })
            .collect();
        let snapshot = object(vec![
            ("version", number(1)),
            ("uptime_ms", number(self.created.elapsed().as_millis())),
            ("policy", Value::String(policy.into())),
            ("devices", Value::Array(devices)),
            ("workers", Value::Array(workers)),
            (
                "stats",
                object(vec![
                    ("local_dispatches", number(stats.local_dispatches)),
                    ("remote_dispatches", number(stats.remote_dispatches)),
                    ("failovers", number(stats.failovers)),
                    ("local_recoveries", number(stats.local_recoveries)),
                    ("split_dispatches", number(stats.split_dispatches)),
                    ("slices", number(stats.slices)),
                    ("device_rescans", number(stats.device_rescans)),
                ]),
            ),
            ("events", Value::Array(events)),
        ]);
        let mut out = String::new();
        json::write_value(&mut out, &snapshot);
        out
    }

    pub(super) fn status_loop(&self, listener: &TcpListener) {
        let mut viewers: Vec<TcpStream> = Vec::new();
        let mut next = Instant::now();
        while !self.status_stop.load(Ordering::Acquire) {
            if let Ok(Some(viewer)) = accept(listener, Duration::from_millis(50))
                && peer_is_loopback(&viewer)
                && viewers.len() < MAX_VIEWERS
            {
                viewers.push(viewer);
            }
            if Instant::now() < next {
                continue;
            }
            next = Instant::now() + self.options.status_interval;
            if viewers.is_empty() {
                continue;
            }
            let line = self.status_json() + "\n";
            // A viewer that stops reading is dropped instead of stalling this thread.
            viewers
                .retain_mut(|viewer| send_all(viewer, line.as_bytes(), Duration::from_millis(200)).is_ok());
        }
    }
}
