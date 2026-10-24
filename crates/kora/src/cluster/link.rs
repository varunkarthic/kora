//! The client side of one worker connection: connects, authenticates, runs jobs, and fails them
//! over to the local runtime whenever the link misbehaves.

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::{Completion, Job, Shared, WorkerStatus, lock, slice_bytes};
use crate::error::{Error, ErrorCode, Result, Status};
use crate::metadata::KernelAccess;
use crate::remote::buffer_key;
use crate::remote::protocol::{
    Message, MessageType, Reader, Writer, handshake_mac, receive_message, send_message,
};
use crate::remote::socket::{Endpoint, connect, interrupt};

#[derive(Default)]
pub(super) struct LinkState {
    jobs: VecDeque<Box<Job>>,
    releases: Vec<u64>,
    socket: Option<TcpStream>,
    pub(super) unsupported_kernels: HashSet<u64>,
    pub(super) status: WorkerStatus,
}

pub(super) struct WorkerLink {
    pub(super) endpoint: String,
    address: Endpoint,
    pub(super) connected: AtomicBool,
    pub(super) inflight: AtomicU32,
    stop: AtomicBool,
    state: Mutex<LinkState>,
    cv: Condvar,
    thread: Mutex<Option<JoinHandle<()>>>,
}

/// Outcome of one remote attempt. Only `Link` drops the connection.
enum Outcome {
    Done(Option<Vec<Vec<u8>>>),
    Rejected,
    Link,
}

/// Per-connection state, reset on every new session.
struct Session {
    stream: TcpStream,
    loaded_kernels: HashSet<u64>,
    /// Host version whose contents the worker holds for each buffer key, and the token the
    /// worker labels them with.
    versions: HashMap<u64, (u64, u64)>,
    /// Bytes per buffer key ever sent.
    sizes: HashMap<u64, u64>,
    next_request: u64,
    max_message: u64,
}

impl Session {
    fn request(&mut self, kind: MessageType, payload: &[u8], timeout: Duration) -> Result<Message> {
        let id = self.next_request;
        self.next_request += 1;
        send_message(&mut self.stream, kind, id, payload, timeout)?;
        let reply = receive_message(&mut self.stream, self.max_message, timeout)?;
        if reply.request_id != id {
            return Err(Error::new(ErrorCode::IoError, "worker reply is out of order"));
        }
        Ok(reply)
    }

    fn cached_bytes(&self) -> u64 {
        self.versions.keys().filter_map(|key| self.sizes.get(key)).sum()
    }
}

impl WorkerLink {
    pub(super) fn start(shared: &Arc<Shared>, endpoint: &str, address: Endpoint) -> Result<Arc<WorkerLink>> {
        let link = Arc::new(WorkerLink {
            endpoint: endpoint.to_string(),
            address,
            connected: AtomicBool::new(false),
            inflight: AtomicU32::new(0),
            stop: AtomicBool::new(false),
            state: Mutex::new(LinkState {
                status: WorkerStatus {
                    endpoint: endpoint.to_string(),
                    ..WorkerStatus::default()
                },
                ..LinkState::default()
            }),
            cv: Condvar::new(),
            thread: Mutex::new(None),
        });
        let thread = {
            let link = Arc::clone(&link);
            let shared = Arc::clone(shared);
            std::thread::Builder::new()
                .name("kora-cluster-link".into())
                .spawn(move || link.run(&shared))
                .map_err(|e| Error::new(ErrorCode::IoError, format!("cannot start worker link: {e}")))?
        };
        *lock(&link.thread) = Some(thread);
        Ok(link)
    }

    pub(super) fn state(&self) -> MutexGuard<'_, LinkState> {
        lock(&self.state)
    }

    pub(super) fn push(&self, job: Box<Job>) {
        self.inflight.fetch_add(1, Ordering::AcqRel);
        self.state().jobs.push_back(job);
        self.cv.notify_all();
    }

    pub(super) fn release(&self, keys: &[u64]) {
        self.state().releases.extend_from_slice(keys);
        self.cv.notify_all();
    }

    pub(super) fn request_stop(&self) {
        let state = self.state();
        self.stop.store(true, Ordering::Release);
        if let Some(socket) = &state.socket {
            interrupt(socket);
        }
        drop(state);
        self.cv.notify_all();
    }

    pub(super) fn join(&self) {
        let thread = lock(&self.thread).take();
        if let Some(thread) = thread {
            let _ = thread.join();
        }
    }

    pub(super) fn stop_and_join(&self) {
        self.request_stop();
        self.join();
    }

    pub(super) fn status(&self) -> WorkerStatus {
        let mut status = self.state().status.clone();
        status.connected = self.connected.load(Ordering::Acquire);
        status.inflight = self.inflight.load(Ordering::Acquire);
        status
    }

    fn stopping(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn wait_or_stop(&self, duration: Duration) {
        let state = self.state();
        let _ = self
            .cv
            .wait_timeout_while(state, duration, |_| !self.stopping())
            .unwrap_or_else(PoisonError::into_inner);
    }

    /// Hands a job back to the local runtime.
    fn fail_over(&self, shared: &Shared, job: Box<Job>) {
        self.inflight.fetch_sub(1, Ordering::AcqRel);
        shared.fail_over(job, Some(self));
    }

    fn drain_to_local(&self, shared: &Shared) {
        let pending = {
            let mut state = self.state();
            state.releases.clear();
            std::mem::take(&mut state.jobs)
        };
        for job in pending {
            self.fail_over(shared, job);
        }
    }

    fn run(&self, shared: &Shared) {
        let options = &shared.options;
        let mut last_refusal = String::new();
        while !self.stopping() {
            let Ok(stream) = connect(&self.address, options.connect_timeout) else {
                self.wait_or_stop(options.reconnect_interval);
                continue;
            };
            self.state().socket = stream.try_clone().ok();
            if self.stopping() {
                break;
            }
            let mut session = Session {
                stream,
                loaded_kernels: HashSet::new(),
                versions: HashMap::new(),
                sizes: HashMap::new(),
                next_request: 1,
                max_message: 0,
            };
            if let Err(error) = self.handshake(shared, &mut session) {
                if error.message() != last_refusal {
                    last_refusal = error.message().to_string();
                    shared.note(format!("worker {} refused: {last_refusal}", self.endpoint));
                }
                interrupt(&session.stream);
                self.state().socket = None;
                self.wait_or_stop(options.reconnect_interval);
                continue;
            }
            last_refusal.clear();
            let device = {
                let mut state = self.state();
                state.status.connects += 1;
                state.status.cached_bytes = 0;
                state.releases.clear(); // A new session starts with an empty worker cache.
                state.status.device.clone()
            };
            self.connected.store(true, Ordering::Release);
            shared.note(format!("worker {} connected ({device})", self.endpoint));
            self.serve(shared, &mut session);
            self.connected.store(false, Ordering::Release);
            interrupt(&session.stream);
            {
                let mut state = self.state();
                state.status.disconnects += 1;
                state.status.cached_bytes = 0;
                state.socket = None;
            }
            if !self.stopping() {
                shared.note(format!("worker {} disconnected", self.endpoint));
            }
            self.drain_to_local(shared);
            if !self.stopping() {
                self.wait_or_stop(options.reconnect_interval);
            }
        }
        self.connected.store(false, Ordering::Release);
        self.drain_to_local(shared);
    }

    /// Runs jobs and heartbeats until the link fails or the link is stopped.
    fn serve(&self, shared: &Shared, session: &mut Session) {
        let options = &shared.options;
        let mut last_activity = Instant::now();
        loop {
            let (job, releases) = {
                let state = self.state();
                let mut state = self
                    .cv
                    .wait_timeout_while(state, options.heartbeat_interval, |s| {
                        !self.stopping() && s.jobs.is_empty() && s.releases.is_empty()
                    })
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
                (state.jobs.pop_front(), std::mem::take(&mut state.releases))
            };
            if self.stopping() {
                if let Some(job) = job {
                    self.fail_over(shared, job);
                }
                return;
            }
            if !releases.is_empty() {
                let mut writer = Writer::default();
                writer.u32(releases.len() as u32);
                for key in releases {
                    writer.u64(key);
                    session.versions.remove(&key);
                }
                if send_message(
                    &mut session.stream,
                    MessageType::Release,
                    0,
                    &writer.bytes,
                    options.request_timeout,
                )
                .is_err()
                {
                    if let Some(job) = job {
                        self.fail_over(shared, job);
                    }
                    return;
                }
            }
            if let Some(job) = job {
                let started = Instant::now();
                let outcome = self.run_remote(shared, session, &job);
                self.state().status.cached_bytes = session.cached_bytes();
                match outcome {
                    Outcome::Done(collected) => {
                        self.inflight.fetch_sub(1, Ordering::AcqRel);
                        self.state().status.dispatches += 1;
                        last_activity = Instant::now();
                        let Job { work, completion, .. } = *job;
                        match completion {
                            Completion::Slice(slice) => shared.finish_slice(
                                slice,
                                &work,
                                Ok(collected.unwrap_or_default()),
                                Some((&self.endpoint, started)),
                            ),
                            Completion::Whole(promise) => {
                                shared.record_latency(&self.endpoint, &work, started.elapsed());
                                shared.counters.remote_dispatches.fetch_add(1, Ordering::Relaxed);
                                promise.complete(Ok(()));
                            }
                        }
                    }
                    Outcome::Rejected => self.fail_over(shared, job),
                    Outcome::Link => {
                        self.fail_over(shared, job);
                        return;
                    }
                }
                continue;
            }
            if last_activity.elapsed() >= options.heartbeat_interval {
                match session.request(MessageType::Ping, &[], options.request_timeout) {
                    Ok(pong) if pong.kind == MessageType::Pong => last_activity = Instant::now(),
                    _ => return,
                }
            }
        }
    }

    fn handshake(&self, shared: &Shared, session: &mut Session) -> Status {
        let timeout = shared.options.connect_timeout + Duration::from_secs(4);
        let challenge = receive_message(&mut session.stream, 1024, timeout)?;
        if challenge.kind != MessageType::Challenge {
            return Err(Error::new(ErrorCode::IoError, "worker did not send a challenge"));
        }
        let nonce = Reader::new(&challenge.payload).bytes()?;
        let mut auth = Writer::default();
        auth.bytes(&handshake_mac(&shared.options.token, nonce));
        send_message(&mut session.stream, MessageType::Auth, 0, &auth.bytes, timeout)?;
        let welcome = receive_message(&mut session.stream, 1 << 20, timeout)?;
        let mut fields = Reader::new(&welcome.payload);
        match welcome.kind {
            MessageType::Welcome => {}
            MessageType::Reject => {
                let reason = fields.string().unwrap_or_default();
                return Err(Error::new(
                    ErrorCode::IoError,
                    format!("worker rejected connection: {reason}"),
                ));
            }
            _ => return Err(Error::new(ErrorCode::IoError, "unexpected handshake reply")),
        }
        let malformed = |_| Error::new(ErrorCode::IoError, "malformed worker welcome");
        let _session_id = fields.u64().map_err(malformed)?;
        let name = fields.string().map_err(malformed)?;
        let device = fields.string().map_err(malformed)?;
        let max_message = fields.u64().map_err(malformed)?;
        session.max_message = max_message.min(shared.options.max_message_bytes);
        let mut state = self.state();
        state.status.name = name;
        state.status.device = device;
        Ok(())
    }

    fn run_remote(&self, shared: &Shared, session: &mut Session, job: &Job) -> Outcome {
        let timeout = shared.options.request_timeout;
        let work = &job.work;
        let kernel = &work.kernel;
        if !session.loaded_kernels.contains(&kernel.key) {
            let mut load = Writer::default();
            load.u64(kernel.key);
            load.string(&kernel.metadata_json);
            load.string(&kernel.entry_point);
            let words: Vec<u8> = kernel.words.iter().flat_map(|w| w.to_le_bytes()).collect();
            load.bytes(&words);
            match session.request(MessageType::LoadKernel, &load.bytes, timeout) {
                Ok(reply) if reply.kind == MessageType::KernelLoaded => {
                    session.loaded_kernels.insert(kernel.key);
                }
                Ok(reply) if reply.kind == MessageType::Failure => {
                    // e.g. missing device features
                    self.state().unsupported_kernels.insert(kernel.key);
                    return Outcome::Rejected;
                }
                _ => return Outcome::Link,
            }
        }

        // Host copies are authoritative: bring back anything newer on a local device, then capture
        // each buffer's version so a concurrent host write is never overwritten by a stale result.
        if job.slice().is_none() {
            shared.ensure_host_current(work); // A split dispatch did this before slicing.
        }
        let versions: Vec<u64> = work.buffers.iter().map(|b| b.host_version()).collect();
        let keys: Vec<u64> = work.buffers.iter().map(|b| buffer_key(b.id())).collect();
        // Tokens label the worker's copy of a buffer. Inputs are labelled with their host version;
        // results with the version they started from plus the top bit, since the host version a
        // result commits as is only known after the commit.
        let result_token = |version: u64| version | (1 << 63);
        let held = |session: &Session, i: usize| {
            session
                .versions
                .get(&keys[i])
                .filter(|(host, _)| *host == versions[i])
                .map(|(_, token)| *token)
        };
        let encode = |session: &Session, force: &[bool]| -> Result<Vec<u8>> {
            let mut writer = Writer::default();
            writer.u64(kernel.key);
            for value in work.global {
                writer.u64(value);
            }
            writer.bytes(&work.push);
            writer.u32(work.buffers.len() as u32);
            for (i, (buffer, access)) in work.buffers.iter().zip(&work.accesses).enumerate() {
                let token = held(session, i).filter(|_| !force[i]);
                let send = *access != KernelAccess::Write && token.is_none();
                writer.u64(keys[i]);
                writer.u64(token.unwrap_or(versions[i]));
                writer.u64(buffer.size_bytes());
                writer.u8(*access as u8);
                writer.u64(result_token(versions[i]));
                writer.u8(u8::from(send));
                if send {
                    writer.bytes(&buffer.map_read()?);
                }
            }
            Ok(writer.bytes)
        };
        let mut force = vec![false; work.buffers.len()];
        let Ok(payload) = encode(session, &force) else {
            return Outcome::Rejected;
        };
        let mut reply = session.request(MessageType::Dispatch, &payload, timeout);
        if let Some(need) = reply.as_ref().ok().filter(|r| r.kind == MessageType::NeedData) {
            let mut reader = Reader::new(&need.payload);
            let Ok(count) = reader.u32() else {
                return Outcome::Link;
            };
            for _ in 0..count {
                match reader.u32() {
                    Ok(index) if (index as usize) < force.len() => {
                        force[index as usize] = true;
                        session.versions.remove(&keys[index as usize]);
                    }
                    _ => return Outcome::Link,
                }
            }
            let Ok(payload) = encode(session, &force) else {
                return Outcome::Rejected;
            };
            reply = session.request(MessageType::Dispatch, &payload, timeout);
        }
        let reply = match reply {
            Ok(reply) if reply.kind == MessageType::Outputs => reply,
            Ok(reply) if reply.kind == MessageType::Failure => return Outcome::Rejected,
            _ => return Outcome::Link,
        };

        // Validate the whole reply before committing anything, so a bad reply changes no buffer.
        let mut reader = Reader::new(&reply.payload);
        let Ok(count) = reader.u32() else {
            return Outcome::Link;
        };
        let mut outputs: Vec<(usize, &[u8])> = Vec::new();
        for _ in 0..count {
            let (Ok(index), Ok(bytes)) = (reader.u32(), reader.bytes()) else {
                return Outcome::Link;
            };
            let index = index as usize;
            if index >= work.buffers.len()
                || work.accesses[index] == KernelAccess::Read
                || bytes.len() as u64 != work.buffers[index].size_bytes()
                || outputs.iter().any(|(seen, _)| *seen == index)
            {
                return Outcome::Link;
            }
            outputs.push((index, bytes));
        }
        let expected = work.accesses.iter().filter(|a| **a != KernelAccess::Read).count();
        if !reader.is_done() || outputs.len() != expected {
            return Outcome::Link;
        }
        for (i, buffer) in work.buffers.iter().enumerate() {
            if work.accesses[i] != KernelAccess::Write {
                let token = if force[i] {
                    versions[i]
                } else {
                    held(session, i).unwrap_or(versions[i])
                };
                session.versions.insert(keys[i], (versions[i], token));
            }
            session.sizes.insert(keys[i], buffer.size_bytes());
        }
        if let Some(slice) = job.slice() {
            // A slice returns only its own range; the worker's copy now differs from the host's.
            let mut collected = Vec::with_capacity(slice.split.outputs.len());
            for output in &slice.split.outputs {
                let Some((_, bytes)) = outputs.iter().find(|(index, _)| index == output) else {
                    return Outcome::Link;
                };
                let (offset, size) = slice_bytes(slice, bytes.len() as u64);
                collected.push(bytes[offset as usize..(offset + size) as usize].to_vec());
                session.versions.remove(&keys[*output]);
            }
            return Outcome::Done(Some(collected));
        }
        for (index, bytes) in outputs {
            let committed = work.buffers[index].write_bytes_if_version(bytes, versions[index]);
            match committed {
                Ok(c) if c.written => {
                    session
                        .versions
                        .insert(keys[index], (c.host_version, result_token(versions[index])));
                }
                // The host changed meanwhile; the host copy wins.
                _ => {
                    session.versions.remove(&keys[index]);
                }
            }
        }
        Outcome::Done(None)
    }
}
