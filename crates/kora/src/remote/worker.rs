//! The worker side: a process that executes dispatches on behalf of a remote Cluster.

use std::collections::HashMap;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use super::protocol::{
    Message, MessageType, Reader, Writer, receive_message, send_message, verify_handshake_mac,
};
use super::socket::{Endpoint, accept, interrupt, listen, peer_is_loopback};
use crate::buffer::{BufferHandle, BufferUsage};
use crate::error::{Error, ErrorCode, Result, Status};
use crate::metadata::{KernelAccess, parse_kernel_metadata};
use crate::options::RuntimeOptions;
use crate::runtime::{Kernel, Runtime};

// ponytail: fixed handshake timeout and accept poll; make configurable if slow links need it.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
const ACCEPT_POLL: Duration = Duration::from_millis(100);
const MAX_BINDINGS: u32 = 1024;

/// Configuration for a [`WorkerServer`].
#[derive(Debug, Clone)]
pub struct WorkerOptions {
    /// Address to listen on, as `host:port` or `[ipv6]:port`. Port 0 picks a free port.
    pub listen: String,
    /// Shared secret for the HMAC-SHA256 handshake. Empty uses `KORA_WORKER_TOKEN`; when that is
    /// also empty, only loopback clients are accepted.
    pub token: String,
    /// Options for the worker's own runtime.
    pub runtime: RuntimeOptions,
    /// Name reported to clients; empty reports the selected device name.
    pub name: String,
    /// Upper bound on buffer bytes cached for a client; least-recently-used entries are evicted.
    pub max_cache_bytes: u64,
    /// Largest accepted protocol frame.
    pub max_message_bytes: u64,
    /// Deadline for any single send or receive once a request has started. A client that stays
    /// silent (no heartbeat) for longer is considered gone.
    pub io_timeout: Duration,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:7400".into(),
            token: String::new(),
            runtime: RuntimeOptions::default(),
            name: String::new(),
            max_cache_bytes: 1 << 30,
            max_message_bytes: 1 << 30,
            io_timeout: Duration::from_secs(30),
        }
    }
}

/// Counters of a [`WorkerServer`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkerStats {
    /// Authenticated sessions served.
    pub sessions: u64,
    /// Connections that failed authentication.
    pub rejected_connections: u64,
    /// Dispatches completed.
    pub dispatches: u64,
    /// Kernels loaded.
    pub kernels_loaded: u64,
    /// Buffer bytes cached for the current client.
    pub cached_bytes: u64,
    /// A client is connected.
    pub session_active: bool,
}

struct CachedBuffer {
    handle: BufferHandle,
    version: u64,
    size: u64,
    last_used: u64,
}

/// The runtime plus the current client's kernels and cached buffers. Only the active session
/// thread holds it.
struct Engine {
    runtime: Runtime,
    kernels: HashMap<u64, Kernel>,
    buffers: HashMap<u64, CachedBuffer>,
    cached_bytes: u64,
    tick: u64,
}

impl Engine {
    fn reset(&mut self) {
        self.kernels.clear();
        self.buffers.clear();
        self.cached_bytes = 0;
    }

    fn remove(&mut self, key: u64) {
        if let Some(buffer) = self.buffers.remove(&key) {
            self.cached_bytes -= buffer.size;
        }
    }

    /// Evicts least-recently-used buffers outside `keep` until `needed` more bytes fit.
    fn evict_until(&mut self, needed: u64, limit: u64, keep: &[u64]) {
        while self.cached_bytes + needed > limit {
            let victim = self
                .buffers
                .iter()
                .filter(|(key, _)| !keep.contains(key))
                .min_by_key(|(_, buffer)| buffer.last_used)
                .map(|(key, _)| *key);
            let Some(victim) = victim else { return };
            self.remove(victim);
        }
    }
}

#[derive(Default)]
struct Session {
    socket: Option<TcpStream>,
    thread: Option<JoinHandle<()>>,
}

struct Inner {
    options: WorkerOptions,
    device_name: String,
    engine: Mutex<Engine>,
    session: Mutex<Session>,
    stats: Mutex<WorkerStats>,
    stopping: AtomicBool,
    next_session: AtomicU64,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Serves one authenticated client at a time; a newly authenticated client replaces an idle or
/// vanished one, so a host that reconnects after a link drop is never locked out. Dropping the
/// server stops it.
pub struct WorkerServer {
    inner: Arc<Inner>,
    port: u16,
    accept_thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for WorkerServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerServer")
            .field("port", &self.port)
            .field("name", &self.inner.options.name)
            .finish()
    }
}

impl WorkerServer {
    /// Creates the worker's runtime and starts listening.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a malformed `listen` address, [`ErrorCode::IoError`] when
    /// the address cannot be bound, and the errors of [`Runtime::new`].
    pub fn start(mut options: WorkerOptions) -> Result<WorkerServer> {
        if options.token.is_empty() {
            options.token = std::env::var("KORA_WORKER_TOKEN").unwrap_or_default();
        }
        let endpoint = Endpoint::parse(&options.listen)?;
        let runtime = Runtime::new(&options.runtime)?;
        let listener = listen(&endpoint)?;
        let port = listener
            .local_addr()
            .map_err(|e| super::socket::io_error("cannot read the bound port", &e))?
            .port();
        let device_name = runtime
            .devices()
            .first()
            .and_then(|device| runtime.device_info(device).ok())
            .map_or_else(|| "unknown".to_string(), |info| info.name.clone());
        if options.name.is_empty() {
            options.name = device_name.clone();
        }
        let inner = Arc::new(Inner {
            options,
            device_name,
            engine: Mutex::new(Engine {
                runtime,
                kernels: HashMap::new(),
                buffers: HashMap::new(),
                cached_bytes: 0,
                tick: 0,
            }),
            session: Mutex::default(),
            stats: Mutex::default(),
            stopping: AtomicBool::new(false),
            next_session: AtomicU64::new(1),
        });
        let accept_thread = {
            let inner = Arc::clone(&inner);
            std::thread::Builder::new()
                .name("kora-worker-accept".into())
                .spawn(move || inner.accept_loop(&listener))
                .map_err(|e| Error::new(ErrorCode::IoError, format!("cannot start worker thread: {e}")))?
        };
        Ok(WorkerServer {
            inner,
            port,
            accept_thread: Some(accept_thread),
        })
    }

    /// Port actually bound (useful with port 0).
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Current counters.
    pub fn stats(&self) -> WorkerStats {
        lock(&self.inner.stats).clone()
    }

    /// Closes the listener and any active session; outstanding client work fails over to the
    /// client. Idempotent.
    pub fn stop(&mut self) {
        self.inner.stopping.store(true, Ordering::Release);
        if let Some(thread) = self.accept_thread.take() {
            let _ = thread.join();
        }
        let thread = {
            let mut session = lock(&self.inner.session);
            if let Some(socket) = &session.socket {
                interrupt(socket);
            }
            session.thread.take()
        };
        if let Some(thread) = thread {
            let _ = thread.join();
        }
    }
}

impl Drop for WorkerServer {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Inner {
    fn accept_loop(self: &Arc<Self>, listener: &TcpListener) {
        while !self.stopping.load(Ordering::Acquire) {
            let Ok(Some(mut socket)) = accept(listener, ACCEPT_POLL) else {
                continue;
            };
            // Authenticate before touching the active session, so an unauthenticated peer cannot
            // displace a working client.
            if !self.authenticate(&mut socket) {
                lock(&self.stats).rejected_connections += 1;
                continue;
            }
            let previous = {
                let mut session = lock(&self.session);
                if let Some(active) = &session.socket {
                    interrupt(active); // Replace a vanished or idle client.
                }
                session.thread.take()
            };
            if let Some(previous) = previous {
                let _ = previous.join();
            }
            let mut session = lock(&self.session);
            if self.stopping.load(Ordering::Acquire) {
                break;
            }
            session.socket = socket.try_clone().ok();
            let inner = Arc::clone(self);
            session.thread = std::thread::Builder::new()
                .name("kora-worker-session".into())
                .spawn(move || inner.serve(socket))
                .ok();
        }
    }

    fn authenticate(&self, socket: &mut TcpStream) -> bool {
        let mut nonce = [0_u8; 32];
        if getrandom::fill(&mut nonce).is_err() {
            return false;
        }
        let mut challenge = Writer::default();
        challenge.bytes(&nonce);
        let accepted = send_message(
            socket,
            MessageType::Challenge,
            0,
            &challenge.bytes,
            HANDSHAKE_TIMEOUT,
        )
        .and_then(|()| receive_message(socket, 1024, HANDSHAKE_TIMEOUT))
        .is_ok_and(|reply| {
            if reply.kind != MessageType::Auth {
                return false;
            }
            if self.options.token.is_empty() {
                return peer_is_loopback(socket);
            }
            Reader::new(&reply.payload)
                .bytes()
                .is_ok_and(|mac| verify_handshake_mac(&self.options.token, &nonce, mac))
        });
        if !accepted {
            let mut reason = Writer::default();
            reason.string(if self.options.token.is_empty() {
                "worker accepts only loopback clients without a token"
            } else {
                "worker authentication failed"
            });
            let _ = send_message(socket, MessageType::Reject, 0, &reason.bytes, HANDSHAKE_TIMEOUT);
        }
        accepted
    }

    fn serve(&self, mut socket: TcpStream) {
        let mut engine = lock(&self.engine);
        engine.reset();
        {
            let mut stats = lock(&self.stats);
            stats.sessions += 1;
            stats.session_active = true;
            stats.cached_bytes = 0;
        }
        let mut welcome = Writer::default();
        welcome.u64(self.next_session.fetch_add(1, Ordering::Relaxed));
        welcome.string(&self.options.name);
        welcome.string(&self.device_name);
        welcome.u64(self.options.max_message_bytes);
        if send_message(
            &mut socket,
            MessageType::Welcome,
            0,
            &welcome.bytes,
            HANDSHAKE_TIMEOUT,
        )
        .is_ok()
        {
            while !self.stopping.load(Ordering::Acquire) {
                // Idle clients send heartbeats; silence beyond the I/O timeout means the peer is gone.
                let Ok(message) = receive_message(
                    &mut socket,
                    self.options.max_message_bytes,
                    self.options.io_timeout,
                ) else {
                    break;
                };
                if self.handle(&mut engine, &mut socket, &message).is_err() {
                    break;
                }
            }
        }
        interrupt(&socket);
        engine.reset();
        let mut stats = lock(&self.stats);
        stats.session_active = false;
        stats.cached_bytes = 0;
    }

    fn reply(&self, socket: &mut TcpStream, kind: MessageType, request: u64, payload: &[u8]) -> Status {
        send_message(socket, kind, request, payload, self.options.io_timeout)
    }

    fn reply_failure(&self, socket: &mut TcpStream, request: u64, message: &str) -> Status {
        let mut writer = Writer::default();
        writer.string(message);
        self.reply(socket, MessageType::Failure, request, &writer.bytes)
    }

    fn handle(&self, engine: &mut Engine, socket: &mut TcpStream, message: &Message) -> Status {
        match message.kind {
            MessageType::Ping => self.reply(socket, MessageType::Pong, message.request_id, &[]),
            MessageType::LoadKernel => self.load_kernel(engine, socket, message),
            MessageType::Dispatch => self.dispatch(engine, socket, message),
            MessageType::Release => {
                let mut reader = Reader::new(&message.payload);
                for _ in 0..reader.u32()? {
                    engine.remove(reader.u64()?);
                }
                lock(&self.stats).cached_bytes = engine.cached_bytes;
                Ok(())
            }
            _ => Err(Error::new(ErrorCode::IoError, "unexpected worker message")),
        }
    }

    fn load_kernel(&self, engine: &mut Engine, socket: &mut TcpStream, message: &Message) -> Status {
        let mut reader = Reader::new(&message.payload);
        let key = reader.u64()?;
        let metadata = reader.string()?;
        let entry = reader.string()?;
        let words = reader.bytes()?;
        if words.len() % 4 != 0 || !reader.is_done() {
            return Err(Error::new(ErrorCode::IoError, "malformed kernel load request"));
        }
        let words: Vec<u32> = words
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        let loaded = parse_kernel_metadata(&metadata).and_then(|metadata| {
            let device = engine
                .runtime
                .devices()
                .into_iter()
                .next()
                .ok_or_else(|| Error::new(ErrorCode::NoEligibleDevice, "worker has no device"))?;
            engine.runtime.load_kernel_on(&device, &words, &entry, metadata)
        });
        match loaded {
            Ok(kernel) => {
                engine.kernels.insert(key, kernel);
                lock(&self.stats).kernels_loaded += 1;
                self.reply(socket, MessageType::KernelLoaded, message.request_id, &[])
            }
            Err(error) => self.reply_failure(socket, message.request_id, &error.describe()),
        }
    }

    fn dispatch(&self, engine: &mut Engine, socket: &mut TcpStream, message: &Message) -> Status {
        struct Binding<'a> {
            key: u64,
            version: u64,
            size: u64,
            access: KernelAccess,
            result_version: u64,
            data: Option<&'a [u8]>,
        }
        let request = message.request_id;
        let malformed = || Error::new(ErrorCode::IoError, "malformed dispatch request");
        let mut reader = Reader::new(&message.payload);
        let kernel_key = reader.u64()?;
        let global = [reader.u64()?, reader.u64()?, reader.u64()?];
        let push = reader.bytes()?;
        let count = reader.u32()?;
        if count > MAX_BINDINGS {
            return Err(malformed());
        }
        let mut bindings = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let key = reader.u64()?;
            let version = reader.u64()?;
            let size = reader.u64()?;
            let access = match reader.u8()? {
                0 => KernelAccess::Read,
                1 => KernelAccess::Write,
                2 => KernelAccess::ReadWrite,
                _ => return Err(malformed()),
            };
            let result_version = reader.u64()?;
            let data = if reader.u8()? != 0 {
                let data = reader.bytes()?;
                if data.len() as u64 != size {
                    return Err(malformed());
                }
                Some(data)
            } else {
                None
            };
            if size == 0 || size > self.options.max_message_bytes {
                return Err(malformed());
            }
            bindings.push(Binding {
                key,
                version,
                size,
                access,
                result_version,
                data,
            });
        }
        if !reader.is_done() {
            return Err(malformed());
        }
        let Some(kernel) = engine.kernels.get(&kernel_key).cloned() else {
            return self.reply_failure(socket, request, "kernel is not loaded on this worker");
        };

        // Ask for any input whose current contents this worker does not hold.
        let mut missing = Vec::new();
        let mut incoming = 0;
        for (index, binding) in bindings.iter().enumerate() {
            let cached = engine
                .buffers
                .get(&binding.key)
                .filter(|c| c.size == binding.size);
            if cached.is_none() {
                incoming += binding.size;
            }
            let current = cached.is_some_and(|c| c.version == binding.version);
            if binding.access != KernelAccess::Write && binding.data.is_none() && !current {
                missing.push(index as u32);
            }
        }
        if !missing.is_empty() {
            let mut reply = Writer::default();
            reply.u32(missing.len() as u32);
            for index in missing {
                reply.u32(index);
            }
            return self.reply(socket, MessageType::NeedData, request, &reply.bytes);
        }
        let keys: Vec<u64> = bindings.iter().map(|b| b.key).collect();
        engine.evict_until(incoming, self.options.max_cache_bytes, &keys);
        if engine.cached_bytes + incoming > self.options.max_cache_bytes {
            return self.reply_failure(socket, request, "dispatch exceeds the worker cache limit");
        }

        let mut builder = engine
            .runtime
            .dispatch(&kernel)
            .global_3d(global[0], global[1], global[2])
            .push_bytes(push);
        let mut handles = Vec::with_capacity(bindings.len());
        for binding in &bindings {
            engine.tick += 1;
            let tick = engine.tick;
            if engine
                .buffers
                .get(&binding.key)
                .is_none_or(|c| c.size != binding.size)
            {
                engine.remove(binding.key);
                let handle = match engine
                    .runtime
                    .create_buffer_bytes(binding.size, BufferUsage::Storage)
                {
                    Ok(handle) => handle,
                    Err(error) => return self.reply_failure(socket, request, &error.describe()),
                };
                engine.buffers.insert(
                    binding.key,
                    CachedBuffer {
                        handle,
                        version: 0,
                        size: binding.size,
                        last_used: tick,
                    },
                );
                engine.cached_bytes += binding.size;
            }
            let cached = engine.buffers.get_mut(&binding.key).expect("inserted above");
            cached.last_used = tick;
            if let Some(data) = binding.data {
                if let Err(error) = cached.handle.write_bytes(data) {
                    return self.reply_failure(socket, request, &error.describe());
                }
                cached.version = binding.version;
            }
            let handle = cached.handle.clone();
            builder = match binding.access {
                KernelAccess::Read => builder.read(&handle),
                KernelAccess::Write => builder.write(&handle),
                KernelAccess::ReadWrite => builder.read_write(&handle),
            };
            handles.push(handle);
        }
        lock(&self.stats).cached_bytes = engine.cached_bytes;
        if let Err(error) = builder.submit().and_then(|event| event.wait()) {
            // The worker's own device may have been lost; drop state so the next request starts clean.
            if error.code() == ErrorCode::DeviceLost {
                engine.reset();
                let _ = engine.runtime.recover();
            }
            return self.reply_failure(socket, request, &error.describe());
        }
        let mut outputs = Writer::default();
        let written = bindings.iter().filter(|b| b.access != KernelAccess::Read).count();
        outputs.u32(written as u32);
        for (index, binding) in bindings.iter().enumerate() {
            if binding.access == KernelAccess::Read {
                continue;
            }
            match handles[index].map_read() {
                Ok(bytes) => {
                    outputs.u32(index as u32);
                    outputs.bytes(&bytes);
                }
                Err(error) => return self.reply_failure(socket, request, &error.describe()),
            }
            if let Some(cached) = engine.buffers.get_mut(&binding.key) {
                cached.version = binding.result_version;
            }
        }
        lock(&self.stats).dispatches += 1;
        self.reply(socket, MessageType::Outputs, request, &outputs.bytes)
    }
}
