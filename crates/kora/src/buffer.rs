//! Logical buffers: a host shadow plus per-device replicas behind a cheap, cloneable handle.
//!
//! A [`BufferHandle`] owns host bytes (the *host shadow*) and a content version that changes on
//! every host write. The runtime keeps device copies (*replicas*) next to the shadow and compares
//! versions to decide whether a dispatch must upload. Everything is released when the last handle
//! and the last in-flight submission using the buffer are gone; there is no registry to reap.
//!
//! Host access is exclusive per buffer: a mapping guard holds the shadow lock, so mapped bytes can
//! never race another host write (the C++ runtime handed out unlocked raw pointers). A thread that
//! holds a mapping and calls back into the same buffer gets an error instead of a deadlock.
//! Device-to-host completions never block on a user's mapping: when the shadow is busy, the
//! downloaded bytes are parked and applied by the next host access.
//!
//! # Examples
//!
//! ```
//! use kora::buffer::BufferManager;
//!
//! let buffers = BufferManager::new();
//! let values = buffers.create_typed::<f32>(4).unwrap();
//! values.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
//! {
//!     let mut mapped = values.map_write().unwrap();
//!     mapped[0] = 10.0;
//! }
//! assert_eq!(values.read().unwrap(), [10.0, 2.0, 3.0, 4.0]);
//! ```

use std::any::Any;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};
use std::thread::ThreadId;

use bytemuck::Pod;

use crate::checked;
use crate::error::{Error, ErrorCode, Result, Status};
use crate::ids::BufferId;
use crate::memory::{DirtyRange, DirtyRangeSet};

/// Intended use of a buffer (a hint for memory placement).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BufferUsage {
    /// No particular use.
    Generic,
    /// Shader storage buffer.
    #[default]
    Storage,
    /// Uniform buffer.
    Uniform,
    /// Mostly a copy source.
    TransferSource,
    /// Mostly a copy destination.
    TransferDestination,
}

/// Snapshot of a buffer's bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferInfo {
    /// Identity.
    pub id: BufferId,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Usage hint.
    pub usage: BufferUsage,
    /// Live handles (including runtime submissions that keep the buffer alive).
    pub handles: usize,
    /// A mapping guard is alive.
    pub mapped: bool,
    /// Current host content version.
    pub host_version: u64,
    /// Contents were lost with a device and must be rewritten.
    pub data_lost: bool,
}

/// Result of [`BufferHandle::write_bytes_if_version`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditionalWrite {
    /// The bytes were accepted (possibly parked until the shadow is unmapped).
    pub written: bool,
    /// Host version after the call.
    pub host_version: u64,
}

/// Versions are unique across all buffers, so a parked device result can never be confused with a
/// later host write that happened to produce the same counter value.
static NEXT_VERSION: AtomicU64 = AtomicU64::new(1);

fn next_version() -> u64 {
    NEXT_VERSION.fetch_add(1, Ordering::Relaxed)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// 16-byte aligned storage unit, so typed views of any `Pod` type up to 16-byte alignment are
/// sound.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Chunk([u8; 16]);

// SAFETY: `Chunk` is a plain byte array with no padding (size 16, align 16) and every bit pattern
// is valid.
unsafe impl bytemuck::Zeroable for Chunk {}
// SAFETY: as above; `Chunk` is `Copy`, `repr(C)`, and contains only `u8`.
unsafe impl Pod for Chunk {}

struct HostStorage {
    chunks: Box<[Chunk]>,
    len: usize,
}

impl HostStorage {
    fn bytes(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.chunks)[..self.len]
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut bytemuck::cast_slice_mut(&mut self.chunks)[..self.len]
    }
}

struct Parked {
    bytes: Box<[u8]>,
    expected: u64,
    version: u64,
}

#[derive(Default)]
struct Meta {
    version: u64,
    dirty: DirtyRangeSet,
    data_lost: bool,
    mapped_by: Option<ThreadId>,
    parked: Option<Parked>,
}

impl Meta {
    /// The version the shadow logically has, counting a parked device result.
    fn logical_version(&self) -> u64 {
        match &self.parked {
            Some(p) if p.expected == self.version => p.version,
            _ => self.version,
        }
    }
}

type Replica = Arc<dyn Any + Send + Sync>;

pub(crate) struct BufferInner {
    id: BufferId,
    size: u64,
    usage: BufferUsage,
    registry: Arc<Registry>,
    host: Mutex<HostStorage>,
    meta: Mutex<Meta>,
    /// Device copies keyed by the owning device context's ID.
    replicas: Mutex<Vec<(u64, Replica)>>,
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        self.registry.release(self.id);
    }
}

#[derive(Default)]
struct Registry {
    slots: Mutex<Slots>,
}

#[derive(Default)]
struct Slots {
    generations: Vec<u64>,
    free: Vec<u32>,
    live: usize,
}

impl Registry {
    fn acquire(&self) -> Result<BufferId> {
        let mut slots = lock(&self.slots);
        let slot = match slots.free.pop() {
            Some(slot) => slot,
            None => {
                let slot = u32::try_from(slots.generations.len())
                    .map_err(|_| Error::new(ErrorCode::Overflow, "buffer slot index overflow"))?;
                slots.generations.push(1);
                slot
            }
        };
        slots.live += 1;
        Ok(BufferId::new(slot, slots.generations[slot as usize]))
    }

    fn release(&self, id: BufferId) {
        let mut slots = lock(&self.slots);
        slots.live -= 1;
        let generation = &mut slots.generations[id.slot as usize];
        // A slot whose generation would wrap is retired instead of reused.
        if let Some(next) = generation.checked_add(1) {
            *generation = next;
            slots.free.push(id.slot);
        }
    }
}

/// Creates logical buffers. Cheap to clone; every clone shares one ID space.
#[derive(Clone, Default)]
pub struct BufferManager {
    registry: Arc<Registry>,
}

impl fmt::Debug for BufferManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferManager")
            .field("live", &self.live_buffers())
            .finish()
    }
}

impl BufferManager {
    /// A manager with no buffers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Buffers with at least one live handle.
    pub fn live_buffers(&self) -> usize {
        lock(&self.registry.slots).live
    }

    /// Creates a zero-filled buffer of `size_bytes`.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for zero bytes, [`ErrorCode::Overflow`] when the size is not
    /// host addressable, and [`ErrorCode::AllocationFailed`] when the host shadow cannot be
    /// allocated.
    pub fn create(&self, size_bytes: u64, usage: BufferUsage) -> Result<BufferHandle> {
        if size_bytes == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "buffer size must be greater than zero",
            ));
        }
        let len = usize::try_from(size_bytes)
            .ok()
            .filter(|&len| len <= isize::MAX as usize - 16)
            .ok_or_else(|| Error::new(ErrorCode::Overflow, "buffer size exceeds host addressable size"))?;
        let mut chunks = Vec::new();
        chunks.try_reserve_exact(len.div_ceil(16)).map_err(|_| {
            Error::new(
                ErrorCode::AllocationFailed,
                "unable to allocate buffer host shadow",
            )
        })?;
        chunks.resize(len.div_ceil(16), Chunk([0; 16]));
        let id = self.registry.acquire()?;
        Ok(BufferHandle {
            inner: Arc::new(BufferInner {
                id,
                size: size_bytes,
                usage,
                registry: Arc::clone(&self.registry),
                host: Mutex::new(HostStorage {
                    chunks: chunks.into_boxed_slice(),
                    len,
                }),
                meta: Mutex::new(Meta {
                    version: next_version(),
                    ..Meta::default()
                }),
                replicas: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Creates a zero-filled buffer of `count` elements.
    ///
    /// # Errors
    ///
    /// As [`create`](Self::create); [`ErrorCode::Overflow`] when `count * size_of::<T>()`
    /// overflows.
    pub fn create_typed<T: Pod>(&self, count: u64) -> Result<Buffer<T>> {
        const {
            assert!(
                align_of::<T>() <= 16,
                "Buffer<T> supports alignment up to 16 bytes"
            )
        };
        let bytes = checked::byte_size_of::<T>(count, "typed buffer size")?;
        Ok(Buffer {
            handle: self.create(bytes, BufferUsage::Storage)?,
            count,
            _marker: PhantomData,
        })
    }
}

/// Cloneable, reference-counted handle to one logical buffer.
#[derive(Clone)]
pub struct BufferHandle {
    inner: Arc<BufferInner>,
}

impl fmt::Debug for BufferHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferHandle")
            .field("id", &self.inner.id)
            .field("size", &self.inner.size)
            .finish()
    }
}

impl PartialEq for BufferHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for BufferHandle {}

fn data_lost() -> Error {
    Error::new(
        ErrorCode::DataLoss,
        "buffer contents were lost with the Vulkan device",
    )
    .with_recovery("rewrite the complete buffer from an authoritative host source")
}

impl BufferHandle {
    /// Identity (slot and generation).
    pub fn id(&self) -> BufferId {
        self.inner.id
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.inner.size
    }

    /// Usage hint.
    pub fn usage(&self) -> BufferUsage {
        self.inner.usage
    }

    /// Bookkeeping snapshot.
    pub fn info(&self) -> BufferInfo {
        let meta = lock(&self.inner.meta);
        BufferInfo {
            id: self.inner.id,
            size_bytes: self.inner.size,
            usage: self.inner.usage,
            handles: Arc::strong_count(&self.inner),
            mapped: meta.mapped_by.is_some(),
            host_version: meta.logical_version(),
            data_lost: meta.data_lost,
        }
    }

    /// Opaque content version. It changes on every host write, writable mapping, and accepted
    /// device result, and is never reused by any buffer.
    pub fn host_version(&self) -> u64 {
        lock(&self.inner.meta).logical_version()
    }

    /// Merged byte ranges written by the host since creation.
    pub fn dirty_ranges(&self) -> Vec<DirtyRange> {
        lock(&self.inner.meta).dirty.ranges().to_vec()
    }

    fn check_not_mapped_here(&self) -> Status {
        if lock(&self.inner.meta).mapped_by == Some(std::thread::current().id()) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "buffer is mapped by this thread; drop the mapping before accessing it again",
            ));
        }
        Ok(())
    }

    /// Locks the shadow for host access, applying a parked device result first.
    fn lock_host(&self) -> Result<MutexGuard<'_, HostStorage>> {
        self.check_not_mapped_here()?;
        let mut host = lock(&self.inner.host);
        let mut meta = lock(&self.inner.meta);
        if let Some(parked) = meta.parked.take()
            && parked.expected == meta.version
        {
            host.bytes_mut().copy_from_slice(&parked.bytes);
            meta.version = parked.version;
            meta.data_lost = false;
        }
        Ok(host)
    }

    /// Copies the host shadow.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::DataLoss`] after device loss, [`ErrorCode::InvalidArgument`] when this thread
    /// holds a mapping of the buffer.
    pub fn read_bytes(&self) -> Result<Vec<u8>> {
        let host = self.lock_host()?;
        if lock(&self.inner.meta).data_lost {
            return Err(data_lost());
        }
        Ok(host.bytes().to_vec())
    }

    /// Copies the host shadow into `destination`.
    ///
    /// # Errors
    ///
    /// As [`read_bytes`](Self::read_bytes); [`ErrorCode::InvalidArgument`] for a length mismatch.
    pub fn read_into(&self, destination: &mut [u8]) -> Status {
        if destination.len() as u64 != self.inner.size {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "buffer read byte count does not match its capacity",
            ));
        }
        let host = self.lock_host()?;
        if lock(&self.inner.meta).data_lost {
            return Err(data_lost());
        }
        destination.copy_from_slice(host.bytes());
        Ok(())
    }

    /// Replaces the whole host shadow (clearing a data-loss mark).
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a length mismatch or when this thread holds a mapping.
    pub fn write_bytes(&self, bytes: &[u8]) -> Status {
        if bytes.len() as u64 != self.inner.size {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "buffer write byte count does not match its capacity",
            ));
        }
        let mut host = self.lock_host()?;
        host.bytes_mut().copy_from_slice(bytes);
        let mut meta = lock(&self.inner.meta);
        meta.dirty.add(DirtyRange::new(0, self.inner.size))?;
        meta.version = next_version();
        meta.data_lost = false;
        Ok(())
    }

    /// Writes device results only if the host shadow still has `expected_version`. Never blocks
    /// on a mapping: a busy shadow parks the bytes until its next access.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a length mismatch.
    pub fn write_bytes_if_version(&self, bytes: &[u8], expected_version: u64) -> Result<ConditionalWrite> {
        if bytes.len() as u64 != self.inner.size {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "buffer write byte count does not match its capacity",
            ));
        }
        let mut meta = lock(&self.inner.meta);
        if meta.logical_version() != expected_version {
            return Ok(ConditionalWrite {
                written: false,
                host_version: meta.logical_version(),
            });
        }
        let version = next_version();
        // Lock order elsewhere is host → meta; a non-blocking attempt here cannot deadlock.
        match self.inner.host.try_lock() {
            Ok(mut host) => {
                host.bytes_mut().copy_from_slice(bytes);
                meta.version = version;
                meta.parked = None;
                meta.data_lost = false;
            }
            Err(TryLockError::Poisoned(poisoned)) => {
                poisoned.into_inner().bytes_mut().copy_from_slice(bytes);
                meta.version = version;
                meta.parked = None;
                meta.data_lost = false;
            }
            Err(TryLockError::WouldBlock) => {
                meta.parked = Some(Parked {
                    bytes: bytes.into(),
                    expected: meta.version,
                    version,
                });
            }
        }
        Ok(ConditionalWrite {
            written: true,
            host_version: version,
        })
    }

    /// Marks the contents unavailable after a device-side write failed in flight. Reads fail
    /// with [`ErrorCode::DataLoss`] until the whole buffer is rewritten.
    pub fn mark_data_lost(&self) {
        let mut meta = lock(&self.inner.meta);
        meta.parked = None;
        meta.data_lost = true;
    }

    fn checked_range(&self, offset: u64, size: Option<u64>) -> Result<(usize, usize)> {
        let size =
            match size {
                Some(size) => size,
                None => self.inner.size.checked_sub(offset).ok_or_else(|| {
                    Error::new(ErrorCode::InvalidArgument, "buffer map offset exceeds size")
                })?,
            };
        if !checked::in_range(offset, size, self.inner.size) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "buffer map range exceeds buffer size",
            ));
        }
        // In range of a host-addressable allocation, so both fit in usize.
        Ok((offset as usize, (offset + size) as usize))
    }

    /// Maps the whole shadow for reading. The guard holds the buffer's host lock.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::DataLoss`] after device loss, [`ErrorCode::InvalidArgument`] when this thread
    /// already holds a mapping of the buffer.
    pub fn map_read(&self) -> Result<MappedRead<'_>> {
        self.map_read_range(0, None)
    }

    /// Maps `[offset, offset + size)` for reading (`None` = to the end).
    ///
    /// # Errors
    ///
    /// As [`map_read`](Self::map_read); [`ErrorCode::InvalidArgument`] for an out-of-range span.
    pub fn map_read_range(&self, offset: u64, size: Option<u64>) -> Result<MappedRead<'_>> {
        let (start, end) = self.checked_range(offset, size)?;
        let host = self.lock_host()?;
        let mut meta = lock(&self.inner.meta);
        if meta.data_lost {
            return Err(data_lost());
        }
        meta.mapped_by = Some(std::thread::current().id());
        drop(meta);
        Ok(MappedRead {
            host,
            buffer: &self.inner,
            start,
            end,
        })
    }

    /// Maps the whole shadow for writing. The content version advances immediately and the
    /// range is marked dirty. A full-range write mapping also clears a data-loss mark, since the
    /// caller is rewriting the whole buffer.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] when this thread already holds a mapping of the buffer.
    pub fn map_write(&self) -> Result<MappedWrite<'_>> {
        self.map_write_range(0, None)
    }

    /// Maps `[offset, offset + size)` for writing (`None` = to the end). Partial write mappings
    /// of a buffer with lost contents fail, because the remaining bytes would stay invalid.
    ///
    /// # Errors
    ///
    /// As [`map_write`](Self::map_write); [`ErrorCode::DataLoss`] for a partial mapping after
    /// device loss.
    pub fn map_write_range(&self, offset: u64, size: Option<u64>) -> Result<MappedWrite<'_>> {
        let (start, end) = self.checked_range(offset, size)?;
        let host = self.lock_host()?;
        let mut meta = lock(&self.inner.meta);
        let full = start == 0 && end as u64 == self.inner.size;
        if meta.data_lost && !full {
            return Err(data_lost());
        }
        if end > start {
            meta.dirty
                .add(DirtyRange::new(start as u64, (end - start) as u64))?;
            meta.version = next_version();
        }
        meta.data_lost &= !full;
        meta.mapped_by = Some(std::thread::current().id());
        drop(meta);
        Ok(MappedWrite {
            host,
            buffer: &self.inner,
            start,
            end,
        })
    }

    /// Runs `f` with the current logical contents and their version without applying a parked
    /// device result, for uploads. Fails when this thread holds a mapping of the buffer.
    pub(crate) fn with_contents<R>(&self, f: impl FnOnce(&[u8], u64) -> R) -> Result<R> {
        self.check_not_mapped_here()?;
        {
            let meta = lock(&self.inner.meta);
            if meta.data_lost {
                return Err(data_lost());
            }
            if let Some(parked) = meta.parked.as_ref().filter(|p| p.expected == meta.version) {
                return Ok(f(&parked.bytes, parked.version));
            }
        }
        let host = lock(&self.inner.host);
        let meta = lock(&self.inner.meta);
        if meta.data_lost {
            return Err(data_lost());
        }
        let version = meta.logical_version();
        if let Some(parked) = meta.parked.as_ref().filter(|p| p.expected == meta.version) {
            return Ok(f(&parked.bytes, parked.version));
        }
        drop(meta);
        Ok(f(host.bytes(), version))
    }

    /// The replica a device context keeps for this buffer.
    pub(crate) fn replica<R: Any + Send + Sync>(&self, context: u64) -> Option<Arc<R>> {
        lock(&self.inner.replicas)
            .iter()
            .find(|(owner, _)| *owner == context)
            .and_then(|(_, replica)| Arc::clone(replica).downcast::<R>().ok())
    }

    /// Returns the existing replica for `context` or stores the one `create` builds.
    pub(crate) fn replica_or_insert<R: Any + Send + Sync>(
        &self,
        context: u64,
        create: impl FnOnce() -> Result<R>,
    ) -> Result<Arc<R>> {
        let mut replicas = lock(&self.inner.replicas);
        if let Some((_, replica)) = replicas.iter().find(|(owner, _)| *owner == context) {
            return Arc::clone(replica).downcast::<R>().map_err(|_| {
                Error::new(
                    ErrorCode::InternalInvariantViolation,
                    "buffer replica has an unexpected type",
                )
            });
        }
        let replica = Arc::new(create()?);
        replicas.push((context, Arc::clone(&replica) as Replica));
        Ok(replica)
    }

    /// Drops the replica of `context` (it is destroyed once in-flight work releases it).
    pub(crate) fn remove_replica(&self, context: u64) {
        lock(&self.inner.replicas).retain(|(owner, _)| *owner != context);
    }

    /// Context IDs that hold a replica.
    pub(crate) fn replica_owners(&self) -> Vec<u64> {
        lock(&self.inner.replicas)
            .iter()
            .map(|(owner, _)| *owner)
            .collect()
    }

    /// A non-owning reference.
    pub(crate) fn downgrade(&self) -> WeakBufferHandle {
        WeakBufferHandle(Arc::downgrade(&self.inner))
    }

    /// A keep-alive token for submissions that use this buffer.
    pub(crate) fn keepalive(&self) -> Arc<dyn Any + Send + Sync> {
        Arc::clone(&self.inner) as Arc<dyn Any + Send + Sync>
    }
}

/// Non-owning reference to a buffer, used by device contexts to release their replicas.
#[derive(Clone)]
pub(crate) struct WeakBufferHandle(std::sync::Weak<BufferInner>);

impl WeakBufferHandle {
    pub(crate) fn upgrade(&self) -> Option<BufferHandle> {
        self.0.upgrade().map(|inner| BufferHandle { inner })
    }
    pub(crate) fn is_alive(&self) -> bool {
        self.0.strong_count() != 0
    }
}

/// Read guard over a host-shadow range. Holds the buffer's host lock.
pub struct MappedRead<'a> {
    host: MutexGuard<'a, HostStorage>,
    buffer: &'a BufferInner,
    start: usize,
    end: usize,
}

impl Deref for MappedRead<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.host.bytes()[self.start..self.end]
    }
}

impl Drop for MappedRead<'_> {
    fn drop(&mut self) {
        lock(&self.buffer.meta).mapped_by = None;
    }
}

impl fmt::Debug for MappedRead<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedRead")
            .field("range", &(self.start..self.end))
            .finish()
    }
}

/// Write guard over a host-shadow range. Holds the buffer's host lock.
pub struct MappedWrite<'a> {
    host: MutexGuard<'a, HostStorage>,
    buffer: &'a BufferInner,
    start: usize,
    end: usize,
}

impl Deref for MappedWrite<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.host.bytes()[self.start..self.end]
    }
}

impl DerefMut for MappedWrite<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        let (start, end) = (self.start, self.end);
        &mut self.host.bytes_mut()[start..end]
    }
}

impl Drop for MappedWrite<'_> {
    fn drop(&mut self) {
        lock(&self.buffer.meta).mapped_by = None;
    }
}

impl fmt::Debug for MappedWrite<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedWrite")
            .field("range", &(self.start..self.end))
            .finish()
    }
}

/// Typed view of a logical buffer holding `count` elements of `T`.
pub struct Buffer<T: Pod> {
    handle: BufferHandle,
    count: u64,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Pod> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            count: self.count,
            _marker: PhantomData,
        }
    }
}

impl<T: Pod> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("id", &self.handle.id())
            .field("count", &self.count)
            .field("element", &std::any::type_name::<T>())
            .finish()
    }
}

impl<T: Pod> Buffer<T> {
    /// Wraps an untyped handle whose size is a whole number of elements.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] when the size is not a multiple of `size_of::<T>()` or `T`
    /// is zero-sized.
    pub fn from_handle(handle: BufferHandle) -> Result<Self> {
        const {
            assert!(
                align_of::<T>() <= 16,
                "Buffer<T> supports alignment up to 16 bytes"
            )
        };
        let element = size_of::<T>() as u64;
        if element == 0 || !handle.size_bytes().is_multiple_of(element) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "buffer size is not a whole number of elements",
            ));
        }
        Ok(Self {
            count: handle.size_bytes() / element,
            handle,
            _marker: PhantomData,
        })
    }

    /// Element count.
    pub fn len(&self) -> u64 {
        self.count
    }

    /// `true` for zero elements (never, for buffers created by KORA).
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.handle.size_bytes()
    }

    /// The untyped handle.
    pub fn handle(&self) -> &BufferHandle {
        &self.handle
    }

    /// Identity.
    pub fn id(&self) -> BufferId {
        self.handle.id()
    }

    /// Replaces every element.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] when `values.len()` differs from the element count.
    pub fn write(&self, values: &[T]) -> Status {
        if values.len() as u64 != self.count {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "buffer write element count does not match its capacity",
            ));
        }
        self.handle.write_bytes(bytemuck::cast_slice(values))
    }

    /// Copies every element out.
    ///
    /// # Errors
    ///
    /// As [`BufferHandle::read_bytes`].
    pub fn read(&self) -> Result<Vec<T>> {
        let mapped = self.map_read()?;
        Ok(mapped.to_vec())
    }

    /// Copies every element into `values`.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for a length mismatch; otherwise as
    /// [`BufferHandle::read_bytes`].
    pub fn read_into(&self, values: &mut [T]) -> Status {
        self.handle.read_into(bytemuck::cast_slice_mut(values))
    }

    /// Maps the elements for reading.
    ///
    /// # Errors
    ///
    /// As [`BufferHandle::map_read`].
    pub fn map_read(&self) -> Result<TypedRead<'_, T>> {
        Ok(TypedRead {
            mapped: self.handle.map_read()?,
            _marker: PhantomData,
        })
    }

    /// Maps the elements for writing.
    ///
    /// # Errors
    ///
    /// As [`BufferHandle::map_write`].
    pub fn map_write(&self) -> Result<TypedWrite<'_, T>> {
        Ok(TypedWrite {
            mapped: self.handle.map_write()?,
            _marker: PhantomData,
        })
    }
}

impl<T: Pod> From<Buffer<T>> for BufferHandle {
    fn from(buffer: Buffer<T>) -> Self {
        buffer.handle
    }
}

impl<T: Pod> AsRef<BufferHandle> for Buffer<T> {
    fn as_ref(&self) -> &BufferHandle {
        &self.handle
    }
}

impl AsRef<BufferHandle> for BufferHandle {
    fn as_ref(&self) -> &BufferHandle {
        self
    }
}

/// Typed read guard.
pub struct TypedRead<'a, T: Pod> {
    mapped: MappedRead<'a>,
    _marker: PhantomData<&'a [T]>,
}

impl<T: Pod> Deref for TypedRead<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // The shadow is 16-byte aligned and whole-buffer sized, so the cast cannot fail.
        bytemuck::cast_slice(&self.mapped)
    }
}

impl<T: Pod> fmt::Debug for TypedRead<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypedRead").field("len", &self.len()).finish()
    }
}

/// Typed write guard.
pub struct TypedWrite<'a, T: Pod> {
    mapped: MappedWrite<'a>,
    _marker: PhantomData<&'a mut [T]>,
}

impl<T: Pod> Deref for TypedWrite<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        bytemuck::cast_slice(&self.mapped)
    }
}

impl<T: Pod> DerefMut for TypedWrite<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        bytemuck::cast_slice_mut(&mut self.mapped)
    }
}

impl<T: Pod> fmt::Debug for TypedWrite<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypedWrite").field("len", &self.len()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager() -> BufferManager {
        BufferManager::new()
    }

    #[test]
    fn cloned_handles_share_one_allocation_and_slots_are_reused_with_new_generations() {
        let buffers = manager();
        let handle = buffers.create(64, BufferUsage::Storage).unwrap();
        let id = handle.id();
        let copy = handle.clone();
        assert_eq!(handle, copy);
        let info = handle.info();
        assert_eq!(info.handles, 2);
        assert_eq!(info.size_bytes, 64);
        assert_eq!(info.usage, BufferUsage::Storage);
        assert_eq!(buffers.live_buffers(), 1);
        drop(handle);
        assert_eq!(copy.info().handles, 1);
        drop(copy);
        assert_eq!(buffers.live_buffers(), 0);

        let replacement = buffers.create(16, BufferUsage::Generic).unwrap();
        assert_eq!(replacement.id().slot, id.slot);
        assert_eq!(replacement.id().generation, id.generation + 1);
        let other = buffers.create(16, BufferUsage::Generic).unwrap();
        assert_ne!(other.id().slot, id.slot);
    }

    #[test]
    fn rejects_zero_and_overflowing_sizes() {
        let buffers = manager();
        assert_eq!(
            buffers.create(0, BufferUsage::Generic).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            buffers.create_typed::<u64>(u64::MAX).unwrap_err().code(),
            ErrorCode::Overflow
        );
        assert_eq!(
            buffers.create(u64::MAX, BufferUsage::Generic).unwrap_err().code(),
            ErrorCode::Overflow
        );
    }

    #[test]
    fn writable_mappings_track_dirty_ranges_and_advance_versions() {
        let buffers = manager();
        let handle = buffers.create(128, BufferUsage::Storage).unwrap();
        let v0 = handle.host_version();
        {
            let mut mapped = handle.map_write_range(24, Some(32)).unwrap();
            assert_eq!(mapped.len(), 32);
            mapped.fill(0x2a);
            assert!(handle.info().mapped);
        }
        assert!(!handle.info().mapped);
        let v1 = handle.host_version();
        assert_ne!(v0, v1);
        assert_eq!(handle.dirty_ranges(), [DirtyRange::new(24, 32)]);
        {
            let read = handle.map_read_range(20, Some(8)).unwrap();
            assert_eq!(&*read, &[0, 0, 0, 0, 0x2a, 0x2a, 0x2a, 0x2a]);
        }
        assert_eq!(handle.host_version(), v1, "read mappings keep the version");
        handle.map_write_range(10, Some(0)).unwrap();
        assert_eq!(handle.host_version(), v1, "empty write mappings keep the version");
        assert_eq!(
            handle.map_read_range(129, None).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            handle.map_read_range(48, Some(81)).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(handle.map_read_range(128, None).unwrap().len(), 0);
    }

    #[test]
    fn typed_buffers_map_elements_with_alignment() {
        let buffers = manager();
        let values = buffers.create_typed::<u32>(4).unwrap();
        assert_eq!((values.len(), values.size_bytes()), (4, 16));
        {
            let mut mapped = values.map_write().unwrap();
            mapped.copy_from_slice(&[10, 20, 30, 40]);
        }
        assert_eq!(values.read().unwrap(), [10, 20, 30, 40]);
        let mut out = [0_u32; 4];
        values.read_into(&mut out).unwrap();
        assert_eq!(out, [10, 20, 30, 40]);
        assert!(values.write(&[1, 2, 3]).is_err());
        assert!(values.read_into(&mut [0; 3]).is_err());

        // 16-byte aligned element types map without copying.
        let wide = buffers.create_typed::<u128>(3).unwrap();
        wide.write(&[1, u128::MAX, 3]).unwrap();
        assert_eq!(wide.map_read().unwrap()[1], u128::MAX);

        let untyped = buffers.create(12, BufferUsage::Storage).unwrap();
        assert_eq!(Buffer::<u32>::from_handle(untyped.clone()).unwrap().len(), 3);
        assert!(Buffer::<u64>::from_handle(untyped).is_err());
    }

    #[test]
    fn conditional_writes_only_accept_the_expected_version() {
        let buffers = manager();
        let handle = buffers.create(16, BufferUsage::Storage).unwrap();
        handle.write_bytes(&[0; 16]).unwrap();
        let current = handle.host_version();
        let stale = handle.write_bytes_if_version(&[0x55; 16], current - 1).unwrap();
        assert!(!stale.written);
        assert_eq!(stale.host_version, current);
        let accepted = handle.write_bytes_if_version(&[0x55; 16], current).unwrap();
        assert!(accepted.written);
        assert!(accepted.host_version > current);
        assert_eq!(handle.host_version(), accepted.host_version);
        assert_eq!(handle.read_bytes().unwrap(), [0x55; 16]);
        assert!(
            handle
                .write_bytes_if_version(&[0; 3], accepted.host_version)
                .is_err()
        );
    }

    #[test]
    fn versions_are_unique_across_buffers() {
        let buffers = manager();
        let a = buffers.create(4, BufferUsage::Storage).unwrap();
        let b = buffers.create(4, BufferUsage::Storage).unwrap();
        a.write_bytes(&[1; 4]).unwrap();
        b.write_bytes(&[1; 4]).unwrap();
        assert_ne!(a.host_version(), b.host_version());
    }

    #[test]
    fn device_results_park_while_mapped_and_apply_on_next_access() {
        let buffers = manager();
        let handle = buffers.create(4, BufferUsage::Storage).unwrap();
        let before = handle.host_version();
        let other = handle.clone();
        let result = {
            let mapped = handle.map_read().unwrap();
            // The completion thread must not block on the user's mapping.
            let result = std::thread::spawn(move || other.write_bytes_if_version(&[7; 4], before).unwrap())
                .join()
                .unwrap();
            assert_eq!(&*mapped, &[0; 4], "the mapping keeps its snapshot");
            result
        };
        assert!(result.written);
        assert_eq!(handle.host_version(), result.host_version);
        // Uploads see the parked bytes without applying them.
        let seen = handle
            .with_contents(|bytes, version| (bytes.to_vec(), version))
            .unwrap();
        assert_eq!(seen, (vec![7; 4], result.host_version));
        assert_eq!(handle.read_bytes().unwrap(), [7; 4]);
        assert_eq!(handle.host_version(), result.host_version);
    }

    #[test]
    fn a_host_write_supersedes_a_parked_result() {
        let buffers = manager();
        let handle = buffers.create(4, BufferUsage::Storage).unwrap();
        let before = handle.host_version();
        let worker = handle.clone();
        let parked = {
            let _mapped = handle.map_read().unwrap();
            std::thread::spawn(move || worker.write_bytes_if_version(&[7; 4], before).unwrap())
                .join()
                .unwrap()
        };
        handle.write_bytes(&[9; 4]).unwrap();
        assert_ne!(handle.host_version(), parked.host_version);
        assert_eq!(handle.read_bytes().unwrap(), [9; 4]);
    }

    #[test]
    fn same_thread_reentry_errors_instead_of_deadlocking() {
        let buffers = manager();
        let handle = buffers.create(8, BufferUsage::Storage).unwrap();
        let _mapped = handle.map_write().unwrap();
        for error in [
            handle.read_bytes().unwrap_err(),
            handle.write_bytes(&[0; 8]).unwrap_err(),
            handle.map_read().unwrap_err(),
            handle.with_contents(|_, _| ()).unwrap_err(),
        ] {
            assert_eq!(error.code(), ErrorCode::InvalidArgument);
            assert!(error.message().contains("mapped by this thread"), "{error}");
        }
        // Other threads wait for the mapping instead of failing.
        let other = handle.clone();
        let reader = std::thread::spawn(move || other.read_bytes().unwrap());
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!reader.is_finished());
        drop(_mapped);
        assert_eq!(reader.join().unwrap(), [0; 8]);
    }

    #[test]
    fn lost_contents_require_a_complete_rewrite() {
        let buffers = manager();
        let handle = buffers.create(4, BufferUsage::Storage).unwrap();
        handle.mark_data_lost();
        assert!(handle.info().data_lost);
        assert_eq!(handle.read_bytes().unwrap_err().code(), ErrorCode::DataLoss);
        assert_eq!(handle.map_read().unwrap_err().code(), ErrorCode::DataLoss);
        assert_eq!(
            handle.map_write_range(0, Some(2)).unwrap_err().code(),
            ErrorCode::DataLoss
        );
        assert_eq!(
            handle.with_contents(|_, _| ()).unwrap_err().code(),
            ErrorCode::DataLoss
        );
        handle.write_bytes(&[1, 2, 3, 4]).unwrap();
        assert_eq!(handle.read_bytes().unwrap(), [1, 2, 3, 4]);

        handle.mark_data_lost();
        {
            let mut full = handle.map_write().unwrap();
            full.copy_from_slice(&[5, 6, 7, 8]);
        }
        assert_eq!(
            handle.read_bytes().unwrap(),
            [5, 6, 7, 8],
            "a full write mapping restores"
        );
    }

    #[test]
    fn marking_data_lost_discards_parked_results() {
        let buffers = manager();
        let handle = buffers.create(4, BufferUsage::Storage).unwrap();
        let version = handle.host_version();
        let worker = handle.clone();
        {
            let _mapped = handle.map_read().unwrap();
            std::thread::spawn(move || worker.write_bytes_if_version(&[3; 4], version).unwrap())
                .join()
                .unwrap();
        }
        handle.mark_data_lost();
        assert_eq!(handle.read_bytes().unwrap_err().code(), ErrorCode::DataLoss);
    }

    #[derive(Debug, PartialEq)]
    struct Replica(u32);

    #[test]
    fn replicas_are_keyed_by_context_and_released_with_the_buffer() {
        let buffers = manager();
        let handle = buffers.create(4, BufferUsage::Storage).unwrap();
        assert!(handle.replica::<Replica>(1).is_none());
        let first = handle.replica_or_insert(1, || Ok(Replica(10))).unwrap();
        let again = handle
            .replica_or_insert(1, || -> Result<Replica> { unreachable!() })
            .unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        handle.replica_or_insert(2, || Ok(Replica(20))).unwrap();
        assert_eq!(handle.replica_owners(), [1, 2]);
        assert!(
            handle
                .replica_or_insert(3, || Err::<Replica, _>(Error::new(
                    ErrorCode::AllocationFailed,
                    "no"
                )))
                .is_err()
        );
        assert_eq!(handle.replica_owners(), [1, 2], "failed creation stores nothing");
        assert!(handle.replica::<u64>(1).is_none(), "wrong type");
        assert!(handle.replica_or_insert(1, || Ok(5_u64)).is_err());
        handle.remove_replica(1);
        assert_eq!(handle.replica_owners(), [2]);
        let weak = Arc::downgrade(&handle.replica::<Replica>(2).unwrap());
        let keepalive = handle.keepalive();
        drop(handle);
        assert!(
            weak.upgrade().is_some(),
            "an in-flight keep-alive retains replicas"
        );
        assert_eq!(buffers.live_buffers(), 1);
        drop(keepalive);
        assert!(weak.upgrade().is_none());
        assert_eq!(buffers.live_buffers(), 0);
    }

    #[test]
    fn concurrent_create_map_release_cycles_stay_consistent() {
        let buffers = manager();
        let workers: Vec<_> = (0..8)
            .map(|thread| {
                let buffers = buffers.clone();
                std::thread::spawn(move || {
                    for iteration in 0..if cfg!(miri) { 8 } else { 250_u64 } {
                        let handle = buffers
                            .create(64 + iteration, BufferUsage::TransferDestination)
                            .unwrap();
                        let mut mapped = handle.map_write().unwrap();
                        mapped[0] = (iteration & 0xff) as u8;
                        mapped[1] = thread;
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(buffers.live_buffers(), 0);
    }

    #[test]
    fn concurrent_writers_and_conditional_writers_never_tear() {
        let buffers = manager();
        let handle = buffers.create(4096, BufferUsage::Storage).unwrap();
        let writers: Vec<_> = (0..4_u8)
            .map(|value| {
                let handle = handle.clone();
                std::thread::spawn(move || {
                    for _ in 0..if cfg!(miri) { 4 } else { 200 } {
                        if value % 2 == 0 {
                            handle.write_bytes(&[value; 4096]).unwrap();
                        } else {
                            let version = handle.host_version();
                            handle.write_bytes_if_version(&[value; 4096], version).unwrap();
                        }
                        let bytes = handle.read_bytes().unwrap();
                        assert!(bytes.iter().all(|&b| b == bytes[0]), "torn write");
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }
    }
}
