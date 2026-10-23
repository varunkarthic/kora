//! Native CPU implementations that run between GPU dispatches.

use std::sync::Arc;

use bytemuck::Pod;

use crate::buffer::{MappedRead, MappedWrite};
use crate::error::{Error, ErrorCode, Result, Status};
use crate::metadata::KernelAccess;

/// A registered CPU implementation of a kernel.
pub type CpuFallback = Arc<dyn Fn(&mut CpuTaskContext<'_>) -> Status + Send + Sync>;

pub(crate) enum Mapping<'a> {
    Read(MappedRead<'a>),
    Write(MappedWrite<'a>),
}

/// Host views of a CPU task's buffers, in declaration order. Access is checked against each
/// buffer's declared [`KernelAccess`]; writable buffers get a new host version when the task ends,
/// so later GPU dispatches upload the new bytes.
pub struct CpuTaskContext<'a> {
    pub(crate) mappings: Vec<Mapping<'a>>,
    pub(crate) accesses: Vec<KernelAccess>,
    pub(crate) push: &'a [u8],
}

impl std::fmt::Debug for CpuTaskContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuTaskContext")
            .field("accesses", &self.accesses)
            .field("push_bytes", &self.push.len())
            .finish()
    }
}

fn out_of_range() -> Error {
    Error::new(
        ErrorCode::InvalidArgument,
        "CPU task buffer index is out of range",
    )
}

fn cast_error(error: bytemuck::PodCastError) -> Error {
    Error::new(
        ErrorCode::InvalidArgument,
        format!("buffer bytes cannot be viewed as the requested type: {error}"),
    )
}

impl<'a> CpuTaskContext<'a> {
    /// Number of buffers.
    pub fn buffer_count(&self) -> usize {
        self.mappings.len()
    }

    /// Push-constant bytes.
    pub fn push_constants(&self) -> &[u8] {
        self.push
    }

    /// Push constants as a `Pod` value.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] when the size differs from `size_of::<T>()`.
    pub fn push_as<T: Pod>(&self) -> Result<T> {
        bytemuck::try_pod_read_unaligned(self.push).map_err(cast_error)
    }

    /// Reads buffer `index` (declared `Read` or `ReadWrite`).
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernelBinding`] for a write-only buffer, [`ErrorCode::InvalidArgument`]
    /// for an index out of range.
    pub fn read(&self, index: usize) -> Result<&[u8]> {
        let access = *self.accesses.get(index).ok_or_else(out_of_range)?;
        if access == KernelAccess::Write {
            return Err(Error::new(
                ErrorCode::InvalidKernelBinding,
                "CPU task requested read access to a write-only buffer",
            ));
        }
        Ok(match &self.mappings[index] {
            Mapping::Read(m) => m,
            Mapping::Write(m) => m,
        })
    }

    /// Writes buffer `index` (declared `Write` or `ReadWrite`).
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernelBinding`] for a read-only buffer, [`ErrorCode::InvalidArgument`]
    /// for an index out of range.
    pub fn write(&mut self, index: usize) -> Result<&mut [u8]> {
        let access = *self.accesses.get(index).ok_or_else(out_of_range)?;
        if access == KernelAccess::Read {
            return Err(Error::new(
                ErrorCode::InvalidKernelBinding,
                "CPU task requested write access to a read-only buffer",
            ));
        }
        match &mut self.mappings[index] {
            Mapping::Write(m) => Ok(m),
            Mapping::Read(_) => Err(Error::new(
                ErrorCode::InternalInvariantViolation,
                "writable CPU task buffer was mapped read-only",
            )),
        }
    }

    /// Reads and writes buffer `index` (declared `ReadWrite`).
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidKernelBinding`] unless the buffer is declared `ReadWrite`.
    pub fn read_write(&mut self, index: usize) -> Result<&mut [u8]> {
        let access = *self.accesses.get(index).ok_or_else(out_of_range)?;
        if access != KernelAccess::ReadWrite {
            return Err(Error::new(
                ErrorCode::InvalidKernelBinding,
                "CPU task requested read_write on a differently declared buffer",
            ));
        }
        self.write(index)
    }

    /// [`read`](Self::read) viewed as `T` elements.
    ///
    /// # Errors
    ///
    /// As [`read`](Self::read); [`ErrorCode::InvalidArgument`] when the size is not a whole
    /// number of elements.
    pub fn read_as<T: Pod>(&self, index: usize) -> Result<&[T]> {
        bytemuck::try_cast_slice(self.read(index)?).map_err(cast_error)
    }

    /// [`write`](Self::write) viewed as `T` elements.
    ///
    /// # Errors
    ///
    /// As [`write`](Self::write); [`ErrorCode::InvalidArgument`] when the size is not a whole
    /// number of elements.
    pub fn write_as<T: Pod>(&mut self, index: usize) -> Result<&mut [T]> {
        bytemuck::try_cast_slice_mut(self.write(index)?).map_err(cast_error)
    }

    /// Reads an input and writes an output at once (`input` declared readable, `output`
    /// writable, distinct indices).
    ///
    /// # Errors
    ///
    /// As [`read_as`](Self::read_as) and [`write_as`](Self::write_as);
    /// [`ErrorCode::InvalidArgument`] when the indices are equal.
    pub fn input_output<I: Pod, O: Pod>(&mut self, input: usize, output: usize) -> Result<(&[I], &mut [O])> {
        if input == output {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "input and output must be different buffers",
            ));
        }
        self.read(input)?;
        self.write(output)?;
        let (first, second) = self.mappings.split_at_mut(input.max(output));
        let (low, high) = (&mut first[input.min(output)], &mut second[0]);
        let (in_map, out_map) = if input < output { (low, high) } else { (high, low) };
        let input_bytes: &[u8] = match in_map {
            Mapping::Read(m) => m,
            Mapping::Write(m) => m,
        };
        let Mapping::Write(output_bytes) = out_map else {
            return Err(Error::new(
                ErrorCode::InternalInvariantViolation,
                "writable CPU task buffer was mapped read-only",
            ));
        };
        Ok((
            bytemuck::try_cast_slice(input_bytes).map_err(cast_error)?,
            bytemuck::try_cast_slice_mut(output_bytes).map_err(cast_error)?,
        ))
    }
}
