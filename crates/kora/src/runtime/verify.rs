//! Duplicate verification: run a dispatch on several devices and compare the outputs.

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use super::context::{DeviceContext, lock};
use super::dispatch::DispatchBuilder;
use super::kernel::Kernel;
use super::replica::{DeviceAccess, Replica};
use crate::buffer::BufferHandle;
use crate::error::{Error, ErrorCode, Result, Status};
use crate::event::{Event, EventPromise};
use crate::verification::{VerificationDataType, VerificationPolicy};

type Snapshots = Arc<Mutex<Vec<Vec<Option<Vec<u8>>>>>>;

struct Verification {
    targets: Vec<(Arc<DeviceContext>, Kernel)>,
    buffers: Vec<BufferHandle>,
    outputs: Vec<usize>,
    host_versions: Vec<u64>,
    policy: VerificationPolicy,
    data_type: VerificationDataType,
    promise: EventPromise,
}

impl Verification {
    fn fail(self, error: Error) {
        for &index in &self.outputs {
            self.buffers[index].mark_data_lost();
        }
        self.promise.complete(Err(error));
    }
}

pub(super) fn submit_verified(builder: DispatchBuilder) -> Result<Event> {
    let (policy, data_type) = builder.verification.clone().ok_or_else(|| {
        Error::new(
            ErrorCode::InternalInvariantViolation,
            "verified dispatch has no policy",
        )
    })?;
    policy.validate()?;
    let metadata = &builder.kernel.state.metadata;
    if builder.bindings.len() != metadata.bindings.len() {
        return Err(Error::new(
            ErrorCode::InvalidKernelBinding,
            "dispatch buffer count does not match kernel metadata",
        ));
    }
    let mut seen = HashSet::new();
    let outputs: Vec<usize> = (0..builder.bindings.len())
        .filter(|&i| builder.bindings[i].1.writes() && seen.insert(builder.bindings[i].0.id()))
        .collect();
    if outputs.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "verified dispatch requires at least one writable output",
        ));
    }
    let element = match data_type {
        VerificationDataType::Float32 => 4,
        VerificationDataType::Float64 => 8,
    };
    if !matches!(policy, VerificationPolicy::Custom(_))
        && outputs
            .iter()
            .any(|&i| !builder.bindings[i].0.size_bytes().is_multiple_of(element))
    {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "verified output size is not a multiple of the selected floating-point type",
        ));
    }
    let mut targets = Vec::new();
    for peer in builder.peers.iter() {
        if peer.quarantined.load(Ordering::Acquire) {
            continue;
        }
        if !peer.accepting.load(Ordering::Acquire) {
            return Err(Error::new(
                ErrorCode::Cancelled,
                "a verification target is shutting down",
            ));
        }
        // Devices that cannot run the kernel simply do not participate.
        if let Ok(kernel) = builder.kernel.on(peer) {
            targets.push((Arc::clone(peer), kernel));
        }
    }
    if targets.len() < 2 {
        return Err(Error::new(
            ErrorCode::UnsupportedFeature,
            "duplicate verification requires at least two initialized, non-quarantined devices that support the kernel",
        ));
    }
    let buffers: Vec<BufferHandle> = builder.bindings.iter().map(|(b, _)| b.clone()).collect();
    let host_versions: Vec<u64> = buffers.iter().map(BufferHandle::host_version).collect();
    let mut events = Vec::with_capacity(targets.len());
    for (_, kernel) in &targets {
        let duplicate = DispatchBuilder {
            kernel: kernel.clone(),
            verification: None,
            adaptive: None,
            defer_readback: true,
            duplicate: true,
            ..builder.clone()
        };
        match duplicate.submit() {
            Ok(event) => events.push(event),
            Err(error) => {
                // Duplicates already enqueued finish on their own; their unverified results are
                // never published, and the outputs are marked lost.
                for &index in &outputs {
                    buffers[index].mark_data_lost();
                }
                return Err(error);
            }
        }
    }
    let promise = EventPromise::new();
    let event = promise.event();
    let verification = Verification {
        targets,
        buffers,
        outputs,
        host_versions,
        policy,
        data_type,
        promise,
    };
    let all = Event::when_all(&events)?;
    all.on_complete(move |status| match status {
        Ok(()) => download(verification),
        Err(error) => verification.fail(error),
    })?;
    Ok(event)
}

/// Reads every output back from every participant.
fn download(verification: Verification) {
    let targets = verification.targets.len();
    let outputs = verification.outputs.len();
    let snapshots: Snapshots = Arc::new(Mutex::new(vec![vec![None; outputs]; targets]));
    let mut events = Vec::with_capacity(targets * outputs);
    for (t, (ctx, _)) in verification.targets.iter().enumerate() {
        for (o, &index) in verification.outputs.iter().enumerate() {
            let buffer = &verification.buffers[index];
            let Some(replica) = buffer.replica::<Replica>(ctx.key) else {
                verification.fail(Error::new(
                    ErrorCode::InternalInvariantViolation,
                    "a verification participant has no copy of an output",
                ));
                return;
            };
            let sink = {
                let snapshots = Arc::clone(&snapshots);
                let ctx = Arc::clone(ctx);
                move |bytes: &[u8]| -> Status {
                    ctx.downloads.fetch_add(1, Ordering::Relaxed);
                    lock(&snapshots)[t][o] = Some(bytes.to_vec());
                    Ok(())
                }
            };
            let enqueued = {
                let _order = lock(&ctx.submit_lock);
                let result = ctx
                    .transfers
                    .download(&replica.transfer_view(), 0, replica.size(), None, sink);
                if result.is_ok() {
                    let mut state = replica.state();
                    if state.access == DeviceAccess::None {
                        state.access = DeviceAccess::Read;
                    }
                }
                result
            };
            match enqueued {
                Ok(event) => events.push(event),
                Err(error) => {
                    verification.fail(error);
                    return;
                }
            }
        }
    }
    match Event::when_all(&events) {
        Ok(all) => {
            let registered = all.on_complete(move |status| match status {
                Ok(()) => compare(verification, &snapshots),
                Err(error) => verification.fail(error),
            });
            drop(registered);
        }
        Err(error) => verification.fail(error),
    }
}

/// Compares the participants' outputs and publishes or quarantines.
fn compare(verification: Verification, snapshots: &Snapshots) {
    let snapshots = std::mem::take(&mut *lock(snapshots));
    for (o, reference) in snapshots[0].iter().enumerate() {
        let Some(reference) = reference else {
            verification.fail(Error::new(
                ErrorCode::InternalInvariantViolation,
                "a verification download produced no bytes",
            ));
            return;
        };
        for other in &snapshots[1..] {
            let Some(other) = other[o].as_ref() else {
                verification.fail(Error::new(
                    ErrorCode::InternalInvariantViolation,
                    "a verification download produced no bytes",
                ));
                return;
            };
            match verification
                .policy
                .compare_bytes(reference, other, verification.data_type)
            {
                Ok(true) => {}
                Ok(false) => {
                    for (ctx, _) in &verification.targets {
                        ctx.quarantined.store(true, Ordering::Release);
                    }
                    verification.fail(Error::new(
                        ErrorCode::ValidationError,
                        "duplicate dispatch outputs differ; all participants quarantined",
                    ));
                    return;
                }
                Err(error) => {
                    verification.fail(error);
                    return;
                }
            }
        }
    }
    for (o, &index) in verification.outputs.iter().enumerate() {
        let buffer = &verification.buffers[index];
        let reference = snapshots[0][o].as_deref().unwrap_or_default();
        match buffer.write_bytes_if_version(reference, verification.host_versions[index]) {
            Ok(written) if written.written => {
                for (ctx, _) in &verification.targets {
                    if let Some(replica) = buffer.replica::<Replica>(ctx.key) {
                        let mut state = replica.state();
                        state.resident_version = Some(written.host_version);
                        state.device_newer = false;
                    }
                }
            }
            Ok(_) => {
                verification.promise.complete(Err(Error::new(
                    ErrorCode::StaleHandle,
                    "host buffer changed during verified dispatch",
                )));
                return;
            }
            Err(error) => {
                verification.fail(error);
                return;
            }
        }
    }
    verification.promise.complete(Ok(()));
}
