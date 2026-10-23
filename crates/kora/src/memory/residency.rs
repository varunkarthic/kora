//! Versioned per-copy validity and transfer planning for one logical allocation.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, ErrorCode, Result, Status};

/// Device identity within a residency tracker.
pub type ResidencyDeviceId = u32;

/// Derived residency state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidencyState {
    /// Only the host copy exists.
    HostOnly,
    /// Only one device copy exists.
    DeviceOnly,
    /// Several current device copies (host may be current too).
    Shared,
    /// Host and exactly one device are current.
    HostAndDeviceClean,
    /// The host is newer than existing device copies.
    HostNewer,
    /// A device is newer than the host.
    DeviceNewer,
    /// Device copies were evicted; the host is current.
    Evicted,
    /// A transfer is in flight.
    InTransfer,
    /// Terminal.
    Disposed,
}

/// Residency transition event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidencyEvent {
    /// Host wrote.
    HostWrite,
    /// A device wrote.
    DeviceWrite,
    /// A transfer started.
    BeginTransfer,
    /// A transfer to the host completed.
    CompleteToHost,
    /// A transfer to a device completed.
    CompleteToDevice,
    /// The host copy was evicted.
    EvictHost,
    /// A device copy was evicted.
    EvictDevice,
    /// The allocation was disposed.
    Dispose,
}

impl fmt::Display for ResidencyState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl fmt::Display for ResidencyEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Pure transition table; validates legality without mutation.
pub fn transition_residency(state: ResidencyState, event: ResidencyEvent) -> Result<ResidencyState> {
    use ResidencyEvent as E;
    use ResidencyState as S;
    let illegal = || {
        Err(Error::new(
            ErrorCode::InvalidArgument,
            format!("illegal residency transition: {state} + {event}"),
        ))
    };
    if event == E::Dispose {
        return if matches!(state, S::Disposed | S::InTransfer) {
            illegal()
        } else {
            Ok(S::Disposed)
        };
    }
    Ok(match (state, event) {
        (S::HostOnly, E::HostWrite) => S::HostOnly,
        (S::HostOnly, E::DeviceWrite) => S::DeviceNewer,
        (S::DeviceOnly, E::DeviceWrite) => S::DeviceOnly,
        (S::DeviceOnly, E::HostWrite) => S::HostNewer,
        (S::Shared | S::HostAndDeviceClean, E::HostWrite) => S::HostNewer,
        (S::Shared | S::HostAndDeviceClean, E::DeviceWrite) => S::DeviceNewer,
        (S::Shared | S::HostAndDeviceClean, E::EvictHost) => S::DeviceOnly,
        (S::Shared | S::HostAndDeviceClean, E::EvictDevice) => S::Evicted,
        (S::HostNewer, E::HostWrite) => S::HostNewer,
        (S::HostNewer, E::DeviceWrite) => S::DeviceNewer,
        (S::HostNewer, E::EvictDevice) => S::Evicted,
        (S::DeviceNewer, E::HostWrite) => S::HostNewer,
        (S::DeviceNewer, E::DeviceWrite) => S::DeviceNewer,
        (S::DeviceNewer, E::EvictHost) => S::DeviceOnly,
        (S::DeviceNewer, E::EvictDevice) => S::DeviceNewer,
        (S::Evicted, E::HostWrite) => S::Evicted,
        (S::Evicted, E::DeviceWrite) => S::DeviceNewer,
        (
            S::HostOnly
            | S::DeviceOnly
            | S::Shared
            | S::HostAndDeviceClean
            | S::HostNewer
            | S::DeviceNewer
            | S::Evicted,
            E::BeginTransfer,
        ) => S::InTransfer,
        (S::InTransfer, E::CompleteToHost) => S::HostAndDeviceClean,
        (S::InTransfer, E::CompleteToDevice) => S::Shared,
        _ => return illegal(),
    })
}

/// A copy location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidencyLocation {
    /// The host shadow.
    Host,
    /// A device replica.
    Device(ResidencyDeviceId),
}

/// A planned transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferPlan {
    /// A copy is needed.
    pub required: bool,
    /// Copy source.
    pub source: ResidencyLocation,
    /// Copy destination.
    pub destination: ResidencyLocation,
    /// Content version being moved.
    pub version: u64,
}

/// Snapshot of a tracker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencySnapshot {
    /// State.
    pub state: ResidencyState,
    /// Current content version.
    pub current_version: u64,
    /// Host copy is current.
    pub host_current: bool,
    /// Per device: copy is current.
    pub device_current: BTreeMap<ResidencyDeviceId, bool>,
    /// Disposed.
    pub disposed: bool,
}

/// Versioned per-copy validity and transfer planner for one logical allocation.
#[derive(Debug, Clone)]
pub struct ResidencyTracker {
    current_version: u64,
    host_version: Option<u64>,
    device_versions: BTreeMap<ResidencyDeviceId, u64>,
    state: ResidencyState,
    ever_had_device_copy: bool,
    disposed: bool,
    pending: Option<(TransferPlan, ResidencyState)>,
}

impl ResidencyTracker {
    /// Data initialized on the host.
    pub fn host_initialized() -> Self {
        Self {
            current_version: 1,
            host_version: Some(1),
            device_versions: BTreeMap::new(),
            state: ResidencyState::HostOnly,
            ever_had_device_copy: false,
            disposed: false,
            pending: None,
        }
    }

    /// Data initialized on `device`.
    pub fn device_initialized(device: ResidencyDeviceId) -> Self {
        let mut tracker = Self::host_initialized();
        tracker.host_version = None;
        tracker.device_versions.insert(device, 1);
        tracker.ever_had_device_copy = true;
        tracker.state = ResidencyState::DeviceOnly;
        tracker
    }

    /// State.
    pub fn state(&self) -> ResidencyState {
        self.state
    }

    /// Current version.
    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    /// Snapshot.
    pub fn snapshot(&self) -> ResidencySnapshot {
        ResidencySnapshot {
            state: self.state,
            current_version: self.current_version,
            host_current: self.host_version == Some(self.current_version),
            device_current: self
                .device_versions
                .iter()
                .map(|(&d, &v)| (d, v == self.current_version))
                .collect(),
            disposed: self.disposed,
        }
    }

    fn current(&self, location: ResidencyLocation) -> bool {
        match location {
            ResidencyLocation::Host => self.host_version == Some(self.current_version),
            ResidencyLocation::Device(device) => {
                self.device_versions.get(&device) == Some(&self.current_version)
            }
        }
    }

    fn current_devices(&self) -> usize {
        self.device_versions
            .values()
            .filter(|&&v| v == self.current_version)
            .count()
    }

    fn derive_state(&self) -> ResidencyState {
        if self.disposed {
            return ResidencyState::Disposed;
        }
        if self.pending.is_some() {
            return ResidencyState::InTransfer;
        }
        let host_current = self.current(ResidencyLocation::Host);
        let devices = self.current_devices();
        if devices > 1 {
            ResidencyState::Shared
        } else if host_current && devices == 1 {
            ResidencyState::HostAndDeviceClean
        } else if host_current {
            if !self.device_versions.is_empty() {
                ResidencyState::HostNewer
            } else if self.ever_had_device_copy {
                ResidencyState::Evicted
            } else {
                ResidencyState::HostOnly
            }
        } else if devices > 0 {
            if self.host_version.is_some() {
                ResidencyState::DeviceNewer
            } else {
                ResidencyState::DeviceOnly
            }
        } else {
            // An invariant violation is represented as terminal, never as valid data.
            ResidencyState::Disposed
        }
    }

    fn advance_version(&mut self) -> Status {
        self.current_version = self
            .current_version
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorCode::Overflow, "residency content version overflow"))?;
        Ok(())
    }

    /// Records a host write.
    pub fn write_host(&mut self) -> Status {
        transition_residency(self.state, ResidencyEvent::HostWrite)?;
        self.advance_version()?;
        self.host_version = Some(self.current_version);
        self.state = self.derive_state();
        Ok(())
    }

    /// Records a device write.
    pub fn write_device(&mut self, device: ResidencyDeviceId) -> Status {
        transition_residency(self.state, ResidencyEvent::DeviceWrite)?;
        self.advance_version()?;
        self.device_versions.insert(device, self.current_version);
        self.ever_had_device_copy = true;
        self.state = self.derive_state();
        Ok(())
    }

    /// Plans a copy to `destination` (not required when it is already current). Device sources are
    /// chosen deterministically (lowest device ID).
    pub fn plan_transfer(&self, destination: ResidencyLocation) -> Result<TransferPlan> {
        if self.disposed {
            return Err(Error::new(
                ErrorCode::StaleHandle,
                "disposed residency has no current copy to transfer",
            ));
        }
        if self.pending.is_some() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "cannot plan another transfer while one is in flight",
            ));
        }
        let plan = |required, source| TransferPlan {
            required,
            source,
            destination,
            version: self.current_version,
        };
        if self.current(destination) {
            return Ok(plan(false, destination));
        }
        let source = if self.current(ResidencyLocation::Host) {
            ResidencyLocation::Host
        } else {
            self.device_versions
                .iter()
                .find(|&(&device, &version)| {
                    version == self.current_version && destination != ResidencyLocation::Device(device)
                })
                .map(|(&device, _)| ResidencyLocation::Device(device))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InternalInvariantViolation,
                        "no current residency copy exists",
                    )
                })?
        };
        Ok(plan(source != destination, source))
    }

    /// Starts a planned transfer (rejecting stale plans).
    pub fn begin_transfer(&mut self, plan: &TransferPlan) -> Status {
        if !plan.required {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "cannot begin an unnecessary residency transfer",
            ));
        }
        if self.plan_transfer(plan.destination)? != *plan {
            return Err(Error::new(
                ErrorCode::StaleHandle,
                "residency transfer plan is stale",
            ));
        }
        let next = transition_residency(self.state, ResidencyEvent::BeginTransfer)?;
        self.pending = Some((*plan, self.state));
        self.state = next;
        Ok(())
    }

    /// Completes the in-flight transfer.
    pub fn complete_transfer(&mut self) -> Status {
        let Some((plan, _)) = self.pending else {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "no residency transfer is in flight",
            ));
        };
        let event = if plan.destination == ResidencyLocation::Host {
            ResidencyEvent::CompleteToHost
        } else {
            ResidencyEvent::CompleteToDevice
        };
        transition_residency(self.state, event)?;
        if plan.version != self.current_version || !self.current(plan.source) {
            return Err(Error::new(
                ErrorCode::StaleHandle,
                "residency transfer source changed before completion",
            ));
        }
        match plan.destination {
            ResidencyLocation::Host => self.host_version = Some(plan.version),
            ResidencyLocation::Device(device) => {
                self.device_versions.insert(device, plan.version);
                self.ever_had_device_copy = true;
            }
        }
        self.pending = None;
        self.state = self.derive_state();
        Ok(())
    }

    /// Abandons the in-flight transfer, restoring the prior state.
    pub fn fail_transfer(&mut self) -> Status {
        let Some((_, previous)) = self.pending.take() else {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "no residency transfer is in flight",
            ));
        };
        self.state = previous;
        Ok(())
    }

    /// Drops the host copy (requires a current device copy).
    pub fn evict_host(&mut self) -> Status {
        transition_residency(self.state, ResidencyEvent::EvictHost)?;
        if self.current_devices() == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "cannot evict the last current copy of a buffer",
            ));
        }
        self.host_version = None;
        self.state = self.derive_state();
        Ok(())
    }

    /// Drops a device copy (never the last current copy).
    pub fn evict_device(&mut self, device: ResidencyDeviceId) -> Status {
        if self.disposed || self.pending.is_some() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "cannot evict a copy in this residency state",
            ));
        }
        let Some(&version) = self.device_versions.get(&device) else {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "device copy does not exist",
            ));
        };
        if version == self.current_version
            && !self.current(ResidencyLocation::Host)
            && self.current_devices() < 2
        {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "cannot evict the last current copy of a buffer",
            ));
        }
        self.device_versions.remove(&device);
        self.state = self.derive_state();
        Ok(())
    }

    /// Terminal disposal.
    pub fn dispose(&mut self) -> Status {
        transition_residency(self.state, ResidencyEvent::Dispose)?;
        self.host_version = None;
        self.device_versions.clear();
        self.pending = None;
        self.disposed = true;
        self.state = ResidencyState::Disposed;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task_graph::tests::SplitMix;
    use ResidencyEvent as E;
    use ResidencyLocation::{Device, Host};
    use ResidencyState as S;

    const CASES: [(S, E, S); 30] = [
        (S::HostOnly, E::HostWrite, S::HostOnly),
        (S::HostOnly, E::DeviceWrite, S::DeviceNewer),
        (S::HostOnly, E::BeginTransfer, S::InTransfer),
        (S::DeviceOnly, E::HostWrite, S::HostNewer),
        (S::DeviceOnly, E::DeviceWrite, S::DeviceOnly),
        (S::DeviceOnly, E::BeginTransfer, S::InTransfer),
        (S::Shared, E::HostWrite, S::HostNewer),
        (S::Shared, E::DeviceWrite, S::DeviceNewer),
        (S::Shared, E::BeginTransfer, S::InTransfer),
        (S::Shared, E::EvictHost, S::DeviceOnly),
        (S::Shared, E::EvictDevice, S::Evicted),
        (S::HostAndDeviceClean, E::HostWrite, S::HostNewer),
        (S::HostAndDeviceClean, E::DeviceWrite, S::DeviceNewer),
        (S::HostAndDeviceClean, E::BeginTransfer, S::InTransfer),
        (S::HostAndDeviceClean, E::EvictHost, S::DeviceOnly),
        (S::HostAndDeviceClean, E::EvictDevice, S::Evicted),
        (S::HostNewer, E::HostWrite, S::HostNewer),
        (S::HostNewer, E::DeviceWrite, S::DeviceNewer),
        (S::HostNewer, E::BeginTransfer, S::InTransfer),
        (S::HostNewer, E::EvictDevice, S::Evicted),
        (S::DeviceNewer, E::HostWrite, S::HostNewer),
        (S::DeviceNewer, E::DeviceWrite, S::DeviceNewer),
        (S::DeviceNewer, E::BeginTransfer, S::InTransfer),
        (S::DeviceNewer, E::EvictHost, S::DeviceOnly),
        (S::DeviceNewer, E::EvictDevice, S::DeviceNewer),
        (S::Evicted, E::HostWrite, S::Evicted),
        (S::Evicted, E::DeviceWrite, S::DeviceNewer),
        (S::Evicted, E::BeginTransfer, S::InTransfer),
        (S::InTransfer, E::CompleteToHost, S::HostAndDeviceClean),
        (S::InTransfer, E::CompleteToDevice, S::Shared),
    ];
    const STATES: [S; 9] = [
        S::HostOnly,
        S::DeviceOnly,
        S::Shared,
        S::HostAndDeviceClean,
        S::HostNewer,
        S::DeviceNewer,
        S::Evicted,
        S::InTransfer,
        S::Disposed,
    ];
    const EVENTS: [E; 8] = [
        E::HostWrite,
        E::DeviceWrite,
        E::BeginTransfer,
        E::CompleteToHost,
        E::CompleteToDevice,
        E::EvictHost,
        E::EvictDevice,
        E::Dispose,
    ];

    fn assert_current_copy_invariant(tracker: &ResidencyTracker) {
        let snapshot = tracker.snapshot();
        if snapshot.disposed {
            assert_eq!(snapshot.state, S::Disposed);
            assert!(!snapshot.host_current && snapshot.device_current.values().all(|c| !c));
        } else {
            assert!(snapshot.host_current || snapshot.device_current.values().any(|&c| c));
        }
    }

    fn transfer(tracker: &mut ResidencyTracker, destination: ResidencyLocation) -> TransferPlan {
        let plan = tracker.plan_transfer(destination).unwrap();
        assert!(plan.required);
        tracker.begin_transfer(&plan).unwrap();
        assert_eq!(tracker.state(), S::InTransfer);
        tracker.complete_transfer().unwrap();
        plan
    }

    #[test]
    fn exhaustive_table_accepts_only_documented_transitions() {
        for state in STATES {
            for event in EVENTS {
                let actual = transition_residency(state, event);
                match CASES.iter().find(|(s, e, _)| *s == state && *e == event) {
                    Some((_, _, to)) => assert_eq!(actual.unwrap(), *to, "{state} + {event}"),
                    None if event == E::Dispose && !matches!(state, S::InTransfer | S::Disposed) => {
                        assert_eq!(actual.unwrap(), S::Disposed)
                    }
                    None => assert_eq!(
                        actual.unwrap_err().code(),
                        ErrorCode::InvalidArgument,
                        "{state} + {event}"
                    ),
                }
            }
        }
    }

    #[test]
    fn plans_only_for_absent_or_stale_destinations() {
        let mut tracker = ResidencyTracker::host_initialized();
        assert_eq!(transfer(&mut tracker, Device(3)).source, Host);
        assert_eq!(tracker.state(), S::HostAndDeviceClean);
        assert!(!tracker.plan_transfer(Device(3)).unwrap().required);
        transfer(&mut tracker, Device(8));
        assert_eq!(tracker.state(), S::Shared);
        tracker.write_host().unwrap();
        assert_eq!(tracker.state(), S::HostNewer);
        assert_eq!(transfer(&mut tracker, Device(3)).source, Host);
        assert_eq!(tracker.state(), S::HostAndDeviceClean);
        assert_current_copy_invariant(&tracker);
    }

    #[test]
    fn device_writes_read_back_and_failed_transfers_preserve_current_copy() {
        let mut tracker = ResidencyTracker::device_initialized(5);
        tracker.write_host().unwrap();
        assert_eq!(tracker.state(), S::HostNewer);
        tracker.write_device(5).unwrap();
        assert_eq!(tracker.state(), S::DeviceNewer);
        let readback = tracker.plan_transfer(Host).unwrap();
        assert!(readback.required && readback.source != Host);
        tracker.begin_transfer(&readback).unwrap();
        tracker.fail_transfer().unwrap();
        assert_eq!(tracker.state(), S::DeviceNewer);
        assert_current_copy_invariant(&tracker);
        transfer(&mut tracker, Host);
        assert_eq!(tracker.state(), S::HostAndDeviceClean);
        let snapshot = tracker.snapshot();
        assert!(snapshot.host_current && snapshot.device_current[&5]);
    }

    #[test]
    fn multi_device_copies_remain_versioned_and_last_copy_cannot_be_evicted() {
        let mut tracker = ResidencyTracker::host_initialized();
        transfer(&mut tracker, Device(1));
        transfer(&mut tracker, Device(2));
        assert_eq!(tracker.state(), S::Shared);
        tracker.write_device(1).unwrap();
        assert_eq!(tracker.state(), S::DeviceNewer);
        let peer = tracker.plan_transfer(Device(2)).unwrap();
        assert!(peer.required);
        assert_eq!(peer.source, Device(1));
        assert!(tracker.evict_device(1).is_err());
        tracker.begin_transfer(&peer).unwrap();
        tracker.complete_transfer().unwrap();
        assert_eq!(tracker.state(), S::Shared);
        tracker.evict_host().unwrap();
        assert_eq!(tracker.state(), S::Shared);
        tracker.evict_device(1).unwrap();
        assert_eq!(tracker.state(), S::DeviceOnly);
        assert!(tracker.evict_device(2).is_err());
        assert_current_copy_invariant(&tracker);
    }

    #[test]
    fn eviction_and_disposal_preserve_the_current_copy_invariant() {
        let mut tracker = ResidencyTracker::host_initialized();
        transfer(&mut tracker, Device(7));
        tracker.evict_host().unwrap();
        assert_eq!(tracker.state(), S::DeviceOnly);
        assert!(tracker.evict_device(7).is_err());
        transfer(&mut tracker, Host);
        tracker.evict_device(7).unwrap();
        assert_eq!(tracker.state(), S::Evicted);
        tracker.write_host().unwrap();
        assert_eq!(tracker.state(), S::Evicted);
        tracker.dispose().unwrap();
        assert_eq!(tracker.state(), S::Disposed);
        assert_current_copy_invariant(&tracker);
    }

    #[test]
    fn deterministic_operation_sequence_always_retains_current_data() {
        let mut tracker = ResidencyTracker::host_initialized();
        let mut random = SplitMix(0x26A11);
        for _ in 0..1000 {
            let device = random.below(4) as u32;
            match random.below(5) {
                0 => drop(tracker.write_host()),
                1 => drop(tracker.write_device(device)),
                2 => {
                    let destination = if random.below(2) == 1 {
                        Host
                    } else {
                        Device(device)
                    };
                    if let Ok(plan) = tracker.plan_transfer(destination)
                        && plan.required
                        && tracker.begin_transfer(&plan).is_ok()
                    {
                        tracker.complete_transfer().unwrap();
                    }
                }
                3 => drop(tracker.evict_host()),
                _ => drop(tracker.evict_device(device)),
            }
            assert_current_copy_invariant(&tracker);
        }
    }

    #[test]
    fn rejects_stale_plans_concurrent_transfers_and_disposed_copies() {
        let mut tracker = ResidencyTracker::host_initialized();
        let stale = tracker.plan_transfer(Device(4)).unwrap();
        tracker.write_host().unwrap();
        assert_eq!(
            tracker.begin_transfer(&stale).unwrap_err().code(),
            ErrorCode::StaleHandle
        );
        let plan = tracker.plan_transfer(Device(4)).unwrap();
        tracker.begin_transfer(&plan).unwrap();
        assert_eq!(
            tracker.plan_transfer(Host).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(tracker.dispose().unwrap_err().code(), ErrorCode::InvalidArgument);
        tracker.fail_transfer().unwrap();
        tracker.dispose().unwrap();
        assert_eq!(
            tracker.plan_transfer(Host).unwrap_err().code(),
            ErrorCode::StaleHandle
        );
        assert_eq!(
            tracker.write_host().unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert!(tracker.complete_transfer().is_err());
        assert!(tracker.fail_transfer().is_err());
    }
}
