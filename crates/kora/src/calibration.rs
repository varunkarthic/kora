//! Startup calibration records and the cost model built on them.
//!
//! A calibration record holds per-device measurements (submission overhead, tiny-dispatch latency,
//! transfer bandwidths, streaming bandwidth, and ALU throughput). Records are bound to the device
//! UUID, driver version, and schema; any mismatch is a cache miss, never a wrong estimate.

use std::path::Path;

use crate::error::{Error, ErrorCode, Result, Status};
use crate::json::{self, Value};
use crate::version::SCHEMA_VERSION;

/// Calibration record schema.
pub const RECORD_SCHEMA: u32 = 2;
const MAXIMUM_RECORD_BYTES: u64 = 64 * 1024;

/// SPIR-V of the built-in compute probe (`shaders/calibration_probe.comp`).
pub(crate) const PROBE_SHADER: &[u8] = include_bytes!("../shaders/calibration_probe.spv");

/// Measurements for one device and driver. A zero measurement means "not measured".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationRecord {
    /// Record schema.
    pub schema: u32,
    /// Device UUID (lower-case hex).
    pub device_uuid: String,
    /// Driver version.
    pub driver_version: u32,
    /// KORA schema that wrote the record.
    pub kora_schema: u32,
    /// Empty submission round trip.
    pub submit_overhead_ns: u64,
    /// One-element dispatch round trip.
    pub tiny_dispatch_ns: u64,
    /// Host-to-device bandwidth.
    pub h2d_bytes_per_second: u64,
    /// Device-to-host bandwidth.
    pub d2h_bytes_per_second: u64,
    /// Device-to-device copy bandwidth.
    pub device_copy_bytes_per_second: u64,
    /// Kernel streaming bandwidth.
    pub streaming_bytes_per_second: u64,
    /// Kernel integer ALU throughput.
    pub alu_operations_per_second: u64,
}

impl Default for CalibrationRecord {
    fn default() -> Self {
        Self {
            schema: RECORD_SCHEMA,
            device_uuid: String::new(),
            driver_version: 0,
            kora_schema: SCHEMA_VERSION,
            submit_overhead_ns: 0,
            tiny_dispatch_ns: 0,
            h2d_bytes_per_second: 0,
            d2h_bytes_per_second: 0,
            device_copy_bytes_per_second: 0,
            streaming_bytes_per_second: 0,
            alu_operations_per_second: 0,
        }
    }
}

const MEASUREMENTS: [&str; 7] = [
    "submit_overhead_ns",
    "tiny_dispatch_ns",
    "h2d_bytes_per_second",
    "d2h_bytes_per_second",
    "device_copy_bytes_per_second",
    "streaming_bytes_per_second",
    "alu_operations_per_second",
];

impl CalibrationRecord {
    /// An empty record for a device identity.
    pub fn new(device_uuid: impl Into<String>, driver_version: u32) -> Self {
        Self {
            device_uuid: device_uuid.into(),
            driver_version,
            ..Self::default()
        }
    }

    fn measurements(&self) -> [u64; 7] {
        [
            self.submit_overhead_ns,
            self.tiny_dispatch_ns,
            self.h2d_bytes_per_second,
            self.d2h_bytes_per_second,
            self.device_copy_bytes_per_second,
            self.streaming_bytes_per_second,
            self.alu_operations_per_second,
        ]
    }

    fn measurements_mut(&mut self) -> [&mut u64; 7] {
        [
            &mut self.submit_overhead_ns,
            &mut self.tiny_dispatch_ns,
            &mut self.h2d_bytes_per_second,
            &mut self.d2h_bytes_per_second,
            &mut self.device_copy_bytes_per_second,
            &mut self.streaming_bytes_per_second,
            &mut self.alu_operations_per_second,
        ]
    }

    /// Every measurement is present.
    pub fn is_complete(&self) -> bool {
        self.measurements().iter().all(|&m| m != 0)
    }

    /// The compute probes (tiny dispatch, streaming, ALU) still need measuring.
    pub fn compute_pending(&self) -> bool {
        self.tiny_dispatch_ns == 0
            || self.streaming_bytes_per_second == 0
            || self.alu_operations_per_second == 0
    }

    /// Serializes to the on-disk JSON form.
    pub fn encode(&self) -> String {
        let mut members = vec![
            ("schema".to_owned(), Value::uint(u64::from(self.schema))),
            ("device_uuid".to_owned(), Value::string(&self.device_uuid)),
            (
                "driver_version".to_owned(),
                Value::uint(u64::from(self.driver_version)),
            ),
            ("kora_schema".to_owned(), Value::uint(u64::from(self.kora_schema))),
        ];
        members.extend(
            MEASUREMENTS
                .iter()
                .zip(self.measurements())
                .map(|(name, value)| ((*name).to_owned(), Value::uint(value))),
        );
        let mut out = String::new();
        json::write_value(&mut out, &Value::Object(members));
        out
    }

    /// Parses a record and checks it belongs to `expected_uuid` and `expected_driver` under the
    /// current schemas.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::InvalidArgument`] for malformed, foreign, or stale records.
    pub fn decode(text: &str, expected_uuid: &str, expected_driver: u32) -> Result<Self> {
        let invalid = |message: &str| Error::new(ErrorCode::InvalidArgument, message);
        if text.len() as u64 > MAXIMUM_RECORD_BYTES {
            return Err(invalid("calibration record exceeds size limit"));
        }
        let value = json::parse(text.as_bytes()).map_err(|_| invalid("calibration record is malformed"))?;
        let number = |name: &str| value.find(name).and_then(Value::as_u64);
        let identity = (
            number("schema"),
            value.find("device_uuid").and_then(Value::as_str),
            number("driver_version").and_then(|d| u32::try_from(d).ok()),
            number("kora_schema"),
        );
        let (Some(schema), Some(uuid), Some(driver), Some(kora_schema)) = identity else {
            return Err(invalid("calibration record identity is incomplete"));
        };
        if schema != u64::from(RECORD_SCHEMA)
            || kora_schema != u64::from(SCHEMA_VERSION)
            || uuid != expected_uuid
            || driver != expected_driver
        {
            return Err(invalid("calibration record identity or schema mismatch"));
        }
        let mut record = Self::new(uuid, driver);
        for (name, slot) in MEASUREMENTS.iter().zip(record.measurements_mut()) {
            *slot = number(name).ok_or_else(|| invalid("calibration record is missing a measurement"))?;
        }
        Ok(record)
    }

    /// Reads a record file; any failure is a cache miss (`None`).
    pub fn read(path: &Path, expected_uuid: &str, expected_driver: u32) -> Option<Self> {
        let bytes = crate::fsutil::read_bounded(path, MAXIMUM_RECORD_BYTES).ok()?;
        let text = std::str::from_utf8(&bytes).ok()?;
        Self::decode(text, expected_uuid, expected_driver).ok()
    }

    /// Writes the record atomically.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::IoError`] when the file cannot be written.
    pub fn write(&self, path: &Path) -> Status {
        crate::fsutil::write_atomic(path, self.encode().as_bytes())
    }
}

/// Kernel cost class for [`estimate_calibration_cost`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CalibratedKernelKind {
    /// No kernel.
    #[default]
    None,
    /// `kernel_work` tiny dispatches.
    TinyDispatch,
    /// `kernel_work` bytes streamed through a kernel.
    Streaming,
    /// `kernel_work` integer operations.
    Alu,
}

/// Workload terms for [`estimate_calibration_cost`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CalibrationCostWorkload {
    /// Host-to-device bytes.
    pub input_bytes: u64,
    /// Device-to-host bytes.
    pub output_bytes: u64,
    /// Device-to-device bytes.
    pub device_copy_bytes: u64,
    /// Kernel work in units of `kernel_kind`.
    pub kernel_work: u64,
    /// Kernel cost class.
    pub kernel_kind: CalibratedKernelKind,
}

/// Wall-time estimates in nanoseconds. Empty components mean "use your fallback model".
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CalibrationCostComponents {
    /// The record was complete and matched the device identity.
    pub complete_record: bool,
    /// Host-to-device transfer.
    pub input_transfer_ns: Option<f64>,
    /// Device-to-host transfer.
    pub output_transfer_ns: Option<f64>,
    /// Device copy.
    pub device_copy_ns: Option<f64>,
    /// Kernel execution.
    pub kernel_ns: Option<f64>,
    /// Submission round trip for work with no other terms.
    pub submit_completion_ns: Option<f64>,
}

fn time_ns(amount: u64, rate: u64, floor_ns: u64) -> f64 {
    if amount == 0 {
        return 0.0;
    }
    (amount as f64 * 1e9 / rate as f64).max(floor_ns as f64)
}

/// Converts a complete, identity-matched record into workload-sized terms. Small operations are
/// floored at the measured latency (submission overhead for transfers, tiny-dispatch time for
/// kernels), so a tiny workload is never estimated as free.
///
/// # Errors
///
/// [`ErrorCode::InvalidArgument`] when a kernel kind is not paired with nonzero work (or vice
/// versa); [`ErrorCode::Overflow`] for non-finite estimates.
pub fn estimate_calibration_cost(
    record: &CalibrationRecord,
    expected_uuid: &str,
    expected_driver: u32,
    workload: &CalibrationCostWorkload,
) -> Result<CalibrationCostComponents> {
    if (workload.kernel_kind == CalibratedKernelKind::None) != (workload.kernel_work == 0) {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "calibration cost workload must pair a kernel kind with a nonzero work amount",
        ));
    }
    let compatible = !expected_uuid.is_empty()
        && record.schema == RECORD_SCHEMA
        && record.kora_schema == SCHEMA_VERSION
        && record.device_uuid == expected_uuid
        && record.driver_version == expected_driver
        && record.is_complete();
    if !compatible {
        return Ok(CalibrationCostComponents::default());
    }
    let r = record;
    let kernel_ns = match workload.kernel_kind {
        CalibratedKernelKind::None => None,
        CalibratedKernelKind::TinyDispatch => Some(workload.kernel_work as f64 * r.tiny_dispatch_ns as f64),
        CalibratedKernelKind::Streaming => Some(time_ns(
            workload.kernel_work,
            r.streaming_bytes_per_second,
            r.tiny_dispatch_ns,
        )),
        CalibratedKernelKind::Alu => Some(time_ns(
            workload.kernel_work,
            r.alu_operations_per_second,
            r.tiny_dispatch_ns,
        )),
    };
    let only_submission = workload.input_bytes == 0
        && workload.output_bytes == 0
        && workload.device_copy_bytes == 0
        && workload.kernel_kind == CalibratedKernelKind::None;
    let result = CalibrationCostComponents {
        complete_record: true,
        input_transfer_ns: Some(time_ns(
            workload.input_bytes,
            r.h2d_bytes_per_second,
            r.submit_overhead_ns,
        )),
        output_transfer_ns: Some(time_ns(
            workload.output_bytes,
            r.d2h_bytes_per_second,
            r.submit_overhead_ns,
        )),
        device_copy_ns: Some(time_ns(
            workload.device_copy_bytes,
            r.device_copy_bytes_per_second,
            r.submit_overhead_ns,
        )),
        kernel_ns,
        submit_completion_ns: only_submission.then_some(r.submit_overhead_ns as f64),
    };
    let finite = [
        result.input_transfer_ns,
        result.output_transfer_ns,
        result.device_copy_ns,
        result.kernel_ns,
    ]
    .iter()
    .flatten()
    .all(|v| v.is_finite());
    if !finite {
        return Err(Error::new(
            ErrorCode::Overflow,
            "calibration cost estimate exceeds finite nanosecond range",
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CalibrationRecord {
        CalibrationRecord {
            submit_overhead_ns: 1234,
            tiny_dispatch_ns: 4567,
            h2d_bytes_per_second: 1_000_000,
            d2h_bytes_per_second: 2_000_000,
            device_copy_bytes_per_second: 3_000_000,
            streaming_bytes_per_second: 4_000_000,
            alu_operations_per_second: 5_000_000,
            ..CalibrationRecord::new("00112233445566778899aabbccddeeff", 17)
        }
    }

    #[test]
    fn records_round_trip_every_measurement() {
        let record = sample();
        let decoded = CalibrationRecord::decode(&record.encode(), &record.device_uuid, 17).unwrap();
        assert_eq!(decoded, record);
        assert!(decoded.is_complete());
        assert!(!decoded.compute_pending());
        let mut partial = sample();
        partial.alu_operations_per_second = 0;
        assert!(partial.compute_pending());
        assert!(!partial.is_complete());
    }

    #[test]
    fn rejects_corruption_identity_and_schema_mismatches() {
        let record = sample();
        let text = record.encode();
        let uuid = &record.device_uuid;
        assert!(CalibrationRecord::decode("{!", uuid, 17).is_err());
        assert!(CalibrationRecord::decode(&text, "ffeeddccbbaa99887766554433221100", 17).is_err());
        assert!(CalibrationRecord::decode(&text, uuid, 18).is_err());
        let newer = text.replace(
            &format!("\"schema\":{RECORD_SCHEMA}"),
            &format!("\"schema\":{}", RECORD_SCHEMA + 1),
        );
        assert!(CalibrationRecord::decode(&newer, uuid, 17).is_err());
        let other_kora = CalibrationRecord {
            kora_schema: SCHEMA_VERSION + 1,
            ..sample()
        };
        assert!(CalibrationRecord::decode(&other_kora.encode(), uuid, 17).is_err());
        let missing = text.replace("\"alu_operations_per_second\"", "\"alu\"");
        assert!(CalibrationRecord::decode(&missing, uuid, 17).is_err());
        assert!(CalibrationRecord::decode(&"x".repeat(70_000), uuid, 17).is_err());
    }

    #[test]
    fn atomic_files_round_trip_and_corrupt_files_are_misses() {
        let directory = crate::fsutil::temp_dir("calibration");
        let path = directory.join("device.json");
        let record = sample();
        record.write(&path).unwrap();
        assert_eq!(
            CalibrationRecord::read(&path, &record.device_uuid, 17),
            Some(record.clone())
        );
        let replacement = CalibrationRecord {
            submit_overhead_ns: 1334,
            ..record.clone()
        };
        replacement.write(&path).unwrap();
        assert_eq!(
            CalibrationRecord::read(&path, &record.device_uuid, 17)
                .unwrap()
                .submit_overhead_ns,
            1334
        );
        std::fs::write(&path, "not json").unwrap();
        assert_eq!(CalibrationRecord::read(&path, &record.device_uuid, 17), None);
        assert_eq!(
            CalibrationRecord::read(&directory.join("missing.json"), &record.device_uuid, 17),
            None
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn embedded_probe_shader_is_structurally_valid() {
        let words = crate::kernel::spirv::words_from_bytes(PROBE_SHADER).unwrap();
        assert_eq!(words[0], 0x0723_0203);
        crate::kernel::validate_module_structure(&words).unwrap();
    }

    fn estimate(workload: CalibrationCostWorkload) -> CalibrationCostComponents {
        let record = sample();
        estimate_calibration_cost(&record, &record.device_uuid, 17, &workload).unwrap()
    }

    #[test]
    fn maps_workload_terms_to_distinct_components() {
        let streaming = estimate(CalibrationCostWorkload {
            input_bytes: 1024,
            output_bytes: 2048,
            device_copy_bytes: 3072,
            kernel_work: 10_000,
            kernel_kind: CalibratedKernelKind::Streaming,
        });
        assert!(streaming.complete_record);
        assert_eq!(streaming.input_transfer_ns, Some(1_024_000.0));
        assert_eq!(streaming.output_transfer_ns, Some(1_024_000.0));
        assert!((streaming.device_copy_ns.unwrap() - 1_024_000.0).abs() < 1e-6);
        assert_eq!(streaming.kernel_ns, Some(2_500_000.0));
        assert_eq!(streaming.submit_completion_ns, None);

        let alu = estimate(CalibrationCostWorkload {
            kernel_work: 5000,
            kernel_kind: CalibratedKernelKind::Alu,
            ..CalibrationCostWorkload::default()
        });
        assert_eq!(alu.kernel_ns, Some(1_000_000.0));
        assert_eq!(alu.submit_completion_ns, None);

        let tiny = estimate(CalibrationCostWorkload {
            kernel_work: 3,
            kernel_kind: CalibratedKernelKind::TinyDispatch,
            ..CalibrationCostWorkload::default()
        });
        assert_eq!(tiny.kernel_ns, Some(13_701.0));

        let submit_only = estimate(CalibrationCostWorkload::default());
        assert_eq!(submit_only.submit_completion_ns, Some(1234.0));
    }

    #[test]
    fn incomplete_stale_or_foreign_records_fall_back() {
        let record = sample();
        let workload = CalibrationCostWorkload {
            input_bytes: 1000,
            ..CalibrationCostWorkload::default()
        };
        let check = |record: &CalibrationRecord, uuid: &str, driver| {
            let result = estimate_calibration_cost(record, uuid, driver, &workload).unwrap();
            assert!(!result.complete_record);
            assert_eq!(result.input_transfer_ns, None);
        };
        check(&record, "different-device", 17);
        check(&record, &record.device_uuid, 18);
        check(&record, "", 17);
        let zero = CalibrationRecord {
            streaming_bytes_per_second: 0,
            ..sample()
        };
        check(&zero, &record.device_uuid, 17);
        let stale = CalibrationRecord {
            schema: RECORD_SCHEMA - 1,
            ..sample()
        };
        check(&stale, &record.device_uuid, 17);
        let stale_kora = CalibrationRecord {
            kora_schema: SCHEMA_VERSION + 1,
            ..sample()
        };
        check(&stale_kora, &record.device_uuid, 17);
    }

    #[test]
    fn rejects_ambiguous_kernel_workloads_and_applies_latency_floors() {
        let record = sample();
        for workload in [
            CalibrationCostWorkload {
                kernel_work: 100,
                ..CalibrationCostWorkload::default()
            },
            CalibrationCostWorkload {
                kernel_kind: CalibratedKernelKind::Streaming,
                ..CalibrationCostWorkload::default()
            },
        ] {
            assert!(estimate_calibration_cost(&record, &record.device_uuid, 17, &workload).is_err());
        }
        let floored = estimate(CalibrationCostWorkload {
            input_bytes: 1,
            kernel_work: 1,
            kernel_kind: CalibratedKernelKind::Streaming,
            ..CalibrationCostWorkload::default()
        });
        assert_eq!(floored.input_transfer_ns, Some(1234.0));
        assert_eq!(floored.kernel_ns, Some(4567.0));
    }
}
