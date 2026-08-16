//! Release and benchmark evidence envelopes derived from a telemetry
//! snapshot: the required IO/residency metric families and the caller-owned
//! decode capacity record.

use super::ring_state::RingOccupancy;

/// Schema version for IO/runtime evidence emitted from megakernel telemetry.
pub const RUNTIME_IO_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Required metric families for runtime IO/residency evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEvidenceMetricFamily {
    /// Ring occupancy metrics are present.
    Ring,
    /// Control-buffer decode metrics are present.
    Control,
    /// Host/device copy accounting metrics are present.
    Copy,
    /// Resident device-byte metrics are present.
    Residency,
}

impl RuntimeEvidenceMetricFamily {
    /// Stable evidence-family token.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ring => "ring",
            Self::Control => "control",
            Self::Copy => "copy",
            Self::Residency => "residency",
        }
    }
}

/// Coverage bits for the required runtime evidence metric families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimeEvidenceMetricCoverage {
    /// Ring occupancy metrics are present.
    pub ring: bool,
    /// Control-buffer decode metrics are present.
    pub control: bool,
    /// Host/device copy accounting metrics are present.
    pub copy: bool,
    /// Resident device-byte metrics are present.
    pub residency: bool,
}

impl RuntimeEvidenceMetricCoverage {
    /// Coverage with every runtime evidence family present.
    #[must_use]
    pub const fn complete() -> Self {
        Self {
            ring: true,
            control: true,
            copy: true,
            residency: true,
        }
    }

    /// Missing required metric families.
    #[must_use]
    pub fn missing_families(self) -> Vec<RuntimeEvidenceMetricFamily> {
        let mut missing = Vec::new();
        if !self.ring {
            missing.push(RuntimeEvidenceMetricFamily::Ring);
        }
        if !self.control {
            missing.push(RuntimeEvidenceMetricFamily::Control);
        }
        if !self.copy {
            missing.push(RuntimeEvidenceMetricFamily::Copy);
        }
        if !self.residency {
            missing.push(RuntimeEvidenceMetricFamily::Residency);
        }
        missing
    }
}

/// Runtime-owned IO/residency evidence envelope for release and benchmark artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentRuntimeEvidence {
    /// Runtime evidence schema version.
    pub schema_version: u32,
    /// Device-resident bytes retained by the dispatch family.
    pub resident_device_bytes: u64,
    /// Host-visible copy bytes still required for this evidence sample.
    pub host_copy_bytes: u64,
    /// Host copy bytes avoided by resident handles or device-side IO.
    pub host_copy_avoided_bytes: u64,
    /// Ring occupancy snapshot.
    pub ring_occupancy: RingOccupancy,
    /// Control-buffer decode cost in nanoseconds.
    pub control_decode_ns: u64,
    /// Ring decode cost in nanoseconds.
    pub ring_decode_ns: u64,
    /// Required metric-family coverage.
    pub coverage: RuntimeEvidenceMetricCoverage,
}

impl ResidentRuntimeEvidence {
    /// Construct a complete runtime evidence envelope.
    #[must_use]
    pub const fn complete(
        resident_device_bytes: u64,
        host_copy_bytes: u64,
        host_copy_avoided_bytes: u64,
        ring_occupancy: RingOccupancy,
        control_decode_ns: u64,
        ring_decode_ns: u64,
    ) -> Self {
        Self {
            schema_version: RUNTIME_IO_EVIDENCE_SCHEMA_VERSION,
            resident_device_bytes,
            host_copy_bytes,
            host_copy_avoided_bytes,
            ring_occupancy,
            control_decode_ns,
            ring_decode_ns,
            coverage: RuntimeEvidenceMetricCoverage::complete(),
        }
    }

    /// Metric families missing from this evidence envelope.
    #[must_use]
    pub fn missing_metric_families(&self) -> Vec<RuntimeEvidenceMetricFamily> {
        self.coverage.missing_families()
    }

    /// Whether all required runtime evidence families are present.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.schema_version == RUNTIME_IO_EVIDENCE_SCHEMA_VERSION
            && self.missing_metric_families().is_empty()
    }

    /// Avoided host-copy bytes in basis points of total relevant copy volume.
    #[must_use]
    pub fn host_copy_avoidance_bps(&self) -> u16 {
        let total = u128::from(self.host_copy_bytes)
            .saturating_add(u128::from(self.host_copy_avoided_bytes));
        if total == 0 {
            return 0;
        }
        let bps = u128::from(self.host_copy_avoided_bytes).saturating_mul(10_000) / total;
        bps.min(10_000) as u16
    }
}

// Inline: `vyre_runtime::resident_work_queue::telemetry::evidence` is `private`, so no integration
// test can reach what this suite exercises.
#[cfg(test)]
mod evidence_tests {
    use super::*;

    #[test]
    fn runtime_evidence_reports_missing_metric_families() {
        let evidence = ResidentRuntimeEvidence {
            schema_version: RUNTIME_IO_EVIDENCE_SCHEMA_VERSION,
            resident_device_bytes: 0,
            host_copy_bytes: 0,
            host_copy_avoided_bytes: 0,
            ring_occupancy: RingOccupancy::default(),
            control_decode_ns: 0,
            ring_decode_ns: 0,
            coverage: RuntimeEvidenceMetricCoverage {
                ring: true,
                control: false,
                copy: true,
                residency: false,
            },
        };

        let missing = evidence
            .missing_metric_families()
            .into_iter()
            .map(RuntimeEvidenceMetricFamily::as_str)
            .collect::<Vec<_>>();

        assert_eq!(missing, vec!["control", "residency"]);
        assert!(!evidence.is_complete());
    }

    /// The record carries the snapshot it was handed, and derives the copy ratio.
    ///
    /// What the two occupancy accessors answer over a snapshot is
    /// `RingOccupancy`'s contract, proved in
    /// `tests/resident_work_queue_adversarial_overflow.rs`. Restating it here
    /// asserts the same sum twice and says nothing about the record.
    #[test]
    fn runtime_evidence_records_copy_avoidance_and_occupancy() {
        let occupancy = RingOccupancy {
            empty: 1,
            published: 2,
            claimed: 3,
            ..RingOccupancy::default()
        };
        let evidence = ResidentRuntimeEvidence::complete(4096, 1024, 3072, occupancy, 11, 13);

        assert!(evidence.is_complete());
        assert_eq!(evidence.resident_device_bytes, 4096);
        assert_eq!(evidence.ring_occupancy, occupancy);
        assert_eq!(evidence.host_copy_avoidance_bps(), 7500);
    }
}

/// Schema version for telemetry decode capacity evidence.
pub const TELEMETRY_DECODE_CAPACITY_SCHEMA_VERSION: u32 = 1;

/// Evidence that a telemetry decode used caller-owned output and scratch buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TelemetryDecodeCapacityEvidence {
    /// Evidence schema version.
    pub schema_version: u32,
    /// Number of decoded ring slots in the output snapshot.
    pub decoded_slot_count: usize,
    /// Capacity of the caller-owned ring-slot output buffer.
    pub slot_output_capacity: usize,
    /// Number of decoded route-window rows in the output snapshot.
    pub decoded_window_count: usize,
    /// Capacity of the caller-owned route-window output buffer.
    pub window_output_capacity: usize,
    /// Capacity retained for sorted window-opcode scratch.
    pub window_opcode_scratch_capacity: usize,
    /// Capacity retained for route-window accumulator scratch.
    pub window_accumulator_scratch_capacity: usize,
    /// True when evidence was produced from caller-owned scratch.
    pub uses_caller_owned_scratch: bool,
}

impl TelemetryDecodeCapacityEvidence {
    /// Return true when output and scratch capacities cover the decoded rows.
    #[must_use]
    pub fn is_complete(self) -> bool {
        self.schema_version == TELEMETRY_DECODE_CAPACITY_SCHEMA_VERSION
            && self.uses_caller_owned_scratch
            && self.slot_output_capacity >= self.decoded_slot_count
            && self.window_output_capacity >= self.decoded_window_count
            && self.window_accumulator_scratch_capacity >= self.decoded_window_count
    }
}
