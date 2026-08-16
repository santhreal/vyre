//! Target-specific sanitizer correctness failures and PMU performance models.
//!
//! Hard correctness failures (NVIDIA Compute Sanitizer, Vulkan validation,
//! data race violations, out-of-bounds accesses) are target-specific correctness
//! errors and must never be treated as performance metrics.
//!
//! Hardware PMU metrics (spills, bank conflicts, coalescing, occupancy, bandwidth)
//! belong to performance evidence and are judged against workload-specific
//! expectations (e.g. gather/scatter kernels are permitted uncoalesced traffic).

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use vyre_foundation::diagnostics::{
    Diagnostic, DiagnosticCause, DiagnosticCode, DiagnosticStage, RetryClass, Severity,
};
/// Target-specific sanitizer defect family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SanitizerKind {
    /// NVIDIA Compute Sanitizer (memcheck, racecheck, synccheck, initcheck).
    ComputeSanitizer,
    /// Vulkan Validation Layer callback error.
    VulkanValidation,
    /// Memory data race across threads/invocations.
    DataRace,
    /// Out-of-bounds buffer or pointer memory access.
    OutOfBoundsMemory,
    /// Target hardware illegal instruction or unsupported opcode.
    IllegalInstruction,
}

impl SanitizerKind {
    /// Stable diagnostic code for this defect family.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::ComputeSanitizer => "SAN001_COMPUTE_SANITIZER",
            Self::VulkanValidation => "SAN002_VULKAN_VALIDATION",
            Self::DataRace => "SAN003_DATA_RACE",
            Self::OutOfBoundsMemory => "SAN004_OUT_OF_BOUNDS",
            Self::IllegalInstruction => "SAN005_ILLEGAL_INSTRUCTION",
        }
    }

    /// Actionable fix hint for this defect family.
    #[must_use]
    pub const fn suggested_fix(self) -> &'static str {
        match self {
            Self::ComputeSanitizer => "run with compute-sanitizer --tool memcheck and inspect buffer bounds",
            Self::VulkanValidation => "inspect Vulkan descriptor set layouts and buffer usage flags",
            Self::DataRace => "insert Barrier { ordering: SeqCst } or use atomic operations with valid ordering",
            Self::OutOfBoundsMemory => "clamp index expressions or increase declared buffer count",
            Self::IllegalInstruction => "verify backend capability matrix and target ISA profile before dispatch",
        }
    }
}

/// Hard target-specific correctness failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SanitizerFailure {
    /// Family of sanitizer or hardware error.
    pub kind: SanitizerKind,
    /// Human-readable failure message.
    pub message: String,
    /// Faulting device virtual address if known.
    pub device_address: Option<u64>,
    /// Faulting thread / global invocation coordinates [x, y, z] if known.
    pub invocation_coords: Option<[u32; 3]>,
    /// Faulting instruction offset in binary if known.
    pub instruction_offset: Option<u32>,
    /// Raw diagnostic output from the hardware tool.
    pub raw_tool_output: Option<String>,
}

impl SanitizerFailure {
    /// Construct a Compute Sanitizer failure.
    #[must_use]
    pub fn compute_sanitizer(message: impl Into<String>) -> Self {
        Self {
            kind: SanitizerKind::ComputeSanitizer,
            message: message.into(),
            device_address: None,
            invocation_coords: None,
            instruction_offset: None,
            raw_tool_output: None,
        }
    }

    /// Construct a Vulkan validation failure.
    #[must_use]
    pub fn vulkan_validation(message: impl Into<String>) -> Self {
        Self {
            kind: SanitizerKind::VulkanValidation,
            message: message.into(),
            device_address: None,
            invocation_coords: None,
            instruction_offset: None,
            raw_tool_output: None,
        }
    }

    /// Construct a data race failure.
    #[must_use]
    pub fn data_race(message: impl Into<String>, address: u64, thread: [u32; 3]) -> Self {
        Self {
            kind: SanitizerKind::DataRace,
            message: message.into(),
            device_address: Some(address),
            invocation_coords: Some(thread),
            instruction_offset: None,
            raw_tool_output: None,
        }
    }

    /// Construct an out-of-bounds memory violation.
    #[must_use]
    pub fn out_of_bounds(message: impl Into<String>, address: u64) -> Self {
        Self {
            kind: SanitizerKind::OutOfBoundsMemory,
            message: message.into(),
            device_address: Some(address),
            invocation_coords: None,
            instruction_offset: None,
            raw_tool_output: None,
        }
    }

    /// Convert this hard correctness failure into a versioned structured Diagnostic.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        let mut diag = Diagnostic {
            severity: Severity::Error,
            code: DiagnosticCode::new(self.kind.code()),
            stage: DiagnosticStage::Materialize,
            message: self.message.clone().into(),
            location: None,
            suggested_fix: Some(Cow::Borrowed(self.kind.suggested_fix())),
            cause: Some(DiagnosticCause {
                kind: format!("{:?}", self.kind),
                detail: self.message.clone(),
            }),
            retry: RetryClass::RecompileSource,
            doc_url: None,
            notes: Vec::new(),
        };

        if let Some(addr) = self.device_address {
            diag.notes.push(Cow::Owned(format!("faulting device address: 0x{addr:016x}")));
        }
        if let Some([x, y, z]) = self.invocation_coords {
            diag.notes.push(Cow::Owned(format!("faulting invocation ID: [{x}, {y}, {z}]")));
        }
        if let Some(offset) = self.instruction_offset {
            diag.notes.push(Cow::Owned(format!("instruction offset: +0x{offset:04x}")));
        }
        if let Some(raw) = &self.raw_tool_output {
            diag.notes.push(Cow::Owned(format!("tool raw output: {raw}")));
        }

        diag
    }
}

/// Workload classification for PMU performance expectations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PmuWorkloadClass {
    /// Contiguous, dense array arithmetic and linear streams.
    DenseRegular,
    /// Non-contiguous indexing: gather, scatter, sparse graphs, irregular lookups.
    SparseOrGather,
    /// Cooperative workgroup reduction tree.
    ReductionTree,
}

/// Workload-specific PMU expectations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PmuExpectation {
    /// Workload family.
    pub workload: PmuWorkloadClass,
    /// Maximum allowed register spill bytes (0 for dense kernels).
    pub max_spill_bytes: u64,
    /// Maximum allowed shared memory bank conflicts.
    pub max_bank_conflicts: u64,
    /// Whether uncoalesced memory traffic is permitted for this workload.
    pub allow_uncoalesced_traffic: bool,
    /// Minimum expected warp / wave occupancy percentage.
    pub min_occupancy_pct: f64,
}

impl PmuExpectation {
    /// Standard expectations for dense, regular compute kernels.
    #[must_use]
    pub fn dense_regular() -> Self {
        Self {
            workload: PmuWorkloadClass::DenseRegular,
            max_spill_bytes: 0,
            max_bank_conflicts: 0,
            allow_uncoalesced_traffic: false,
            min_occupancy_pct: 50.0,
        }
    }

    /// Standard expectations for sparse, gather, or scatter kernels.
    #[must_use]
    pub fn sparse_or_gather() -> Self {
        Self {
            workload: PmuWorkloadClass::SparseOrGather,
            max_spill_bytes: 0,
            max_bank_conflicts: 32,
            allow_uncoalesced_traffic: true,
            min_occupancy_pct: 25.0,
        }
    }

    /// Standard expectations for reduction trees.
    #[must_use]
    pub fn reduction_tree() -> Self {
        Self {
            workload: PmuWorkloadClass::ReductionTree,
            max_spill_bytes: 0,
            max_bank_conflicts: 0,
            allow_uncoalesced_traffic: false,
            min_occupancy_pct: 50.0,
        }
    }
}

/// Measured PMU performance metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PmuMeasurement {
    /// Bytes spilled to local memory.
    pub spill_bytes: u64,
    /// Shared memory bank conflict events.
    pub bank_conflicts: u64,
    /// Uncoalesced global memory transactions.
    pub uncoalesced_transactions: u64,
    /// Measured warp / wave occupancy percentage (0.0 .. 100.0).
    pub occupancy_pct: f64,
    /// Achieved device memory bandwidth in GB/s.
    pub achieved_bandwidth_gb_s: f64,
}

/// Performance expectation discrepancy (not a correctness error).
#[derive(Debug, Clone, PartialEq)]
pub enum PmuWarning {
    /// Spill exceeded expectation.
    SpillExceeded {
        /// Observed spill in bytes.
        observed: u64,
        /// Allowed spill in bytes.
        allowed: u64,
    },
    /// Shared memory bank conflict exceeded expectation.
    BankConflictExceeded {
        /// Observed conflicts.
        observed: u64,
        /// Allowed conflicts.
        allowed: u64,
    },
    /// Uncoalesced transactions observed on dense workload.
    UncoalescedTrafficOnDenseWorkload {
        /// Observed uncoalesced transactions.
        observed: u64,
    },
    /// Occupancy below threshold.
    LowOccupancy {
        /// Observed occupancy percentage.
        observed: f64,
        /// Expected occupancy percentage.
        expected: f64,
    },
}

impl std::fmt::Display for PmuWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SpillExceeded { observed, allowed } => {
                write!(f, "register spill of {observed} bytes exceeds expectation of {allowed} bytes")
            }
            Self::BankConflictExceeded { observed, allowed } => {
                write!(f, "shared memory bank conflicts {observed} exceed expectation of {allowed}")
            }
            Self::UncoalescedTrafficOnDenseWorkload { observed } => {
                write!(f, "uncoalesced memory transactions ({observed}) observed on dense workload where coalesced access is required")
            }
            Self::LowOccupancy { observed, expected } => {
                write!(f, "measured occupancy {observed:.1}% is below expected threshold {expected:.1}%")
            }
        }
    }
}

impl std::error::Error for PmuWarning {}
impl PmuMeasurement {
    /// Evaluate measured PMU counters against workload expectations.
    pub fn evaluate(&self, expectation: &PmuExpectation) -> Vec<PmuWarning> {
        let mut warnings = Vec::new();

        if self.spill_bytes > expectation.max_spill_bytes {
            warnings.push(PmuWarning::SpillExceeded {
                observed: self.spill_bytes,
                allowed: expectation.max_spill_bytes,
            });
        }

        if self.bank_conflicts > expectation.max_bank_conflicts {
            warnings.push(PmuWarning::BankConflictExceeded {
                observed: self.bank_conflicts,
                allowed: expectation.max_bank_conflicts,
            });
        }

        if !expectation.allow_uncoalesced_traffic && self.uncoalesced_transactions > 0 {
            warnings.push(PmuWarning::UncoalescedTrafficOnDenseWorkload {
                observed: self.uncoalesced_transactions,
            });
        }

        if self.occupancy_pct < expectation.min_occupancy_pct {
            warnings.push(PmuWarning::LowOccupancy {
                observed: self.occupancy_pct,
                expected: expectation.min_occupancy_pct,
            });
        }

        warnings
    }
}
