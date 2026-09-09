//! Native kernel comparison equality conditions and invariants for BACKLOG row 47.
//!
//! BACKLOG row 47 requires:
//! "Version-pinned expert-written native kernels are compared under identical:
//! 1. semantics
//! 2. dtype
//! 3. shapes
//! 4. raggedness
//! 5. initial and final state
//! 6. target
//! 7. stream
//! 8. toolchain and flags
//! 9. clock and power state
//! 10. warmup
//! 11. interleaving
//! 12. repetitions
//! 13. cache state
//! 14. objective
//! Every one of those is recorded with the measurement, so a comparison that did not
//! hold them equal is detectable rather than assumed."

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The 14 required equality condition dimensions for native baseline comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EqualityDimension {
    /// Mathematical and numeric semantics (e.g. "exact", "fp32_ulp_tol:4").
    Semantics,
    /// Data type representation (e.g. "f32", "u32", "f16").
    Dtype,
    /// Tensor dimensions and layout extents (e.g. "[4096, 4096]").
    Shapes,
    /// Regularity versus ragged/skewed layout structure (e.g. "uniform", "ragged_power_law").
    Raggedness,
    /// Initial and final memory/buffer state contracts (e.g. "clean_unaliased").
    InitialAndFinalState,
    /// Execution target architecture (e.g. "sm_90a", "sm_89", "sm_80").
    Target,
    /// Execution queue / stream isolation (e.g. "cuda_stream_non_blocking_0").
    Stream,
    /// Compiler toolchain and compilation flags (e.g. "nvcc_12.4_-O3").
    ToolchainAndFlags,
    /// Operating frequency, clock locks, and power state (e.g. "locked_base_clock_tdp_100pct").
    ClockAndPowerState,
    /// Warmup iteration count and discard policy (e.g. "300_warmup_iterations_discarded").
    Warmup,
    /// Sample interleaving protocol (e.g. "ab_ba_round_robin_interleaving").
    Interleaving,
    /// Repetition sample count for CLT bounds (e.g. "30_measured_samples_clt").
    Repetitions,
    /// Hardware cache residency and invalidation state (e.g. "flushed_l2_between_samples").
    CacheState,
    /// Optimization objective function (e.g. "minimize_p50_latency").
    Objective,
}

impl EqualityDimension {
    /// String name of the equality dimension.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Semantics => "semantics",
            Self::Dtype => "dtype",
            Self::Shapes => "shapes",
            Self::Raggedness => "raggedness",
            Self::InitialAndFinalState => "initial_and_final_state",
            Self::Target => "target",
            Self::Stream => "stream",
            Self::ToolchainAndFlags => "toolchain_and_flags",
            Self::ClockAndPowerState => "clock_and_power_state",
            Self::Warmup => "warmup",
            Self::Interleaving => "interleaving",
            Self::Repetitions => "repetitions",
            Self::CacheState => "cache_state",
            Self::Objective => "objective",
        }
    }

    /// All 14 required equality dimensions in canonical order.
    pub const ALL: &'static [EqualityDimension] = &[
        Self::Semantics,
        Self::Dtype,
        Self::Shapes,
        Self::Raggedness,
        Self::InitialAndFinalState,
        Self::Target,
        Self::Stream,
        Self::ToolchainAndFlags,
        Self::ClockAndPowerState,
        Self::Warmup,
        Self::Interleaving,
        Self::Repetitions,
        Self::CacheState,
        Self::Objective,
    ];

    /// Derive the set of equality dimension names dynamically at runtime.
    #[must_use]
    pub fn dimension_names() -> BTreeSet<&'static str> {
        Self::ALL.iter().map(|dim| dim.as_str()).collect()
    }
}

/// The complete record of 14 comparison conditions holding between a compiler output and native kernel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeComparisonConditions {
    /// 1. Mathematical and numeric semantics contract.
    pub semantics: Option<String>,
    /// 2. Data type representation.
    pub dtype: Option<String>,
    /// 3. Tensor shapes and layout extents.
    pub shapes: Option<String>,
    /// 4. Regularity versus raggedness description.
    pub raggedness: Option<String>,
    /// 5. Initial and final memory / state invariant.
    pub initial_and_final_state: Option<String>,
    /// 6. Target device architecture / compute capability.
    pub target: Option<String>,
    /// 7. Execution stream / queue identifier.
    pub stream: Option<String>,
    /// 8. Compiler toolchain version and compilation flags.
    pub toolchain_and_flags: Option<String>,
    /// 9. Clock locking and power profile state.
    pub clock_and_power_state: Option<String>,
    /// 10. Warmup protocol description.
    pub warmup: Option<String>,
    /// 11. Interleaving sequence (e.g. ABBA round-robin).
    pub interleaving: Option<String>,
    /// 12. Measured sample repetition count.
    pub repetitions: Option<String>,
    /// 13. Cache flushing / warm-cache state.
    pub cache_state: Option<String>,
    /// 14. Target optimization objective.
    pub objective: Option<String>,
}

impl NativeComparisonConditions {
    /// Return the value of a dimension by enum.
    #[must_use]
    pub fn get(&self, dim: EqualityDimension) -> Option<&str> {
        match dim {
            EqualityDimension::Semantics => self.semantics.as_deref(),
            EqualityDimension::Dtype => self.dtype.as_deref(),
            EqualityDimension::Shapes => self.shapes.as_deref(),
            EqualityDimension::Raggedness => self.raggedness.as_deref(),
            EqualityDimension::InitialAndFinalState => self.initial_and_final_state.as_deref(),
            EqualityDimension::Target => self.target.as_deref(),
            EqualityDimension::Stream => self.stream.as_deref(),
            EqualityDimension::ToolchainAndFlags => self.toolchain_and_flags.as_deref(),
            EqualityDimension::ClockAndPowerState => self.clock_and_power_state.as_deref(),
            EqualityDimension::Warmup => self.warmup.as_deref(),
            EqualityDimension::Interleaving => self.interleaving.as_deref(),
            EqualityDimension::Repetitions => self.repetitions.as_deref(),
            EqualityDimension::CacheState => self.cache_state.as_deref(),
            EqualityDimension::Objective => self.objective.as_deref(),
        }
    }

    /// Check if all 14 required dimensions are set with non-empty values.
    #[must_use]
    pub fn unset_dimensions(&self) -> Vec<EqualityDimension> {
        EqualityDimension::ALL
            .iter()
            .copied()
            .filter(|dim| self.get(*dim).map(|v| v.trim().is_empty()).unwrap_or(true))
            .collect()
    }
}

/// Refusal generated when comparison conditions are unset or mismatched between Vyre and native baseline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EqualityConditionRefusal {
    /// A required equality dimension was unset or empty.
    Unset {
        /// Case identifier.
        case_id: String,
        /// The unset equality dimension.
        dimension: EqualityDimension,
        /// Diagnostic reason.
        reason: String,
    },
    /// An equality dimension value differs between Vyre and native baseline measurements.
    Mismatched {
        /// Case identifier.
        case_id: String,
        /// The mismatched equality dimension.
        dimension: EqualityDimension,
        /// Vyre condition value.
        vyre_value: String,
        /// Native baseline condition value.
        native_value: String,
        /// Diagnostic reason.
        reason: String,
    },
}

impl std::fmt::Display for EqualityConditionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unset { case_id, dimension, reason } => write!(
                f,
                "refused: case `{case_id}` comparison condition `{}` is unset: {reason}",
                dimension.as_str()
            ),
            Self::Mismatched { case_id, dimension, vyre_value, native_value, reason } => write!(
                f,
                "refused: case `{case_id}` comparison condition `{}` mismatch (vyre: `{vyre_value}` vs native: `{native_value}`): {reason}",
                dimension.as_str()
            ),
        }
    }
}

impl std::error::Error for EqualityConditionRefusal {}

/// Validate that all 14 required equality conditions are set and strictly equal between Vyre and the native baseline.
pub fn validate_equality_conditions(
    case_id: &str,
    vyre_cond: &NativeComparisonConditions,
    native_cond: &NativeComparisonConditions,
) -> Result<(), EqualityConditionRefusal> {
    for dim in EqualityDimension::ALL {
        let vyre_val = match vyre_cond.get(*dim) {
            Some(val) if !val.trim().is_empty() => val,
            _ => {
                return Err(EqualityConditionRefusal::Unset {
                    case_id: case_id.to_string(),
                    dimension: *dim,
                    reason: format!(
                        "Fix: Vyre measurement must explicitly record equality condition `{}`",
                        dim.as_str()
                    ),
                });
            }
        };

        let native_val = match native_cond.get(*dim) {
            Some(val) if !val.trim().is_empty() => val,
            _ => {
                return Err(EqualityConditionRefusal::Unset {
                    case_id: case_id.to_string(),
                    dimension: *dim,
                    reason: format!(
                        "Fix: native baseline measurement must explicitly record equality condition `{}`",
                        dim.as_str()
                    ),
                });
            }
        };

        if vyre_val != native_val {
            return Err(EqualityConditionRefusal::Mismatched {
                case_id: case_id.to_string(),
                dimension: *dim,
                vyre_value: vyre_val.to_string(),
                native_value: native_val.to_string(),
                reason: format!(
                    "Fix: comparisons must hold `{}` strictly identical to be valid",
                    dim.as_str()
                ),
            });
        }
    }
    Ok(())
}
