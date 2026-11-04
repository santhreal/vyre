//! Compiler level and lifecycle phases for causal spans.

use core::fmt;
use serde::{Deserialize, Serialize};

/// The six canonical compiler and runtime lifecycle phases.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalPhase {
    /// Level 0: Foundation IR, Semantic SSA, Logical ProgramGraph, and Region construction.
    SemanticFrontend,
    /// Level 1: Semantic optimization, pass scheduling, rewrite rules, and loop analysis.
    SemanticOptimizer,
    /// Level 2: Target-neutral physical kernel lowering and schedule legality.
    TargetLowering,
    /// Level 3: Megakernel compilation, candidate search, funnel ranking, and emission.
    MegakernelCompilation,
    /// Level 4: Backend driver artifact admission, residency binding, and queue submission.
    DriverSubmission,
    /// Level 5: Runtime session execution, hardware synchronization, and completion.
    RuntimeExecution,
}

impl CausalPhase {
    /// All canonical phases in topological execution order.
    pub const ALL: [Self; 6] = [
        Self::SemanticFrontend,
        Self::SemanticOptimizer,
        Self::TargetLowering,
        Self::MegakernelCompilation,
        Self::DriverSubmission,
        Self::RuntimeExecution,
    ];

    /// Numeric tier (0 to 5) corresponding to compiler level.
    pub const fn level_tier(self) -> u8 {
        match self {
            Self::SemanticFrontend => 0,
            Self::SemanticOptimizer => 1,
            Self::TargetLowering => 2,
            Self::MegakernelCompilation => 3,
            Self::DriverSubmission => 4,
            Self::RuntimeExecution => 5,
        }
    }

    /// Human-readable label.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SemanticFrontend => "semantic_frontend",
            Self::SemanticOptimizer => "semantic_optimizer",
            Self::TargetLowering => "target_lowering",
            Self::MegakernelCompilation => "megakernel_compilation",
            Self::DriverSubmission => "driver_submission",
            Self::RuntimeExecution => "runtime_execution",
        }
    }
}

impl fmt::Display for CausalPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Exhaustiveness verification ensuring every enum variant is accounted for.
#[inline]
pub fn exhaustiveness_check_causal_phase(phase: CausalPhase) -> u8 {
    match phase {
        CausalPhase::SemanticFrontend => 0,
        CausalPhase::SemanticOptimizer => 1,
        CausalPhase::TargetLowering => 2,
        CausalPhase::MegakernelCompilation => 3,
        CausalPhase::DriverSubmission => 4,
        CausalPhase::RuntimeExecution => 5,
    }
}
