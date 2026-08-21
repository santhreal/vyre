//! What a pass answers the scheduler with: whether to run, what it produced,
//! and why it declined.
//!
//! These three types are the pass-to-scheduler side of the framework contract.
//! The framework itself lives in the parent module; keeping the answer types
//! here means a change to refusal reporting touches one file.

use crate::ir_inner::model::program::Program;

/// Lightweight pass analysis result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassAnalysis {
    /// Whether the scheduler should invoke `transform`.
    pub should_run: bool,
}

impl PassAnalysis {
    /// Analysis result that asks the scheduler to run the pass.
    pub const RUN: Self = Self { should_run: true };

    /// Analysis result that asks the scheduler to skip the pass.
    pub const SKIP: Self = Self { should_run: false };

    /// Run when the predicate holds, skip when it does not.
    ///
    /// A pass decides by evaluating one condition, so the answer is that
    /// condition rather than a branch around two constants.
    #[must_use]
    pub const fn run_if(condition: bool) -> Self {
        Self {
            should_run: condition,
        }
    }
}

/// Result of one pass transformation.
#[derive(Debug, Clone, PartialEq)]
pub struct PassResult {
    /// Rewritten program.
    pub program: Program,
    /// Whether the program changed.
    pub changed: bool,
}

/// Why a pass declined to apply a transformation it would otherwise have run.
///
/// Refusal is the principled alternative to "silently emit the same program back"  -  it lets
/// the scheduler tell the user *why* a transformation was skipped (cost would go up, effect
/// lattice forbids the fusion, the wire contract would be broken). Cost-certificate-bounded
/// fusion, effect-lattice fusion, and divergence-aware barrier insertion all produce these.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RefusalReason {
    /// The pass's cost certificate predicts the rewrite would increase total cost beyond the
    /// declared monotone-down budget. The scheduler treats this as a hard refusal  -  it must
    /// not run the rewrite, even if `analyze` returned `RUN`.
    CostIncrease {
        /// Predicted cost delta (post − pre); positive means cost goes up.
        delta: i64,
        /// Free-form reason naming what increased.
        detail: &'static str,
    },
    /// The effect lattice composition rule forbids the rewrite. Surfaced when a pass would
    /// fuse two ops whose effect profiles don't compose (e.g. `Pure ∘ Diverging` without an
    /// explicit `GridSync`). Carries a suggested fix string the user can act on.
    EffectLatticeViolation {
        /// Producer `op_id` whose effect is incompatible with the consumer.
        producer: &'static str,
        /// Consumer `op_id` whose effect is incompatible with the producer.
        consumer: &'static str,
        /// Concrete fix the user can apply (insert barrier, refuse to fuse, etc.).
        suggested_fix: &'static str,
    },
    /// The pass would break the wire contract  -  `op_id` drift, Region-chain break, or any
    /// invariant the scheduler enforces by construction. The scheduler converts this into a
    /// hard error (the pass is buggy), not a soft refusal.
    WireContractViolation {
        /// Free-form description of the violation.
        detail: &'static str,
    },
    /// Catch-all refusal with a free-form reason. Use this only when none of the above fits;
    /// preferred path is to add a typed variant.
    Other {
        /// Free-form reason.
        detail: &'static str,
    },
}

impl RefusalReason {
    /// Stable kind tag for diagnostics + scheduler telemetry.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::CostIncrease { .. } => "cost_increase",
            Self::EffectLatticeViolation { .. } => "effect_lattice_violation",
            Self::WireContractViolation { .. } => "wire_contract_violation",
            Self::Other { .. } => "other",
        }
    }
}

impl std::fmt::Display for RefusalReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CostIncrease { delta, detail } => {
                write!(f, "cost_increase: delta={delta} reason={detail}")
            }
            Self::EffectLatticeViolation {
                producer,
                consumer,
                suggested_fix,
            } => write!(
                f,
                "effect_lattice_violation: producer={producer} consumer={consumer} fix={suggested_fix}"
            ),
            Self::WireContractViolation { detail } => {
                write!(f, "wire_contract_violation: {detail}")
            }
            Self::Other { detail } => write!(f, "other: {detail}"),
        }
    }
}

impl PassResult {
    /// Build a transformation result by comparing before and after programs.
    ///
    /// Change detection is a structural comparison, never a content digest. A
    /// pass engine returns a freshly built `Program` whose memos are cold, so
    /// hashing it forces a full canonical wire encode plus BLAKE3 on every pass
    /// invocation; that idiom measured as roughly half of the optimizer
    /// pipeline's instruction count. `Program`'s structural equality answers the
    /// same question: it ignores buffer declaration order for the same reason
    /// the canonical fingerprint does, short-circuits on shared `Arc` identity
    /// when an engine handed its input straight back, and stops at the first
    /// difference.
    ///
    /// `before` is taken by value so that an unchanged pass returns the program
    /// it received. That program's fingerprint, statistics and shape memos then
    /// survive into the next pass instead of being dropped with the discarded
    /// rewrite.
    #[must_use]
    #[inline]
    pub fn from_programs(before: Program, program: Program) -> Self {
        if program == before {
            Self {
                program: before,
                changed: false,
            }
        } else {
            Self {
                program,
                changed: true,
            }
        }
    }

    /// Declare the pass left the program unchanged.
    ///
    /// [`from_programs`](Self::from_programs) still pays a `Program` clone plus
    /// an O(N) structural comparison on every no-op call. A pass that has
    /// already proven it will not rewrite the program should return
    /// `PassResult::unchanged(program)` and move the input through without
    /// cloning or comparing at all.
    #[must_use]
    #[inline]
    pub fn unchanged(program: Program) -> Self {
        Self {
            program,
            changed: false,
        }
    }
}
