//! What a lens reports after running against one op.

/// Outcome of running one lens against one op.
#[derive(Debug)]
pub enum LensOutcome {
    /// Lens passed  -  op output matched the oracle for every case.
    Pass {
        /// Number of input cases that were compared.
        cases: usize,
    },
    /// Lens failed  -  op diverged from the oracle on the referenced case.
    Fail {
        /// Zero-based case index of the first divergence.
        case_index: usize,
        /// Rendered failure detail.
        detail: String,
    },
}

impl LensOutcome {
    /// True only when the lens passed (ran and matched the oracle).
    ///
    /// Missing coverage is represented as [`LensOutcome::Fail`], so a
    /// passing lens always performed real comparisons.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, LensOutcome::Pass { .. })
    }

    /// True only when the lens actually ran and matched the oracle.
    #[must_use]
    pub fn is_pass(&self) -> bool {
        matches!(self, LensOutcome::Pass { .. })
    }
}
