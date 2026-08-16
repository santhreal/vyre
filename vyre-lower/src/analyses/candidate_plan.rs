use serde::{Deserialize, Serialize};

/// Generic candidate list emitted by a lowered-IR analysis.
///
/// Analysis-specific modules own the candidate payload type; this shared
/// container keeps the report shape stable across optimization analyses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidatePlan<Candidate> {
    /// Kernel descriptor id analyzed.
    pub kernel_id: String,
    /// Ordered candidates found by the analysis.
    pub candidates: Vec<Candidate>,
}

impl<Candidate> CandidatePlan<Candidate> {
    /// Number of optimization candidates in this plan.
    #[must_use]
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }
}
