//! Decides which cached pipeline entries a rule-graph change reaches.
//!
//! The rule graph, its intervention mask and the pipeline lineage cell are one
//! query, not eight loose arguments. Both the in-memory and the on-disk
//! invalidation paths used to carry the eight-argument list and their own
//! error mapping around the shared reachability walk, so the two could disagree
//! about which entries a change touched.

use vyre_driver::BackendError;
use vyre_megakernel::{SemanticExecutionPolicy, SemanticExecutor};

/// The rule-graph state one invalidation is evaluated against.
pub(crate) struct RuleImpactQuery<'a> {
    /// Non-zero per rule that the intervention directly perturbs.
    pub(crate) intervention_mask: &'a [u32],
    /// Rule adjacency in the packed form the reachability walk expects.
    pub(crate) rule_adj: &'a [u32],
    /// Current rule activation state.
    pub(crate) state: &'a [u32],
    /// Join rules that propagate activation.
    pub(crate) join_rules: &'a [u32],
    /// Rule count the packed slices are sized for.
    pub(crate) n: u32,
    /// Fixpoint iteration cap for the propagation walk.
    pub(crate) max_iterations: u32,
    /// Rule handle each cache entry was compiled from, one per entry.
    pub(crate) pipeline_lineage_cell: &'a [u32],
}

impl RuleImpactQuery<'_> {
    /// One entry per cache slot: non-zero when this change reaches it.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the packed rule graph is inconsistent with
    /// the declared rule count or the walk exceeds its iteration cap.
    pub(crate) fn impact_mask(
        &self,
        executor: &dyn SemanticExecutor,
        policy: &SemanticExecutionPolicy,
    ) -> Result<Vec<u32>, BackendError> {
        vyre_driver::cache_invalidation::impacted_entries(
            executor,
            policy,
            self.intervention_mask,
            self.rule_adj,
            self.state,
            self.join_rules,
            self.n,
            self.max_iterations,
            self.pipeline_lineage_cell,
        )
        .map_err(|error| BackendError::new(error.to_string()))
    }
}
