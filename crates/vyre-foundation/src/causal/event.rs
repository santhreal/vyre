//! Causal events, metrics, cache hit tracking, and counterfactual decisions.

use serde::{Deserialize, Serialize};

use super::id::{CausalSpanId, SourceSpanRef};
use super::phase::CausalPhase;

/// Reason why a schedule candidate or pass was pruned or rejected.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PruneReason {
    /// Estimated or measured execution cost exceeded incumbent winner.
    CostInferior {
        /// Cost of incumbent schedule.
        incumbent_cost: u64,
        /// Cost of candidate schedule.
        candidate_cost: u64,
    },
    /// Schedule violates target hardware or memory model legality.
    LegalityViolation {
        /// Explanation of invariant or legality rule broken.
        rule: String,
    },
    /// Candidate is strictly dominated by an existing partition/schedule.
    DominanceInferior {
        /// Id or name of dominating candidate.
        dominator: String,
    },
    /// Shared memory, registers, or buffer binding capacity exceeded.
    ResourceExhaustion {
        /// Resource kind that exceeded limits.
        resource_kind: String,
        /// Required capacity.
        required: u64,
        /// Hardware or quota limit.
        limit: u64,
    },
    /// Quota or compilation time budget exhausted.
    BudgetExhausted {
        /// Limit that was hit.
        budget_name: String,
    },
    /// Tenant authorization or security label violation.
    TenantPolicyViolation {
        /// Reason for policy denial.
        policy_reason: String,
    },
    /// Custom prune reason.
    Other(String),
}

/// Cache hit record for compilation and kernel caches.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub struct CacheHitRecord {
    /// Semantic classification of the cache (e.g. "pass_order", "lowered_module", "compiled_artifact").
    pub key_class: String,
    /// Whether the lookup hit.
    pub hit: bool,
    /// Cryptographic digest of the cache key.
    pub key_digest: String,
}

/// Description of an unselected counterfactual schedule alternative.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct AlternativeSchedule {
    /// Name or descriptor of alternative schedule.
    pub schedule_name: String,
    /// Predicted or measured cost for the alternative.
    pub estimated_cost: f64,
    /// Why this alternative was not selected over the winner.
    pub rejection_reason: String,
}

/// Decision rationale recording why a winner was chosen over alternatives.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct CounterfactualDecision {
    /// Name or descriptor of chosen schedule.
    pub chosen_schedule: String,
    /// Estimated cost of the chosen schedule.
    pub chosen_cost: f64,
    /// Summary explanation of why this schedule was selected.
    pub winning_reason: String,
    /// Evaluated alternatives that were rejected.
    pub alternatives: Vec<AlternativeSchedule>,
}

/// Single causal event representing a deterministic unit of compiler or runtime work.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct CausalEvent {
    /// Unique span identifier.
    pub span_id: CausalSpanId,
    /// Parent span identifier, forming a causal DAG.
    pub parent_id: Option<CausalSpanId>,
    /// Compiler or runtime lifecycle phase.
    pub phase: CausalPhase,
    /// Specific pass, lowering arm, or runtime stage name.
    pub stage_name: String,
    /// Source region identifier in IR, if applicable.
    pub region_id: Option<u64>,
    /// High-level source span reference, if available.
    pub source_span: Option<SourceSpanRef>,
    /// Schedule operator being planned or executed.
    pub schedule_operator: Option<String>,
    /// Target physical instruction or dialect opcode.
    pub physical_instruction: Option<String>,
    /// Emitted artifact payload entry symbol or offset.
    pub payload_entry: Option<String>,
    /// Resource or buffer identifier.
    pub resource_id: Option<u64>,
    /// Deterministic work units processed (e.g. node count, instructions, bytes).
    pub work_units: u64,
    /// Wall-clock time spent in nanoseconds.
    pub wall_time_ns: u64,
    /// Device execution time in nanoseconds, when available.
    pub device_time_ns: Option<u64>,
    /// Number of heap allocations performed.
    pub allocations: usize,
    /// Retained bytes allocated.
    pub retained_bytes: usize,
    /// Cache hit record, if this stage consulted a cache.
    pub cache_hit: Option<CacheHitRecord>,
    /// Prune reason if this candidate was pruned.
    pub prune_reason: Option<PruneReason>,
    /// Counterfactual decision explanation.
    pub counterfactual_decision: Option<CounterfactualDecision>,
    /// Diagnostic message or warning associated with this event.
    pub diagnostic: Option<String>,
}

impl CausalEvent {
    /// Create a new minimal causal event.
    pub fn new(
        span_id: CausalSpanId,
        parent_id: Option<CausalSpanId>,
        phase: CausalPhase,
        stage_name: impl Into<String>,
    ) -> Self {
        Self {
            span_id,
            parent_id,
            phase,
            stage_name: stage_name.into(),
            region_id: None,
            source_span: None,
            schedule_operator: None,
            physical_instruction: None,
            payload_entry: None,
            resource_id: None,
            work_units: 1,
            wall_time_ns: 0,
            device_time_ns: None,
            allocations: 0,
            retained_bytes: 0,
            cache_hit: None,
            prune_reason: None,
            counterfactual_decision: None,
            diagnostic: None,
        }
    }
}
