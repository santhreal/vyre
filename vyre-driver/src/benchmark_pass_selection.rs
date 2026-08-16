//! Benchmark-driven optimization pass selection.
//!
//! Expensive passes must not fire because a static list says so. They need
//! graph/frontier/reuse evidence showing that the launch, memory, or readback
//! cost they remove is larger than their own planning cost. This module makes
//! that decision explicit and deterministic.

use crate::accounting::{
    checked_add_u64_count as checked_add, checked_add_usize_count as checked_add_usize,
    ArithmeticOverflow,
};
use crate::numeric::checked_compose_basis_points_u64;
use crate::reservation_policy::{
    reserved_typed_vec as reserved_vec, storage_reserve_failure_adapter, ReservationPolicy,
    ReusableIndexScratch,
};

const BENCHMARK_PASS_SELECTION_RESERVATION: ReservationPolicy = ReservationPolicy::new(
    "benchmark pass selection",
    "shard the optimization candidate set before pass selection",
);

/// One optimization candidate with benchmark-derived thresholds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchmarkPassCandidate {
    /// Registered optimization pass id.
    pub pass_id: &'static str,
    /// Minimum active frontier items required before this pass is profitable.
    pub min_frontier_items: u64,
    /// Minimum repeated graph executions required before this pass is profitable.
    pub min_reuse_count: u64,
    /// Minimum readback bytes avoided before this pass is profitable.
    pub min_avoided_readback_bytes: u64,
    /// Estimated planning/compile cost in nanoseconds.
    pub planning_cost_ns: u64,
    /// Scratch bytes needed by the pass while planning/executing.
    pub scratch_bytes: u64,
    /// Expected speedup in basis points from committed benchmark evidence.
    pub expected_speedup_bps: u32,
    /// Whether the pass is mandatory when its thresholds are met.
    pub mandatory_when_profitable: bool,
}

/// Runtime benchmark sample used to select optimization passes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchmarkPassSelectionSample {
    /// Active frontier items in the current graph/query batch.
    pub frontier_items: u64,
    /// Number of repeated executions over the same resident graph shape.
    pub reuse_count: u64,
    /// Readback bytes the workload can avoid with compaction/aggregation.
    pub avoidable_readback_bytes: u64,
    /// Maximum total planning cost allowed.
    pub planning_budget_ns: u64,
    /// Maximum scratch bytes allowed for selected passes.
    pub scratch_budget_bytes: u64,
}

/// One skipped optimization with a stable reason.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SkippedBenchmarkPass {
    /// Registered optimization pass id.
    pub pass_id: &'static str,
    /// Stable reason.
    pub reason: BenchmarkPassSkipReason,
}

/// Stable skip reason for an optimization candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchmarkPassSkipReason {
    /// Frontier is too small for this pass to pay for itself.
    FrontierBelowThreshold,
    /// Graph reuse is too low for residency/cache/fusion work to amortize.
    ReuseBelowThreshold,
    /// Readback pressure is too low for compaction/aggregation to pay off.
    ReadbackBelowThreshold,
    /// Planning budget would be exceeded.
    PlanningBudgetExceeded,
    /// Scratch budget would be exceeded.
    ScratchBudgetExceeded,
}

/// Pass-selection output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkPassSelectionPlan {
    /// Selected pass ids in benchmark-value order.
    pub selected_pass_ids: Vec<&'static str>,
    /// Skipped pass ids with stable reasons.
    pub skipped_passes: Vec<SkippedBenchmarkPass>,
    /// Total selected planning cost.
    pub total_planning_cost_ns: u64,
    /// Total selected scratch bytes.
    pub total_scratch_bytes: u64,
    /// Product of selected speedup multipliers in basis points.
    pub projected_speedup_bps: u64,
}

/// Caller-owned scratch for repeated benchmark pass selection.
#[derive(Debug, Default)]
pub struct BenchmarkPassSelectionScratch {
    index_scratch: ReusableIndexScratch<&'static str>,
}

impl BenchmarkPassSelectionScratch {
    /// Allocate empty reusable pass-selection scratch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate reusable pass-selection scratch for a known candidate count.
    ///
    /// # Errors
    ///
    /// Returns [`BenchmarkPassSelectionError`] when scratch storage cannot be reserved.
    pub fn try_with_capacity(candidate_count: usize) -> Result<Self, BenchmarkPassSelectionError> {
        let mut scratch = Self::default();
        scratch.try_reserve_candidates(candidate_count)?;
        Ok(scratch)
    }

    /// Reserve reusable pass-selection scratch for a known candidate count.
    ///
    /// # Errors
    ///
    /// Returns [`BenchmarkPassSelectionError`] when scratch storage cannot be reserved.
    pub fn try_reserve_candidates(
        &mut self,
        candidate_count: usize,
    ) -> Result<(), BenchmarkPassSelectionError> {
        self.index_scratch.try_reserve_with(
            BENCHMARK_PASS_SELECTION_RESERVATION,
            candidate_count,
            "scratch.seen",
            "scratch.ordered_indices",
            storage_reserve_failed,
        )
    }

    /// Retained duplicate-detection capacity.
    #[must_use]
    pub fn seen_capacity(&self) -> usize {
        self.index_scratch.seen_capacity()
    }

    /// Retained candidate-ordering capacity.
    #[must_use]
    pub fn ordered_index_capacity(&self) -> usize {
        self.index_scratch.ordered_index_capacity()
    }
}

/// Benchmark-driven pass-selection errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BenchmarkPassSelectionError {
    /// Candidate pass id is empty.
    EmptyPassId,
    /// Duplicate candidate pass id.
    DuplicatePassId {
        /// Duplicate pass id.
        pass_id: &'static str,
    },
    /// Candidate has no benchmark speedup evidence.
    MissingSpeedupEvidence {
        /// Invalid pass id.
        pass_id: &'static str,
    },
    /// Mandatory profitable pass could not fit the explicit budgets.
    MandatoryProfitablePassOverBudget {
        /// Pass id.
        pass_id: &'static str,
        /// Reason it could not fit.
        reason: BenchmarkPassSkipReason,
    },
    /// Arithmetic overflowed.
    CountOverflow {
        /// Field being computed.
        field: &'static str,
    },
    /// Scratch or result-vector storage reservation failed before pass selection.
    StorageReserveFailed {
        /// Field being reserved.
        field: &'static str,
        /// Requested total capacity.
        requested: usize,
        /// Allocator failure details.
        message: String,
    },
}

impl ArithmeticOverflow for BenchmarkPassSelectionError {
    fn arithmetic_overflow(field: &'static str) -> Self {
        Self::CountOverflow { field }
    }
}

impl std::fmt::Display for BenchmarkPassSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPassId => write!(
                f,
                "benchmark pass selection received an empty pass id. Fix: register every pass before selection."
            ),
            Self::DuplicatePassId { pass_id } => write!(
                f,
                "benchmark pass selection received duplicate pass `{pass_id}`. Fix: keep one benchmark row per pass."
            ),
            Self::MissingSpeedupEvidence { pass_id } => write!(
                f,
                "benchmark pass `{pass_id}` has no positive speedup evidence. Fix: add committed benchmark evidence or remove the candidate."
            ),
            Self::MandatoryProfitablePassOverBudget { pass_id, reason } => write!(
                f,
                "mandatory profitable pass `{pass_id}` was blocked by {reason:?}. Fix: raise the explicit budget or shard before pass selection."
            ),
            Self::CountOverflow { field } => write!(
                f,
                "benchmark pass selection overflowed while computing {field}. Fix: shard the optimization candidate set."
            ),
            Self::StorageReserveFailed {
                field,
                requested,
                message,
            } => write!(
                f,
                "benchmark pass selection failed to reserve {field} for {requested} entries: {message}. Fix: shard the optimization candidate set before pass selection."
            ),
        }
    }
}

impl std::error::Error for BenchmarkPassSelectionError {}

/// Select optimization passes from benchmark evidence and workload stats.
///
/// # Errors
///
/// Returns [`BenchmarkPassSelectionError`] when candidates are invalid, budget
/// accounting overflows, mandatory profitable passes cannot fit the budget, or
/// planner storage cannot be reserved.
pub fn select_benchmark_passes(
    candidates: &[BenchmarkPassCandidate],
    sample: BenchmarkPassSelectionSample,
) -> Result<BenchmarkPassSelectionPlan, BenchmarkPassSelectionError> {
    let mut scratch = BenchmarkPassSelectionScratch::try_with_capacity(candidates.len())?;
    select_benchmark_passes_with_scratch(candidates, sample, &mut scratch)
}

/// Select optimization passes using caller-owned temporary storage.
///
/// # Errors
///
/// Returns [`BenchmarkPassSelectionError`] when candidates are invalid, budget
/// accounting overflows, mandatory profitable passes cannot fit the budget, or
/// planner storage cannot be reserved.
pub fn select_benchmark_passes_with_scratch(
    candidates: &[BenchmarkPassCandidate],
    sample: BenchmarkPassSelectionSample,
    scratch: &mut BenchmarkPassSelectionScratch,
) -> Result<BenchmarkPassSelectionPlan, BenchmarkPassSelectionError> {
    scratch.index_scratch.clear();
    scratch.try_reserve_candidates(candidates.len())?;
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.pass_id.is_empty() {
            return Err(BenchmarkPassSelectionError::EmptyPassId);
        }
        if !scratch.index_scratch.insert_seen(candidate.pass_id) {
            return Err(BenchmarkPassSelectionError::DuplicatePassId {
                pass_id: candidate.pass_id,
            });
        }
        if candidate.expected_speedup_bps <= 10_000 {
            return Err(BenchmarkPassSelectionError::MissingSpeedupEvidence {
                pass_id: candidate.pass_id,
            });
        }
        scratch.index_scratch.push_index(index);
    }
    scratch
        .index_scratch
        .ordered_indices_mut()
        .sort_unstable_by(|&left, &right| {
            candidates[right]
                .mandatory_when_profitable
                .cmp(&candidates[left].mandatory_when_profitable)
                .then_with(|| {
                    pass_value(&candidates[right])
                        .cmp(&pass_value(&candidates[left]))
                        .then_with(|| candidates[left].pass_id.cmp(candidates[right].pass_id))
                })
        });

    let (selected_pass_capacity, skipped_pass_capacity) =
        count_final_pass_buckets(candidates, sample, scratch.index_scratch.ordered_indices())?;
    let mut selected_pass_ids =
        reserved_selection_vec(selected_pass_capacity, "selected_pass_ids")?;
    let mut skipped_passes = reserved_selection_vec(skipped_pass_capacity, "skipped_passes")?;
    let mut total_planning_cost_ns = 0_u64;
    let mut total_scratch_bytes = 0_u64;
    let mut projected_speedup_bps = 10_000_u64;

    for &index in scratch.index_scratch.ordered_indices() {
        let candidate = candidates[index];
        if sample.frontier_items < candidate.min_frontier_items {
            skipped_passes.push(skipped(
                candidate.pass_id,
                BenchmarkPassSkipReason::FrontierBelowThreshold,
            ));
            continue;
        }
        if sample.reuse_count < candidate.min_reuse_count {
            skipped_passes.push(skipped(
                candidate.pass_id,
                BenchmarkPassSkipReason::ReuseBelowThreshold,
            ));
            continue;
        }
        if sample.avoidable_readback_bytes < candidate.min_avoided_readback_bytes {
            skipped_passes.push(skipped(
                candidate.pass_id,
                BenchmarkPassSkipReason::ReadbackBelowThreshold,
            ));
            continue;
        }

        let next_planning = checked_add(
            total_planning_cost_ns,
            candidate.planning_cost_ns,
            "planning cost",
        )?;
        if next_planning > sample.planning_budget_ns {
            handle_budget_skip(
                candidate,
                BenchmarkPassSkipReason::PlanningBudgetExceeded,
                &mut skipped_passes,
            )?;
            continue;
        }
        let next_scratch = checked_add(
            total_scratch_bytes,
            candidate.scratch_bytes,
            "scratch bytes",
        )?;
        if next_scratch > sample.scratch_budget_bytes {
            handle_budget_skip(
                candidate,
                BenchmarkPassSkipReason::ScratchBudgetExceeded,
                &mut skipped_passes,
            )?;
            continue;
        }

        selected_pass_ids.push(candidate.pass_id);
        total_planning_cost_ns = next_planning;
        total_scratch_bytes = next_scratch;
        projected_speedup_bps = checked_compose_basis_points_u64(
            projected_speedup_bps,
            u64::from(candidate.expected_speedup_bps),
        )
        .ok_or(BenchmarkPassSelectionError::CountOverflow {
            field: "projected speedup product",
        })?;
    }

    Ok(BenchmarkPassSelectionPlan {
        selected_pass_ids,
        skipped_passes,
        total_planning_cost_ns,
        total_scratch_bytes,
        projected_speedup_bps,
    })
}

fn pass_value(candidate: &BenchmarkPassCandidate) -> u128 {
    u128::from(candidate.expected_speedup_bps)
        * (u128::from(candidate.min_frontier_items)
            + u128::from(candidate.min_reuse_count)
            + u128::from(candidate.min_avoided_readback_bytes))
}

fn count_final_pass_buckets(
    candidates: &[BenchmarkPassCandidate],
    sample: BenchmarkPassSelectionSample,
    ordered_indices: &[usize],
) -> Result<(usize, usize), BenchmarkPassSelectionError> {
    let mut selected = 0usize;
    let mut skipped = 0usize;
    let mut total_planning_cost_ns = 0_u64;
    let mut total_scratch_bytes = 0_u64;
    for &index in ordered_indices {
        let candidate = candidates[index];
        if sample.frontier_items < candidate.min_frontier_items
            || sample.reuse_count < candidate.min_reuse_count
            || sample.avoidable_readback_bytes < candidate.min_avoided_readback_bytes
        {
            skipped = checked_add_usize(skipped, 1, "skipped pass count")?;
            continue;
        }
        let next_planning = checked_add(
            total_planning_cost_ns,
            candidate.planning_cost_ns,
            "planning cost",
        )?;
        if next_planning > sample.planning_budget_ns {
            if candidate.mandatory_when_profitable {
                return Err(
                    BenchmarkPassSelectionError::MandatoryProfitablePassOverBudget {
                        pass_id: candidate.pass_id,
                        reason: BenchmarkPassSkipReason::PlanningBudgetExceeded,
                    },
                );
            }
            skipped = checked_add_usize(skipped, 1, "skipped pass count")?;
            continue;
        }
        let next_scratch = checked_add(
            total_scratch_bytes,
            candidate.scratch_bytes,
            "scratch bytes",
        )?;
        if next_scratch > sample.scratch_budget_bytes {
            if candidate.mandatory_when_profitable {
                return Err(
                    BenchmarkPassSelectionError::MandatoryProfitablePassOverBudget {
                        pass_id: candidate.pass_id,
                        reason: BenchmarkPassSkipReason::ScratchBudgetExceeded,
                    },
                );
            }
            skipped = checked_add_usize(skipped, 1, "skipped pass count")?;
            continue;
        }
        selected = checked_add_usize(selected, 1, "selected pass count")?;
        total_planning_cost_ns = next_planning;
        total_scratch_bytes = next_scratch;
    }
    Ok((selected, skipped))
}

fn skipped(pass_id: &'static str, reason: BenchmarkPassSkipReason) -> SkippedBenchmarkPass {
    SkippedBenchmarkPass { pass_id, reason }
}

fn handle_budget_skip(
    candidate: BenchmarkPassCandidate,
    reason: BenchmarkPassSkipReason,
    skipped_passes: &mut Vec<SkippedBenchmarkPass>,
) -> Result<(), BenchmarkPassSelectionError> {
    if candidate.mandatory_when_profitable {
        return Err(
            BenchmarkPassSelectionError::MandatoryProfitablePassOverBudget {
                pass_id: candidate.pass_id,
                reason,
            },
        );
    }
    skipped_passes.push(skipped(candidate.pass_id, reason));
    Ok(())
}

fn reserved_selection_vec<T>(
    capacity: usize,
    field: &'static str,
) -> Result<Vec<T>, BenchmarkPassSelectionError> {
    reserved_vec(
        BENCHMARK_PASS_SELECTION_RESERVATION,
        capacity,
        field,
        storage_reserve_failed,
    )
}

storage_reserve_failure_adapter!(BenchmarkPassSelectionError);
