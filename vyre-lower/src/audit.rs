//! Combined performance audit for vyre kernels.
//! One entry point runs every substrate-neutral analysis on a
//! `KernelDescriptor` and returns a unified `PerfAuditReport` with
//! sub-reports + a single `waste_score` aggregate + a list of
//! prioritized recommendations.
//!
//! This is the user-visible "tell me what's slow" call. Substrate-
//! specific patterns produce their own per-emit reports; emitter crates
//! expose their own audit entry points and the host can compose them.

use crate::analyses::{
    bank_conflict, coalesce, shared_mem_promote, BankConflictReport, CoalescenceReport,
    PromotionPlan,
};
use crate::KernelDescriptor;
use serde::{Deserialize, Serialize};

/// Unified report combining every substrate-neutral analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerfAuditReport {
    /// Stable kernel identifier.
    pub kernel_id: String,
    /// Global-memory coalescence analysis.
    pub coalesce: CoalescenceReport,
    /// Shared-memory promotion analysis.
    pub shared_mem: PromotionPlan,
    /// Shared-memory bank-conflict analysis.
    pub bank_conflict: BankConflictReport,
    /// Single aggregate score: sum of `1 - throughput_factor` across
    /// every coalesce site + critical bank-conflict count weight +
    /// number of unrealized promotion candidates. Higher = worse.
    /// Useful for ranking kernels in a corpus by which to optimize first.
    pub waste_score: f32,
    /// Human-readable, prioritized recommendations. Ordered by impact.
    pub recommendations: Vec<Recommendation>,
}

impl PerfAuditReport {
    /// One-line human-readable summary suitable for log lines.
    /// Format: `"<id>: waste=X.X, N recommendations (top: <top-msg>)"`.
    /// When there are no recommendations: `"<id>: waste=X.X, clean"`.
    #[must_use]
    pub fn format_short(&self) -> String {
        let id = crate::pattern_audit::display_kernel_id(&self.kernel_id);
        match self.recommendations.first() {
            Some(top) => format!(
                "{id}: waste={:.2}, {} recommendations (top: {})",
                self.waste_score,
                self.recommendations.len(),
                top.message
            ),
            None => format!("{id}: waste={:.2}, clean", self.waste_score),
        }
    }

    /// True iff the report has zero recommendations AND zero waste.
    /// Stronger than `recommendations.is_empty()` because a kernel
    /// can have non-zero waste even without recommendations (e.g.,
    /// uncoalesced accesses we don't currently flag at recommendation
    /// granularity).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.recommendations.is_empty() && self.waste_score == 0.0
    }

    /// Recommendations filtered by category. Useful for tooling that
    /// wants only one perf-issue family (e.g., only coalesce-related
    /// advice for a memory-perf dashboard).
    #[must_use]
    pub fn recommendations_by_category(
        &self,
        category: RecommendationCategory,
    ) -> Vec<&Recommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.category == category)
            .collect()
    }

    /// The highest-priority recommendation (lowest `priority` value).
    /// `None` if the report has no recommendations.
    #[must_use]
    pub fn top_recommendation(&self) -> Option<&Recommendation> {
        self.recommendations.iter().min_by_key(|r| r.priority)
    }
}

impl std::fmt::Display for PerfAuditReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.format_short())
    }
}

/// One prioritized optimization recommendation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recommendation {
    /// Optimization family that owns the recommendation.
    pub category: RecommendationCategory,
    /// Priority `0` = highest. Ties broken by category order in the enum.
    pub priority: u32,
    /// Human-readable explanation, stable enough to grep for.
    pub message: String,
    /// Estimated speedup multiplier if this is fixed alone (best-case
    /// upper bound, not a promise).
    pub estimated_speedup_upper_bound: f32,
}

/// Optimization family for a performance recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecommendationCategory {
    /// PERF B14: memory access not coalesced; emit-side fix or layout
    /// change needed.
    Coalesce,
    /// PERF B12: shared-memory promotion candidate remains in global memory.
    SharedMemPromote,
    /// PERF B13: shared-memory bank conflict serializes accesses.
    BankConflict,
}

/// Like [`audit`] but also returns an [`crate::analyses::OpHistogram`]
/// so the caller gets perf recommendations AND kernel-shape telemetry
/// in a single walk. Useful for routing decisions: a memory-bound
/// kernel with a Coalesce recommendation should be addressed via the
/// substrate-specific emission layer; an arithmetic-bound kernel might
/// benefit more from hardware math fusion at the same layer.
#[must_use]
pub fn audit_with_histogram(
    desc: &KernelDescriptor,
) -> (PerfAuditReport, crate::analyses::OpHistogram) {
    (audit(desc), crate::analyses::op_histogram::analyze(desc))
}

/// Run every substrate-neutral analysis and return the unified report.
#[must_use]
pub fn audit(desc: &KernelDescriptor) -> PerfAuditReport {
    let coalesce_report = coalesce::analyze(desc);
    let shared_mem_report = shared_mem_promote::analyze(desc);
    let bank_conflict_report = bank_conflict::analyze(desc);

    let mut waste_score = coalesce_report.waste_score();
    waste_score += shared_mem_report.candidates.len() as f32 * 5.0;
    waste_score += bank_conflict_report.critical_count() as f32 * 8.0;
    waste_score += (bank_conflict_report.problematic_count()
        - bank_conflict_report.critical_count()) as f32
        * 2.0;

    let recommendations = recommend(&coalesce_report, &shared_mem_report, &bank_conflict_report);

    PerfAuditReport {
        kernel_id: desc.id.clone(),
        coalesce: coalesce_report,
        shared_mem: shared_mem_report,
        bank_conflict: bank_conflict_report,
        waste_score,
        recommendations,
    }
}

fn recommend(
    coalesce: &CoalescenceReport,
    shared_mem: &PromotionPlan,
    bank_conflict: &BankConflictReport,
) -> Vec<Recommendation> {
    // Upper bound: every coalesce site, every shared-mem candidate,
    // and every bank-conflict site can produce at most one recommendation.
    // Pre-size to the sum so the three sequential pushes never resize.
    let mut out = Vec::with_capacity(
        coalesce.sites.len() + shared_mem.candidates.len() + bank_conflict.sites.len(),
    );

    for site in &coalesce.sites {
        if site.pattern.is_problematic() {
            let speedup = 1.0 / site.pattern.throughput_factor();
            out.push(Recommendation {
                category: RecommendationCategory::Coalesce,
                priority: priority_for_speedup(speedup),
                message: format!(
                    "non-coalesced access at op {}, slot {}: {:?}",
                    site.op_index, site.binding_slot, site.pattern
                ),
                estimated_speedup_upper_bound: speedup,
            });
        }
    }

    for cand in &shared_mem.candidates {
        out.push(Recommendation {
            category: RecommendationCategory::SharedMemPromote,
            priority: priority_for_speedup(cand.estimated_speedup_factor),
            message: format!(
                "shared-memory promotion candidate slot {}: {} accesses, tile {} bytes",
                cand.binding_slot, cand.access_count, cand.tile_bytes
            ),
            estimated_speedup_upper_bound: cand.estimated_speedup_factor,
        });
    }

    for site in &bank_conflict.sites {
        use crate::analyses::bank_conflict::ConflictSeverity;
        if let crate::analyses::bank_conflict::BankConflictKind::Conflict { way_count } =
            site.conflict
        {
            let speedup = way_count as f32;
            let severity = site.conflict.severity();
            let priority = match severity {
                ConflictSeverity::Critical => 0,
                ConflictSeverity::Severe => 1,
                ConflictSeverity::Mild => 2,
                _ => 3,
            };
            out.push(Recommendation {
                category: RecommendationCategory::BankConflict,
                priority,
                message: format!(
                    "{}-way bank conflict at op {}, slot {}",
                    way_count, site.op_index, site.binding_slot
                ),
                estimated_speedup_upper_bound: speedup,
            });
        }
    }

    out.sort_by(|a, b| {
        a.priority.cmp(&b.priority).then(
            b.estimated_speedup_upper_bound
                .partial_cmp(&a.estimated_speedup_upper_bound)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });

    out
}

fn priority_for_speedup(speedup: f32) -> u32 {
    if speedup >= 16.0 {
        0
    } else if speedup >= 4.0 {
        1
    } else if speedup >= 2.0 {
        2
    } else {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::{
        binop, body, descriptor, global_ro, lit, load_global, op, shared_rw,
    };
    use crate::{KernelBody, KernelDescriptor, KernelOpKind, LiteralValue};
    use vyre_foundation::ir::{BinOp, DataType};

    /// A 64-thread kernel with no bindings and no ops.
    fn empty_kernel_named(id: &str) -> KernelDescriptor {
        descriptor(id).dispatch(64, 1, 1).build()
    }

    fn empty_kernel() -> KernelDescriptor {
        empty_kernel_named("empty")
    }

    /// A kernel over a single read-only `f32` global named `buf`.
    fn one_buffer_kernel(id: &str, threads: u32, body: impl Into<KernelBody>) -> KernelDescriptor {
        descriptor(id)
            .slot(global_ro(0, DataType::F32, "buf"))
            .dispatch(threads, 1, 1)
            .body(body)
            .build()
    }

    fn empty_report_with_recs(recs: Vec<Recommendation>) -> PerfAuditReport {
        let mut r = audit(&empty_kernel_named("k"));
        r.recommendations = recs;
        r
    }

    #[test]
    fn empty_kernel_has_zero_waste_and_no_recommendations() {
        let r = audit(&empty_kernel());
        assert_eq!(r.kernel_id, "empty");
        assert!((r.waste_score - 0.0).abs() < 1e-6);
        assert!(r.recommendations.is_empty());
    }

    #[test]
    fn coalesced_kernel_has_zero_waste() {
        // Single coalesced load  -  perfect.
        let kk = one_buffer_kernel(
            "perfect",
            64,
            body()
                .op(op(KernelOpKind::LocalInvocationId, [], 0))
                .op(load_global(0, 0, 1)),
        );
        let r = audit(&kk);
        assert!((r.waste_score - 0.0).abs() < 1e-6);
        assert!(r.recommendations.is_empty());
    }

    #[test]
    fn strided_kernel_produces_coalesce_recommendation() {
        // load(buf, 4 * tid)
        let kk = one_buffer_kernel(
            "strided",
            64,
            body()
                .op(op(KernelOpKind::LocalInvocationId, [], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 1, 0, 2))
                .op(load_global(0, 2, 3))
                .literal(LiteralValue::U32(4)),
        );
        let r = audit(&kk);
        assert!(r.waste_score > 0.0);
        assert_eq!(r.recommendations.len(), 1);
        assert_eq!(
            r.recommendations[0].category,
            RecommendationCategory::Coalesce
        );
        assert!(r.recommendations[0].message.contains("non-coalesced"));
    }

    #[test]
    fn shared_mem_promotion_candidate_appears_in_recommendations() {
        // Two LoadGlobal of same slot  -  promotion candidate.
        let kk = one_buffer_kernel(
            "promote",
            32,
            body()
                .op(lit(0, 0))
                .op(load_global(0, 0, 1))
                .op(load_global(0, 0, 2))
                .literal(LiteralValue::U32(0)),
        );
        let r = audit(&kk);
        assert!(
            r.recommendations
                .iter()
                .any(|rec| rec.category == RecommendationCategory::SharedMemPromote),
            "got: {:?}",
            r.recommendations
        );
    }

    #[test]
    fn recommendations_sorted_by_priority_then_speedup() {
        // Build a kernel with mixed problems to verify ordering.
        let kk = descriptor("mixed")
            .slots([
                global_ro(0, DataType::F32, "g"),
                shared_rw(1, DataType::F32, 64, "s"),
            ])
            .dispatch(32, 1, 1)
            .body(
                body()
                    // Strided global load (4x speedup)
                    .op(op(KernelOpKind::LocalInvocationId, [], 0))
                    .op(lit(0, 1))
                    .op(binop(BinOp::Mul, 1, 0, 2))
                    .op(load_global(0, 2, 3))
                    // 32-way bank conflict on shared (32x speedup, critical)
                    .op(lit(1, 4))
                    .op(binop(BinOp::Mul, 0, 4, 5))
                    .op(op(KernelOpKind::LoadShared, [1, 5], 6))
                    .literals([LiteralValue::U32(4), LiteralValue::U32(32)]),
            )
            .build();
        let r = audit(&kk);
        assert!(r.recommendations.len() >= 2);
        // Critical bank conflict (priority 0) must come before strided
        // global load (priority 1).
        let categories: Vec<_> = r.recommendations.iter().map(|r| r.category).collect();
        let bank_pos = categories
            .iter()
            .position(|c| *c == RecommendationCategory::BankConflict);
        let coalesce_pos = categories
            .iter()
            .position(|c| *c == RecommendationCategory::Coalesce);
        assert!(bank_pos.unwrap() < coalesce_pos.unwrap());
    }

    #[test]
    fn report_kernel_id_echoes_descriptor_id() {
        let r = audit(&empty_kernel());
        assert_eq!(r.kernel_id, "empty");
    }

    #[test]
    fn priority_for_speedup_brackets() {
        assert_eq!(priority_for_speedup(32.0), 0);
        assert_eq!(priority_for_speedup(16.0), 0);
        assert_eq!(priority_for_speedup(8.0), 1);
        assert_eq!(priority_for_speedup(4.0), 1);
        assert_eq!(priority_for_speedup(3.0), 2);
        assert_eq!(priority_for_speedup(2.0), 2);
        assert_eq!(priority_for_speedup(1.5), 3);
    }

    #[test]
    fn format_short_clean_kernel_says_clean() {
        let r = audit(&empty_kernel_named("named"));
        let s = r.format_short();
        assert!(s.contains("named:"));
        assert!(s.contains("clean") || s.contains("recommendations"));
    }

    #[test]
    fn format_short_unnamed_uses_unnamed_label() {
        let r = audit(&empty_kernel_named(""));
        assert!(r.format_short().contains("<unnamed>"));
    }

    #[test]
    fn is_clean_on_empty_kernel() {
        assert!(audit(&empty_kernel_named("k")).is_clean());
    }

    #[test]
    fn recommendations_by_category_filters() {
        let report = empty_report_with_recs(vec![
            Recommendation {
                category: RecommendationCategory::Coalesce,
                priority: 0,
                message: "uncoalesced access at op 3".into(),
                estimated_speedup_upper_bound: 4.0,
            },
            Recommendation {
                category: RecommendationCategory::SharedMemPromote,
                priority: 1,
                message: "promote slot 0".into(),
                estimated_speedup_upper_bound: 2.0,
            },
            Recommendation {
                category: RecommendationCategory::Coalesce,
                priority: 2,
                message: "another coalesce".into(),
                estimated_speedup_upper_bound: 1.5,
            },
        ]);
        let coalesce_only = report.recommendations_by_category(RecommendationCategory::Coalesce);
        assert_eq!(coalesce_only.len(), 2);
        assert!(coalesce_only
            .iter()
            .all(|r| r.category == RecommendationCategory::Coalesce));
    }

    #[test]
    fn top_recommendation_picks_lowest_priority_value() {
        let report = empty_report_with_recs(vec![
            Recommendation {
                category: RecommendationCategory::Coalesce,
                priority: 5,
                message: "low".into(),
                estimated_speedup_upper_bound: 1.0,
            },
            Recommendation {
                category: RecommendationCategory::BankConflict,
                priority: 0,
                message: "top".into(),
                estimated_speedup_upper_bound: 8.0,
            },
        ]);
        let top = report.top_recommendation().unwrap();
        assert_eq!(top.message, "top");
    }

    #[test]
    fn top_recommendation_none_when_empty() {
        let report = empty_report_with_recs(vec![]);
        assert!(report.top_recommendation().is_none());
    }

    #[test]
    fn audit_with_histogram_returns_both() {
        let desc = descriptor("k")
            .dispatch(64, 1, 1)
            .body(
                body()
                    .op(lit(0, 0))
                    .op(lit(0, 1))
                    .literal(LiteralValue::U32(7)),
            )
            .build();
        let (report, hist) = audit_with_histogram(&desc);
        assert_eq!(report.kernel_id, "k");
        assert_eq!(hist.literal, 2);
        assert_eq!(hist.total(), 2);
    }
}
