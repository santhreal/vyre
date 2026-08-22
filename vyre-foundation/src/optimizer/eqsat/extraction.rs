//! Greedy bottom-up extraction of the lowest-cost equivalent representation.

use super::class_index::{eclass_index, reserve_vec_exact};
use super::{
    log_egraph_compat_error, EClassId, EGraph, EGraphError, ENodeLang, ExtractionReport,
    ExtractionStopReason, DEFAULT_EXTRACTION_ITER_BUDGET,
};

/// Extract the lowest-cost equivalent representation of `class_id` under
/// `cost_fn`. Returns the chosen `ENode` and its computed cost.
///
/// Greedy bottom-up extraction: cost of each `EClass` is the min over its
/// nodes of `cost_fn(node) + sum(cost_of_child_classes)`. Iterates to
/// fixed point on the cost map.
pub fn extract_best<L: ENodeLang>(
    egraph: &EGraph<L>,
    class_id: EClassId,
    cost_fn: impl Fn(&L) -> u64,
) -> Option<(L, u64)> {
    match try_extract_best(egraph, class_id, cost_fn) {
        Ok(best) => best,
        Err(error) => {
            log_egraph_compat_error("egraph extract_best", &error);
            None
        }
    }
}

/// Fallible variant of [`extract_best`].
pub fn try_extract_best<L: ENodeLang>(
    egraph: &EGraph<L>,
    class_id: EClassId,
    cost_fn: impl Fn(&L) -> u64,
) -> Result<Option<(L, u64)>, EGraphError> {
    try_extract_best_with_budget(egraph, class_id, cost_fn, DEFAULT_EXTRACTION_ITER_BUDGET)
        .map(|report| report.best)
}

/// Extract with a caller-supplied fixed-point iteration budget and return
/// telemetry.
pub fn try_extract_best_with_budget<L: ENodeLang>(
    egraph: &EGraph<L>,
    class_id: EClassId,
    cost_fn: impl Fn(&L) -> u64,
    iter_budget: usize,
) -> Result<ExtractionReport<L>, EGraphError> {
    // VYRE_IR_HOTSPOTS HIGH: extract_best is the inner loop of every
    // optimizer extraction (called per device per root by
    // device_extraction). The previous FxHashMap<EClassId, (L,u64)>
    // hashed-lookup'd costs three times per node per iteration
    // (canon_cid, every child, and the insert check). Class ids are
    // dense u32s in [0, class_count); a direct Vec<Option<(L,u64)>>
    // cuts every lookup to a u32 deref. Plus iter_nodes already
    // filters for canonical (parent[idx] == idx), so the find_immut
    // on `cid` was redundant work  -  drop it.
    let class_count = egraph.class_count();
    let mut costs: Vec<Option<(L, u64)>> = Vec::new();
    reserve_vec_exact(&mut costs, class_count, "egraph extraction cost table")?;
    costs.resize_with(class_count, || None);
    let mut changed = true;
    let mut iters = 0;
    while changed && iters < iter_budget {
        changed = false;
        iters += 1;
        for (cid, node) in egraph.iter_nodes() {
            // cid is already canonical  -  iter_nodes filters parent[idx] == idx.
            let canon_cid_idx = eclass_index(cid, class_count, "egraph extraction class")?;
            let mut node_cost = cost_fn(node);
            let mut child_overflow = false;
            for child in node.children() {
                let canon_child = egraph.try_find_immut(child)?;
                let canon_child_idx =
                    eclass_index(canon_child, class_count, "egraph extraction child class")?;
                if let Some((_, c)) = costs.get(canon_child_idx).and_then(Option::as_ref) {
                    node_cost = node_cost.saturating_add(*c);
                } else {
                    child_overflow = true;
                    break;
                }
            }
            if child_overflow {
                continue;
            }
            let Some(slot) = costs.get_mut(canon_cid_idx) else {
                continue;
            };
            match slot {
                Some((_, existing_cost)) if *existing_cost <= node_cost => {}
                _ => {
                    *slot = Some((node.clone(), node_cost));
                    changed = true;
                }
            }
        }
    }
    let canon = egraph.try_find_immut(class_id)?;
    let canon_idx = eclass_index(canon, class_count, "egraph extraction root class")?;
    let best = costs.get(canon_idx).and_then(Clone::clone);
    let stop_reason = if changed {
        ExtractionStopReason::IterationBudget
    } else if best.is_some() {
        ExtractionStopReason::FixedPoint
    } else {
        ExtractionStopReason::MissingCost
    };
    Ok(ExtractionReport {
        class_id,
        best,
        iters_used: iters,
        budget: iter_budget,
        stop_reason,
        class_count,
    })
}

#[cfg(test)]
mod tests {
    use super::super::arith_fixture::{arith_cost, Arith};
    use super::super::{EGraph, ExtractionStopReason};
    use super::{extract_best, try_extract_best_with_budget};

    #[test]
    fn extract_best_picks_cheapest_equivalent() {
        // Build two equivalent representations: Add(1, 2) and Const(3).
        // Equate them. Extract should pick Const(3) (cost 1) over Add (cost 4).
        let mut egraph: EGraph<Arith> = EGraph::new();
        let one = egraph.add(Arith::Const(1));
        let two = egraph.add(Arith::Const(2));
        let three = egraph.add(Arith::Const(3));
        let add_12 = egraph.add(Arith::Add(one, two));
        egraph.union(add_12, three);
        let _ = egraph.rebuild();
        let (best, cost) = extract_best(&egraph, add_12, arith_cost).expect("Fix: must extract");
        assert_eq!(best, Arith::Const(3));
        assert_eq!(cost, 1);
    }

    #[test]
    fn eqsat_extraction_report_records_budget_stop_reason_class_count_and_cost() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let one = egraph.add(Arith::Const(1));
        let two = egraph.add(Arith::Const(2));
        let three = egraph.add(Arith::Const(3));
        let add_12 = egraph.add(Arith::Add(one, two));
        egraph.union(add_12, three);
        let _ = egraph.rebuild();
        let report = try_extract_best_with_budget(&egraph, add_12, arith_cost, 16)
            .expect("Fix: valid extraction report must be produced");
        let (best, cost) = report
            .best
            .expect("Fix: equivalent constant must be extractable");
        assert_eq!(best, Arith::Const(3));
        assert_eq!(cost, 1);
        assert_eq!(report.class_id, add_12);
        assert_eq!(report.budget, 16);
        assert!(report.iters_used > 0 && report.iters_used <= 16);
        assert_eq!(report.stop_reason, ExtractionStopReason::FixedPoint);
        assert_eq!(report.class_count, egraph.class_count());
    }

    #[test]
    fn eqsat_extraction_zero_budget_reports_budget_stop_without_best() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let class_id = egraph.add(Arith::Const(42));
        let report = try_extract_best_with_budget(&egraph, class_id, arith_cost, 0)
            .expect("Fix: valid zero-budget extraction report must be produced");
        assert_eq!(report.class_id, class_id);
        assert_eq!(report.best, None);
        assert_eq!(report.iters_used, 0);
        assert_eq!(report.budget, 0);
        assert_eq!(report.stop_reason, ExtractionStopReason::IterationBudget);
        assert_eq!(report.class_count, egraph.class_count());
    }

    #[test]
    fn extract_best_returns_only_node_when_no_alternatives() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let a = egraph.add(Arith::Const(42));
        let (best, cost) = extract_best(&egraph, a, arith_cost).expect("Fix: must extract");
        assert_eq!(best, Arith::Const(42));
        assert_eq!(cost, 1);
    }
}
