//! Merging discovered equivalences back into the CPU e-graph.
//!
//! Whatever found the equivalence, it is applied through the same `EGraph`
//! merge the CPU uses, so saturation invariants hold by construction. An
//! out-of-range e-class id is counted and not applied: a backend result must
//! not be able to panic the optimizer.

use super::error::u32_len;
use super::snapshot::Equivalence;
use crate::optimizer::eqsat::{EClassId, EGraph, ENodeLang};

/// Report returned after applying discovered equivalences to an `EGraph`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ApplyEquivalencesReport {
    /// Input equivalence count.
    pub requested: usize,
    /// Equivalences whose e-class ids existed in the target `EGraph`.
    pub valid: usize,
    /// Direct union operations that changed the union-find root.
    pub merged: usize,
    /// Additional unions discovered during `EGraph::rebuild`.
    pub rebuild_unions: usize,
}

/// Apply a batch of GPU-discovered equivalences to a CPU-side
/// merge sink. The `merger` closure receives `(left, right)` and
/// performs the canonical `EGraph` merge. Returns the number of
/// merges that actually changed the union-find state (the merger
/// returns `true` for a state-changing merge, `false` for a no-op
/// where left and right were already in the same e-class).
pub fn apply_equivalences<F>(equivalences: &[Equivalence], mut merger: F) -> usize
where
    F: FnMut(u32, u32) -> bool,
{
    let mut applied = 0usize;
    for eq in equivalences {
        if merger(eq.left, eq.right) {
            applied += 1;
        }
    }
    applied
}

/// Apply discovered equivalences to the CPU `EGraph` and rebuild it once.
///
/// Invalid e-class ids are counted as requested but not applied; user input
/// must not be able to panic the optimizer by returning an out-of-range merge.
pub fn apply_equivalences_to_egraph<L>(
    egraph: &mut EGraph<L>,
    equivalences: &[Equivalence],
) -> ApplyEquivalencesReport
where
    L: ENodeLang,
{
    let mut report = ApplyEquivalencesReport {
        requested: equivalences.len(),
        ..ApplyEquivalencesReport::default()
    };
    let Ok(class_count) = u32_len(egraph.class_count(), "CPU egraph class count") else {
        return report;
    };
    for eq in equivalences {
        if eq.left >= class_count || eq.right >= class_count {
            continue;
        }
        report.valid += 1;
        let left = EClassId(eq.left);
        let right = EClassId(eq.right);
        if egraph.find(left) != egraph.find(right) {
            egraph.union(left, right);
            report.merged += 1;
        }
    }
    report.rebuild_unions = egraph.rebuild();
    report
}
