//! The bounded expansion loop that applies derived rewrites.
//!
//! The `Rule` seam of the e-graph substrate returns pairs of existing classes,
//! so it can close a congruence but cannot introduce the right-hand side of an
//! equality. A law-derived alternative is exactly that: `f(b, a)` is not in the
//! graph until the commutativity law puts it there. This loop materialises each
//! matched right-hand side, unions it with the matched class, rebuilds, and
//! repeats until nothing new is derived or a bound is reached.
//!
//! Both bounds are the caller's: iterations, and the class count the graph may
//! grow to. An unbounded run over an associative and commutative operator does
//! not converge in any useful time, so a budget is not an optimization here, it
//! is what makes the mechanism usable at all.

use crate::optimizer::eqsat::{EClassId, EGraphError};

use super::expr_lang::{ExprLang, ExprMirror};
use super::law_rule::{DerivedRewrite, DerivedRewriteKind};

/// Why a saturation run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawSaturationStop {
    /// The declared laws authorized no rewrite for this expression's operators.
    NoRewrites,
    /// The caller supplied a zero-iteration budget.
    ZeroBudget,
    /// A full pass over the rewrite set derived nothing new.
    FixedPoint,
    /// The run consumed its iteration budget with derivations still arriving.
    IterationBudget,
    /// The graph reached the class count the caller allowed.
    ExpansionBudget,
}

/// Telemetry for one saturation run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LawSaturationReport {
    /// Rewrites the declared laws authorized.
    pub rewrite_count: usize,
    /// Iterations actually executed.
    pub iters_used: usize,
    /// Iteration budget the caller supplied.
    pub budget: usize,
    /// Equalities the rewrites derived and handed to union.
    pub applied_equivalences: usize,
    /// Extra unions congruence closure discovered during rebuild.
    pub rebuild_unions: usize,
    /// Class count before the first rewrite.
    pub class_count_before: usize,
    /// Class count after the last rewrite.
    pub class_count_after: usize,
    /// Why the run stopped.
    pub stop_reason: LawSaturationStop,
    /// Rewrites that changed the graph, in first-application order, once each.
    ///
    /// `applied_equivalences` counts how many equalities landed and cannot say
    /// which law authorized them. A caller that carries a derivation as
    /// evidence needs the names, and recomputing them from the mirror after the
    /// run would be a second implementation of the match this loop already
    /// performed.
    pub applied_rewrites: Vec<&'static str>,
}

/// Apply `rewrites` to `mirror` until a fixed point or a bound.
///
/// # Errors
///
/// Returns the substrate's allocation and class-id errors.
pub fn saturate_laws(
    mirror: &mut ExprMirror,
    rewrites: &[DerivedRewrite],
    iteration_budget: usize,
    class_budget: usize,
) -> Result<LawSaturationReport, EGraphError> {
    let class_count_before = mirror.egraph().class_count();
    let mut report = LawSaturationReport {
        rewrite_count: rewrites.len(),
        iters_used: 0,
        budget: iteration_budget,
        applied_equivalences: 0,
        rebuild_unions: 0,
        class_count_before,
        class_count_after: class_count_before,
        applied_rewrites: Vec::new(),
        stop_reason: LawSaturationStop::FixedPoint,
    };
    if rewrites.is_empty() {
        report.stop_reason = LawSaturationStop::NoRewrites;
        return Ok(report);
    }
    if iteration_budget == 0 {
        report.stop_reason = LawSaturationStop::ZeroBudget;
        return Ok(report);
    }

    while report.iters_used < iteration_budget {
        if mirror.egraph().class_count() >= class_budget {
            report.stop_reason = LawSaturationStop::ExpansionBudget;
            break;
        }
        report.iters_used += 1;
        let mut derived = 0;
        for rewrite in rewrites {
            let landed = apply_rewrite(mirror, rewrite, class_budget)?;
            if landed > 0 && !report.applied_rewrites.contains(&rewrite.name) {
                report.applied_rewrites.push(rewrite.name);
            }
            derived += landed;
        }
        report.applied_equivalences += derived;
        report.rebuild_unions += mirror.egraph_mut().try_rebuild()?;
        if derived == 0 {
            report.stop_reason = LawSaturationStop::FixedPoint;
            break;
        }
        if report.iters_used == iteration_budget {
            report.stop_reason = LawSaturationStop::IterationBudget;
        }
    }

    report.class_count_after = mirror.egraph().class_count();
    Ok(report)
}

/// Materialise and union every match of one rewrite. Returns the number of
/// equalities that changed the graph.
fn apply_rewrite(
    mirror: &mut ExprMirror,
    rewrite: &DerivedRewrite,
    class_budget: usize,
) -> Result<usize, EGraphError> {
    let matches = matched_nodes(mirror, rewrite);
    let mut applied = 0;
    for (class, node) in matches {
        if mirror.egraph().class_count() >= class_budget {
            break;
        }
        let Some(equivalent) = equivalent_class(mirror, rewrite.kind, &node)? else {
            continue;
        };
        let before = mirror.egraph_mut().try_find(class)?;
        let other = mirror.egraph_mut().try_find(equivalent)?;
        if before == other {
            continue;
        }
        mirror.egraph_mut().try_union(before, other)?;
        applied += 1;
    }
    Ok(applied)
}

/// The `(class, node)` pairs one rewrite matches, snapshotted so the loop can
/// add nodes while it walks them.
fn matched_nodes(mirror: &ExprMirror, rewrite: &DerivedRewrite) -> Vec<(EClassId, ExprLang)> {
    mirror
        .egraph()
        .iter_nodes()
        .filter(|(_, node)| matches!(node, ExprLang::Bin { op, .. } if *op == rewrite.op))
        .map(|(class, node)| (class, node.clone()))
        .collect()
}

/// The class the rewrite's right-hand side lives in, adding it when the graph
/// does not hold it yet. `None` when the match does not satisfy the rewrite's
/// precondition.
fn equivalent_class(
    mirror: &mut ExprMirror,
    kind: DerivedRewriteKind,
    node: &ExprLang,
) -> Result<Option<EClassId>, EGraphError> {
    let ExprLang::Bin { op, left, right } = *node else {
        return Ok(None);
    };
    match kind {
        DerivedRewriteKind::Commute => {
            if mirror.egraph_mut().try_find(left)? == mirror.egraph_mut().try_find(right)? {
                return Ok(None);
            }
            mirror
                .egraph_mut()
                .try_add(ExprLang::Bin {
                    op,
                    left: right,
                    right: left,
                })
                .map(Some)
        }
        DerivedRewriteKind::ReassociateRight => {
            // f(f(a, b), c) = f(a, f(b, c))
            let Some((a, b)) = same_operator_children(mirror, op, left) else {
                return Ok(None);
            };
            let inner = mirror
                .egraph_mut()
                .try_add(ExprLang::Bin { op, left: b, right })?;
            mirror
                .egraph_mut()
                .try_add(ExprLang::Bin {
                    op,
                    left: a,
                    right: inner,
                })
                .map(Some)
        }
        DerivedRewriteKind::ReassociateLeft => {
            // f(a, f(b, c)) = f(f(a, b), c)
            let Some((b, c)) = same_operator_children(mirror, op, right) else {
                return Ok(None);
            };
            let inner = mirror
                .egraph_mut()
                .try_add(ExprLang::Bin { op, left, right: b })?;
            mirror
                .egraph_mut()
                .try_add(ExprLang::Bin {
                    op,
                    left: inner,
                    right: c,
                })
                .map(Some)
        }
        DerivedRewriteKind::RightIdentity { element } => {
            Ok((mirror.literal_u32(right) == Some(element)).then_some(left))
        }
        DerivedRewriteKind::LeftIdentity { element } => {
            Ok((mirror.literal_u32(left) == Some(element)).then_some(right))
        }
        DerivedRewriteKind::Idempotent => {
            let same =
                mirror.egraph_mut().try_find(left)? == mirror.egraph_mut().try_find(right)?;
            Ok(same.then_some(left))
        }
        DerivedRewriteKind::RightAbsorbing { element } => {
            Ok((mirror.literal_u32(right) == Some(element)).then_some(right))
        }
        DerivedRewriteKind::LeftAbsorbing { element } => {
            Ok((mirror.literal_u32(left) == Some(element)).then_some(left))
        }
    }
}

/// The two operands of a node in `class` that applies `op`, if the class holds
/// one.
fn same_operator_children(
    mirror: &ExprMirror,
    op: vyre_spec::BinOp,
    class: EClassId,
) -> Option<(EClassId, EClassId)> {
    let canonical = mirror.egraph().find_immut(class);
    mirror
        .egraph()
        .class(canonical)?
        .nodes
        .iter()
        .find_map(|node| match node {
            ExprLang::Bin {
                op: held,
                left,
                right,
            } if *held == op => Some((*left, *right)),
            _ => None,
        })
}
