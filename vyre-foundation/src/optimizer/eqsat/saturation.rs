//! Running rewrite rules to a fixed point, per-family budgets, and the
//! telemetry each run reports.

use super::class_index::reserve_vec_exact;
use super::{
    log_egraph_compat_error, EGraph, EGraphError, ENodeLang, Family, FamilySaturationReport,
    FamilySaturationTelemetry, Rule, SaturationReport, SaturationStopReason,
};

impl SaturationReport {
    /// Signed dense-class delta from before to after.
    #[must_use]
    pub fn class_count_delta(&self) -> isize {
        self.class_count_after as isize - self.class_count_before as isize
    }
}

/// Run rules to fixed point or `max_iters`, whichever comes first.
/// Returns the iteration count actually used.
pub fn saturate<L: ENodeLang>(
    egraph: &mut EGraph<L>,
    rules: &[Box<dyn Rule<L>>],
    max_iters: usize,
) -> usize {
    match try_saturate(egraph, rules, max_iters) {
        Ok(iters) => iters,
        Err(error) => {
            log_egraph_compat_error("egraph saturate", &error);
            0
        }
    }
}

/// Fallible variant of [`saturate`].
pub fn try_saturate<L: ENodeLang>(
    egraph: &mut EGraph<L>,
    rules: &[Box<dyn Rule<L>>],
    max_iters: usize,
) -> Result<usize, EGraphError> {
    try_saturate_with_report(egraph, rules, max_iters).map(|report| report.iters_used)
}

/// Run a raw rule set and return saturation telemetry.
pub fn saturate_with_report<L: ENodeLang>(
    egraph: &mut EGraph<L>,
    rules: &[Box<dyn Rule<L>>],
    max_iters: usize,
) -> SaturationReport {
    match try_saturate_with_report(egraph, rules, max_iters) {
        Ok(report) => report,
        Err(error) => {
            log_egraph_compat_error("egraph saturate_with_report", &error);
            finalize_saturation_report(
                egraph,
                saturation_report(
                    "global",
                    rules.len(),
                    0,
                    max_iters,
                    SaturationStopReason::IterationBudget,
                    egraph.class_count(),
                    0,
                    0,
                ),
            )
        }
    }
}

/// Fallible variant of [`saturate_with_report`].
pub fn try_saturate_with_report<L: ENodeLang>(
    egraph: &mut EGraph<L>,
    rules: &[Box<dyn Rule<L>>],
    max_iters: usize,
) -> Result<SaturationReport, EGraphError> {
    try_saturate_named(egraph, "global", rules, max_iters)
}

/// Run a named rewrite family and return saturation telemetry.
pub fn try_saturate_named<L: ENodeLang>(
    egraph: &mut EGraph<L>,
    rewrite_family: &'static str,
    rules: &[Box<dyn Rule<L>>],
    max_iters: usize,
) -> Result<SaturationReport, EGraphError> {
    let class_count_before = egraph.class_count();
    if rules.is_empty() {
        return Ok(finalize_saturation_report(
            egraph,
            saturation_report(
                rewrite_family,
                0,
                0,
                max_iters,
                SaturationStopReason::EmptyRuleSet,
                class_count_before,
                0,
                0,
            ),
        ));
    }
    if max_iters == 0 {
        return Ok(finalize_saturation_report(
            egraph,
            saturation_report(
                rewrite_family,
                rules.len(),
                0,
                0,
                SaturationStopReason::ZeroBudget,
                class_count_before,
                0,
                0,
            ),
        ));
    }
    let mut equivalences = Vec::new();
    reserve_vec_exact(
        &mut equivalences,
        egraph.class_count(),
        "egraph saturation equivalence staging",
    )?;
    let mut applied_equivalences = 0usize;
    let mut rebuild_unions = 0usize;
    for iter in 0..max_iters {
        equivalences.clear();
        for rule in rules {
            let matches = rule.matches(egraph);
            reserve_vec_exact(
                &mut equivalences,
                matches.len(),
                "egraph saturation rule-match staging",
            )?;
            equivalences.extend(matches);
        }
        if equivalences.is_empty() {
            return Ok(finalize_saturation_report(
                egraph,
                saturation_report(
                    rewrite_family,
                    rules.len(),
                    iter,
                    max_iters,
                    SaturationStopReason::FixedPoint,
                    class_count_before,
                    applied_equivalences,
                    rebuild_unions,
                ),
            ));
        }
        applied_equivalences = applied_equivalences.saturating_add(equivalences.len());
        for (a, b) in equivalences.drain(..) {
            egraph.try_union(a, b)?;
        }
        let extra = egraph.try_rebuild()?;
        rebuild_unions = rebuild_unions.saturating_add(extra);
        if extra == 0 && egraph.pending.is_empty() {
            // Nothing else to propagate; still need to check if rules find
            // anything new on the next iter.
        }
    }
    Ok(finalize_saturation_report(
        egraph,
        saturation_report(
            rewrite_family,
            rules.len(),
            max_iters,
            max_iters,
            SaturationStopReason::IterationBudget,
            class_count_before,
            applied_equivalences,
            rebuild_unions,
        ),
    ))
}

fn saturation_report(
    rewrite_family: &'static str,
    rule_count: usize,
    iters_used: usize,
    budget: usize,
    stop_reason: SaturationStopReason,
    class_count_before: usize,
    applied_equivalences: usize,
    rebuild_unions: usize,
) -> SaturationReport {
    SaturationReport {
        rewrite_family,
        rule_count,
        iters_used,
        budget,
        stop_reason,
        class_count_before,
        class_count_after: class_count_before,
        applied_equivalences,
        rebuild_unions,
    }
}

fn finalize_saturation_report<L: ENodeLang>(
    egraph: &EGraph<L>,
    mut report: SaturationReport,
) -> SaturationReport {
    report.class_count_after = egraph.class_count();
    report
}

/// Run each family with its own iteration budget.
///
/// Saturate-per-family is the prerequisite for ROADMAP A8: a global
/// `max_iters` punishes algebraic families (which converge in 2-3 iters)
/// for sharing a budget with slow rewrite families (which may need 50+).
/// The fix is to give each family its own cap  -  algebraic gets the small
/// cap it needs, structural rewrite gets the larger one, and neither
/// starves the other.
///
/// Order: families run in the order they appear in `families`. Earlier
/// families' merges are visible to later families (the `EGraph` carries
/// state across calls). Re-running this wrapper after a third-party
/// pass mutates the `EGraph` is safe  -  each call is independent.
///
/// `budget_for` is queried once per family to allow callers to pull
/// per-family caps from a TOML config or cost model. Returning 0 skips
/// the family without running it.
pub fn saturate_per_family<L: ENodeLang>(
    egraph: &mut EGraph<L>,
    families: &[&dyn Family<L>],
    budget_for: impl Fn(&str) -> usize,
) -> Vec<FamilySaturationReport> {
    match try_saturate_per_family(egraph, families, budget_for) {
        Ok(report) => report,
        Err(error) => {
            log_egraph_compat_error("egraph saturate_per_family", &error);
            Vec::new()
        }
    }
}

/// Fallible variant of [`saturate_per_family`].
pub fn try_saturate_per_family<L: ENodeLang>(
    egraph: &mut EGraph<L>,
    families: &[&dyn Family<L>],
    budget_for: impl Fn(&str) -> usize,
) -> Result<Vec<FamilySaturationReport>, EGraphError> {
    let detailed = try_saturate_per_family_detailed(egraph, families, budget_for)?;
    let mut out = Vec::new();
    reserve_vec_exact(
        &mut out,
        detailed.len(),
        "egraph family saturation legacy report staging",
    )?;
    out.extend(detailed.into_iter().map(|entry| FamilySaturationReport {
        family: entry.family,
        iters_used: entry.saturation.iters_used,
        budget: entry.saturation.budget,
    }));
    Ok(out)
}

/// Run each family with its own iteration budget and return full telemetry.
pub fn saturate_per_family_detailed<L: ENodeLang>(
    egraph: &mut EGraph<L>,
    families: &[&dyn Family<L>],
    budget_for: impl Fn(&str) -> usize,
) -> Vec<FamilySaturationTelemetry> {
    match try_saturate_per_family_detailed(egraph, families, budget_for) {
        Ok(report) => report,
        Err(error) => {
            log_egraph_compat_error("egraph saturate_per_family_detailed", &error);
            Vec::new()
        }
    }
}

/// Fallible variant of [`saturate_per_family_detailed`].
pub fn try_saturate_per_family_detailed<L: ENodeLang>(
    egraph: &mut EGraph<L>,
    families: &[&dyn Family<L>],
    budget_for: impl Fn(&str) -> usize,
) -> Result<Vec<FamilySaturationTelemetry>, EGraphError> {
    let mut out = Vec::new();
    reserve_vec_exact(
        &mut out,
        families.len(),
        "egraph family saturation telemetry staging",
    )?;
    for family in families {
        let name = family.name();
        let budget = budget_for(name);
        let rules = family.rules();
        let saturation = try_saturate_named(egraph, name, &rules, budget)?;
        out.push(FamilySaturationTelemetry {
            family: name,
            saturation,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::arith_fixture::{
        arith_cost, Arith, ConstUnionFamily, ForeignClassRule, PairConstSelfRule,
        UnionEqualConstsRule,
    };
    use super::super::extraction::try_extract_best;
    use super::super::EGraphError;
    use super::super::{EClassId, EGraph, Rule, SaturationStopReason};
    use super::{
        saturate, saturate_per_family, try_saturate, try_saturate_named,
        try_saturate_per_family_detailed,
    };

    #[test]
    fn saturate_runs_to_fixed_point() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        // Hashcons should already prevent two Const(7)s, but this exercises
        // the saturate loop end-to-end with a real rule.
        let _a = egraph.add(Arith::Const(7));
        let _b = egraph.add(Arith::Const(8));
        let rules: Vec<Box<dyn Rule<Arith>>> = vec![Box::new(UnionEqualConstsRule)];
        let iters = saturate(&mut egraph, &rules, 10);
        assert!(iters <= 10);
        // No new equivalences past the first iter (hashcons already
        // dedupes), so saturate returns 0 or 1.
        assert!(iters <= 1);
    }

    #[test]
    fn eqsat_saturation_report_records_fixed_point_class_count_and_family() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _a = egraph.add(Arith::Const(7));
        let _b = egraph.add(Arith::Const(8));
        let rules: Vec<Box<dyn Rule<Arith>>> = vec![Box::new(UnionEqualConstsRule)];
        let report = try_saturate_named(&mut egraph, "arith_identities", &rules, 10)
            .expect("Fix: valid saturation report must be produced");
        assert_eq!(report.rewrite_family, "arith_identities");
        assert_eq!(report.rule_count, 1);
        assert_eq!(report.stop_reason, SaturationStopReason::FixedPoint);
        assert_eq!(report.class_count_before, 2);
        assert_eq!(report.class_count_after, egraph.class_count());
        assert_eq!(report.class_count_delta(), 0);
        assert_eq!(report.applied_equivalences, 0);
        assert_eq!(report.rebuild_unions, 0);
    }

    #[test]
    fn eqsat_saturation_report_exposes_budget_stop_without_silent_fallback() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _ = egraph.add(Arith::Const(7));
        let rules: Vec<Box<dyn Rule<Arith>>> = vec![Box::new(PairConstSelfRule)];
        let report = try_saturate_named(&mut egraph, "self_pairs", &rules, 2)
            .expect("Fix: valid budgeted saturation report must be produced");
        assert_eq!(report.rewrite_family, "self_pairs");
        assert_eq!(report.iters_used, 2);
        assert_eq!(report.budget, 2);
        assert_eq!(report.stop_reason, SaturationStopReason::IterationBudget);
        assert_eq!(report.class_count_before, 1);
        assert_eq!(report.class_count_after, egraph.class_count());
        assert_eq!(report.applied_equivalences, 2);

        let invalid_rules: Vec<Box<dyn Rule<Arith>>> = vec![Box::new(ForeignClassRule)];
        let err = try_saturate_named(&mut egraph, "invalid_foreign", &invalid_rules, 1)
            .expect_err("Fix: fallible saturation must reject invalid equivalence ids");
        assert!(
            matches!(
                err,
                EGraphError::ClassIdOutOfBounds {
                    context: "egraph find",
                    id: EClassId(999),
                    len: 1
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn saturate_per_family_skips_zero_budget() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _ = egraph.add(Arith::Const(7));
        let fam = ConstUnionFamily { name: "f0" };
        let report = saturate_per_family(&mut egraph, &[&fam], |_| 0);
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].family, "f0");
        assert_eq!(report[0].iters_used, 0);
        assert_eq!(report[0].budget, 0);
    }

    #[test]
    fn saturate_per_family_runs_each_family_independently() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _ = egraph.add(Arith::Const(1));
        let _ = egraph.add(Arith::Const(2));
        let fam_a = ConstUnionFamily { name: "alpha" };
        let fam_b = ConstUnionFamily { name: "beta" };
        let report = saturate_per_family(&mut egraph, &[&fam_a, &fam_b], |name| match name {
            "alpha" => 3,
            "beta" => 5,
            _ => 0,
        });
        assert_eq!(report.len(), 2);
        assert_eq!(report[0].family, "alpha");
        assert_eq!(report[0].budget, 3);
        assert!(report[0].iters_used <= 3);
        assert_eq!(report[1].family, "beta");
        assert_eq!(report[1].budget, 5);
        assert!(report[1].iters_used <= 5);
    }

    #[test]
    fn eqsat_per_family_detailed_report_keeps_family_identity_and_stop_reason() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _ = egraph.add(Arith::Const(1));
        let fam = ConstUnionFamily { name: "alpha" };
        let report = try_saturate_per_family_detailed(&mut egraph, &[&fam], |_| 0)
            .expect("Fix: detailed per-family report must be produced");
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].family, "alpha");
        assert_eq!(report[0].saturation.rewrite_family, "alpha");
        assert_eq!(report[0].saturation.budget, 0);
        assert_eq!(
            report[0].saturation.stop_reason,
            SaturationStopReason::ZeroBudget
        );
        assert_eq!(report[0].saturation.class_count_after, egraph.class_count());
    }

    #[test]
    fn saturate_per_family_empty_input_returns_empty() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let report = saturate_per_family(&mut egraph, &[], |_| 10);
        assert!(report.is_empty());
    }

    #[test]
    fn saturate_per_family_reports_iters_used_le_budget() {
        let mut egraph: EGraph<Arith> = EGraph::new();
        let _ = egraph.add(Arith::Const(1));
        let _ = egraph.add(Arith::Const(2));
        let fam = ConstUnionFamily { name: "single" };
        let report = saturate_per_family(&mut egraph, &[&fam], |_| 100);
        assert_eq!(report.len(), 1);
        assert!(
            report[0].iters_used <= report[0].budget,
            "iters_used ({}) must not exceed budget ({})",
            report[0].iters_used,
            report[0].budget
        );
    }

    #[test]
    fn fallible_saturate_and_extract_match_infallible_contracts() {
        let mut egraph: EGraph<Arith> = EGraph::try_with_capacity(4)
            .expect("Fix: unit-test oracle precondition - small egraph reservation must succeed");
        let one = egraph
            .try_add(Arith::Const(1))
            .expect("Fix: unit-test oracle precondition - insert one");
        let two = egraph
            .try_add(Arith::Const(2))
            .expect("Fix: unit-test oracle precondition - insert two");
        let three = egraph
            .try_add(Arith::Const(3))
            .expect("Fix: unit-test oracle precondition - insert three");
        let add_12 = egraph
            .try_add(Arith::Add(one, two))
            .expect("Fix: unit-test oracle precondition - insert add");
        egraph
            .try_union(add_12, three)
            .expect("Fix: unit-test oracle precondition - union equivalent nodes");
        egraph
            .try_rebuild()
            .expect("Fix: unit-test oracle precondition - rebuild equivalent nodes");
        let rules: Vec<Box<dyn Rule<Arith>>> = vec![Box::new(UnionEqualConstsRule)];
        let iters = try_saturate(&mut egraph, &rules, 10)
            .expect("Fix: unit-test oracle precondition - fallible saturation");
        assert!(iters <= 10);
        let (best, cost) = try_extract_best(&egraph, add_12, arith_cost)
            .expect("Fix: unit-test oracle precondition - fallible extraction")
            .expect("Fix: unit-test oracle precondition - best node must exist");
        assert_eq!(best, Arith::Const(3));
        assert_eq!(cost, 1);
    }
}
