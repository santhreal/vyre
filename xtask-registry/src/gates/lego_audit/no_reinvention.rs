//! Check 1: no operation reinvents another.
//!
//! Two operations whose fingerprints agree above the threshold are either one
//! algorithm written twice or a shared IR idiom. The check reports the pair; the
//! reviewer records which it is, as a shared builder or as a reviewed-distinct
//! row.

use super::*;

pub(super) const FINGERPRINT_SIM_THRESHOLD: f64 = 0.88;

/// Check 1: flag pairs of ops with near-identical fingerprints whose
/// Region chains don't indicate one calls the other.
///
/// Uses bigram-frequency cosine similarity  -  captures ordered
/// structure, not just node-kind sets.
pub(super) fn check_1_no_reinvention(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note(format!(
        "[1/10] No-reinvention check (bigram cosine ≥ {FINGERPRINT_SIM_THRESHOLD:.2})"
    ));
    let pairs = no_reinvention_pairs(ops);
    for (sim, a, b) in &pairs {
        report.find(violation(format!("  ✗ reinvention: `{}` and `{}` are {:.0}% structurally similar (cross-dialect) but neither composes the other. Extract the shared body into a Tier 2.5 primitive.",
            a.id,
            b.id,
            sim * 100.0)));
    }
    if pairs.is_empty() {
        report.note("  ✓ no cross-dialect duplication".to_string());
    }
    pairs.len()
}

pub(super) fn has_machine_checkable_distinction(a: &OpInfo, b: &OpInfo) -> bool {
    a.effects != b.effects
        || a.required_caps != b.required_caps
        || a.laws != b.laws
        || a.tolerance != b.tolerance
        || a.buffer_signature != b.buffer_signature
}

pub(super) fn no_reinvention_pairs(ops: &[OpInfo]) -> Vec<(f64, &OpInfo, &OpInfo)> {
    let mut pairs = Vec::new();
    let mut reported: BTreeSet<(String, String)> = BTreeSet::new();
    for (i, a) in ops.iter().enumerate() {
        if is_internal_phase_op(&a.id) {
            continue;
        }
        if a.fingerprint.len() < 40 {
            continue;
        }
        for b in ops.iter().skip(i + 1) {
            if is_internal_phase_op(&b.id) {
                continue;
            }
            if a.tier != Tier::T3 && b.tier != Tier::T3 {
                continue;
            }
            if b.fingerprint.len() < 40 {
                continue;
            }
            // Direct composition: if one composes the other, it is composed.
            if a.children.contains(&b.id) || b.children.contains(&a.id) {
                continue;
            }
            let sim = structural_similarity(&a.fingerprint, &b.fingerprint);
            if sim < FINGERPRINT_SIM_THRESHOLD {
                continue;
            }
            if has_machine_checkable_distinction(a, b) {
                continue;
            }
            if !first_report_of_pair(&mut reported, &a.id, &b.id) {
                continue;
            }
            pairs.push((sim, a, b));
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// IR duplicate analysis judges exactly the registrations that carry a
    /// program.
    ///
    /// WHY: `collect_ops` fingerprints a program, so a registration without one
    /// cannot be compared and has to be left out rather than fingerprinted as
    /// an empty body. Set equality is what makes this non-vacuous: dropping a
    /// program-carrying operation would shrink the analysis in silence, and
    /// admitting a signature-only one would compare it against nothing. This
    /// used to assert that a signature-only registration exists, which the
    /// design then removed: `OperationRegistry` refuses the dotted
    /// host-capability ids that were the last of them, so that assertion could
    /// only ever fail.
    #[test]
    fn ir_duplicate_analysis_judges_exactly_the_operations_that_carry_a_program() {
        let registry = vyre_registry_link::operation::live_operation_registry();
        let mut expected: Vec<&str> = registry
            .iter()
            .filter(|entry| entry.program().is_some())
            .map(|entry| entry.id)
            .collect();
        expected.sort_unstable();
        let ops = collect_ops(&mut Report::clean());
        let mut analysed: Vec<&str> = ops.iter().map(|op| op.id.as_str()).collect();
        analysed.sort_unstable();
        assert_eq!(
            analysed, expected,
            "Fix: the duplicate analysis must judge every registration that carries a program and no registration that does not"
        );
    }

    fn op_with_fingerprint(id: &'static str, fp: Vec<u8>) -> OpInfo {
        crate::gates::lego_audit::test_ops::op_with_fingerprint(id, fp)
    }

    /// WHY: Section 182.13.7 requires that same-subdialect pairs cannot bypass semantic comparison.
    #[test]
    fn same_subdialect_without_machine_distinction_is_reported() {
        let entry: Vec<u8> = (0..60u8).collect();
        let a = op_with_fingerprint("vyre-libs::math::foo", entry.clone());
        let b = op_with_fingerprint("vyre-libs::math::bar", entry);
        let ops = vec![a, b];
        let pairs = no_reinvention_pairs(&ops);
        assert_eq!(
            pairs.len(),
            1,
            "same-subdialect operations without machine distinction must not be bypassed"
        );
    }

    /// WHY: Section 182.13.7 requires that shared-child operations without machine distinction cannot bypass comparison.
    #[test]
    fn shared_child_without_machine_distinction_is_reported() {
        let entry: Vec<u8> = (0..60u8).collect();
        let mut a = op_with_fingerprint("vyre-libs::math::foo", entry.clone());
        let mut b = op_with_fingerprint("vyre-libs::graph::bar", entry);
        a.children
            .insert("vyre-libs::builder::common_child".to_string());
        b.children
            .insert("vyre-libs::builder::common_child".to_string());
        let ops = vec![a, b];
        let pairs = no_reinvention_pairs(&ops);
        assert_eq!(
            pairs.len(),
            1,
            "shared-child operations without machine distinction must not be bypassed"
        );
    }

    /// WHY: Section 182.13.6 allows machine-checkable distinct effects to close a candidate.
    #[test]
    fn distinct_effects_close_candidate() {
        let entry: Vec<u8> = (0..60u8).collect();
        let mut a = op_with_fingerprint("vyre-libs::math::foo", entry.clone());
        let mut b = op_with_fingerprint("vyre-libs::graph::bar", entry);
        a.effects.writes = true;
        b.effects.writes = false;
        let ops = vec![a, b];
        let pairs = no_reinvention_pairs(&ops);
        assert!(
            pairs.is_empty(),
            "distinct effects must close structural candidate"
        );
    }

    /// WHY: Section 182.13.6 allows machine-checkable distinct capabilities to close a candidate.
    #[test]
    fn distinct_capabilities_close_candidate() {
        let entry: Vec<u8> = (0..60u8).collect();
        let mut a = op_with_fingerprint("vyre-libs::math::foo", entry.clone());
        let mut b = op_with_fingerprint("vyre-libs::graph::bar", entry);
        a.required_caps.trap = true;
        b.required_caps.trap = false;
        let ops = vec![a, b];
        let pairs = no_reinvention_pairs(&ops);
        assert!(
            pairs.is_empty(),
            "distinct capabilities must close structural candidate"
        );
    }

    /// WHY: Section 182.13.6 allows machine-checkable distinct algebraic laws to close a candidate.
    #[test]
    fn distinct_laws_close_candidate() {
        let entry: Vec<u8> = (0..60u8).collect();
        let mut a = op_with_fingerprint("vyre-libs::math::foo", entry.clone());
        let b = op_with_fingerprint("vyre-libs::graph::bar", entry);
        a.laws.insert("associative".to_string());
        let ops = vec![a, b];
        let pairs = no_reinvention_pairs(&ops);
        assert!(
            pairs.is_empty(),
            "distinct algebraic laws must close structural candidate"
        );
    }

    #[test]
    fn global_reductions_have_no_reinvention_pairs() {
        let ops = collect_ops(&mut Report::clean());
        assert_no_global_reduce_pairs(
            &no_reinvention_pairs(&ops),
            "all global reduction and indexed move pairs must be closed",
        );
    }
}
