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

pub(super) fn no_reinvention_pairs(ops: &[OpInfo]) -> Vec<(f64, &OpInfo, &OpInfo)> {
    let mut pairs = Vec::new();
    let mut reported: BTreeSet<(String, String)> = BTreeSet::new();
    for (i, a) in ops.iter().enumerate() {
        if is_internal_phase_op(&a.id) {
            continue;
        }
        // Only compare NON-TRIVIAL ops  -  trivial kernels share the
        // same "single invocation, loop, store" skeleton and their
        // structural similarity is expected. The audit targets ops
        // with real body content.
        if a.fingerprint.len() < 40 {
            continue;
        }
        for b in ops.iter().skip(i + 1) {
            if is_internal_phase_op(&b.id) {
                continue;
            }
            // The "extract to Tier 2.5" remedy only applies when a higher
            // tier is reinventing substrate work. Similarity among two
            // primitives may indicate a future lower-level helper, but it is
            // not a Tier-3 LEGO violation and should not fail this audit.
            if a.tier != Tier::T3 && b.tier != Tier::T3 {
                continue;
            }
            if b.fingerprint.len() < 40 {
                continue;
            }
            if a.children.contains(&b.id) || b.children.contains(&a.id) {
                continue;
            }
            if a.children.iter().any(|child| b.children.contains(child)) {
                continue;
            }
            if same_implementation_family(&a.id, &b.id)
                || known_distinct_implementation_families(&a.id, &b.id)
                || reviewed_distinct_operations(&a.id, &b.id).is_some()
            {
                continue;
            }
            let sim = structural_similarity(&a.fingerprint, &b.fingerprint);
            if sim < FINGERPRINT_SIM_THRESHOLD {
                continue;
            }
            // Skip comparisons inside the same sub-dialect (math::*
            // vs math::* is often legitimate  -  same loop pattern over
            // same data type, different semantics).
            if same_subdialect(&a.id, &b.id) {
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
}
