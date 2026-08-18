//! Check 10: the operand-shape advisory.
//!
//! Two operations whose fingerprints agree over the bucket key and then score
//! above the cosine threshold past it share an operand shape. The verdict is
//! `unreviewed` rather than `duplicate`, because a shape cannot tell a shared
//! algorithm from a shared idiom.

use super::*;

pub(super) const OPERAND_DUP_MIN_COSINE: f64 = 0.55;

pub(super) fn check_10_operand_shape_duplicate(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note(format!("[10/10] Operand-shape advisory (same fingerprint prefix, then cosine ≥ {OPERAND_DUP_MIN_COSINE:.2} past that prefix)"));
    let pairs = operand_shape_duplicate_pairs(ops);
    for (cos, a, b) in &pairs {
        report.find(violation(format!("  ⚠ unreviewed shape pair: `{}` and `{}` share their entry shape and {:.0}% cosine over the rest of the body. Fix: extract the shared body to one builder and record both in `IMPLEMENTATION_FAMILY_ROWS`, or read the two algorithms side by side and record the pair in `REVIEWED_DISTINCT_OPERATIONS` with the reason the shape cannot express.",
            a.id,
            b.id,
            cos * 100.0)));
    }
    if pairs.is_empty() {
        report.note("  ✓ every shape pair is reviewed".to_string());
    }
    0
}

pub(super) fn operand_shape_duplicate_pairs(ops: &[OpInfo]) -> Vec<(f64, &OpInfo, &OpInfo)> {
    let mut buckets: HashMap<Vec<u8>, Vec<&OpInfo>> = HashMap::new();
    for op in ops {
        if is_internal_phase_op(&op.id) {
            continue;
        }
        if op.fingerprint.len() < PREFIX_LEN {
            continue;
        }
        let prefix: Vec<u8> = op.fingerprint[..PREFIX_LEN].to_vec();
        buckets.entry(prefix).or_default().push(op);
    }
    let mut pairs = Vec::new();
    let mut reported: BTreeSet<(String, String)> = BTreeSet::new();
    for ops_in_bucket in buckets.values() {
        if ops_in_bucket.len() < 2 {
            continue;
        }
        for (i, a) in ops_in_bucket.iter().enumerate() {
            for b in ops_in_bucket.iter().skip(i + 1) {
                if a.children.contains(&b.id) || b.children.contains(&a.id) {
                    continue;
                }
                if crate::gates::lego_audit::no_reinvention::has_machine_checkable_distinction(a, b)
                {
                    continue;
                }
                if xtask::gates::implementation_family::same_implementation_family(&a.id, &b.id)
                    || xtask::gates::implementation_family::known_distinct_implementation_families(
                        &a.id, &b.id,
                    )
                    || xtask::gates::implementation_family::reviewed_distinct_operations(
                        &a.id, &b.id,
                    )
                    .is_some()
                {
                    continue;
                }
                let cos = structural_similarity(
                    fingerprint_past_prefix(&a.fingerprint),
                    fingerprint_past_prefix(&b.fingerprint),
                );
                if cos < OPERAND_DUP_MIN_COSINE {
                    continue;
                }
                if !first_report_of_pair(&mut reported, &a.id, &b.id) {
                    continue;
                }
                pairs.push((cos, *a, *b));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::lego_audit::test_ops::op_with_fingerprint;

    /// WHY: the bucket key fixes the first `PREFIX_LEN` bytes identical for
    /// every pair in a bucket. Scoring those bytes again measures the key, so a
    /// pair whose bodies diverge everywhere the key did not reach used to score
    /// above the threshold on the strength of the agreement that bucketed it.
    /// This test fails the moment the score reads the whole fingerprint again:
    /// with a 16-byte shared entry and remainders that share no bigram, whole
    /// fingerprint cosine is over 0.55 and remainder cosine is 0.
    #[test]
    fn a_pair_that_agrees_only_where_the_bucket_key_reaches_is_not_a_duplicate() {
        let entry: Vec<u8> = (0..PREFIX_LEN as u8).collect();
        let mut left = entry.clone();
        left.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2]);
        let mut right = entry;
        right.extend([0xB1, 0xB3, 0xB1, 0xB3, 0xB1, 0xB3, 0xB1, 0xB3]);
        let ops = vec![
            op_with_fingerprint("vyre-libs::alpha::left", left),
            op_with_fingerprint("vyre-primitives::beta::right", right),
        ];
        assert!(operand_shape_duplicate_pairs(&ops).is_empty());
    }

    /// WHY: a body that ends inside the key window leaves no evidence the key
    /// did not already spend, so it cannot be judged either way. Two four-node
    /// operations used to pair at 88% because the key had made them identical.
    #[test]
    fn a_body_that_ends_inside_the_bucket_key_is_not_compared() {
        let entry: Vec<u8> = (0..PREFIX_LEN as u8).collect();
        let ops = vec![
            op_with_fingerprint("vyre-libs::alpha::left", entry.clone()),
            op_with_fingerprint("vyre-primitives::beta::right", entry),
        ];
        assert!(operand_shape_duplicate_pairs(&ops).is_empty());
    }

    /// WHY: the correction must keep the duplicates it was built to find. Two
    /// bodies that agree past the key still pair.
    #[test]
    fn a_pair_that_agrees_past_the_bucket_key_is_still_a_duplicate() {
        let entry: Vec<u8> = (0..PREFIX_LEN as u8).collect();
        let mut left = entry.clone();
        left.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2]);
        let mut right = entry;
        right.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA3]);
        let ops = vec![
            op_with_fingerprint("vyre-libs::alpha::left", left),
            op_with_fingerprint("vyre-primitives::beta::right", right),
        ];
        let pairs = operand_shape_duplicate_pairs(&ops);
        assert_eq!(pairs.len(), 1, "the pair past the key must still be found");
        assert!(pairs[0].0 >= OPERAND_DUP_MIN_COSINE);
    }

    /// WHY: Section 182.13.7 requires that same-subdialect operand-shape pairs without machine distinction cannot bypass.
    #[test]
    fn same_subdialect_operand_shape_without_machine_distinction_is_reported() {
        let entry: Vec<u8> = (0..PREFIX_LEN as u8).collect();
        let mut left = entry.clone();
        left.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2]);
        let mut right = entry;
        right.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA3]);
        let ops = vec![
            op_with_fingerprint("vyre-libs::math::foo", left),
            op_with_fingerprint("vyre-libs::math::bar", right),
        ];
        let pairs = operand_shape_duplicate_pairs(&ops);
        assert_eq!(
            pairs.len(),
            1,
            "same-subdialect shape pair without machine distinction must be reported"
        );
    }

    /// WHY: Section 182.13.6 allows machine-checkable distinct effects to close an operand shape candidate.
    #[test]
    fn distinct_effects_close_operand_shape_candidate() {
        let entry: Vec<u8> = (0..PREFIX_LEN as u8).collect();
        let mut left = entry.clone();
        left.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2]);
        let mut right = entry;
        right.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA3]);
        let mut a = op_with_fingerprint("vyre-libs::math::foo", left);
        let mut b = op_with_fingerprint("vyre-libs::graph::bar", right);
        a.effects.writes = true;
        b.effects.writes = false;
        let ops = vec![a, b];
        let pairs = operand_shape_duplicate_pairs(&ops);
        assert!(
            pairs.is_empty(),
            "distinct effects must close operand shape candidate"
        );
    }

    /// WHY: Section 182.13.8 allows shared implementation families to close an operand shape candidate.
    #[test]
    fn same_implementation_family_closes_operand_shape_candidate() {
        let entry: Vec<u8> = (0..PREFIX_LEN as u8).collect();
        let mut left = entry.clone();
        left.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2]);
        let mut right = entry;
        right.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA3]);
        let ops = vec![
            op_with_fingerprint("vyre-primitives::hardware::subgroup_add", left),
            op_with_fingerprint("vyre-primitives::hardware::subgroup_ballot", right),
        ];
        let pairs = operand_shape_duplicate_pairs(&ops);
        assert!(
            pairs.is_empty(),
            "same implementation family must close operand shape candidate"
        );
    }

    /// WHY: Section 182.13.9 allows reviewed distinct operations to close an operand shape candidate.
    #[test]
    fn reviewed_distinct_operations_close_operand_shape_candidate() {
        let entry: Vec<u8> = (0..PREFIX_LEN as u8).collect();
        let mut left = entry.clone();
        left.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2]);
        let mut right = entry;
        right.extend([0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA2, 0xA1, 0xA3]);
        let ops = vec![
            op_with_fingerprint("vyre-libs::math::fft::fft4_complex", left),
            op_with_fingerprint("vyre-libs::hash::blake3_g", right),
        ];
        let pairs = operand_shape_duplicate_pairs(&ops);
        assert!(
            pairs.is_empty(),
            "reviewed distinct operations must close operand shape candidate"
        );
    }

    #[test]
    fn global_reductions_have_no_operand_shape_pairs() {
        let ops = collect_ops(&mut Report::clean());
        assert_no_global_reduce_pairs(
            &operand_shape_duplicate_pairs(&ops),
            "all global reduction and indexed move operand-shape pairs must be closed",
        );
    }
}
