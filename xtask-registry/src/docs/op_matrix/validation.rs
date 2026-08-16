//! Every rule a merged set of matrix rows has to satisfy.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::operation::{
    classify_operation_id as classify_op_id, OperationTier as OpTier,
};

use super::record::OpRecord;

/// Every rule an op matrix row breaks.
///
/// This used to return the first violation and abort the run, so a second
/// duplicate family was invisible until the first was fixed. The count of
/// sentences returned here is what the gate's pin holds level.
///
/// `registered` is every id the live operation registry declares. A row naming
/// something outside it is a blocker rather than a silent line, because a
/// generated document that carries a name nothing registers reads as coverage
/// the tree does not have.
pub(super) fn validate_records(records: &[OpRecord], registered: &BTreeSet<&str>) -> Vec<String> {
    let mut problems = Vec::new();
    let mut families = BTreeSet::new();
    let mut ops = BTreeMap::<&str, &str>::new();
    for record in records {
        if !families.insert(record.family.as_str()) {
            problems.push(format!(
                "Fix: duplicate OP_MATRIX family `{}`.",
                record.family
            ));
        }
        if record.owners.is_empty() {
            problems.push(format!(
                "Fix: OP_MATRIX row `{}` has no owners.",
                record.family
            ));
        }
        if record.tests.is_empty() {
            problems.push(format!(
                "Fix: OP_MATRIX row `{}` has no tests.",
                record.family
            ));
        }
        for op in &record.ops {
            if let Some(first_family) = ops.insert(op, record.family.as_str()) {
                problems.push(format!(
                    "Fix: op `{op}` appears in both OP_MATRIX families `{first_family}` and `{}`.",
                    record.family
                ));
            }
            if !registered.contains(op.as_str()) {
                problems.push(format!(
                    "Fix: op `{op}` in OP_MATRIX family `{}` has no live registration. The matrix \
                     carries op families whose rows resolve to a registered id. An IR-level \
                     rewrite belongs to the optimizer pass catalog and its generated pass \
                     artifact, not here.",
                    record.family
                ));
            }
            // ROADMAP S7: an op id's namespace classification must match
            // its row's declared tier. A Category C record must not carry
            // `vyre-libs::` ops, and a Category A record must not carry
            // `vyre-primitives::` ops. Mismatches were the root cause of
            // the original S7 finding (some intrinsics shipped under
            // Category A ids, making op truth ambiguous to the matrix).
            let observed = classify_op_id(op);
            if observed != OpTier::Unknown && tier_id_mismatch(record.tier, observed) {
                problems.push(format!(
                    "Fix: op `{op}` is namespaced as {observed:?} but lives in OP_MATRIX family \
                     `{}` declared as {:?}. Move the id to the matching namespace, change the \
                     row tier, or split the row.",
                    record.family, record.tier,
                ));
            }
        }
    }
    problems
}

/// Two operation tiers mismatch when one is `Intrinsic` and the other
/// is `Library` (or vice versa), the ownership distinction guarded here.
fn tier_id_mismatch(declared: OpTier, observed: OpTier) -> bool {
    matches!(
        (declared, observed),
        (OpTier::Intrinsic, OpTier::Library) | (OpTier::Library, OpTier::Intrinsic)
    )
}

/// The rules live in a private module of the gate, so no integration test can
/// call [`validate_records`] and prove a rule is able to fail.
#[cfg(test)]
mod tests {
    use super::*;

    fn row(family: &str, op: &str) -> OpRecord {
        OpRecord {
            family: family.to_string(),
            tier: OpTier::Foundation,
            owners: vec!["vyre-foundation".to_string()],
            ops: vec![op.to_string()],
            registry_sources: vec!["vyre-foundation::operation".to_string()],
            duplicate_ok: false,
            reference: "supported",
            foundation_ir: "supported",
            cuda: "supported",
            wgpu: "supported",
            spirv: "experimental",
            release_blocking_notes: String::new(),
            tests: vec!["vyre-foundation/tests/op.rs".to_string()],
        }
    }

    /// WHY: the matrix understated the op surface by carrying an IR rewrite as an op
    /// name nothing registered. This proves the rule goes red on such a name, and
    /// silent on a registered one. It proves nothing about what the registry contains.
    #[test]
    fn an_op_with_no_live_registration_blocks_the_matrix() {
        let registered = BTreeSet::from(["vyre-primitives::bitset::and"]);

        let registered_row = [row("bitset_and", "vyre-primitives::bitset::and")];
        let live = validate_records(&registered_row, &registered);
        assert!(live.is_empty(), "registered op must not block: {live:?}");

        let invented_row = [row("integer_strength_reduction", "mul_power_of_two_to_shift")];
        let invented = validate_records(&invented_row, &registered);
        assert_eq!(
            invented,
            vec![
                "Fix: op `mul_power_of_two_to_shift` in OP_MATRIX family \
                 `integer_strength_reduction` has no live registration. The matrix carries op \
                 families whose rows resolve to a registered id. An IR-level rewrite belongs to \
                 the optimizer pass catalog and its generated pass artifact, not here."
                    .to_string()
            ]
        );
    }
}
