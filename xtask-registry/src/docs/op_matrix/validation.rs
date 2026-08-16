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
pub(super) fn validate_records(records: &[OpRecord]) -> Vec<String> {
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
