//! Every rule a merged set of matrix rows has to satisfy.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::operation::OperationTier as OpTier;

use super::record::OpRecord;

/// Every rule an op matrix row breaks.
///
/// This used to return the first violation and abort the run, so a second
/// duplicate family was invisible until the first was fixed. The count of
/// sentences returned here is what the gate's pin holds level.
pub(super) fn validate_records(records: &[OpRecord]) -> Vec<String> {
    let mut problems = Vec::new();
    let registry = vyre_registry_link::operation::live_operation_registry();
    let declared: BTreeMap<&str, OpTier> = registry
        .iter()
        .map(|entry| (entry.id, entry.tier))
        .collect();
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
            // A row's tier must be the tier its registration declares. Both
            // sides used to read the id namespace, which is frozen at mint
            // time, so the rule compared one fact with itself and could not
            // fire: 154 rows recorded `intrinsic` for compositions that had
            // moved to `vyre-libs`. The registration is the independent
            // authority, and a stale row now reports.
            match declared.get(op.as_str()) {
                Some(registered) if *registered != record.tier => {
                    problems.push(format!(
                        "Fix: op `{op}` is registered as {registered:?} but OP_MATRIX family `{}` records {:?}. Regenerate the matrix through `op-matrix --write`.",
                        record.family, record.tier,
                    ));
                }
                Some(_) => {}
                None => problems.push(format!(
                    "Fix: op `{op}` in OP_MATRIX family `{}` has no live registration. Delete the row or restore the registration.",
                    record.family
                )),
            }
        }
    }
    problems
}
