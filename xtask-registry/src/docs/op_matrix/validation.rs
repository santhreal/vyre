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
            //
            // Which authority a row answers to is the row's own declaration. A
            // `manual.` source says the ops are not registrations: an IR
            // rewrite the optimizer applies and a benchmark case are named in
            // the matrix and minted by nobody, so demanding a registration for
            // them reported six rows that are correct as written. The other
            // direction is the real defect, and it is reported: an op the
            // registry does mint has no business being declared by hand.
            let from_registry = record
                .registry_sources
                .iter()
                .any(|source| !source.starts_with("manual."));
            match declared.get(op.as_str()) {
                Some(registered) if *registered != record.tier => {
                    problems.push(format!(
                        "Fix: op `{op}` is registered as {registered:?} but OP_MATRIX family `{}` records {:?}. Regenerate the matrix through `op-matrix --write`.",
                        record.family, record.tier,
                    ));
                }
                Some(_) if !from_registry => problems.push(format!(
                    "Fix: op `{op}` in OP_MATRIX family `{}` is declared by hand and the registry mints it. Take the row from the registry instead, or drop the registration.",
                    record.family
                )),
                Some(_) => {}
                None if from_registry => problems.push(format!(
                    "Fix: op `{op}` in OP_MATRIX family `{}` has no live registration. Delete the row or restore the registration.",
                    record.family
                )),
                None => {}
            }
        }
    }
    problems
}

/// `validate_records` and `OpRecord` are crate-private, so no integration test
/// can hand this rule a row. What the gate reports over the live matrix is
/// asserted in `tests/registry_contracts/op_matrix.rs`.
#[cfg(test)]
mod tests {
    use super::*;

    fn row(family: &str, op: &str, source: &str) -> OpRecord {
        OpRecord {
            family: family.to_string(),
            tier: OpTier::Foundation,
            owners: vec!["vyre-foundation/src/optimizer".to_string()],
            ops: vec![op.to_string()],
            registry_sources: vec![source.to_string()],
            duplicate_ok: false,
            reference: "not_applicable",
            foundation_ir: "supported",
            cuda: "not_applicable",
            wgpu: "not_applicable",
            spirv: "not_applicable",
            release_blocking_notes: "IR rewrite".to_string(),
            tests: vec!["vyre-foundation/src/optimizer/tests/mod.rs".to_string()],
            bench_targets: Vec::new(),
        }
    }

    /// An IR rewrite the optimizer applies is minted by nobody, and the row
    /// says so. Demanding a registration for it reported six correct rows.
    #[test]
    fn a_hand_declared_op_needs_no_registration() {
        let problems = validate_records(&[row(
            "integer_strength_reduction",
            "mul_power_of_two_to_shift",
            "manual.foundation_ir",
        )]);

        assert_eq!(problems, Vec::<String>::new());
    }

    /// A row that answers to the registry still has to name a registration.
    #[test]
    fn a_registry_sourced_op_that_nothing_mints_is_reported() {
        let problems = validate_records(&[row(
            "departed",
            "vyre-libs::math::departed",
            "vyre-foundation::operation",
        )]);

        assert_eq!(
            problems,
            vec![
                "Fix: op `vyre-libs::math::departed` in OP_MATRIX family `departed` has no live registration. Delete the row or restore the registration.".to_string()
            ]
        );
    }

    /// The other direction: an op the registry mints is not declared by hand.
    #[test]
    fn a_hand_declared_op_the_registry_mints_is_reported() {
        let registry = vyre_registry_link::operation::live_operation_registry();
        let minted = registry
            .iter()
            .next()
            .expect("Fix: the live registry mints at least one operation");
        let mut record = row("hand_written", minted.id, "manual.foundation_ir");
        record.tier = minted.tier;

        assert_eq!(
            validate_records(&[record]),
            vec![format!(
                "Fix: op `{}` in OP_MATRIX family `hand_written` is declared by hand and the registry mints it. Take the row from the registry instead, or drop the registration.",
                minted.id
            )]
        );
    }
}
