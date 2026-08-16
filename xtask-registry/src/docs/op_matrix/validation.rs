//! Every rule a merged set of matrix rows has to satisfy.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::operation::OperationTier as OpTier;

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

        let invented_row = [row(
            "integer_strength_reduction",
            "mul_power_of_two_to_shift",
        )];
        let invented = validate_records(&invented_row, &registered);
        assert_eq!(
            invented,
            vec!["Fix: op `mul_power_of_two_to_shift` in OP_MATRIX family \
                 `integer_strength_reduction` has no live registration. The matrix carries op \
                 families whose rows resolve to a registered id. An IR-level rewrite belongs to \
                 the optimizer pass catalog and its generated pass artifact, not here."
                .to_string()]
        );
    }
}
