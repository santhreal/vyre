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
/// `declared` is the tier the live operation registry mints for every id it
/// declares, supplied by the caller. It used to be an unread argument beside a
/// second read of the live registry inside the rule, which made every test here
/// a test against whatever this checkout mints: the fixture registry a case
/// passed in decided nothing. A row naming something outside `declared` is a
/// blocker rather than a silent line, because a generated document that carries
/// a name nothing registers reads as coverage the tree does not have.
pub(super) fn validate_records(
    records: &[OpRecord],
    declared: &BTreeMap<&str, OpTier>,
) -> Vec<String> {
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
            tier: OpTier::Library,
            owners: vec!["vyre-libs".to_string()],
            ops: vec![op.to_string()],
            registry_sources: vec!["vyre-libs::bitset".to_string()],
            duplicate_ok: false,
            reference: "supported",
            foundation_ir: "supported",
            cuda: "supported",
            wgpu: "supported",
            spirv: "experimental",
            release_blocking_notes: String::new(),
            tests: vec!["vyre-libs/tests/op.rs".to_string()],
        }
    }

    /// The registry a case judges its rows against.
    fn declared(entries: &[(&'static str, OpTier)]) -> BTreeMap<&'static str, OpTier> {
        entries.iter().copied().collect()
    }

    /// WHY: the matrix understated the op surface by carrying an IR rewrite as an
    /// op name nothing registered. This proves the rule goes red on such a name
    /// and silent on a registered one, against the registry the case states
    /// rather than against whatever this checkout mints.
    #[test]
    fn an_op_with_no_live_registration_blocks_the_matrix() {
        let registry = declared(&[("vyre-libs::bitset::and", OpTier::Library)]);

        let registered_row = [row("bitset_and", "vyre-libs::bitset::and")];
        let live = validate_records(&registered_row, &registry);
        assert!(live.is_empty(), "registered op must not block: {live:?}");

        let invented_row = [row(
            "integer_strength_reduction",
            "mul_power_of_two_to_shift",
        )];
        let invented = validate_records(&invented_row, &registry);
        assert_eq!(
            invented,
            vec!["Fix: op `mul_power_of_two_to_shift` in OP_MATRIX family \
                 `integer_strength_reduction` has no live registration. Delete the row or restore the registration."
                .to_string()]
        );
    }

    /// WHY: 154 rows recorded `intrinsic` for compositions that had moved to
    /// `vyre-libs`, and the rule that should have caught it compared the id
    /// namespace with itself. The registration is the independent authority, so a
    /// row whose tier disagrees with the stated registry must report. This case
    /// could not exist while the rule read the live registry: no fixture could
    /// state a tier that disagreed with a real registration.
    #[test]
    fn a_row_whose_tier_disagrees_with_the_registration_blocks_the_matrix() {
        let registry = declared(&[("vyre-libs::bitset::and", OpTier::Intrinsic)]);
        let rows = [row("bitset_and", "vyre-libs::bitset::and")];

        let problems = validate_records(&rows, &registry);
        assert_eq!(
            problems,
            vec![
                "Fix: op `vyre-libs::bitset::and` is registered as Intrinsic but OP_MATRIX \
                 family `bitset_and` records Library. Regenerate the matrix through \
                 `op-matrix --write`."
                    .to_string()
            ]
        );
    }

    /// WHY: a `manual.` source states the ops are not registrations, which is how
    /// six correct rows for IR rewrites and benchmark cases stopped being
    /// reported. The other direction is the defect: an id the registry mints has
    /// no business being declared by hand, and only a stated registry can tell
    /// the two apart.
    #[test]
    fn a_hand_declared_row_for_a_minted_op_blocks_and_an_unminted_one_does_not() {
        let registry = declared(&[("vyre-libs::bitset::and", OpTier::Library)]);
        let mut minted = row("bitset_and", "vyre-libs::bitset::and");
        minted.registry_sources = vec!["manual.optimizer".to_string()];
        let mut unminted = row("integer_strength_reduction", "mul_power_of_two_to_shift");
        unminted.registry_sources = vec!["manual.optimizer".to_string()];

        assert_eq!(
            validate_records(&[minted], &registry),
            vec![
                "Fix: op `vyre-libs::bitset::and` in OP_MATRIX family `bitset_and` is declared \
                 by hand and the registry mints it. Take the row from the registry instead, or \
                 drop the registration."
                    .to_string()
            ]
        );
        assert!(
            validate_records(&[unminted], &registry).is_empty(),
            "a hand-declared row for an id the registry never mints is correct as written"
        );
    }
}
