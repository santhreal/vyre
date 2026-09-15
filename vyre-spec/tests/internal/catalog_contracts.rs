//! Test: catalog contracts.
use crate::{
    by_category, by_id, catalog_is_complete, expr_variants, invariants, Category,
    InvariantCategory, InvariantId,
};
use std::collections::BTreeSet;

#[test]
fn catalog_is_complete_holds() {
    assert!(
        catalog_is_complete(),
        "INVARIANTS must contain exactly I1..I15 in order"
    );
}

#[test]
#[allow(clippy::expect_used, clippy::clone_on_copy)]
fn by_id_roundtrips_every_invariant() {
    for inv in invariants() {
        let resolved = by_id(inv.id.clone())
            .expect("Fix: every invariant id in the catalog must round-trip through by_id");
        assert_eq!(resolved.id, inv.id, "by_id lost identity for {}", inv.id);
        assert!(
            !resolved.name.is_empty(),
            "invariant {} has empty name",
            inv.id
        );
        assert!(
            !resolved.description.is_empty(),
            "invariant {} has empty description",
            inv.id
        );
    }
}

#[test]
fn every_invariant_declares_real_test_family() {
    for inv in invariants() {
        let family = (inv.test_family)();
        assert!(
            family.len() >= 2,
            "Fix: invariant {} must declare at least happy and adversarial TestDescriptors",
            inv.id
        );

        for descriptor in family {
            assert_eq!(
                descriptor.invariant, inv.id,
                "Fix: descriptor {} is attached to the wrong invariant",
                descriptor.name
            );
            assert!(
                descriptor.name.starts_with("conform/") && descriptor.name.contains(".rs::"),
                "Fix: descriptor {} must use conform/<path>/<file>.rs::<test_fn>",
                descriptor.name
            );
            assert!(
                !descriptor.purpose.is_empty(),
                "Fix: descriptor {} must document the invariant behavior it probes",
                descriptor.name
            );
        }
    }
}

#[test]
fn invariant_test_descriptor_names_are_unique() {
    let mut names = BTreeSet::new();
    for inv in invariants() {
        for descriptor in (inv.test_family)() {
            assert!(
                names.insert(descriptor.name),
                "Fix: descriptor {} is assigned to multiple invariant slots",
                descriptor.name
            );
        }
    }
}

#[test]
#[allow(clippy::expect_used)]
fn i4_is_wire_format_not_bytecode() {
    let i4 = by_id(InvariantId::I4).expect("Fix: invariant I4 must be present in the catalog");
    let text = format!("{} {}", i4.name, i4.description);
    assert!(
        !text.to_lowercase().contains("bytecode"),
        "I4 must not reference the retired 'bytecode' terminology"
    );
    assert!(
        i4.description.contains("wire"),
        "I4 description must name the wire format"
    );
}

/// Names the conform parity matrix keys its per-variant rows by. A catalog that
/// shrinks silently drops rows from that matrix, and a name that is not an
/// identifier can never match one.
const EXPR_VARIANT_FLOOR: usize = 24;

/// The catalog is the row key set of the conform parity matrix, so it must not
/// shrink and every entry must be shaped like an `Expr` variant name.
///
/// This deliberately does not restate the list. A second copy of the same 24
/// strings would fail only when someone edited one copy, which is a diff nobody
/// can land by accident, while proving nothing about the catalog itself. A
/// renamed variant is caught where it matters, by the parity matrix failing to
/// find a row for the new key.
#[test]
fn expr_variant_catalog_is_unique_and_well_formed() {
    let actual = expr_variants();
    assert!(
        actual.len() >= EXPR_VARIANT_FLOOR,
        "Fix: the expr variant catalog lists {} names, below the {EXPR_VARIANT_FLOOR} the parity \
         matrix expects rows for; restore the removed variant or lower the floor with the \
         removal",
        actual.len()
    );
    let unique = actual.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(
        unique.len(),
        actual.len(),
        "Fix: expr variant catalog contains duplicate entries"
    );
    for variant in actual {
        assert!(
            variant.starts_with(|first: char| first.is_ascii_uppercase())
                && variant.chars().all(char::is_alphanumeric),
            "Fix: expr variant `{variant}` is not an `Expr` variant name, so no parity matrix row \
             can key on it"
        );
    }
}

#[test]
fn invariants_partition_by_category() {
    let exec = by_category(InvariantCategory::Execution).count();
    let alg = by_category(InvariantCategory::Algebra).count();
    let res = by_category(InvariantCategory::Resource).count();
    let stab = by_category(InvariantCategory::Stability).count();
    assert_eq!(
        exec + alg + res + stab,
        invariants().len(),
        "categories must partition the catalog exactly"
    );
}

#[test]
fn category_unclassified_is_round_trippable() {
    let cat = Category::unclassified();
    assert!(cat.is_unclassified());
}
