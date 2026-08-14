//! The shared operator variant tables must cover the frozen surface exactly.
//!
//! WHY: four suites used to enumerate `BinOp`, `UnOp`, `AtomicOp` and
//! `TernaryOp` from four private copies, and every copy was missing variants a
//! different one had. Giving the tables one owner removes the drift between
//! copies but not the drift that matters most, which is between the table and
//! the enum: a variant added to `vyre-spec` and to no table is a wire tag, a
//! round trip and a parity window nobody ever exercised.
//!
//! So the expectation is not a count written here. It is derived at run time
//! from `docs/public-api/vyre-spec.txt`, which `scripts/check_public_api_snapshot.sh`
//! regenerates from rustdoc and holds byte-stable, so a new variant reaches
//! this test through the same gate that already forces a snapshot refresh and
//! turns it red until the tables record a decision for it.
//!
//! What this does NOT catch: a variant added to the enum and never blessed into
//! the snapshot. That case fails in the public-API gate instead, which is the
//! gate that owns it.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

#[path = "../../tests/support/spec_variant_tables.rs"]
mod spec_variant_tables;

use spec_variant_tables::{
    builtin_atomic_ops, builtin_bin_ops, builtin_ternary_ops, builtin_un_ops,
    public_api_variant_names,
};

fn debug_names<T: std::fmt::Debug>(variants: &[T]) -> BTreeSet<String> {
    variants
        .iter()
        .map(|variant| format!("{variant:?}"))
        .collect()
}

fn assert_table_matches_surface<T: std::fmt::Debug>(enum_name: &str, table: &[T]) {
    let listed = debug_names(table);
    assert_eq!(
        listed.len(),
        table.len(),
        "Fix: the {enum_name} table lists a variant twice; each variant appears once."
    );

    let frozen = public_api_variant_names(enum_name);
    let missing: Vec<&String> = frozen.difference(&listed).collect();
    assert!(
        missing.is_empty(),
        "Fix: tests/support/spec_variant_tables.rs omits {enum_name} variant(s) {missing:?}. \
         Add each one, then check whether the suites that read this table (wire round trips, \
         wire tag pins, the random-IR corpus, the f32 parity window) need a decision for it."
    );
    let unknown: Vec<&String> = listed.difference(&frozen).collect();
    assert!(
        unknown.is_empty(),
        "Fix: tests/support/spec_variant_tables.rs lists {enum_name} name(s) {unknown:?} that the \
         frozen public surface does not have. Remove them or refresh the snapshot."
    );
}

#[test]
fn bin_op_table_covers_every_frozen_builtin() {
    assert_table_matches_surface("BinOp", &builtin_bin_ops());
}

#[test]
fn un_op_table_covers_every_frozen_builtin() {
    assert_table_matches_surface("UnOp", &builtin_un_ops());
}

#[test]
fn atomic_op_table_covers_every_frozen_builtin() {
    assert_table_matches_surface("AtomicOp", &builtin_atomic_ops());
}

#[test]
fn ternary_op_table_covers_every_frozen_builtin() {
    assert_table_matches_surface("TernaryOp", &builtin_ternary_ops());
}

#[test]
fn every_table_entry_carries_a_reserved_builtin_wire_tag() {
    // A table entry that is not a builtin has no place here: `Opaque` is the
    // extension escape hatch and each suite draws its own id for it. This also
    // catches a table that reached for a variant from the wrong enum.
    for op in builtin_bin_ops() {
        let tag = op
            .builtin_wire_tag()
            .unwrap_or_else(|| panic!("Fix: BinOp table entry {op:?} has no builtin wire tag."));
        assert!((1..=0x7f).contains(&tag), "BinOp {op:?} tag {tag:#04x}");
    }
    for op in builtin_un_ops() {
        let tag = op
            .builtin_wire_tag()
            .unwrap_or_else(|| panic!("Fix: UnOp table entry {op:?} has no builtin wire tag."));
        assert!((1..=0x7f).contains(&tag), "UnOp {op:?} tag {tag:#04x}");
    }
    for op in builtin_atomic_ops() {
        let tag = op
            .builtin_wire_tag()
            .unwrap_or_else(|| panic!("Fix: AtomicOp table entry {op:?} has no builtin wire tag."));
        assert!((1..=0x7f).contains(&tag), "AtomicOp {op:?} tag {tag:#04x}");
    }
    for op in builtin_ternary_ops() {
        let tag = op.builtin_wire_tag().unwrap_or_else(|| {
            panic!("Fix: TernaryOp table entry {op:?} has no builtin wire tag.")
        });
        assert!((1..=0x7f).contains(&tag), "TernaryOp {op:?} tag {tag:#04x}");
    }
}
