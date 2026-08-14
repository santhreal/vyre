//! Generated property coverage for builtin and opaque operation wire tags.
//!
//! The builtin variant space each strategy draws from is owned by
//! `tests/support/spec_variant_tables.rs`. The round trip and the reserved-tag
//! bound asserted below are this suite's own contract.

mod spec_variants;

#[path = "../../tests/support/spec_variant_tables.rs"]
mod spec_variant_tables;

use proptest::prelude::*;
use spec_variant_tables::{
    builtin_atomic_ops, builtin_bin_ops, builtin_ternary_ops, builtin_un_ops,
};
use spec_variants::collective_op_strategy;
use vyre_spec::extension::{
    ExtensionAtomicOpId, ExtensionBinOpId, ExtensionTernaryOpId, ExtensionUnOpId,
};
use vyre_spec::{AtomicOp, BinOp, CollectiveOp, TernaryOp, UnOp};

fn extension_raw_id() -> impl Strategy<Value = u32> {
    any::<u32>().prop_map(|raw| raw | 0x8000_0000)
}

// Each builtin keeps the weight it had when every variant was its own
// `prop_oneof!` arm, so folding the table into one `select` arm does not hand
// the opaque arm half the corpus.
fn bin_op_strategy() -> impl Strategy<Value = BinOp> {
    let builtins = builtin_bin_ops();
    let weight = builtins.len() as u32;
    prop_oneof![
        weight => prop::sample::select(builtins),
        1 => extension_raw_id().prop_map(|raw| BinOp::Opaque(ExtensionBinOpId(raw))),
    ]
}

fn un_op_strategy() -> impl Strategy<Value = UnOp> {
    let builtins = builtin_un_ops();
    let weight = builtins.len() as u32;
    prop_oneof![
        weight => prop::sample::select(builtins),
        1 => extension_raw_id().prop_map(|raw| UnOp::Opaque(ExtensionUnOpId(raw))),
    ]
}

fn atomic_op_strategy() -> impl Strategy<Value = AtomicOp> {
    let builtins = builtin_atomic_ops();
    let weight = builtins.len() as u32;
    prop_oneof![
        weight => prop::sample::select(builtins),
        1 => extension_raw_id().prop_map(|raw| AtomicOp::Opaque(ExtensionAtomicOpId(raw))),
    ]
}

fn ternary_op_strategy() -> impl Strategy<Value = TernaryOp> {
    let builtins = builtin_ternary_ops();
    let weight = builtins.len() as u32;
    prop_oneof![
        weight => prop::sample::select(builtins),
        1 => extension_raw_id().prop_map(|raw| TernaryOp::Opaque(ExtensionTernaryOpId(raw))),
    ]
}

fn assert_builtin_tag_is_reserved(tag: Option<u8>) -> Result<(), TestCaseError> {
    if let Some(tag) = tag {
        prop_assert!(
            (1..=0x7f).contains(&tag),
            "Fix: builtin op tags must stay in 0x01..=0x7f"
        );
    }
    Ok(())
}

proptest! {
    #[test]
    fn generated_bin_ops_round_trip_and_keep_builtin_tags_reserved(op in bin_op_strategy()) {
        assert_builtin_tag_is_reserved(op.builtin_wire_tag())?;
        let encoded = serde_json::to_string(&op).expect("Fix: BinOp must serialize");
        let decoded: BinOp = serde_json::from_str(&encoded).expect("Fix: BinOp must deserialize");
        prop_assert_eq!(decoded, op);
    }

    #[test]
    fn generated_un_ops_round_trip_and_keep_builtin_tags_reserved(op in un_op_strategy()) {
        assert_builtin_tag_is_reserved(op.builtin_wire_tag())?;
        let encoded = serde_json::to_string(&op).expect("Fix: UnOp must serialize");
        let decoded: UnOp = serde_json::from_str(&encoded).expect("Fix: UnOp must deserialize");
        prop_assert_eq!(decoded, op);
    }

    #[test]
    fn generated_atomic_ops_round_trip_and_keep_builtin_tags_reserved(op in atomic_op_strategy()) {
        assert_builtin_tag_is_reserved(op.builtin_wire_tag())?;
        let encoded = serde_json::to_string(&op).expect("Fix: AtomicOp must serialize");
        let decoded: AtomicOp = serde_json::from_str(&encoded).expect("Fix: AtomicOp must deserialize");
        prop_assert_eq!(decoded, op);
    }

    #[test]
    fn generated_ternary_ops_round_trip_and_keep_builtin_tags_reserved(op in ternary_op_strategy()) {
        assert_builtin_tag_is_reserved(op.builtin_wire_tag())?;
        let encoded = serde_json::to_string(&op).expect("Fix: TernaryOp must serialize");
        let decoded: TernaryOp = serde_json::from_str(&encoded).expect("Fix: TernaryOp must deserialize");
        prop_assert_eq!(decoded, op);
    }

    #[test]
    fn generated_collective_ops_round_trip_and_decode_wire_tags(op in collective_op_strategy()) {
        assert_builtin_tag_is_reserved(Some(op.builtin_wire_tag()))?;
        prop_assert_eq!(
            CollectiveOp::from_wire_tag(op.builtin_wire_tag())
                .expect("Fix: CollectiveOp builtin tag must decode"),
            op
        );
        let encoded = serde_json::to_string(&op).expect("Fix: CollectiveOp must serialize");
        let decoded: CollectiveOp = serde_json::from_str(&encoded).expect("Fix: CollectiveOp must deserialize");
        prop_assert_eq!(decoded, op);
    }
}
