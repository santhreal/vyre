//! Generated property coverage for builtin and opaque operation wire tags.
//!
//! The operator strategies each proptest draws from are owned by
//! `tests/support/spec_op_strategies.rs`. The round trip and the reserved-tag
//! bound asserted below are this suite's own contract.

mod spec_variants;

#[path = "../../tests/support/spec_op_strategies.rs"]
mod spec_op_strategies;

use proptest::prelude::*;
use spec_op_strategies::{arb_atomic_op, arb_bin_op, arb_ternary_op, arb_un_op};
use spec_variants::collective_op_strategy;
use vyre_spec::{AtomicOp, BinOp, CollectiveOp, TernaryOp, UnOp};

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
    fn generated_bin_ops_round_trip_and_keep_builtin_tags_reserved(op in arb_bin_op()) {
        assert_builtin_tag_is_reserved(op.builtin_wire_tag())?;
        let encoded = serde_json::to_string(&op).expect("Fix: BinOp must serialize");
        let decoded: BinOp = serde_json::from_str(&encoded).expect("Fix: BinOp must deserialize");
        prop_assert_eq!(decoded, op);
    }

    #[test]
    fn generated_un_ops_round_trip_and_keep_builtin_tags_reserved(op in arb_un_op()) {
        assert_builtin_tag_is_reserved(op.builtin_wire_tag())?;
        let encoded = serde_json::to_string(&op).expect("Fix: UnOp must serialize");
        let decoded: UnOp = serde_json::from_str(&encoded).expect("Fix: UnOp must deserialize");
        prop_assert_eq!(decoded, op);
    }

    #[test]
    fn generated_atomic_ops_round_trip_and_keep_builtin_tags_reserved(op in arb_atomic_op()) {
        assert_builtin_tag_is_reserved(op.builtin_wire_tag())?;
        let encoded = serde_json::to_string(&op).expect("Fix: AtomicOp must serialize");
        let decoded: AtomicOp = serde_json::from_str(&encoded).expect("Fix: AtomicOp must deserialize");
        prop_assert_eq!(decoded, op);
    }

    #[test]
    fn generated_ternary_ops_round_trip_and_keep_builtin_tags_reserved(op in arb_ternary_op()) {
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
