//! Generated property coverage for collective operation wire contracts.

mod spec_variants;

use proptest::prelude::*;
use spec_variants::collective_op_strategy;
use vyre_spec::{CollectiveOp, CommGroup};

proptest! {
    #[test]
    fn generated_collective_ops_round_trip_through_wire_tags(op in collective_op_strategy()) {
        let tag = op.builtin_wire_tag();

        prop_assert!((1..=6).contains(&tag));
        prop_assert_eq!(
            CollectiveOp::from_wire_tag(tag)
                .expect("Fix: assigned CollectiveOp wire tag must decode"),
            op
        );
    }

    #[test]
    fn generated_collective_wire_decoder_accepts_only_assigned_tags(tag in any::<u8>()) {
        let decoded = CollectiveOp::from_wire_tag(tag);
        if (1..=6).contains(&tag) {
            prop_assert!(decoded.is_ok());
        } else {
            let error = decoded.expect_err("Fix: unassigned CollectiveOp tag must be rejected");
            prop_assert!(error.contains("Fix: unknown CollectiveOp tag"));
        }
    }

    #[test]
    fn generated_comm_group_exposes_exact_raw_id(raw in any::<u32>()) {
        prop_assert_eq!(CommGroup(raw).as_u32(), raw);
        prop_assert_eq!(CommGroup::WORLD.as_u32(), 0);
    }
}
