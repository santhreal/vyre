//! Frozen RFC-0004 collective operation contracts.

#[path = "../../tests/support/spec_variant_tables.rs"]
mod spec_variant_tables;

use spec_variant_tables::builtin_collective_ops;
use vyre_spec::CollectiveOp;

#[test]
fn collective_op_wire_tags_are_dense_and_frozen() {
    let cases = builtin_collective_ops();

    for (op, tag) in cases {
        assert_eq!(
            op.builtin_wire_tag(),
            tag,
            "Fix: RFC-0004 collective op tags are part of the public wire ABI."
        );
        assert_eq!(
            CollectiveOp::from_wire_tag(tag).expect("assigned tag must decode"),
            op,
            "Fix: CollectiveOp tag {tag} must decode to its frozen operator."
        );
    }
}

#[test]
fn collective_op_wire_decoder_rejects_unassigned_tags() {
    for tag in [0, 7, 0xff] {
        let error =
            CollectiveOp::from_wire_tag(tag).expect_err("unassigned collective op tags must fail");
        assert!(
            error.contains("Fix: unknown CollectiveOp tag"),
            "Fix: collective op decode failures must be actionable, got `{error}`."
        );
    }
}
