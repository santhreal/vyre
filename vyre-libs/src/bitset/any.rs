//! `bitset_any`  -  emit 1 when any bit in the packed bitset is set.
//!
//! One workgroup ORs the packed words: every lane walks a strided slice of the
//! bitset and stops loading once it has seen a set bit, then the lane verdicts
//! collapse through a workgroup reduction and lane 0 writes the boolean to
//! `out[0]`. Used by source-query dialect `exists` / `any(...)` aggregate
//! lowerings.

use vyre_foundation::composition::wrap_anonymous_region;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, PORTABLE_WORKGROUP_INVOCATIONS,
};

use crate::builder::cooperative::for_each_index;
use crate::reduce::workgroup_tree::{max_u32_child, WorkgroupReductionScope};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::bitset::any";

/// Workgroup scratch the lane verdicts reduce through, one u32 entry per lane.
const ANY_SCRATCH: &str = "bitset_any_scratch";

/// Build a Program: `out[0] = 1` iff any bit of `input` is set.
///
/// AUDIT_2026-04-24 F-ANY-01: a lane stops loading once it has observed a
/// non-zero word. The IR has no `break`, so the escape is a per-lane `found`
/// flag gating the load: later iterations of that lane's walk become empty
/// bodies and its scan cost degrades to O(first set word in its slice) instead
/// of O(its slice). Bitsets are typically sparse (e.g. taint frontiers with one
/// or two set bits) so the average cut is large, and the walk itself is now
/// one lane's slice of the words rather than all of them.
#[must_use]
pub fn bitset_any(input: &str, out: &str, words: u32) -> Program {
    let mut body = vec![
        Node::let_bind("local", Expr::LocalId { axis: 0 }),
        Node::let_bind("found", Expr::u32(0)),
        Node::if_then(
            Expr::is_first_workgroup(),
            vec![Node::store(ANY_SCRATCH, Expr::var("local"), Expr::u32(0))],
        ),
        Node::barrier(),
        Node::if_then(
            Expr::is_first_workgroup(),
            vec![for_each_index(
                words,
                PORTABLE_WORKGROUP_INVOCATIONS,
                "w",
                vec![Node::if_then(
                    Expr::eq(Expr::var("found"), Expr::u32(0)),
                    vec![Node::if_then(
                        Expr::ne(Expr::load(input, Expr::var("w")), Expr::u32(0)),
                        vec![
                            Node::assign("found", Expr::u32(1)),
                            Node::store(ANY_SCRATCH, Expr::var("local"), Expr::u32(1)),
                        ],
                    )],
                )],
            )],
        ),
        Node::barrier(),
    ];
    // A max over lane verdicts is the OR of them: the slot is 1 exactly when some lane saw a set
    // bit, which is the answer, and it costs a log-depth tree instead of a second serial pass.
    body.push(max_u32_child(
        OP_ID,
        PORTABLE_WORKGROUP_INVOCATIONS,
        ANY_SCRATCH,
        WorkgroupReductionScope::FirstWorkgroup,
    ));
    body.push(Node::if_then(
        Expr::and(
            Expr::is_first_workgroup(),
            Expr::eq(Expr::var("local"), Expr::u32(0)),
        ),
        vec![Node::store(
            out,
            Expr::u32(0),
            Expr::load(ANY_SCRATCH, Expr::u32(0)),
        )],
    ));
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(words),
            BufferDecl::storage(out, 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::workgroup(ANY_SCRATCH, PORTABLE_WORKGROUP_INVOCATIONS, DataType::U32),
        ],
        [PORTABLE_WORKGROUP_INVOCATIONS, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

const EXPECTED_BITSET_ANY_OUTPUT_BYTES: [u8; 4] = [1, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || bitset_any("input", "out", 2),
        Some(|| {
            vec![vec![
                vec![0, 0, 0, 0, 1, 0, 0, 0],
                vec![0, 0, 0, 0],
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_BITSET_ANY_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::composition_witness::reduce_any_witness as reference_bitset_any;

    #[test]
    fn any_true_when_single_bit_set() {
        assert_eq!(reference_bitset_any(&[0, 1]), 1);
    }

    #[test]
    fn any_false_when_all_zero() {
        assert_eq!(reference_bitset_any(&[0, 0]), 0);
    }

    #[test]
    fn registration_fixture_matches_exact_byte_constant() {
        assert_eq!(EXPECTED_BITSET_ANY_OUTPUT_BYTES, [1, 0, 0, 0]);
        let cpu_ref = reference_bitset_any(&[0, 1]);
        assert_eq!(cpu_ref.to_le_bytes(), EXPECTED_BITSET_ANY_OUTPUT_BYTES);
    }

}
