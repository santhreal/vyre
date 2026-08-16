//! Cat-C `subgroup_shuffle`  -  per-lane permutation via source-lane indices.
//! Maps to hardware `subgroupShuffle()`.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::hardware::MAP_WORKGROUP;
use crate::wire::pack_u32_slice as pack_u32;
/// Canonical op id shared by semantics, fixtures, and driver registration.
pub const OP_ID: &str = "vyre-primitives::hardware::subgroup_shuffle";

/// Build a Program that maps `out[i] = values[lanes[i]]` across the subgroup.
///
/// Both operands of a shuffle are evaluated at EVERY lane index in the subgroup, not
/// only at the indices a store guard admits, because every lane publishes a value its
/// peers may select. Writing `load(values, idx)` and `load(lanes, idx)` as the operands
/// therefore reads both buffers at the index of every lane in the subgroup, past the
/// end of them for a dispatch wider than the buffers, no matter what branch the
/// collective sits in. The operands are control-flow-guarded per-lane locals instead,
/// and the collective itself sits in uniform control flow, where a subgroup
/// collective's participating-lane set is well defined rather than dependent on the
/// active mask of a divergent branch.
#[must_use]
pub fn subgroup_shuffle(values: &str, lanes: &str, out: &str, n: u32) -> Program {
    let body = vec![wrap_anonymous_region(
        OP_ID,
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::let_bind("lane_value", Expr::u32(0)),
            Node::let_bind("src_lane", Expr::u32(0)),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(values)),
                vec![Node::assign(
                    "lane_value",
                    Expr::load(values, Expr::var("idx")),
                )],
            ),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(lanes)),
                vec![Node::assign(
                    "src_lane",
                    Expr::load(lanes, Expr::var("idx")),
                )],
            ),
            Node::let_bind(
                "shuffled",
                Expr::SubgroupShuffle {
                    value: Box::new(Expr::var("lane_value")),
                    lane: Box::new(Expr::var("src_lane")),
                },
            ),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len(out)),
                vec![Node::store(out, Expr::var("idx"), Expr::var("shuffled"))],
            ),
        ],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage(values, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(lanes, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::output(out, 2, DataType::U32).with_count(n),
        ],
        MAP_WORKGROUP,
        body,
    )
}

/// CPU reference for the subgroup shuffle intrinsic.
///
/// # Panics
/// Panics when a lane index resolves outside `values`. The oracle cannot produce a
/// meaningful reference value for an out-of-range shuffle, and returning a filler
/// byte would make the GPU comparison pass on invalid input.
fn cpu_ref(values: &[u32], lanes: &[u32]) -> Vec<u8> {
    const SUBGROUP_WIDTH: usize = 32;
    let n = values.len().min(lanes.len());
    let mut out = Vec::with_capacity(n);
    for (i, lane) in lanes.iter().take(n).enumerate() {
        let subgroup_start = (i / SUBGROUP_WIDTH) * SUBGROUP_WIDTH;
        let src = subgroup_start + (*lane as usize);
        out.push(values.get(src).copied().unwrap_or_else(|| {
            panic!(
                "Fix: subgroup_shuffle cpu_ref OOB: lane {lane} in subgroup_start \
                     {subgroup_start} resolves to src index {src} which exceeds values \
                     len {}; the GPU oracle cannot produce a meaningful reference value \
                     for an out-of-bounds lane index",
                values.len()
            )
        }));
    }
    pack_u32(&out)
}

fn test_inputs() -> Vec<Vec<Vec<u8>>> {
    let values = vec![10u32, 20, 30, 40];
    let lanes = vec![0u32, 1, 0, 2];
    let len = values.len() * 4;
    vec![vec![pack_u32(&values), pack_u32(&lanes), vec![0u8; len]]]
}

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    let values = vec![10u32, 20, 30, 40];
    let lanes = vec![0u32, 1, 0, 2];
    vec![vec![cpu_ref(&values, &lanes)]]
}

crate::submit_intrinsic_operation! {
    id: OP_ID,
    signature: Some(crate::hardware::catalog::U32_BINARY_SIGNATURE),
    build: || subgroup_shuffle("values", "lanes", "out", 4),
    inputs: test_inputs,
    expected: expected_output
}

inventory::submit! {
    crate::hardware::catalog::IntrinsicFacet {
        operation_id: OP_ID,
        shape: crate::hardware::catalog::OpShape::new(
            2,
            1,
            4,
            crate::hardware::catalog::HardwareSemantic::SubgroupShuffleU32,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::run_program;

    fn assert_case(values: &[u32], lanes: &[u32]) {
        let n = values.len() as u32;
        let program = subgroup_shuffle("values", "lanes", "out", n.max(1));
        let outputs = run_program(
            &program,
            vec![
                pack_u32(values),
                pack_u32(lanes),
                vec![0u8; (n.max(1) * 4) as usize],
            ],
        );
        assert_eq!(outputs, vec![cpu_ref(values, lanes)]);
    }

    #[test]
    fn lane_zero_passes_through() {
        assert_case(&[7, 9, 11], &[0, 0, 0]);
    }
    #[test]
    fn nonzero_lane_in_bounds() {
        // All lane indices are in-bounds for the 3-element subgroup window.
        assert_case(&[7, 9, 11], &[1, 2, 0]);
    }
    #[test]
    fn mixed() {
        assert_case(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1, 0, 2, 0, 0, 3, 4]);
    }

    #[test]
    fn cpu_ref_oob_lane_panics_with_actionable_message() {
        // lane index 32 in a 32-wide subgroup resolves to src = 0 + 32 = 32,
        // which is out-of-bounds for a 4-element values slice.  The oracle
        // must panic loudly rather than silently substituting 0.
        let values = vec![10u32, 20u32, 30u32, 40u32];
        let lanes = vec![32u32]; // OOB: lane 32 >= SUBGROUP_WIDTH(32)
        let result = std::panic::catch_unwind(|| cpu_ref(&values, &lanes));
        let err = result.expect_err("expected cpu_ref to panic on OOB lane index");
        let msg = err
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| err.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic>");
        assert!(
            msg.contains("Fix:") && msg.contains("OOB"),
            "panic message must contain 'Fix:' and 'OOB' to be actionable, got: {msg}"
        );
    }
}
