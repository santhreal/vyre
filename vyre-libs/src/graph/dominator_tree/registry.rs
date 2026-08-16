use super::depth::{dominator_tree_depth, OP_ID as DEPTH_OP_ID};
use super::intersect_step::{dominator_tree_intersect_step, OP_ID as INTERSECT_STEP_OP_ID};
use super::program::{dominator_tree_program, IDOM_NONE, OP_ID};

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || dominator_tree_program(4, 4, 4, "idom"),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0, 1, 2, 3, 3]),
                vyre_primitives::wire::pack_u32_slice(&[1, 2, 3, 0]),
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 1, 2, 3]),
                vyre_primitives::wire::pack_u32_slice(&[0, 1, 2, 0]),
                vyre_primitives::wire::pack_u32_slice(&[0; 4]),
                vyre_primitives::wire::pack_u32_slice(&[0; 4]),
            ]]
        }),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 1, 2]),
                vyre_primitives::wire::pack_u32_slice(&[0, 1, 2, 3]),
            ]]
        }),
    )
}

// The forest is a chain `0 <- 1 <- 2` with node 3 not yet reached, so the
// witness covers the upward walk and the `IDOM_NONE` guard that stops it.
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        DEPTH_OP_ID,
        || dominator_tree_depth(4, "idom", "dt_depth"),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 1, IDOM_NONE]),
                vyre_primitives::wire::pack_u32_slice(&[0; 4]),
            ]]
        }),
        Some(|| vec![vec![vyre_primitives::wire::pack_u32_slice(&[0, 1, 2, 0])]]),
    )
}

// A diamond `0 -> {1, 2} -> 3` whose node 3 is not yet reached, so one sweep
// takes the first-predecessor branch on nodes 1 and 2, both arms of the LCA
// descent on node 3, and reports movement.
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        INTERSECT_STEP_OP_ID,
        || dominator_tree_intersect_step(4, 4, "idom", "dt_depth"),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 1, 2, 4]),
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 1, 2]),
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 0, IDOM_NONE]),
                vyre_primitives::wire::pack_u32_slice(&[0, 1, 1, 0]),
                vyre_primitives::wire::pack_u32_slice(&[0]),
            ]]
        }),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 0, 0]),
                vyre_primitives::wire::pack_u32_slice(&[1]),
            ]]
        }),
    )
}
