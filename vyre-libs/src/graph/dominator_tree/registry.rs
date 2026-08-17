use super::depth::dominator_tree_depth;
use super::intersect_step::dominator_tree_intersect_step;
use super::program::{dominator_tree_program, IDOM_NONE};

const OP_ID: &str = "vyre-libs::graph::dominator_tree";
const DEPTH_OP_ID: &str = "vyre-libs::graph::dominator_tree_depth";
const INTERSECT_STEP_OP_ID: &str = "vyre-libs::graph::dominator_tree_intersect_step";

const EXPECTED_DOM_TREE_IDOM_BYTES: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0];
const EXPECTED_DOM_TREE_ORDER_BYTES: [u8; 16] = [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0];
const EXPECTED_DOM_TREE_DEPTH_BYTES: [u8; 16] = [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
const EXPECTED_DOM_TREE_STEP_IDOM_BYTES: [u8; 16] =
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const EXPECTED_DOM_TREE_STEP_CHANGED_BYTES: [u8; 4] = [1, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
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
                EXPECTED_DOM_TREE_IDOM_BYTES.to_vec(),
                EXPECTED_DOM_TREE_ORDER_BYTES.to_vec(),
            ]]
        }),
    )
}

// The forest is a chain `0 <- 1 <- 2` with node 3 not yet reached, so the
// witness covers the upward walk and the `IDOM_NONE` guard that stops it.
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        DEPTH_OP_ID,
        || dominator_tree_depth(4, "idom", "dt_depth"),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0, 0, 1, IDOM_NONE]),
                vyre_primitives::wire::pack_u32_slice(&[0; 4]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_DOM_TREE_DEPTH_BYTES.to_vec()]]),
    )
}

// A diamond `0 -> {1, 2} -> 3` whose node 3 is not yet reached, so one sweep
// takes the first-predecessor branch on nodes 1 and 2, both arms of the LCA
// descent on node 3, and reports movement.
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
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
                EXPECTED_DOM_TREE_STEP_IDOM_BYTES.to_vec(),
                EXPECTED_DOM_TREE_STEP_CHANGED_BYTES.to_vec(),
            ]]
        }),
    )
}
