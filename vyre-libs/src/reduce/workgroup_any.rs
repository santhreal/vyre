//! Workgroup-local OR reduction over a u32 scratch buffer.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id for workgroup-local u32 any reduction.
pub const WORKGROUP_ANY_U32_OP_ID: &str = "vyre-libs::reduce::workgroup_any_u32";

/// Build a body that assigns `out_var = bit_or(values[0..count])`.
#[must_use]
pub fn workgroup_any_u32_body(values: &str, out_var: &str, count: u32) -> Vec<Node> {
    workgroup_any_u32_body_prefixed(values, out_var, count, "i")
}

/// Build a body with a caller-selected loop variable name for repeated inlining.
#[must_use]
pub fn workgroup_any_u32_body_prefixed(
    values: &str,
    out_var: &str,
    count: u32,
    iter_var: &str,
) -> Vec<Node> {
    vec![
        Node::assign(out_var, Expr::u32(0)),
        Node::loop_for(
            iter_var,
            Expr::u32(0),
            Expr::u32(count),
            vec![Node::assign(
                out_var,
                Expr::bitor(Expr::var(out_var), Expr::load(values, Expr::var(iter_var))),
            )],
        ),
    ]
}

/// Wrap the workgroup any body as a child of `parent_op_id`.
#[must_use]
pub fn workgroup_any_u32_child(
    parent_op_id: &str,
    values: &str,
    out_var: &str,
    count: u32,
) -> Node {
    workgroup_any_u32_child_prefixed(parent_op_id, values, out_var, count, "i")
}

/// Wrap the workgroup any body with a prefixed loop variable for repeated
/// inlining under no-shadowing validation.
#[must_use]
pub fn workgroup_any_u32_child_prefixed(
    parent_op_id: &str,
    values: &str,
    out_var: &str,
    count: u32,
    iter_var: &str,
) -> Node {
    wrap_child_region(
        WORKGROUP_ANY_U32_OP_ID,
        Ident::from(parent_op_id),
        workgroup_any_u32_body_prefixed(values, out_var, count, iter_var),
    )
}

/// Standalone workgroup-any program for primitive-level conformance.
#[must_use]
pub fn workgroup_any_u32(values: &str, out: &str, count: u32) -> Program {
    let mut body = vec![Node::let_bind("any_changed", Expr::u32(0))];
    body.extend(workgroup_any_u32_body(values, "any_changed", count));
    body.push(Node::store(out, Expr::u32(0), Expr::var("any_changed")));
    Program::wrapped(
        vec![
            BufferDecl::storage(values, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count.max(1)),
            BufferDecl::output(out, 1, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(WORKGROUP_ANY_U32_OP_ID, body)],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        WORKGROUP_ANY_U32_OP_ID,
        || workgroup_any_u32("values", "out", 4),
        Some(|| vec![vec![
            vyre_primitives::wire::pack_u32_slice(&[0u32, 0, 7, 0]),
        ]]),
        Some(|| vec![vec![vec![0x07, 0x00, 0x00, 0x00]]]),
    )
}

#[cfg(test)]
mod tests {
    use vyre_reference::composition_witness::reduce_workgroup_any_witness as reference_workgroup_any;

    #[test]
    fn reference_ors_values() {
        assert_eq!(reference_workgroup_any(&[0, 2, 4, 0]), 6);
    }
}
