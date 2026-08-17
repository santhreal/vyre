//! Histogram range-count primitive.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id for summing a half-open histogram range.
pub const RANGE_COUNTS_U32_OP_ID: &str = "vyre-libs::reduce::range_counts_u32";

/// Build a body that assigns `out_var = sum(histogram[start..end])`.
///
/// The caller owns the `out_var` declaration so this body can be composed as a
/// child Region without relying on declarations leaking across Region scopes.
#[must_use]
pub fn range_counts_u32_body(histogram: &str, out_var: &str, start: u32, end: u32) -> Vec<Node> {
    vec![
        Node::assign(out_var, Expr::u32(0)),
        Node::loop_for(
            "i",
            Expr::u32(start),
            Expr::u32(end),
            vec![Node::assign(
                out_var,
                Expr::add(Expr::var(out_var), Expr::load(histogram, Expr::var("i"))),
            )],
        ),
    ]
}

/// Wrap a range-count body as a child of `parent_op_id`.
#[must_use]
pub fn range_counts_u32_child(
    parent_op_id: &str,
    histogram: &str,
    out_var: &str,
    start: u32,
    end: u32,
) -> Node {
    wrap_child_region(
        RANGE_COUNTS_U32_OP_ID,
        Ident::from(parent_op_id),
        range_counts_u32_body(histogram, out_var, start, end),
    )
}

/// Standalone range-count program for primitive conformance.
#[must_use]
pub fn range_counts_u32(histogram: &str, out: &str, start: u32, end: u32) -> Program {
    let mut body = vec![Node::let_bind("sum", Expr::u32(0))];
    body.extend(range_counts_u32_body(histogram, "sum", start, end));
    body.push(Node::store(out, Expr::u32(0), Expr::var("sum")));
    Program::wrapped(
        vec![
            BufferDecl::storage(histogram, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(256),
            BufferDecl::output(out, 1, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(RANGE_COUNTS_U32_OP_ID, body)],
    )
}


inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        RANGE_COUNTS_U32_OP_ID,
        || range_counts_u32("histogram", "out", 1, 4),
        Some(|| {
            let mut histogram = vec![0u8; 256 * 4];
            for (slot, value) in [(0usize, 9u32), (1, 2), (2, 3), (3, 5), (4, 11)] {
                histogram[slot * 4..slot * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
            vec![vec![histogram, vec![0; 4]]]
        }),
        Some(|| vec![vec![10u32.to_le_bytes().to_vec()]]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ref_sums_half_open_range() {
        assert_eq!(cpu_ref(&[9, 2, 3, 5, 11], 1, 4), 10);
    }
}
