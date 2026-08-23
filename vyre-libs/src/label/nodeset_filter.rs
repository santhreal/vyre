//! Shared u32-per-node to packed-NodeSet filter kernel.
//!
//! Several primitives scan one u32 fact per node, test a compile-time
//! predicate, and atomically set the corresponding bit in a packed
//! NodeSet. Centralizing that skeleton prevents node-kind, label-family,
//! and future tag predicates from drifting at word boundaries.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id for the shared nodeset filter kernel skeleton.
pub const OP_ID: &str = "vyre-libs::label::nodeset_filter";
/// Compile-time predicate applied to each per-node u32 value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeSetFilter {
    /// Match exactly one u32 value.
    Eq(u32),
    /// Match when any bit in the mask is present.
    Intersects(u32),
}

impl NodeSetFilter {
    fn expr(self, value: Expr) -> Expr {
        match self {
            Self::Eq(expected) => Expr::eq(value, Expr::u32(expected)),
            Self::Intersects(mask) => Expr::ne(Expr::bitand(value, Expr::u32(mask)), Expr::u32(0)),
        }
    }
}

/// Build `nodeset_out = { v : filter(values[v]) }`.
#[must_use]
pub fn nodeset_filter(
    values: &str,
    nodeset_out: &str,
    node_count: u32,
    filter: NodeSetFilter,
) -> Program {
    nodeset_filter_program(OP_ID, values, nodeset_out, node_count, filter)
}

#[must_use]
pub(crate) fn nodeset_filter_program(
    op_id: &'static str,
    values: &str,
    nodeset_out: &str,
    node_count: u32,
    filter: NodeSetFilter,
) -> Program {
    let t = Expr::LogicalIndex { axis: 0 };
    let words = node_count.div_ceil(32);
    let value = Expr::load(values, t.clone());
    let body = vec![Node::if_then(
        filter.expr(value),
        vec![
            Node::let_bind("word_idx", Expr::shr(t.clone(), Expr::u32(5))),
            Node::let_bind(
                "bit",
                Expr::shl(Expr::u32(1), Expr::bitand(t.clone(), Expr::u32(31))),
            ),
            Node::let_bind(
                "_",
                Expr::atomic_or(nodeset_out, Expr::var("word_idx"), Expr::var("bit")),
            ),
        ],
    )];
    let inner = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(node_count)),
        body,
    )];
    let region = if op_id == OP_ID {
        wrap_anonymous_region(OP_ID, inner)
    } else {
        wrap_anonymous_region(
            op_id,
            vec![wrap_child_region(OP_ID, Ident::from(op_id), inner)],
        )
    };
    Program::wrapped(
        vec![
            BufferDecl::storage(values, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(node_count),
            BufferDecl::storage(nodeset_out, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(words),
        ],
        [256, 1, 1],
        vec![region],
    )
}

const EXPECTED_NODESET_FILTER_OUTPUT_BYTES: [u8; 4] = [6, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || nodeset_filter("values", "nodeset", 4, NodeSetFilter::Intersects(0b0010)),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0x01, 0x02, 0x06, 0x04]), to_bytes(&[0])]]
        }),
        Some(|| {
            vec![vec![EXPECTED_NODESET_FILTER_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_laws(&["idempotent"])
}

#[cfg(test)]
mod tests {
    use super::*;

    impl NodeSetFilter {
        fn matches(self, value: u32) -> bool {
            match self {
                Self::Eq(expected) => value == expected,
                Self::Intersects(mask) => (value & mask) != 0,
            }
        }
    }

    fn reference_nodeset_filter(values: &[u32], filter: NodeSetFilter) -> Vec<u32> {
        let mut out = vec![0_u32; values.len().div_ceil(32)];
        reference_nodeset_filter_into(values, filter, &mut out);
        out
    }

    fn reference_nodeset_filter_into(values: &[u32], filter: NodeSetFilter, out: &mut Vec<u32>) {
        let needed = values.len().div_ceil(32);
        out.clear();
        out.resize(needed, 0);
        for (node, &value) in values.iter().enumerate() {
            if filter.matches(value) {
                out[node / 32] |= 1_u32 << (node % 32);
            }
        }
    }

    fn scalar_ref(values: &[u32], filter: NodeSetFilter) -> Vec<u32> {
        reference_nodeset_filter(values, filter)
    }

    #[test]
    fn generated_filters_match_scalar_reference() {
        let mut state = 0xF117_EA5E_u32;
        for case in 0..4096_u32 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let len = (state as usize % 257).min(case as usize % 257);
            let exact = state.rotate_left(case & 31);
            let mask = 1_u32 << (case & 31);
            let filters = [NodeSetFilter::Eq(exact), NodeSetFilter::Intersects(mask)];
            let mut values = Vec::with_capacity(len);
            for index in 0..len {
                state = state.rotate_left(9) ^ (index as u32).wrapping_mul(0x9E37_79B9);
                let value = match index % 5 {
                    0 => exact,
                    1 => mask,
                    2 => exact ^ mask,
                    3 => !mask,
                    _ => state,
                };
                values.push(value);
            }
            for filter in filters {
                assert_eq!(
                    reference_nodeset_filter(&values, filter),
                    scalar_ref(&values, filter),
                    "case {case} filter {filter:?}"
                );
            }
        }
    }

    #[test]
    fn reference_into_reuses_output_and_clears_stale_tail() {
        let values = [1_u32, 2, 3, 4, 5, 6, 7, 8];
        let mut out = Vec::with_capacity(4);
        out.extend([u32::MAX; 4]);
        let ptr = out.as_ptr();
        reference_nodeset_filter_into(&values, NodeSetFilter::Intersects(0b1), &mut out);
        assert_eq!(out, vec![0b0101_0101]);
        assert_eq!(out.as_ptr(), ptr);

        reference_nodeset_filter_into(&[], NodeSetFilter::Eq(1), &mut out);
        assert!(out.is_empty());
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn reference_wrapper_matches_reference() {
        let values = [1_u32, 2, 3, 4, 5, 6, 7, 8];
        let filter = NodeSetFilter::Intersects(0b1);
        let mut compat = Vec::with_capacity(4);
        let mut reference = Vec::with_capacity(4);

        reference_nodeset_filter_into(&values, filter, &mut compat);
        reference_nodeset_filter_into(&values, filter, &mut reference);

        assert_eq!(compat, reference);
        assert_eq!(reference_nodeset_filter(&values, filter), reference);
    }
}
