//! Value-to-bitset filter operations over packed `u32` arrays.
//!
//! Scans one `u32` value per element, evaluates a compile-time predicate, and
//! atomically sets the corresponding bit in a packed `u32` bitset.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

/// Canonical op id for the shared value-to-bitset filter kernel.
pub const OP_ID: &str = "vyre-libs::bitset::value_filter";
/// Canonical op id for exact `u32` equality filtering into a bitset.
pub const VALUE_EQ_OP_ID: &str = "vyre-libs::bitset::value_eq";
/// Canonical op id for bitmask intersection filtering into a bitset.
pub const MASK_INTERSECTS_OP_ID: &str = "vyre-libs::bitset::mask_intersects";

/// Compile-time predicate applied to each `u32` element.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueFilter {
    /// Match when `values[i] == expected`.
    Eq(u32),
    /// Match when `(values[i] & mask) != 0`.
    Intersects(u32),
}

impl ValueFilter {
    fn expr(self, value: Expr) -> Expr {
        match self {
            Self::Eq(expected) => Expr::eq(value, Expr::u32(expected)),
            Self::Intersects(mask) => Expr::ne(Expr::bitand(value, Expr::u32(mask)), Expr::u32(0)),
        }
    }
}

/// Build `bitset_out = { i : filter(values[i]) }` over `element_count` items.
#[must_use]
pub fn bitset_value_filter(
    values: &str,
    bitset_out: &str,
    element_count: u32,
    filter: ValueFilter,
) -> Program {
    value_filter_program(OP_ID, values, bitset_out, element_count, filter)
}

/// Build `bitset_out = { i : values[i] == expected }` over `element_count` items.
#[must_use]
pub fn bitset_value_eq(
    values: &str,
    bitset_out: &str,
    element_count: u32,
    expected: u32,
) -> Program {
    value_filter_program(
        VALUE_EQ_OP_ID,
        values,
        bitset_out,
        element_count,
        ValueFilter::Eq(expected),
    )
}

/// Build `bitset_out = { i : (values[i] & mask) != 0 }` over `element_count` items.
#[must_use]
pub fn bitset_mask_intersects(
    values: &str,
    bitset_out: &str,
    element_count: u32,
    mask: u32,
) -> Program {
    value_filter_program(
        MASK_INTERSECTS_OP_ID,
        values,
        bitset_out,
        element_count,
        ValueFilter::Intersects(mask),
    )
}

#[must_use]
pub(crate) fn value_filter_program(
    op_id: &'static str,
    values: &str,
    bitset_out: &str,
    element_count: u32,
    filter: ValueFilter,
) -> Program {
    let words = element_count.div_ceil(32);
    let value = Expr::load(values, Expr::LogicalIndex { axis: 0 });
    let body = vec![Node::if_then(
        filter.expr(value),
        vec![
            Node::let_bind(
                "word_idx",
                Expr::shr(Expr::LogicalIndex { axis: 0 }, Expr::u32(5)),
            ),
            Node::let_bind(
                "bit",
                Expr::shl(
                    Expr::u32(1),
                    Expr::bitand(Expr::LogicalIndex { axis: 0 }, Expr::u32(31)),
                ),
            ),
            Node::let_bind(
                "_",
                Expr::atomic_or(bitset_out, Expr::var("word_idx"), Expr::var("bit")),
            ),
        ],
    )];
    let inner = vec![Node::if_then(
        Expr::lt(Expr::LogicalIndex { axis: 0 }, Expr::u32(element_count)),
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
                .with_count(element_count),
            BufferDecl::storage(bitset_out, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(words),
        ],
        [256, 1, 1],
        vec![region],
    )
}

const EXPECTED_INTERSECTS_BYTES: [u8; 4] = [6, 0, 0, 0];
const EXPECTED_EQ_BYTES: [u8; 4] = [5, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || bitset_value_filter("values", "bitset", 4, ValueFilter::Intersects(0b0010)),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0x01, 0x02, 0x06, 0x04]), to_bytes(&[0])]]
        }),
        Some(|| vec![vec![EXPECTED_INTERSECTS_BYTES.into()]]),
    )
    .with_laws(&["idempotent"])
    .with_uncharacterized()
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        MASK_INTERSECTS_OP_ID,
        || bitset_mask_intersects("values", "bitset", 4, 0b0010),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0x01, 0x02, 0x06, 0x04]), to_bytes(&[0])]]
        }),
        Some(|| vec![vec![EXPECTED_INTERSECTS_BYTES.into()]]),
    )
    .with_laws(&["idempotent"])
    .with_uncharacterized()
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        VALUE_EQ_OP_ID,
        || bitset_value_eq("values", "bitset", 4, 2),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[2, 1, 2, 4]), to_bytes(&[0])]]
        }),
        Some(|| vec![vec![EXPECTED_EQ_BYTES.into()]]),
    )
    .with_laws(&["identity"])
    .with_no_legal_rewrite()
}

#[cfg(test)]
mod tests {
    use super::*;

    impl ValueFilter {
        fn matches(self, value: u32) -> bool {
            match self {
                Self::Eq(expected) => value == expected,
                Self::Intersects(mask) => (value & mask) != 0,
            }
        }
    }

    fn reference_filter_into(values: &[u32], filter: ValueFilter, out: &mut Vec<u32>) {
        let needed = values.len().div_ceil(32);
        out.clear();
        out.resize(needed, 0);
        for (idx, &value) in values.iter().enumerate() {
            if filter.matches(value) {
                out[idx / 32] |= 1_u32 << (idx % 32);
            }
        }
    }

    #[test]
    fn value_eq_and_mask_intersects_wrap_shared_filter_region() {
        let eq_prog = bitset_value_eq("vals", "out", 64, 7);
        let mask_prog = bitset_mask_intersects("vals", "out", 64, 0x3);
        assert_eq!(eq_prog.buffers().len(), 2);
        assert_eq!(eq_prog.buffers()[1].count(), 2);
        assert_eq!(mask_prog.buffers()[1].count(), 2);
    }

    #[test]
    fn reference_filter_reuses_output_and_clears_stale_tail() {
        let values = [1_u32, 2, 3, 4, 5, 6, 7, 8];
        let mut out = Vec::with_capacity(4);
        out.extend([u32::MAX; 4]);
        let ptr = out.as_ptr();
        reference_filter_into(&values, ValueFilter::Intersects(0b1), &mut out);
        assert_eq!(out, vec![0b0101_0101]);
        assert_eq!(out.as_ptr(), ptr);

        reference_filter_into(&[], ValueFilter::Eq(1), &mut out);
        assert!(out.is_empty());
        assert_eq!(out.as_ptr(), ptr);
    }
}
