// The one owner of the random-IR corpus the proptest contract cases draw from.
//
// The wire-roundtrip suite and the ProgramStats suite need the same generator
// for identifiers, data types, literals and expressions. They used to carry a
// copy each, and the copies had already drifted: one sampled seven call ids and
// the other four, and one had lost the adversarial f32 bit patterns. A property
// that holds over a smaller corpus in one suite than the other is a gap nobody
// chose.
//
// The two suites still differ in exactly one place: the opaque expression leaf.
// The wire suite needs an opaque node carrying a payload so the round-trip has
// something to preserve; the stats suite needs only that an opaque leaf exists
// so `opaque_count` moves. That difference is the parameter of
// `arb_expr_with`, not a reason for a second corpus.

#[path = "../../../tests/support/spec_op_strategies.rs"]
mod spec_op_strategies;

use proptest::collection::vec as prop_vec;
use proptest::prelude::*;
use spec_op_strategies::{arb_atomic_op, arb_bin_op, arb_un_op};
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_spec::extension::ExtensionDataTypeId;
use vyre_spec::TypeId;
pub(crate) use vyre_test_support::data_type_elements::flat_buffer_element_types;

pub(crate) const VAR_NAMES: &[&str] = &["", "x", "alpha", "snow_雪", "nul\0name"];
pub(crate) const CALL_IDS: &[&str] = &[
    "",
    "call",
    "筛选",
    "op::雪",
    "subgroup_reduce",
    "my::wave::add",
    "warp_shuffle",
];
pub(crate) const TAG_NAMES: &[&str] = &["", "tag", "stream-雪", "wait\0tag"];
pub(crate) const BUFFER_NAMES: &[&str] = &[
    "out",
    "input",
    "rw",
    "bytes_in",
    "bytes_out",
    "counts",
    "scratch",
];

pub(crate) fn arb_ident() -> BoxedStrategy<String> {
    prop::sample::select(VAR_NAMES.to_vec())
        .prop_map(str::to_string)
        .boxed()
}

pub(crate) fn arb_call_id() -> BoxedStrategy<String> {
    prop::sample::select(CALL_IDS.to_vec())
        .prop_map(str::to_string)
        .boxed()
}

pub(crate) fn arb_tag() -> BoxedStrategy<String> {
    prop::sample::select(TAG_NAMES.to_vec())
        .prop_map(str::to_string)
        .boxed()
}

pub(crate) fn arb_axis() -> BoxedStrategy<u8> {
    prop_oneof![Just(0), Just(1), Just(2), Just(255)].boxed()
}

pub(crate) fn arb_opaque_bytes() -> BoxedStrategy<Vec<u8>> {
    prop_vec(
        prop_oneof![
            Just(0x00u8),
            Just(0xffu8),
            Just(0xc0u8),
            Just(0xafu8),
            any::<u8>(),
        ],
        0..=24,
    )
    .boxed()
}

/// Every `DataType`, including the nested and opaque forms.
///
/// The weights keep the leaf distribution the flat 20-way `prop_oneof!` this
/// replaced had: `arb_buffer_datatype` covers eighteen of those twenty leaves,
/// so it carries eighteen times the weight of `Handle` or `Opaque`.
pub(crate) fn arb_datatype() -> BoxedStrategy<DataType> {
    let leaf = prop_oneof![
        18 => arb_buffer_datatype(),
        1 => any::<u32>().prop_map(|id| DataType::Handle(TypeId(id))),
        1 => any::<u32>().prop_map(|id| DataType::Opaque(ExtensionDataTypeId(id | 0x8000_0000))),
    ];

    leaf.prop_recursive(3, 24, 3, |inner| {
        prop_oneof![
            (inner.clone(), 0u8..=4).prop_map(|(element, count)| DataType::Vec {
                element: Box::new(element),
                count,
            }),
            (inner, prop_vec(any::<u32>(), 0..=4)).prop_map(|(element, shape)| {
                DataType::TensorShaped {
                    element: Box::new(element),
                    shape: shape.into_iter().collect(),
                }
            }),
        ]
    })
    .boxed()
}

/// The flat `DataType` forms a buffer element may take.
///
/// `Array` is the one flat form carrying a payload, so its element size is
/// drawn rather than fixed; the rest of the set comes from the shared table.
pub(crate) fn arb_buffer_datatype() -> BoxedStrategy<DataType> {
    (0usize..=64usize)
        .prop_flat_map(|element_size| prop::sample::select(flat_buffer_element_types(element_size)))
        .boxed()
}

pub(crate) fn arb_literal() -> BoxedStrategy<Expr> {
    let adversarial_f32_bits = prop_oneof![
        Just(0x0000_0000u32),
        Just(0x0000_0001u32),
        Just(0x007f_ffffu32),
        Just(f32::MIN_POSITIVE.to_bits()),
        Just(f32::MIN.to_bits()),
        Just(f32::MAX.to_bits()),
        any::<u32>().prop_filter(
            "general proptest literals exclude NaN and -0.0; targeted tests cover payload preservation and canonical zero encoding",
            |bits| !f32::from_bits(*bits).is_nan() && *bits != (-0.0f32).to_bits()
        ),
    ];

    prop_oneof![
        any::<u32>().prop_map(Expr::LitU32),
        any::<i32>().prop_map(Expr::LitI32),
        any::<bool>().prop_map(Expr::LitBool),
        adversarial_f32_bits.prop_map(|bits| Expr::LitF32(f32::from_bits(bits))),
    ]
    .boxed()
}

/// Every `Expr` variant, with `opaque_leaf` supplying `Expr::Opaque`.
pub(crate) fn arb_expr_with(opaque_leaf: BoxedStrategy<Expr>) -> BoxedStrategy<Expr> {
    let leaf = prop_oneof![
        arb_literal(),
        arb_ident().prop_map(Expr::var),
        prop::sample::select(BUFFER_NAMES.to_vec()).prop_map(Expr::buf_len),
        arb_axis().prop_map(|axis| Expr::InvocationId { axis }),
        arb_axis().prop_map(|axis| Expr::WorkgroupId { axis }),
        arb_axis().prop_map(|axis| Expr::LocalId { axis }),
        opaque_leaf,
    ];

    leaf.prop_recursive(4, 128, 4, |inner| {
        prop_oneof![
            (prop::sample::select(BUFFER_NAMES.to_vec()), inner.clone()).prop_map(
                |(buffer, index)| Expr::Load {
                    buffer: buffer.into(),
                    index: Box::new(index),
                }
            ),
            (arb_bin_op(), inner.clone(), inner.clone()).prop_map(|(op, left, right)| Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }),
            (arb_un_op(), inner.clone()).prop_map(|(op, operand)| Expr::UnOp {
                op,
                operand: Box::new(operand),
            }),
            (arb_call_id(), prop_vec(inner.clone(), 0..=4)).prop_map(|(op_id, args)| Expr::Call {
                op_id: op_id.into(),
                args,
            }),
            (inner.clone(), inner.clone(), inner.clone()).prop_map(
                |(cond, true_val, false_val)| Expr::Select {
                    cond: Box::new(cond),
                    true_val: Box::new(true_val),
                    false_val: Box::new(false_val),
                }
            ),
            (arb_buffer_datatype(), inner.clone()).prop_map(|(target, value)| Expr::Cast {
                target,
                value: Box::new(value),
            }),
            (inner.clone(), inner.clone(), inner.clone()).prop_map(|(a, b, c)| Expr::Fma {
                a: Box::new(a),
                b: Box::new(b),
                c: Box::new(c),
            }),
            (
                arb_atomic_op(),
                prop::sample::select(vec!["rw", "out", "counts", "bytes_out"]),
                inner.clone(),
                proptest::option::of(inner.clone()),
                inner.clone(),
            )
                .prop_map(|(op, buffer, index, expected, value)| Expr::Atomic {
                    op,
                    buffer: buffer.into(),
                    index: Box::new(index),
                    expected: expected.map(Box::new),
                    value: Box::new(value),
                    ordering: MemoryOrdering::SeqCst,
                }),
            inner.clone().prop_map(Expr::subgroup_add),
            (inner.clone(), inner.clone()).prop_map(|(value, lane)| Expr::SubgroupShuffle {
                value: Box::new(value),
                lane: Box::new(lane),
            }),
            inner.prop_map(|cond| Expr::SubgroupBallot {
                cond: Box::new(cond),
            }),
        ]
    })
    .boxed()
}

/// An expression generator, passed as a function so one statement generator can
/// draw as many independent expressions as it needs. `BoxedStrategy` is not
/// `Clone`, so a strategy value could only be used once.
pub(crate) type ExprFactory = fn() -> BoxedStrategy<Expr>;

/// The statement leaves every suite's node generator starts from.
pub(crate) fn arb_statement_leaf(expr: ExprFactory) -> BoxedStrategy<Node> {
    prop_oneof![
        (arb_ident(), expr()).prop_map(|(name, value)| Node::Let {
            name: name.into(),
            value,
        }),
        (arb_ident(), expr()).prop_map(|(name, value)| Node::Assign {
            name: name.into(),
            value,
        }),
        (
            prop::sample::select(vec!["out", "rw", "bytes_out"]),
            expr(),
            expr(),
        )
            .prop_map(|(buffer, index, value)| Node::Store {
                buffer: buffer.into(),
                index,
                value,
            }),
        Just(Node::Return),
        Just(Node::barrier()),
    ]
    .boxed()
}

/// The three body-carrying statements every suite recurses through, over
/// `inner`. A suite that also generates other body-carrying statements weights
/// this at 3 so each of the three keeps the share it had in a flat `prop_oneof!`.
pub(crate) fn arb_control_flow(
    expr: ExprFactory,
    inner: BoxedStrategy<Node>,
) -> BoxedStrategy<Node> {
    prop_oneof![
        (
            expr(),
            prop_vec(inner.clone(), 0..=3),
            prop_vec(inner.clone(), 0..=3),
        )
            .prop_map(|(cond, then, otherwise)| Node::If {
                cond,
                then,
                otherwise,
            }),
        (arb_ident(), expr(), expr(), prop_vec(inner.clone(), 0..=3),).prop_map(
            |(var, from, to, body)| Node::Loop {
                var: var.into(),
                from,
                to,
                body,
            }
        ),
        prop_vec(inner, 0..=3).prop_map(Node::Block),
    ]
    .boxed()
}

/// The nine-buffer program every suite wraps a generated entry body in.
///
/// The buffer table is what makes a generated body valid: `arb_statement_leaf`
/// stores into `out`, `rw` and `bytes_out`, and `arb_expr_with` loads from every
/// name in `BUFFER_NAMES`, so a suite that declared its own table would have to
/// keep it in step with both generators.
pub(crate) fn arb_program_with(node: BoxedStrategy<Node>) -> BoxedStrategy<Program> {
    (
        arb_buffer_datatype(),
        arb_buffer_datatype(),
        prop_vec(node, 0..=6),
        prop_oneof![9 => Just(false), 1 => Just(true)],
    )
        .prop_map(|(extra_a, extra_b, entry, non_composable)| {
            Program::wrapped(
                vec![
                    BufferDecl::output("out", 0, DataType::U32)
                        .with_count(8)
                        .with_output_byte_range(0..16),
                    BufferDecl::read("input", 1, DataType::U32).with_count(8),
                    BufferDecl::read_write("rw", 2, DataType::U32).with_count(8),
                    BufferDecl::read("bytes_in", 3, DataType::Bytes).with_count(16),
                    BufferDecl::read_write("bytes_out", 4, DataType::Bytes).with_count(16),
                    BufferDecl::read("counts", 5, DataType::U32).with_count(8),
                    BufferDecl::workgroup("scratch", 4, DataType::U32),
                    BufferDecl::read("extra_a", 6, extra_a).with_count(1),
                    BufferDecl::read("extra_b", 7, extra_b).with_count(1),
                ],
                [1, 1, 1],
                entry,
            )
            .with_non_composable_with_self(non_composable)
        })
        .boxed()
}
