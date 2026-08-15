//! Shared F32 unary activation Program builder.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Build `output[i] = op(input[i])` for an F32 activation.
#[must_use]
pub(crate) fn f32_unary_activation_program<F>(
    op_id: &'static str,
    input: &str,
    output: &str,
    n: u32,
    op: F,
) -> Program
where
    F: Fn(Expr) -> Expr,
{
    let i = Expr::var("i");
    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::buf_len(input)),
            vec![Node::Store {
                buffer: output.into(),
                index: i.clone(),
                value: op(Expr::load(input, i)),
            }],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
            BufferDecl::output(output, 1, DataType::F32)
                .with_count(n.max(1))
                .with_output_byte_range(0..(n as usize).saturating_mul(4)),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(op_id, body)],
    )
}

/// Build one typed binary activation map with F32 intermediate arithmetic.
pub(crate) fn typed_binary_activation_program(
    op_id: &'static str,
    left: &str,
    right: &str,
    output: &str,
    n: u32,
    dtype: DataType,
    combine: impl Fn(Expr, Expr) -> Expr,
) -> Program {
    let index = Expr::var("index");
    let left_value = Expr::cast(DataType::F32, Expr::load(left, index.clone()));
    let right_value = Expr::cast(DataType::F32, Expr::load(right, index.clone()));
    let body = vec![
        Node::let_bind("index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(n)),
            vec![Node::Store {
                buffer: output.into(),
                index,
                value: Expr::cast(dtype.clone(), combine(left_value, right_value)),
            }],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(left, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
            BufferDecl::storage(right, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
            BufferDecl::output(output, 2, dtype).with_count(n),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(op_id, body)],
    )
}

/// Build a typed sigmoid gate, optionally multiplying by its gate input.
pub(crate) fn typed_sigmoid_gate_program(
    op_id: &'static str,
    gate: &str,
    branch: &str,
    output: &str,
    n: u32,
    dtype: DataType,
    include_gate: bool,
) -> Program {
    typed_binary_activation_program(op_id, gate, branch, output, n, dtype, |gate, branch| {
        let sigmoid = Expr::div(
            Expr::f32(1.0),
            Expr::add(Expr::f32(1.0), Expr::exp(Expr::negate(gate.clone()))),
        );
        let branch = Expr::mul(branch, sigmoid);
        if include_gate {
            Expr::mul(gate, branch)
        } else {
            branch
        }
    })
}
