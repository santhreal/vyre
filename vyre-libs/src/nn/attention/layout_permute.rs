//! The gather program both attention layout conversions dispatch.
//!
//! [`attention_head_to_token_typed`](super::head_to_token::attention_head_to_token_typed)
//! and
//! [`attention_token_to_head_typed`](super::token_to_head::attention_token_to_head_typed)
//! move the same elements between `[batch, heads, sequence, head_dim]` and
//! `[batch, sequence, heads, head_dim]`. They are one program with two different
//! index derivations, and each used to carry its own copy of the guard, the two
//! buffer declarations, and the region wrapper.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::region::wrap_anonymous;

/// Reject the shape and dtype inputs no attention layout conversion can serve.
///
/// A zero dimension makes the flat element count zero, so the dispatch would
/// write nothing and report success. A non-float dtype has no conversion
/// contract on the read side. `label` names the calling conversion so the
/// message still identifies which entry point refused, and it is the only
/// position that differed between the two copies.
///
/// It deliberately does NOT compute or validate the element count: the two
/// conversions derive different strides from the same dimensions and report
/// their overflow differently, and head-major output additionally has token and
/// batch strides that token-major output cannot overflow.
pub(super) fn check_layout_dims(
    label: &str,
    dims: [u32; 4],
    dtype: &DataType,
) -> Result<(), String> {
    if dims.iter().any(|dimension| *dimension == 0) {
        return Err(format!("Fix: {label} requires nonzero dimensions"));
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(format!(
            "Fix: {label} requires a floating dtype; got {dtype:?}"
        ));
    }
    Ok(())
}

/// Emit the layout-permutation gather: one invocation per element, storing
/// `output[index] = input[source]` under an `index < count` guard.
///
/// `source` is the one position that differs between the two conversions. It
/// reads the `index` binding this function establishes, so a caller builds it
/// from `Expr::var("index")` and nothing else.
///
/// The gather direction is fixed on purpose: the guard bounds the OUTPUT index,
/// which is the invocation id, so every output element is written exactly once
/// and a permutation cannot leave a hole. A scatter would bound the input index
/// instead and could not make that promise.
pub(super) fn layout_permute_program(
    op_id: &'static str,
    input: &str,
    output: &str,
    count: u32,
    dtype: DataType,
    source: Expr,
) -> Program {
    let index = Expr::var("index");
    let body = vec![
        Node::let_bind("index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(count)),
            vec![Node::Store {
                buffer: output.into(),
                index,
                value: Expr::load(input, source),
            }],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(count),
            BufferDecl::output(output, 1, dtype).with_count(count),
        ],
        [64, 1, 1],
        vec![wrap_anonymous(op_id, body)],
    )
}
