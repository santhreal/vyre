//! Convert head-major attention output to token-major projection rows.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::region::wrap_anonymous;

const OP_ID: &str = "vyre-libs::nn::attention_head_to_token";

/// Convert `[batch, heads, sequence, head_dim]` into
/// `[batch, sequence, heads, head_dim]` without changing element values.
///
/// # Errors
///
/// Returns `Err` for zero dimensions or flattened element-count overflow.
pub fn attention_head_to_token(
    input: &str,
    output: &str,
    batch: u32,
    heads: u32,
    sequence: u32,
    head_dim: u32,
) -> Result<Program, String> {
    attention_head_to_token_typed(
        input,
        output,
        batch,
        heads,
        sequence,
        head_dim,
        DataType::F32,
    )
}

/// Typed head-major to token-major layout conversion.
#[allow(clippy::too_many_arguments)]
pub fn attention_head_to_token_typed(
    input: &str,
    output: &str,
    batch: u32,
    heads: u32,
    sequence: u32,
    head_dim: u32,
    dtype: DataType,
) -> Result<Program, String> {
    if batch == 0 || heads == 0 || sequence == 0 || head_dim == 0 {
        return Err("Fix: attention_head_to_token requires nonzero dimensions".to_string());
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(format!(
            "Fix: attention_head_to_token requires a floating dtype; got {dtype:?}"
        ));
    }
    let count = [batch, heads, sequence, head_dim]
        .into_iter()
        .try_fold(1_u32, |product, value| product.checked_mul(value))
        .ok_or_else(|| {
            "Fix: attention_head_to_token element count overflows u32; shard the tensor".to_string()
        })?;
    let token_width = sequence.checked_mul(head_dim).ok_or_else(|| {
        "Fix: attention_head_to_token token width overflows u32; shard sequence".to_string()
    })?;
    let batch_width = heads.checked_mul(token_width).ok_or_else(|| {
        "Fix: attention_head_to_token batch width overflows u32; shard batch".to_string()
    })?;
    let index = Expr::var("index");
    let dimension = Expr::rem(index.clone(), Expr::u32(head_dim));
    let feature = Expr::div(index.clone(), Expr::u32(head_dim));
    let head = Expr::rem(feature.clone(), Expr::u32(heads));
    let token_and_batch = Expr::div(feature, Expr::u32(heads));
    let token = Expr::rem(token_and_batch.clone(), Expr::u32(sequence));
    let batch_index = Expr::div(token_and_batch, Expr::u32(sequence));
    let source = Expr::add(
        Expr::mul(batch_index, Expr::u32(batch_width)),
        Expr::add(
            Expr::mul(head, Expr::u32(token_width)),
            Expr::add(Expr::mul(token, Expr::u32(head_dim)), dimension),
        ),
    );
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
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(count),
            BufferDecl::output(output, 1, dtype).with_count(count),
        ],
        [64, 1, 1],
        vec![wrap_anonymous(OP_ID, body)],
    ))
}
