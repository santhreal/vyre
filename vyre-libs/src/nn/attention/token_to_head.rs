//! Token-major to head-major projection layout conversion.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::region::wrap_anonymous;

const LAYOUT_OP_ID: &str = "vyre-libs::nn::attention_token_to_head";

fn checked_count(values: &[u32], label: &str) -> Result<u32, String> {
    values.iter().try_fold(1_u32, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| format!("Fix: {label} element count overflows u32"))
    })
}

/// Convert `[batch, sequence, heads, head_dim]` into
/// `[batch, heads, sequence, head_dim]`.
pub fn attention_token_to_head(
    input: &str,
    output: &str,
    batch: u32,
    sequence: u32,
    heads: u32,
    head_dim: u32,
) -> Result<Program, String> {
    attention_token_to_head_typed(
        input,
        output,
        batch,
        sequence,
        heads,
        head_dim,
        DataType::F32,
    )
}

/// Typed token-major to head-major layout conversion.
#[allow(clippy::too_many_arguments)]
pub fn attention_token_to_head_typed(
    input: &str,
    output: &str,
    batch: u32,
    sequence: u32,
    heads: u32,
    head_dim: u32,
    dtype: DataType,
) -> Result<Program, String> {
    if batch == 0 || sequence == 0 || heads == 0 || head_dim == 0 {
        return Err("Fix: attention_token_to_head requires nonzero dimensions".to_string());
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(format!(
            "Fix: attention_token_to_head requires a floating dtype; got {dtype:?}"
        ));
    }
    let count = checked_count(
        &[batch, sequence, heads, head_dim],
        "attention_token_to_head",
    )?;
    let index = Expr::var("index");
    let feature = Expr::rem(index.clone(), Expr::u32(head_dim));
    let head_row = Expr::div(index.clone(), Expr::u32(head_dim));
    let token = Expr::rem(head_row.clone(), Expr::u32(sequence));
    let head_and_batch = Expr::div(head_row, Expr::u32(sequence));
    let head = Expr::rem(head_and_batch.clone(), Expr::u32(heads));
    let batch_index = Expr::div(head_and_batch, Expr::u32(heads));
    let source = Expr::add(
        Expr::mul(
            Expr::add(Expr::mul(batch_index, Expr::u32(sequence)), token),
            Expr::u32(heads * head_dim),
        ),
        Expr::add(Expr::mul(head, Expr::u32(head_dim)), feature),
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
        vec![wrap_anonymous(LAYOUT_OP_ID, body)],
    ))
}
