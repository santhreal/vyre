//! Embedding lookup: `y[s, d] = embed_table[token[s], d]`.
//!
//! Category A composition  -  gather from weight buffer by token index.
//! Tokens are U32, embedding table is F32.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Program};

use crate::builder::build_indexed_map;

const OP_ID: &str = "vyre-libs::nn::embedding";

/// Build a Program that looks up F32 embeddings for `n` U32 token IDs.
///
/// `embed_table[vocab_size * embed_dim]` (F32), `tokens[n]` (U32),
/// `output[n * embed_dim]` (F32).
#[must_use]
pub fn embedding(embed_table: &str, tokens: &str, output: &str, n: u32, embed_dim: u32) -> Program {
    let total_out = n * embed_dim;

    build_indexed_map(
        OP_ID,
        vec![
            BufferDecl::storage(embed_table, 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage(tokens, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::output(output, 2, DataType::F32)
                .with_count(total_out.max(1))
                .with_output_byte_range(0..(total_out as usize).saturating_mul(4)),
        ],
        output,
        total_out,
        [64, 1, 1],
        |i| {
            let seq_idx = Expr::div(i.clone(), Expr::u32(embed_dim));
            let dim_idx = Expr::sub(i.clone(), Expr::mul(seq_idx.clone(), Expr::u32(embed_dim)));
            let token_id = Expr::load(tokens, seq_idx);
            let table_offset = Expr::add(Expr::mul(token_id, Expr::u32(embed_dim)), dim_idx);
            (i, Expr::load(embed_table, table_offset))
        },
    )
}

/// Build a typed embedding lookup with an explicit checkpoint table extent.
///
/// `table` uses `[vocab_size, embed_dim]`, `tokens` uses `[n]`, and `output`
/// uses `[n, embed_dim]`. Token IDs outside the declared vocabulary retain the
/// backend's bounds-trap semantics.
#[allow(clippy::too_many_arguments)]
pub fn embedding_typed(
    table: &str,
    tokens: &str,
    output: &str,
    n: u32,
    vocab_size: u32,
    embed_dim: u32,
    dtype: DataType,
) -> Result<Program, String> {
    if n == 0 || vocab_size == 0 || embed_dim == 0 {
        return Err(
            "Fix: typed embedding requires nonzero token, vocabulary, and embedding dimensions"
                .to_string(),
        );
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(format!(
            "Fix: typed embedding requires F16, BF16, or F32 table storage; got {dtype:?}"
        ));
    }
    let table_count = vocab_size
        .checked_mul(embed_dim)
        .ok_or_else(|| "Fix: embedding vocabulary table count overflows u32".to_string())?;
    let output_count = n
        .checked_mul(embed_dim)
        .ok_or_else(|| "Fix: embedding output count overflows u32".to_string())?;
    Ok(build_indexed_map(
        OP_ID,
        vec![
            BufferDecl::storage(table, 0, BufferAccess::ReadOnly, dtype.clone())
                .with_count(table_count),
            BufferDecl::storage(tokens, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::output(output, 2, dtype).with_count(output_count),
        ],
        output,
        output_count,
        [64, 1, 1],
        |index| {
            let token = Expr::div(index.clone(), Expr::u32(embed_dim));
            let feature = Expr::rem(index.clone(), Expr::u32(embed_dim));
            let token_id = Expr::load(tokens, token);
            (
                index,
                Expr::load(
                    table,
                    Expr::add(Expr::mul(token_id, Expr::u32(embed_dim)), feature),
                ),
            )
        },
    ))
}

const EXPECTED_EMBEDDING_OUTPUT_BYTES: [u8; 24] = [
    0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xA0, 0x40, 0x00, 0x00, 0xC0, 0x40, 0x00, 0x00, 0x80, 0x3F,
    0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || embedding("table", "tokens", "output", 2, 3),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            let to_u32 = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_f32(&[1.0, 2.0, 3.0,  4.0, 5.0, 6.0]), // table: 2 vocab × 3 dim
                to_u32(&[1, 0]),                             // tokens
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_EMBEDDING_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::decode_f32;
    use crate::fixture_bytes::eval_bytes;
    use crate::fixture_bytes::f32_bytes;
    use crate::fixture_bytes::try_eval_bytes;
    use crate::fixture_bytes::u32_bytes;

    #[test]
    fn embedding_empty_tensor() {
        let program = embedding("table", "tokens", "output", 0, 3);
        let outputs = eval_bytes(
            "embedding",
            &program,
            vec![f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), vec![], vec![]],
        );
        assert!(outputs[0].is_empty());
    }

    #[test]
    fn embedding_single_element() {
        let program = embedding("table", "tokens", "output", 1, 2);
        let outputs = eval_bytes(
            "embedding",
            &program,
            vec![
                f32_bytes(&[10.0, 20.0, 30.0, 40.0]),
                u32_bytes(&[1]),
                vec![0u8; 8],
            ],
        );
        let out = decode_f32(&outputs[0]);
        assert_eq!(out, vec![30.0, 40.0]);
    }

    #[test]
    fn embedding_zero_token_index() {
        let program = embedding("table", "tokens", "output", 2, 2);
        let outputs = eval_bytes(
            "embedding",
            &program,
            vec![
                f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
                u32_bytes(&[0, 0]),
                // Two tokens of two dimensions each: four f32, sixteen bytes.
                // This read `8` (copied from the single-token test above) and
                // went unnoticed because the interpreter used to discard the
                // size of a legacy output initializer entirely.
                vec![0u8; 16],
            ],
        );
        let out = decode_f32(&outputs[0]);
        assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn embedding_nan_in_table_propagates_to_output() {
        let program = embedding("table", "tokens", "output", 1, 2);
        let outputs = eval_bytes(
            "embedding",
            &program,
            vec![f32_bytes(&[f32::NAN, 2.0]), u32_bytes(&[0]), vec![0u8; 8]],
        );
        let out = decode_f32(&outputs[0]);
        assert!(
            out[0].is_nan(),
            "embedding must propagate NaN from table to output"
        );
        assert_eq!(out[1], 2.0);
    }

    #[test]
    fn embedding_out_of_bounds_token_may_trap_or_return_zero() {
        // Adversarial: token index >= vocab_size. The IR does an
        // unguarded load at table_offset = token_id * embed_dim + dim_idx.
        // The reference interpreter may trap or return 0 for OOB.
        // We assert that it does not silently produce a finite non-zero value.
        let program = embedding("table", "tokens", "output", 1, 2);
        let result = try_eval_bytes(
            &program,
            vec![f32_bytes(&[1.0, 2.0]), u32_bytes(&[9999]), vec![0u8; 8]],
        );
        match result {
            Ok(outputs) => {
                let out = decode_f32(&outputs[0]);
                // If the interpreter does not trap, it should at least not
                // silently claim the lookup is valid (0 is acceptable for OOB).
                assert!(
                    out.iter().all(|&v| v == 0.0 || v.is_nan()),
                    "OOB embedding lookup must trap or return 0/NaN, got {:?}",
                    out
                );
            }
            Err(_) => {
                // Trapping is acceptable behavior for OOB.
            }
        }
    }
}
