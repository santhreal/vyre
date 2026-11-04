//! Embedding lookup: `y[s, d] = embed_table[token[s], d]`.
//!
//! Category A composition  -  gather from weight buffer by token index.
//! Tokens are U32, embedding table is F32.

use vyre_foundation::composition::bounded_index_when;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Program};

use vyre_libs_builder::builder::build_indexed_map;

const OP_ID: &str = "vyre-libs::nn::embedding";

/// Build a Program that looks up F32 embeddings for `n` U32 token IDs.
///
/// `embed_table[vocab_size * embed_dim]` (F32), `tokens[n]` (U32),
/// `output[n * embed_dim]` (F32).
///
/// A token id is data, so it is bounded against the table's own extent rather
/// than a declared count: this form takes no vocabulary size, and a buffer
/// declaration whose count is unset states none. A token at or past the
/// vocabulary reads nothing and its output row is zero, which is what
/// `embedding_out_of_bounds_token_may_trap_or_return_zero` states. The index
/// is folded as well as tested, because a select evaluates both arms and the
/// load would otherwise still run for the rejected lane.
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
                .with_output_byte_range(0..u64::from(total_out).saturating_mul(4)),
        ],
        output,
        total_out,
        [64, 1, 1],
        |i| {
            let seq_idx = Expr::div(i.clone(), Expr::u32(embed_dim));
            let dim_idx = Expr::sub(i.clone(), Expr::mul(seq_idx.clone(), Expr::u32(embed_dim)));
            let token_id = Expr::load(tokens, seq_idx);
            let table_offset = Expr::add(Expr::mul(token_id, Expr::u32(embed_dim)), dim_idx);
            (
                i,
                table_gather_or_zero(
                    embed_table,
                    table_offset,
                    Expr::buf_len(embed_table),
                    DataType::F32,
                ),
            )
        },
    )
}

/// `table[offset]`, or a zero of `dtype` when `offset` is at or past `extent`.
///
/// A gather index that comes from an input buffer is bounded here, and the
/// caller names the extent: a declared count where the form takes a vocabulary
/// size, and the bound buffer's own length where it does not. The index is
/// folded as well as tested, because a select evaluates both arms and the load
/// would otherwise still run for the rejected lane, so that lane reads element
/// zero and discards it instead of reading memory the table does not own.
fn table_gather_or_zero(table: &str, offset: Expr, extent: Expr, dtype: DataType) -> Expr {
    let in_table = Expr::lt(offset.clone(), extent);
    Expr::select(
        in_table.clone(),
        Expr::load(table, bounded_index_when(in_table, offset)),
        Expr::cast(dtype, Expr::f32(0.0)),
    )
}

/// Build a typed embedding lookup with an explicit checkpoint table extent.
///
/// `table` uses `[vocab_size, embed_dim]`, `tokens` uses `[n]`, and `output`
/// uses `[n, embed_dim]`. A token id at or past the declared vocabulary reads
/// nothing and its output row is zero.
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
    let value_dtype = dtype.clone();
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
                table_gather_or_zero(
                    table,
                    Expr::add(Expr::mul(token_id, Expr::u32(embed_dim)), feature),
                    Expr::u32(table_count),
                    value_dtype.clone(),
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
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
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
    .with_uncharacterized()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_test_support::test_parity_oracles::decode_f32;
    use vyre_test_support::test_parity_oracles::eval_bytes;
    use vyre_test_support::test_parity_oracles::f32_bytes;
    use vyre_test_support::test_parity_oracles::u32_bytes;

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

    /// WHY: a token id is data. The lookup used to index the table with it
    /// directly, so a token at or past the vocabulary read past the end of the
    /// table and this case accepted either a trap or a zero, which no defect
    /// can turn red. The answer is now determinate: the row is zero, and the
    /// access stays inside the table.
    ///
    /// Does not catch a token that is inside the vocabulary but wrong; that is
    /// the caller's value, not an extent.
    #[test]
    fn a_token_past_the_vocabulary_reads_a_zero_row() {
        let program = embedding("table", "tokens", "output", 1, 2);
        let outputs = eval_bytes(
            "embedding",
            &program,
            vec![f32_bytes(&[1.0, 2.0]), u32_bytes(&[9999]), vec![0u8; 8]],
        );
        assert_eq!(
            decode_f32(&outputs[0]),
            vec![0.0, 0.0],
            "Fix: a token id past the vocabulary must read a zero row, not the table's first row"
        );
    }

    /// WHY: the fold must not move a lookup that was already inside the table.
    /// A fold written as an unconditional clamp would send the last row to the
    /// first one and pass the case above.
    #[test]
    fn the_last_row_of_the_vocabulary_still_reads_itself() {
        let program = embedding("table", "tokens", "output", 1, 2);
        let outputs = eval_bytes(
            "embedding",
            &program,
            vec![
                f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
                u32_bytes(&[1]),
                vec![0u8; 8],
            ],
        );
        assert_eq!(
            decode_f32(&outputs[0]),
            vec![3.0, 4.0],
            "Fix: the highest in-vocabulary token must read its own row"
        );
    }

    /// WHY: the typed form declares its vocabulary, so it bounds against the
    /// declared count rather than the bound buffer's length. Both forms must
    /// answer the same way, or one of them is the unbounded one.
    #[test]
    fn a_typed_token_past_the_declared_vocabulary_reads_a_zero_row() {
        let program = embedding_typed("table", "tokens", "output", 1, 2, 2, DataType::F32)
            .expect("Fix: a two-token F32 vocabulary must build");
        let outputs = eval_bytes(
            "embedding_typed",
            &program,
            vec![
                f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
                u32_bytes(&[7]),
                vec![0u8; 8],
            ],
        );
        assert_eq!(
            decode_f32(&outputs[0]),
            vec![0.0, 0.0],
            "Fix: a typed token id past the declared vocabulary must read a zero row"
        );
    }
}
