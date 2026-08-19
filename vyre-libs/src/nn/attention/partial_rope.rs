//! Partial RoPE: rotary position embedding on first `rope_dims` of
//! each head, identity on the rest.
//!
//! Category A composition. Recipe rotates first 16 of 64 head dims.
//! Standard RoPE: `[x1*cos - x2*sin, x1*sin + x2*cos]` on pairs.

use super::layout::{layout_move_program, IndexMap, LayoutMove};
use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Program};

const OP_ID: &str = "vyre-libs::nn::partial_rope";

/// Build partial RoPE with positions starting at zero.
#[must_use]
pub fn partial_rope(
    input: &str,
    cos_table: &str,
    sin_table: &str,
    output: &str,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    rope_dims: u32,
) -> Program {
    partial_rope_at_offset(
        input, cos_table, sin_table, output, num_heads, seq_len, head_dim, rope_dims, 0, seq_len,
    )
}

/// Build partial RoPE for a prompt or cached decode position range.
///
/// `table_seq_len` is the number of positions in each cosine/sine table;
/// `position_offset` selects the first position consumed by this Program.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn partial_rope_at_offset(
    input: &str,
    cos_table: &str,
    sin_table: &str,
    output: &str,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    rope_dims: u32,
    position_offset: u32,
    table_seq_len: u32,
) -> Program {
    build_partial_rope_at_offset(
        input,
        cos_table,
        sin_table,
        output,
        num_heads,
        seq_len,
        head_dim,
        rope_dims,
        position_offset,
        table_seq_len,
        DataType::F32,
    )
}

/// Build partial RoPE with typed activation storage and F32 lookup tables.
///
/// # Errors
///
/// Returns `Err` when the activation dtype is not F16, BF16, or F32.
#[allow(clippy::too_many_arguments)]
pub fn partial_rope_at_offset_typed(
    input: &str,
    cos_table: &str,
    sin_table: &str,
    output: &str,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    rope_dims: u32,
    position_offset: u32,
    table_seq_len: u32,
    activation_dtype: DataType,
) -> Result<Program, String> {
    if !matches!(
        activation_dtype,
        DataType::F16 | DataType::BF16 | DataType::F32
    ) {
        return Err(format!(
            "Fix: partial_rope_at_offset_typed supports F16, BF16, or F32 activations; got {activation_dtype:?}"
        ));
    }
    Ok(build_partial_rope_at_offset(
        input,
        cos_table,
        sin_table,
        output,
        num_heads,
        seq_len,
        head_dim,
        rope_dims,
        position_offset,
        table_seq_len,
        activation_dtype,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_partial_rope_at_offset(
    input: &str,
    cos_table: &str,
    sin_table: &str,
    output: &str,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    rope_dims: u32,
    position_offset: u32,
    table_seq_len: u32,
    activation_dtype: DataType,
) -> Program {
    if num_heads == 0 || seq_len == 0 || head_dim == 0 {
        return trap_program(OP_ID, Some((output, activation_dtype.clone())), format!(
            "Fix: partial_rope requires positive num_heads, seq_len, and head_dim; got num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}."
        ));
    }
    if rope_dims > head_dim || rope_dims % 2 != 0 {
        return trap_program(OP_ID, Some((output, activation_dtype.clone())), format!(
            "Fix: partial_rope requires an even rope_dims <= head_dim; got rope_dims={rope_dims}, head_dim={head_dim}."
        ));
    }
    if position_offset
        .checked_add(seq_len)
        .is_none_or(|end| end > table_seq_len)
    {
        return trap_program(OP_ID, Some((output, activation_dtype.clone())), format!(
            "Fix: partial_rope position range offset={position_offset}, seq_len={seq_len} exceeds table_seq_len={table_seq_len}."
        ));
    }
    let total = match num_heads
        .checked_mul(seq_len)
        .and_then(|value| value.checked_mul(head_dim))
    {
        Some(total) => total,
        None => {
            return trap_program(OP_ID, Some((output, activation_dtype.clone())), format!(
                "Fix: partial_rope total element count overflows u32 for num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}."
            ));
        }
    };
    let half_rope = rope_dims / 2;
    let table_count = match table_seq_len.checked_mul(half_rope) {
        Some(count) => count,
        None => {
            return trap_program(OP_ID, Some((output, activation_dtype.clone())), format!(
                "Fix: partial_rope table element count overflows u32 for table_seq_len={table_seq_len}, rope_dims={rope_dims}."
            ));
        }
    };

    let i = Expr::var("index");
    let dim = Expr::rem(i.clone(), Expr::u32(head_dim));
    let token = Expr::rem(
        Expr::div(i.clone(), Expr::u32(head_dim)),
        Expr::u32(seq_len),
    );
    let pair = Expr::div(dim.clone(), Expr::u32(2));
    let parity = Expr::rem(dim.clone(), Expr::u32(2));
    let pair_base = Expr::sub(i.clone(), parity.clone());
    let x0 = Expr::cast(DataType::F32, Expr::load(input, pair_base.clone()));
    let x1 = Expr::cast(
        DataType::F32,
        Expr::load(input, Expr::add(pair_base, Expr::u32(1))),
    );
    let table_idx = Expr::add(
        Expr::mul(
            Expr::add(token, Expr::u32(position_offset)),
            Expr::u32(half_rope),
        ),
        pair,
    );
    let cos_v = Expr::load(cos_table, table_idx.clone());
    let sin_v = Expr::load(sin_table, table_idx);
    // Each rotation states one fused multiply-add. The pair a backend would
    // otherwise contract on its own is the product feeding the sum, and the
    // reference takes two roundings where a device takes one. Negating the
    // operand instead of the sum keeps the subtraction exact: multiplying by
    // -1.0 is representable, so the even rotation is the odd one with a flipped
    // sine operand.
    let rotated_even = Expr::fma(
        Expr::mul(x1.clone(), Expr::f32(-1.0)),
        sin_v.clone(),
        Expr::mul(x0.clone(), cos_v.clone()),
    );
    let rotated_odd = Expr::fma(x0, sin_v, Expr::mul(x1, cos_v));
    let rotated = Expr::select(Expr::eq(parity, Expr::u32(0)), rotated_even, rotated_odd);
    let value = Expr::cast(
        activation_dtype.clone(),
        Expr::select(
            Expr::lt(dim, Expr::u32(rope_dims)),
            rotated,
            Expr::cast(DataType::F32, Expr::load(input, i.clone())),
        ),
    );

    layout_move_program(LayoutMove {
        op_id: OP_ID,
        buffers: vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, activation_dtype.clone())
                .with_count(total),
            BufferDecl::storage(cos_table, 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(table_count),
            BufferDecl::storage(sin_table, 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(table_count),
            BufferDecl::output(output, 3, activation_dtype).with_count(total),
        ],
        write: output,
        count: total,
        map: IndexMap::Element { value },
    })
}

const EXPECTED_PARTIAL_ROPE_OUTPUT_BYTES: [u8; 32] = [
    0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40,
    0x00, 0x00, 0xA0, 0x40, 0x00, 0x00, 0xC0, 0x40, 0x00, 0x00, 0xE0, 0x40, 0x00, 0x00, 0x00, 0x41,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || partial_rope("input", "cos", "sin", "output", 1, 2, 4, 2),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), // input
                to_f32(&[1.0, 1.0]),  // cos table
                to_f32(&[0.0, 0.0]),  // sin table
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_PARTIAL_ROPE_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::decode_f32;
    use crate::fixture_bytes::f32_bytes;
    use vyre_reference::value::Value;

    #[test]
    fn rejects_invalid_rope_dims_without_panicking() {
        let p = partial_rope("input", "cos", "sin", "output", 1, 2, 4, 3);
        assert!(p.stats().trap());
    }

    #[test]
    fn rejects_zero_shape_without_panicking() {
        let p = partial_rope("input", "cos", "sin", "output", 0, 2, 4, 2);
        assert!(p.stats().trap());
    }

    #[test]
    fn rejects_overflow_shape_without_panicking() {
        let p = partial_rope("input", "cos", "sin", "output", u32::MAX, 2, 4, 2);
        assert!(p.stats().trap());
    }

    #[test]
    fn partial_rope_nan_in_input_propagates_nan() {
        let input = [f32::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cos = [1.0f32, 1.0];
        let sin = [0.0f32, 0.0];
        let program = partial_rope("input", "cos", "sin", "output", 1, 2, 4, 2);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&input)),
                Value::from(f32_bytes(&cos)),
                Value::from(f32_bytes(&sin)),
                Value::from(vec![0u8; 32]),
            ],
        )
        .expect("Fix: partial_rope must not panic on NaN input");
        let out = decode_f32(&outputs[0].to_bytes());
        assert!(
            out[0].is_nan(),
            "partial_rope must propagate NaN from input"
        );
        // RoPE pairs lanes (0,1): out[0] = in[0]*cos - in[1]*sin and
        // out[1] = in[0]*sin + in[1]*cos. With NaN at in[0] and sin=0,
        // out[1] computes `NaN*0 + 2*1` which is NaN under IEEE 754
        // (any arithmetic involving NaN returns NaN, including NaN*0).
        // Asserting out[1] == 2.0 would require a non-IEEE shortcut.
        assert!(
            out[1].is_nan(),
            "partial_rope NaN at in[0] poisons the paired lane via NaN*0 = NaN per IEEE 754, got {}",
            out[1]
        );
        // Lanes outside the rotated pair must NOT be poisoned.
        assert_eq!(out[2], 3.0, "partial_rope leaves unrotated lanes untouched");
        assert_eq!(out[3], 4.0, "partial_rope leaves unrotated lanes untouched");
    }

    #[test]
    fn partial_rope_zero_sequence_length_rejected() {
        let p = partial_rope("input", "cos", "sin", "output", 1, 0, 4, 2);
        assert!(p.stats().trap(), "partial_rope seq_len=0 must trap");
    }

    #[test]
    fn partial_rope_single_token() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let cos = [1.0f32, 1.0];
        let sin = [0.0f32, 0.0];
        let program = partial_rope("input", "cos", "sin", "output", 1, 1, 4, 2);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&input)),
                Value::from(f32_bytes(&cos)),
                Value::from(f32_bytes(&sin)),
                Value::from(vec![0u8; 16]),
            ],
        )
        .expect("Fix: partial_rope single token must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        // With sin=0, cos=1, RoPE is identity on pairs.
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn partial_rope_nan_in_cos_sin_tables_propagates_nan() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let cos = [f32::NAN, 1.0];
        let sin = [0.0f32, 0.0];
        let program = partial_rope("input", "cos", "sin", "output", 1, 1, 4, 2);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&input)),
                Value::from(f32_bytes(&cos)),
                Value::from(f32_bytes(&sin)),
                Value::from(vec![0u8; 16]),
            ],
        )
        .expect("Fix: partial_rope must not panic on NaN cos table");
        let out = decode_f32(&outputs[0].to_bytes());
        assert!(
            out[0].is_nan() || out[1].is_nan(),
            "partial_rope NaN in cos table must propagate to rotated pair"
        );
    }
}
