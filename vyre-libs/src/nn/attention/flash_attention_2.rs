//! FlashAttention-2 tiling  -  the online-softmax core at a cooperative tile
//! width.
//!
//! [`flash_attention_2`] selects the tiled plan and composes
//! [`online_softmax_attention`](super::tiled_online_softmax::online_softmax_attention),
//! which owns the recurrence. Parity is against
//! [`attention_reference`](super::attention_reference), the offline three-pass
//! schedule: an oracle that shares the implementation under test agrees by
//! construction rather than by being right, and a scalar copy of this
//! recurrence was exactly that.
//!
//! Category-A composition.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use super::planner::plan_flash_attention_tiled;
use super::tiled_online_softmax::compose_online_softmax_attention;

const OP_ID: &str = "vyre-libs::nn::flash_attention_2";

/// Build a Program that computes FlashAttention-2 with explicit
/// sequence tiling.
///
/// Each invocation handles one query row. The KV sequence is iterated in tiles
/// of `tile_size`. For every tile all scores are computed first, then the
/// row-level online-softmax accumulator `(m, l, o_acc)` is updated in one
/// batched step.
///
/// # Parameters
///
/// * `tile_size`  -  number of keys processed per tile (e.g. 64 or 128).
///
/// # Errors
///
/// Returns a trap program when `seq_len == 0`, `head_dim == 0` or
/// `tile_size == 0`.
#[must_use]
pub fn flash_attention_2(
    q: &str,
    k: &str,
    v: &str,
    out: &str,
    seq_len: u32,
    head_dim: u32,
    tile_size: u32,
) -> Program {
    if seq_len == 0 || head_dim == 0 || tile_size == 0 {
        return trap_program(
            OP_ID,
            Some((out, DataType::F32)),
            "Fix: flash_attention_2 seq_len, head_dim, and tile_size must all be > 0".to_string(),
        );
    }
    let plan = match plan_flash_attention_tiled(seq_len, head_dim, tile_size) {
        Ok(plan) => plan,
        Err(error) => {
            return trap_program(OP_ID, Some((out, DataType::F32)), error);
        }
    };
    compose_online_softmax_attention(OP_ID, q, k, v, out, &plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_program(program: Program, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        crate::nn::attention::eval_qkv_program(
            &program,
            q,
            k,
            v,
            "Fix: reference eval must succeed",
        )
    }

    /// Tiled FlashAttention-2 agrees with the offline three-pass schedule on
    /// a non-trivial random fixture. The oracle is a different algorithm, not a
    /// second spelling of the recurrence under test.
    #[test]
    fn flash_attention_2_matches_attention_reference() {
        let seq_len = 8_u32;
        let head_dim = 16_u32;
        let tile_size = 4_u32;
        let elements = (seq_len * head_dim) as usize;

        let (q, k, v) = super::synth_qkv_fixtures(elements);

        let actual = run_program(
            flash_attention_2("q", "k", "v", "out", seq_len, head_dim, tile_size),
            &q,
            &k,
            &v,
        );
        let expected = run_program(
            crate::nn::attention::attention_reference("q", "k", "v", "out", seq_len, head_dim),
            &q,
            &k,
            &v,
        );

        assert_eq!(actual.len(), expected.len(), "output length must match");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-3,
                "flash_attention_2 vs attention_reference mismatch at index {i}: {a} != {e}"
            );
        }
    }

    /// Output shape is `[seq_len, head_dim]`.
    #[test]
    fn flash_attention_2_output_shape() {
        let seq_len = 5_u32;
        let head_dim = 7_u32;
        let tile_size = 3_u32;
        let elements = (seq_len * head_dim) as usize;

        let q = vec![1.0f32; elements];
        let k = vec![0.5f32; elements];
        let v = vec![2.0f32; elements];

        let out = run_program(
            flash_attention_2("q", "k", "v", "out", seq_len, head_dim, tile_size),
            &q,
            &k,
            &v,
        );
        assert_eq!(out.len(), elements);
    }

    /// Edge case: `seq_len == 1` degenerates to passing V through
    /// (softmax of a length-1 vector is `[1.0]`).
    #[test]
    fn flash_attention_2_seq_len_one() {
        let seq_len = 1_u32;
        let head_dim = 4_u32;
        let tile_size = 1_u32;
        let q = vec![1.0f32, 2.0, 3.0, 4.0];
        let k = vec![0.5f32, 1.5, 2.5, 3.5];
        let v = vec![10.0f32, 20.0, 30.0, 40.0];

        let actual = run_program(
            flash_attention_2("q", "k", "v", "out", seq_len, head_dim, tile_size),
            &q,
            &k,
            &v,
        );
        let expected = run_program(
            crate::nn::attention::attention_reference("q", "k", "v", "out", seq_len, head_dim),
            &q,
            &k,
            &v,
        );

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-3,
                "seq_len=1 mismatch at {i}: {a} != {e}"
            );
        }
    }

    /// Edge case: `seq_len == tile_size`.
    #[test]
    fn flash_attention_2_seq_len_eq_tile_size() {
        let seq_len = 4_u32;
        let head_dim = 8_u32;
        let tile_size = 4_u32;
        let elements = (seq_len * head_dim) as usize;

        let q: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.05 + 0.2).collect();
        let v: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.3 - 0.1).collect();

        let actual = run_program(
            flash_attention_2("q", "k", "v", "out", seq_len, head_dim, tile_size),
            &q,
            &k,
            &v,
        );
        let expected = run_program(
            crate::nn::attention::attention_reference("q", "k", "v", "out", seq_len, head_dim),
            &q,
            &k,
            &v,
        );

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-3,
                "seq_len==tile_size mismatch at {i}: {a} != {e}"
            );
        }
    }

    /// Edge case: `seq_len == tile_size + 1`.
    #[test]
    fn flash_attention_2_seq_len_eq_tile_size_plus_one() {
        let seq_len = 5_u32;
        let head_dim = 8_u32;
        let tile_size = 4_u32;
        let elements = (seq_len * head_dim) as usize;

        let q: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.11).collect();
        let k: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.06 + 0.15).collect();
        let v: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.25 - 0.05).collect();

        let actual = run_program(
            flash_attention_2("q", "k", "v", "out", seq_len, head_dim, tile_size),
            &q,
            &k,
            &v,
        );
        let expected = run_program(
            crate::nn::attention::attention_reference("q", "k", "v", "out", seq_len, head_dim),
            &q,
            &k,
            &v,
        );

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-3,
                "seq_len==tile_size+1 mismatch at {i}: {a} != {e}"
            );
        }
    }
}
