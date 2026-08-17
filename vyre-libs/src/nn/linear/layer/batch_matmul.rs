//! Batched matrix multiplication: `out[b, i, j] = sum_k a[b, i, k] * b[b, k, j]`.
//!
//! Category A composition. Each invocation computes one output element.

use vyre_foundation::ir::{DataType, Program};

use crate::builder::gemm::ContractionComposer;
use crate::plumbing::operand::tensor_ref::{TensorRef, TensorRefError};

/// Build a Program that computes batched matmul.
///
/// Shapes: `a: [batch, m, k]`, `b: [batch, k, n]`, `out: [batch, m, n]`.
/// Each invocation computes one `out[b, i, j]` by iterating the `k` dimension.
///
/// # Errors
/// Returns `Err` when any dimension is zero or total elements overflow u32.
pub fn batch_matmul(
    a: &str,
    b: &str,
    out: &str,
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
) -> Result<Program, String> {
    if batch == 0 || m == 0 || k == 0 || n == 0 {
        return Err("Fix: batch_matmul all dims must be > 0".to_string());
    }

    m.checked_mul(k)
        .ok_or("Fix: batch_matmul a_batch_stride overflow")?;
    k.checked_mul(n)
        .ok_or("Fix: batch_matmul b_batch_stride overflow")?;
    m.checked_mul(n)
        .ok_or("Fix: batch_matmul out_batch_stride overflow")?;

    let a_ref = TensorRef::new(a, DataType::F32, vec![batch, m, k]);
    let b_ref = TensorRef::new(b, DataType::F32, vec![batch, k, n]);
    let out_ref = TensorRef::new(out, DataType::F32, vec![batch, m, n]);

    ContractionComposer::batched_matmul_3d(
        "vyre-libs::nn::batch_matmul",
        a_ref,
        b_ref,
        out_ref,
        batch,
        m,
        k,
        n,
    )
    .with_region_generator("anonymous::vyre-libs::nn::batch_matmul")
    .build()
    .map_err(|e| match e {
        TensorRefError::ElementCountOverflow { name, .. } => {
            if name == a {
                "Fix: batch_matmul a_count overflow".to_string()
            } else if name == b {
                "Fix: batch_matmul b_count overflow".to_string()
            } else {
                "Fix: batch_matmul out_count overflow".to_string()
            }
        }
        _ => format!("Fix: batch_matmul build failed: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::decode_f32;
    use crate::fixture_bytes::f32_bytes;
    use vyre_reference::value::Value;

    #[test]
    fn batch_matmul_single_batch_matches_matmul() {
        // batch=1, m=2, k=3, n=2
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [1, 2, 3]
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [1, 3, 2]
                                                       // out[0,0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
                                                       // out[0,0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
                                                       // out[0,1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
                                                       // out[0,1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        let program = batch_matmul("a", "b", "out", 1, 2, 3, 2).unwrap();
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&a)),
                Value::from(f32_bytes(&b)),
                Value::from(vec![0u8; 4 * 4]),
            ],
        )
        .expect("Fix: batch_matmul single batch must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        assert_eq!(out, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn batch_matmul_two_batches() {
        // batch=2, m=2, k=2, n=2
        let a = vec![
            1.0f32, 0.0, 0.0, 1.0, // batch 0: identity
            2.0f32, 0.0, 0.0, 2.0, // batch 1: 2*identity
        ];
        let b = vec![
            1.0f32, 2.0, 3.0, 4.0, // batch 0
            5.0f32, 6.0, 7.0, 8.0, // batch 1
        ];
        // batch 0: identity @ b[0] = b[0] = [1,2,3,4]
        // batch 1: 2*identity @ b[1] = 2*b[1] = [10,12,14,16]
        let program = batch_matmul("a", "b", "out", 2, 2, 2, 2).unwrap();
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&a)),
                Value::from(f32_bytes(&b)),
                Value::from(vec![0u8; 4 * 4 * 2]),
            ],
        )
        .expect("Fix: batch_matmul two batches must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn batch_matmul_zero_dim_errors() {
        for (batch, m, k, n) in [(0, 2, 2, 2), (1, 0, 2, 2), (1, 2, 0, 2), (1, 2, 2, 0)] {
            let err =
                batch_matmul("a", "b", "out", batch, m, k, n).expect_err("zero dim must error");
            assert!(
                err.contains("batch_matmul") && err.contains("> 0"),
                "batch_matmul zero-dim error for ({batch},{m},{k},{n}): {err}"
            );
        }
    }
}
