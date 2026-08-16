//! Live CUDA regression for grouped-query attention primitive composition.

mod harness;

use harness::with_live_backend;
use vyre_driver::DispatchConfig;
use vyre_libs::nn::attention::gqa_attention;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte F32 chunk")))
        .collect()
}

/// Canonical max, sum, and write passes must preserve grouped KV-head broadcasting on live CUDA.
#[test]
fn canonical_gqa_passes_execute_exact_grouped_head_mapping_on_cuda() {
    let program = gqa_attention("q", "k", "v", "out", 4, 2, 1, 2)
        .expect("valid grouped-query dimensions must build");
    let inputs = vec![
        f32_bytes(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0]),
        f32_bytes(&[1.0, 0.0, 0.0, 1.0]),
        f32_bytes(&[10.0, 20.0, 30.0, 40.0]),
    ];

    let output = with_live_backend("canonical grouped-query attention", |backend| {
        let mut config = DispatchConfig::default();
        config.ulp_budget = Some(4);
        backend
            .dispatch(&program, &inputs, &config)
            .expect("live CUDA dispatch must execute canonical GQA passes")
    });

    assert_eq!(
        decode_f32(&output[0]),
        vec![10.0, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0]
    );
}
