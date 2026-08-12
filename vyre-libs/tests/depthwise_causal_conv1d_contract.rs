//! Floating depthwise causal convolution execution contracts.

#![forbid(unsafe_code)]

use vyre::ir::DataType;
use vyre_libs::nn::conv::{
    depthwise_causal_conv1d, CausalConvActivation, DepthwiseCausalConv1dError,
};
use vyre_reference::value::Value;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn execute_f32(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    mask: Option<&[u32]>,
    batch: u32,
    channels: u32,
    sequence: u32,
    kernel: u32,
    activation: CausalConvActivation,
) -> Vec<f32> {
    let program = depthwise_causal_conv1d(
        "input",
        "weight",
        bias.map(|_| "bias"),
        mask.map(|_| "mask"),
        "output",
        batch,
        channels,
        sequence,
        kernel,
        activation,
        DataType::F32,
    )
    .expect("Fix: valid convolution fixture must build");
    let mut inputs = vec![
        Value::from(f32_bytes(input)),
        Value::from(f32_bytes(weight)),
    ];
    if let Some(bias) = bias {
        inputs.push(Value::from(f32_bytes(bias)));
    }
    if let Some(mask) = mask {
        inputs.push(Value::from(u32_bytes(mask)));
    }
    inputs.push(Value::from(vec![0; input.len() * 4]));
    let outputs = vyre_reference::reference_eval(&program, &inputs)
        .expect("Fix: causal convolution must execute");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("Fix: exact f32 word")))
        .collect()
}

/// Locks left padding and PyTorch correlation orientation for a non-symmetric kernel.
#[test]
fn prefill_matches_exact_left_padded_correlation() {
    assert_eq!(
        execute_f32(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0],
            None,
            None,
            1,
            1,
            4,
            3,
            CausalConvActivation::None,
        ),
        vec![3.0, 8.0, 14.0, 20.0]
    );
}

/// Prevents kernels wider than the prompt from reading OOB or emitting a full padded tail.
#[test]
fn sequence_shorter_than_kernel_is_truncated_to_input_length() {
    assert_eq!(
        execute_f32(
            &[1.0, 2.0],
            &[1.0, 2.0, 3.0, 4.0],
            None,
            None,
            1,
            1,
            2,
            4,
            CausalConvActivation::None,
        ),
        vec![4.0, 11.0]
    );
}

/// Proves each batch/channel pair uses only its own signal, filter, and bias.
#[test]
fn batches_channels_and_bias_are_isolated_exactly() {
    let actual = execute_f32(
        &[
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0,
        ],
        &[1.0, 1.0, 2.0, -1.0],
        Some(&[0.5, -0.5]),
        None,
        2,
        2,
        3,
        2,
        CausalConvActivation::None,
    );
    assert_eq!(
        actual,
        vec![1.5, 3.5, 5.5, -10.5, -0.5, 9.5, 4.5, 9.5, 11.5, -40.5, 29.5, 39.5,]
    );
}

/// Ensures masked source positions contribute zero without changing output truncation.
#[test]
fn padding_mask_excludes_each_masked_input_position() {
    assert_eq!(
        execute_f32(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 1.0],
            None,
            Some(&[1, 0, 1, 1]),
            1,
            1,
            4,
            2,
            CausalConvActivation::None,
        ),
        vec![1.0, 1.0, 3.0, 7.0]
    );
}

/// Locks SiLU after bias and convolution rather than applying activation to each tap.
#[test]
fn silu_activation_is_applied_once_after_accumulation() {
    let actual = execute_f32(
        &[1.0, -2.0],
        &[2.0],
        Some(&[0.5]),
        None,
        1,
        1,
        2,
        1,
        CausalConvActivation::Silu,
    );
    assert!((actual[0] - 2.310_354_5).abs() <= 2e-6);
    assert!((actual[1] - -0.102_592_83).abs() <= 2e-6);
}

/// Proves BF16 accumulation converts once to exact source-dtype output words.
#[test]
fn bf16_convolution_matches_exact_output_words() {
    let program = depthwise_causal_conv1d(
        "input",
        "weight",
        None,
        None,
        "output",
        1,
        1,
        2,
        1,
        CausalConvActivation::None,
        DataType::BF16,
    )
    .expect("Fix: BF16 convolution must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(u16_bytes(&[0x3f80, 0xbf80])),
            Value::from(u16_bytes(&[0x4000])),
            Value::from(vec![0; 4]),
        ],
    )
    .expect("Fix: BF16 convolution must execute");
    assert_eq!(outputs[0].to_bytes(), u16_bytes(&[0x4000, 0xc000]));
}

/// Prevents empty, overflowing, and non-floating configurations from becoming Programs.
#[test]
fn invalid_convolution_contracts_fail_closed() {
    assert_eq!(
        depthwise_causal_conv1d(
            "x",
            "w",
            None,
            None,
            "y",
            1,
            1,
            0,
            4,
            CausalConvActivation::None,
            DataType::F32,
        )
        .expect_err("Fix: empty sequence must fail"),
        DepthwiseCausalConv1dError::EmptyShape
    );
    assert_eq!(
        depthwise_causal_conv1d(
            "x",
            "w",
            None,
            None,
            "y",
            u32::MAX,
            2,
            2,
            1,
            CausalConvActivation::None,
            DataType::F32,
        )
        .expect_err("Fix: flattened overflow must fail"),
        DepthwiseCausalConv1dError::ElementCountOverflow
    );
    assert_eq!(
        depthwise_causal_conv1d(
            "x",
            "w",
            None,
            None,
            "y",
            1,
            1,
            2,
            1,
            CausalConvActivation::None,
            DataType::U32,
        )
        .expect_err("Fix: integer source must fail"),
        DepthwiseCausalConv1dError::UnsupportedDtype {
            dtype: DataType::U32
        }
    );
}
