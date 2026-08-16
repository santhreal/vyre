//! Streaming causal convolution state-transition contracts.

#![forbid(unsafe_code)]

mod wire_words;
use wire_words::{f32_bytes, f32_words_of as decode_f32};

use vyre::ir::{
    BufferAccess, DataType, GraphInput, GraphOutput, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
use vyre_libs::nn::conv::{
    depthwise_causal_conv1d, depthwise_causal_conv1d_update, CausalConvActivation,
};
use vyre_reference::value::Value;

fn update(input: &[f32], state: &[f32], weight: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let program = depthwise_causal_conv1d_update(
        "input",
        "weight",
        None,
        "state.in",
        "output",
        "state.out",
        1,
        1,
        input.len() as u32,
        weight.len() as u32,
        CausalConvActivation::None,
        DataType::F32,
    )
    .expect("Fix: valid state update must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(f32_bytes(input)),
            Value::from(f32_bytes(weight)),
            Value::from(f32_bytes(state)),
            Value::from(vec![0; input.len() * 4]),
            Value::from(vec![0; state.len() * 4]),
        ],
    )
    .expect("Fix: state update must execute");
    assert_eq!(outputs.len(), 3);
    assert_eq!(decode_f32(&outputs[0]), state);
    (decode_f32(&outputs[1]), decode_f32(&outputs[2]))
}

/// Proves two short chunks emit the same values and final state as one full causal prefill.
#[test]
fn chunked_state_continuation_matches_full_sequence_convolution() {
    let weight = [1.0, 2.0, 3.0];
    let (first, state) = update(&[1.0, 2.0], &[0.0, 0.0], &weight);
    assert_eq!(first, vec![3.0, 8.0]);
    assert_eq!(state, vec![1.0, 2.0]);
    let (second, state) = update(&[3.0, 4.0], &state, &weight);
    assert_eq!(second, vec![14.0, 20.0]);
    assert_eq!(state, vec![3.0, 4.0]);

    let prefill = depthwise_causal_conv1d(
        "input",
        "weight",
        None,
        None,
        "output",
        1,
        1,
        4,
        3,
        CausalConvActivation::None,
        DataType::F32,
    )
    .expect("Fix: full prefill must build");
    let outputs = vyre_reference::reference_eval(
        &prefill,
        &[
            Value::from(f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
            Value::from(f32_bytes(&weight)),
            Value::from(vec![0; 16]),
        ],
    )
    .expect("Fix: full prefill must execute");
    assert_eq!(decode_f32(&outputs[0]), [first, second].concat());
}

/// Locks continuation across uneven one-token and multi-token partitions.
#[test]
fn arbitrary_chunk_partitions_preserve_every_output_and_tail() {
    let weight = [1.0, 2.0, 3.0];
    let mut state = vec![0.0, 0.0];
    let mut emitted = Vec::new();
    for chunk in [&[1.0][..], &[2.0, 3.0][..], &[4.0][..]] {
        let (output, next) = update(chunk, &state, &weight);
        emitted.extend(output);
        state = next;
    }
    assert_eq!(emitted, vec![3.0, 8.0, 14.0, 20.0]);
    assert_eq!(state, vec![3.0, 4.0]);
}

/// Ensures resetting to the zero state exactly reproduces cold-sequence output.
#[test]
fn cache_reset_restores_cold_convolution_semantics() {
    let weight = [1.0, 2.0, 3.0];
    let cold = update(&[5.0], &[0.0, 0.0], &weight);
    let warm = update(&[5.0], &[9.0, 8.0], &weight);
    let reset = update(&[5.0], &[0.0, 0.0], &weight);
    assert_ne!(cold, warm);
    assert_eq!(cold, reset);
    assert_eq!(cold, (vec![15.0], vec![0.0, 5.0]));
}

fn contract(shape: Vec<ShapeDim>, lifetime: ValueLifetime, access: BufferAccess) -> ValueContract {
    ValueContract {
        dtype: DataType::F32,
        shape,
        access,
        lifetime,
    }
}

/// Proves the executable update can be represented as an explicit type-preserving ProgramGraph state edge.
#[test]
fn program_graph_carries_convolution_state_generation_explicitly() {
    let mut graph = ProgramGraph::new();
    let input_contract = contract(
        vec![ShapeDim::Known(1), ShapeDim::Known(1), ShapeDim::Known(2)],
        ValueLifetime::Invocation,
        BufferAccess::ReadOnly,
    );
    let weight_contract = contract(
        vec![ShapeDim::Known(1), ShapeDim::Known(3)],
        ValueLifetime::Constant,
        BufferAccess::ReadOnly,
    );
    let state_contract = contract(
        vec![ShapeDim::Known(1), ShapeDim::Known(1), ShapeDim::Known(2)],
        ValueLifetime::Retained,
        BufferAccess::ReadWrite,
    );
    let input = graph
        .add_external_value("chunk", input_contract.clone())
        .expect("Fix: chunk must register");
    let weight = graph
        .add_external_value("conv.weight", weight_contract.clone())
        .expect("Fix: weight must register");
    let state = graph
        .add_external_value("conv.state.0", state_contract.clone())
        .expect("Fix: prior state must register");
    let program: Program = depthwise_causal_conv1d_update(
        "input",
        "weight",
        None,
        "state.in",
        "output",
        "state.out",
        1,
        1,
        2,
        3,
        CausalConvActivation::None,
        DataType::F32,
    )
    .expect("Fix: graph update must build");
    let (_, outputs) = graph
        .add_node(
            "conv.update",
            program,
            vec![
                GraphInput {
                    buffer: "input".into(),
                    value: input,
                    contract: input_contract.clone(),
                },
                GraphInput {
                    buffer: "weight".into(),
                    value: weight,
                    contract: weight_contract,
                },
                GraphInput {
                    buffer: "state.in".into(),
                    value: state,
                    contract: state_contract.clone(),
                },
            ],
            vec![
                GraphOutput {
                    buffer: "output".into(),
                    name: "conv.output".into(),
                    contract: contract(
                        input_contract.shape,
                        ValueLifetime::Output,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                },
                GraphOutput {
                    buffer: "state.out".into(),
                    name: "conv.state.1".into(),
                    contract: state_contract,
                    retained_successor_of: Some(state),
                },
            ],
        )
        .expect("Fix: typed convolution state edge must connect");
    assert_eq!(
        graph.values()[outputs[1].0 as usize].retained_successor_of,
        Some(state)
    );
    graph
        .analyze()
        .expect("Fix: stateful convolution graph must analyze");
}
