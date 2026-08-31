//! Recurrent gated delta-rule execution contracts.

#![forbid(unsafe_code)]

mod wire_words;
use wire_words::{default_gated_delta_spec, f32_bytes as bytes, f32_words_of as decode};

use vyre::ir::{
    BufferAccess, DataType, GraphInput, GraphOutput, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_libs::nn::attention::{recurrent_gated_delta, GatedDeltaSpec, RecurrentGatedDeltaError};
use vyre_reference::value::Value;

#[allow(clippy::too_many_arguments)]
fn execute(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    decay: &[f32],
    beta: &[f32],
    state: &[f32],
    sequence: u32,
    key_heads: u32,
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
) -> (Vec<f32>, Vec<f32>) {
    let spec = default_gated_delta_spec(
        sequence,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        DataType::F32,
    );
    let program =
        recurrent_gated_delta(&spec).expect("Fix: valid recurrent delta fixture must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(query)),
            Value::from(bytes(key)),
            Value::from(bytes(value)),
            Value::from(bytes(decay)),
            Value::from(bytes(beta)),
            Value::from(bytes(state)),
            // `state_output` is a plain ReadWrite result: it consumes one host
            // input slot whose incoming contents the schedule overwrites.
            Value::from(bytes(&vec![0.0f32; state.len()])),
        ],
    )
    .expect("Fix: recurrent delta must execute");
    assert_eq!(outputs.len(), 3);
    assert_eq!(decode(&outputs[0]), state);
    (decode(&outputs[1]), decode(&outputs[2]))
}

/// Locks the exact scalar recurrence over two tokens from a cold state.
#[test]
fn cold_scalar_recurrence_matches_hand_oracle() {
    let (output, state) = execute(
        &[1.0, 1.0],
        &[1.0, 1.0],
        &[1.0, 2.0],
        &[0.0, 0.0],
        &[100.0, 100.0],
        &[0.0],
        2,
        1,
        1,
        1,
        1,
    );
    assert_eq!(output, vec![1.0, 2.0]);
    assert_eq!(state, vec![2.0]);
}

/// Proves warm state, exponential decay, and sigmoid beta are applied in authoritative order.
#[test]
fn warm_state_decay_and_beta_match_exact_update_order() {
    let (output, state) = execute(
        &[1.0],
        &[1.0],
        &[2.0],
        &[0.5_f32.ln()],
        &[0.0],
        &[3.0],
        1,
        1,
        1,
        1,
        1,
    );
    assert_eq!(output, vec![1.75]);
    assert_eq!(state, vec![1.75]);
}

/// Ensures one key/query head repeats independently across grouped value heads.
#[test]
fn grouped_value_heads_share_normalized_key_without_sharing_state() {
    let (output, state) = execute(
        &[1.0],
        &[1.0],
        &[2.0, 3.0],
        &[0.0, 0.0],
        &[100.0, 100.0],
        &[0.0, 0.0],
        1,
        1,
        2,
        1,
        1,
    );
    assert_eq!(output, vec![2.0, 3.0]);
    assert_eq!(state, vec![2.0, 3.0]);
}

/// Proves token partitioning with the returned matrix state is identical to one recurrent sequence.
#[test]
fn returned_state_continues_across_token_partitions_exactly() {
    let full = execute(
        &[1.0, 1.0],
        &[1.0, 1.0],
        &[1.0, 2.0],
        &[0.0, 0.0],
        &[100.0, 100.0],
        &[0.0],
        2,
        1,
        1,
        1,
        1,
    );
    let first = execute(
        &[1.0],
        &[1.0],
        &[1.0],
        &[0.0],
        &[100.0],
        &[0.0],
        1,
        1,
        1,
        1,
        1,
    );
    let second = execute(
        &[1.0],
        &[1.0],
        &[2.0],
        &[0.0],
        &[100.0],
        &first.1,
        1,
        1,
        1,
        1,
        1,
    );
    assert_eq!(full.0, [first.0, second.0].concat());
    assert_eq!(full.1, second.1);
}

/// Locks fail-closed dimension, grouping, overflow, and dtype boundaries.
#[test]
fn invalid_recurrent_delta_contracts_are_rejected() {
    assert_eq!(
        recurrent_gated_delta(&GatedDeltaSpec {
            query: "q",
            key: "k",
            value: "v",
            decay_log: "g",
            beta_logits: "b",
            state_input: "s",
            output: "o",
            state_output: "n",
            batch: 0,
            sequence: 1,
            key_heads: 1,
            value_heads: 1,
            key_dim: 1,
            value_dim: 1,
            eps: 1e-6,
            dtype: DataType::F32,
        })
        .expect_err("Fix: zero batch must fail"),
        RecurrentGatedDeltaError::EmptyShape
    );
    assert_eq!(
        recurrent_gated_delta(&GatedDeltaSpec {
            query: "q",
            key: "k",
            value: "v",
            decay_log: "g",
            beta_logits: "b",
            state_input: "s",
            output: "o",
            state_output: "n",
            batch: 1,
            sequence: 1,
            key_heads: 2,
            value_heads: 3,
            key_dim: 1,
            value_dim: 1,
            eps: 1e-6,
            dtype: DataType::F32,
        })
        .expect_err("Fix: invalid head ratio must fail"),
        RecurrentGatedDeltaError::InvalidHeadGrouping {
            key_heads: 2,
            value_heads: 3
        }
    );
    assert_eq!(
        recurrent_gated_delta(&GatedDeltaSpec {
            query: "q",
            key: "k",
            value: "v",
            decay_log: "g",
            beta_logits: "b",
            state_input: "s",
            output: "o",
            state_output: "n",
            batch: u32::MAX,
            sequence: 2,
            key_heads: 1,
            value_heads: 1,
            key_dim: 1,
            value_dim: 1,
            eps: 1e-6,
            dtype: DataType::F32,
        })
        .expect_err("Fix: flattened overflow must fail"),
        RecurrentGatedDeltaError::ElementCountOverflow
    );
    assert_eq!(
        recurrent_gated_delta(&GatedDeltaSpec {
            query: "q",
            key: "k",
            value: "v",
            decay_log: "g",
            beta_logits: "b",
            state_input: "s",
            output: "o",
            state_output: "n",
            batch: 1,
            sequence: 1,
            key_heads: 1,
            value_heads: 1,
            key_dim: 1,
            value_dim: 1,
            eps: 1e-6,
            dtype: DataType::U32,
        })
        .expect_err("Fix: integer dtype must fail"),
        RecurrentGatedDeltaError::UnsupportedDtype {
            dtype: DataType::U32
        }
    );
}

/// Locks decay-to-zero and beta-to-zero boundaries without stale-state leakage.
#[test]
fn extreme_decay_and_beta_erase_state_without_update() {
    let (output, state) = execute(
        &[1.0],
        &[1.0],
        &[5.0],
        &[-100.0],
        &[-100.0],
        &[10.0],
        1,
        1,
        1,
        1,
        1,
    );
    assert_eq!(output, vec![0.0]);
    assert_eq!(state, vec![0.0]);
}

fn tensor(shape: Vec<ShapeDim>, lifetime: ValueLifetime, access: BufferAccess) -> ValueContract {
    ValueContract {
        dtype: DataType::F32,
        shape,
        access,
        lifetime,
    }
}

/// Proves matrix state is a typed, shape-preserving ProgramGraph generation rather than a hidden buffer convention.
#[test]
fn recurrent_matrix_state_has_explicit_graph_successor() {
    let mut graph = ProgramGraph::new();
    let qk = tensor(
        vec![ShapeDim::Known(1); 4],
        ValueLifetime::Invocation,
        BufferAccess::ReadOnly,
    );
    let activation = qk.clone();
    let scalar = tensor(
        vec![ShapeDim::Known(1); 3],
        ValueLifetime::Invocation,
        BufferAccess::ReadOnly,
    );
    let state_contract = tensor(
        vec![ShapeDim::Known(1); 4],
        ValueLifetime::Retained,
        BufferAccess::ReadWrite,
    );
    let mut external = Vec::new();
    for (name, contract) in [
        ("query", qk.clone()),
        ("key", qk.clone()),
        ("value", activation.clone()),
        ("decay", scalar.clone()),
        ("beta", scalar.clone()),
        ("state.0", state_contract.clone()),
    ] {
        external.push(
            graph
                .add_external_value(name, contract)
                .expect("Fix: recurrent graph input must register"),
        );
    }
    let program = recurrent_gated_delta(&GatedDeltaSpec {
        query: "query",
        key: "key",
        value: "value",
        decay_log: "decay",
        beta_logits: "beta",
        state_input: "state.in",
        output: "output",
        state_output: "state.out",
        batch: 1,
        sequence: 1,
        key_heads: 1,
        value_heads: 1,
        key_dim: 1,
        value_dim: 1,
        eps: 0.0,
        dtype: DataType::F32,
    })
    .expect("Fix: recurrent graph Program must build");
    let (_, outputs) = graph
        .add_node(
            "delta.step",
            program,
            [
                ("query", qk),
                ("key", activation.clone()),
                ("value", activation.clone()),
                ("decay", scalar.clone()),
                ("beta", scalar),
                ("state.in", state_contract.clone()),
            ]
            .into_iter()
            .zip(external.iter().copied())
            .map(|((buffer, contract), value)| GraphInput {
                buffer: buffer.into(),
                value,
                contract,
            })
            .collect(),
            vec![
                GraphOutput {
                    buffer: "output".into(),
                    name: "delta.output".into(),
                    contract: tensor(
                        activation.shape,
                        ValueLifetime::Output,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                },
                GraphOutput {
                    buffer: "state.out".into(),
                    name: "state.1".into(),
                    contract: state_contract,
                    retained_successor_of: Some(external[5]),
                },
            ],
        )
        .expect("Fix: recurrent state edge must connect");
    assert_eq!(
        graph.values()[outputs[1].0 as usize].retained_successor_of,
        Some(external[5])
    );
    graph
        .analyze()
        .expect("Fix: recurrent matrix-state graph must analyze");
}
