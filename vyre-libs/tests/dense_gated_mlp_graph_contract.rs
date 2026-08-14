//! Connected dense gated-MLP execution contracts.

#![forbid(unsafe_code)]

mod common;
use common::{bf16_bytes, bf16_word, f32_bytes, f32_words_of as decode_f32};

use std::collections::HashMap;

use vyre::ir::{DataType, ProgramGraph};
use vyre_libs::nn::model::{dense_gated_mlp_graph, DenseGatedMlpSpec};
use vyre_reference::value::Value;

fn decode_bf16_word(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

fn element_bytes(dtype: &DataType) -> usize {
    match dtype {
        DataType::F16 | DataType::BF16 => 2,
        DataType::F32 => 4,
        other => panic!("Fix: dense MLP test received unsupported dtype {other:?}"),
    }
}

fn execute_graph(graph: &ProgramGraph, external: &[(&str, Vec<u8>)]) -> Value {
    let mut values = HashMap::<u32, Value>::new();
    for (name, data) in external {
        let graph_value = graph
            .values()
            .iter()
            .find(|value| value.name == *name)
            .unwrap_or_else(|| panic!("Fix: missing dense MLP external {name}"));
        values.insert(graph_value.id.0, Value::from(data.clone()));
    }
    for node in graph.nodes() {
        let arguments = node
            .program
            .buffers()
            .iter()
            .map(|buffer| {
                if buffer.is_output {
                    return Value::from(vec![
                        0;
                        buffer.count as usize * element_bytes(&buffer.element)
                    ]);
                }
                let input = node
                    .inputs
                    .iter()
                    .find(|input| input.buffer.as_str() == buffer.name.as_ref())
                    .unwrap_or_else(|| {
                        panic!("Fix: {} lacks graph input {}", node.name, buffer.name)
                    });
                let value = values.get(&input.value.0).unwrap_or_else(|| {
                    panic!(
                        "Fix: {} input {} is not materialized",
                        node.name, buffer.name
                    )
                });
                Value::from(value.to_bytes().to_vec())
            })
            .collect::<Vec<_>>();
        let outputs = vyre_reference::reference_eval(&node.program, &arguments)
            .unwrap_or_else(|error| panic!("Fix: dense MLP node {} failed: {error}", node.name));
        let output = outputs
            .last()
            .unwrap_or_else(|| panic!("Fix: dense MLP node {} returned no output", node.name));
        values.insert(node.outputs[0].0, Value::from(output.to_bytes().to_vec()));
    }
    let output = graph
        .values()
        .iter()
        .find(|value| value.name == "mlp.block_output")
        .expect("Fix: dense MLP graph must expose block output");
    values
        .remove(&output.id.0)
        .expect("Fix: dense MLP output must execute")
}

fn identity_external(dtype: DataType, residual: &[f32]) -> Vec<(&'static str, Vec<u8>)> {
    let encode = |values: &[f32]| match &dtype {
        DataType::BF16 => bf16_bytes(values),
        DataType::F32 => f32_bytes(values),
        other => panic!("Fix: unsupported identity fixture dtype {other:?}"),
    };
    vec![
        ("residual", encode(residual)),
        ("post_attention_layernorm.weight", encode(&[1.0, 1.0])),
        ("mlp.gate_proj.weight", encode(&[1.0, 0.0, 0.0, 1.0])),
        ("mlp.up_proj.weight", encode(&[1.0, 0.0, 0.0, 1.0])),
        ("mlp.down_proj.weight", encode(&[1.0, 0.0, 0.0, 1.0])),
    ]
}

/// Proves two prompt rows execute normalization, both projections, SwiGLU, down projection, and residual exactly.
#[test]
fn f32_prompt_rows_match_direct_formula() {
    let spec = DenseGatedMlpSpec {
        batch: 1,
        sequence: 2,
        hidden_dim: 2,
        intermediate_dim: 2,
        norm_eps: 0.0,
        dtype: DataType::F32,
    };
    let graph = dense_gated_mlp_graph(&spec).expect("Fix: valid dense MLP graph must build");
    let output = execute_graph(
        &graph,
        &identity_external(DataType::F32, &[1.0, 1.0, 2.0, 2.0]),
    );
    let silu_one = 1.0_f32 / (1.0 + (-1.0_f32).exp());
    let actual = decode_f32(&output);
    for (value, expected) in actual.iter().zip([
        1.0 + silu_one,
        1.0 + silu_one,
        2.0 + silu_one,
        2.0 + silu_one,
    ]) {
        assert!((*value - expected).abs() < 1e-6, "{value} != {expected}");
    }
}

/// Proves one reusable builder supports a different intermediate width and non-square checkpoint weights.
#[test]
fn non_square_cross_model_fixture_uses_output_major_weights() {
    let spec = DenseGatedMlpSpec {
        batch: 1,
        sequence: 1,
        hidden_dim: 2,
        intermediate_dim: 1,
        norm_eps: 0.0,
        dtype: DataType::F32,
    };
    let graph = dense_gated_mlp_graph(&spec).expect("Fix: non-square dense MLP must build");
    let output = execute_graph(
        &graph,
        &[
            ("residual", f32_bytes(&[1.0, 1.0])),
            ("post_attention_layernorm.weight", f32_bytes(&[1.0, 1.0])),
            ("mlp.gate_proj.weight", f32_bytes(&[1.0, 0.0])),
            ("mlp.up_proj.weight", f32_bytes(&[0.0, 1.0])),
            ("mlp.down_proj.weight", f32_bytes(&[2.0, 3.0])),
        ],
    );
    let silu_one = 1.0_f32 / (1.0 + (-1.0_f32).exp());
    let actual = decode_f32(&output);
    assert!((actual[0] - (1.0 + 2.0 * silu_one)).abs() < 1e-6);
    assert!((actual[1] - (1.0 + 3.0 * silu_one)).abs() < 1e-6);
}

/// Locks BF16 rounding after each primitive while all reductions and nonlinear math remain F32.
#[test]
fn bf16_path_matches_exact_rounding_policy() {
    let spec = DenseGatedMlpSpec {
        batch: 1,
        sequence: 1,
        hidden_dim: 1,
        intermediate_dim: 1,
        norm_eps: 0.0,
        dtype: DataType::BF16,
    };
    let graph = dense_gated_mlp_graph(&spec).expect("Fix: BF16 dense MLP graph must build");
    let one = bf16_bytes(&[1.0]);
    let output = execute_graph(
        &graph,
        &[
            ("residual", bf16_bytes(&[2.0])),
            ("post_attention_layernorm.weight", one.clone()),
            ("mlp.gate_proj.weight", one.clone()),
            ("mlp.up_proj.weight", one.clone()),
            ("mlp.down_proj.weight", one),
        ],
    );
    let silu_rounded = decode_bf16_word(bf16_word(1.0 / (1.0 + (-1.0_f32).exp())));
    let expected = bf16_word(2.0 + silu_rounded);
    assert_eq!(
        u16::from_le_bytes(output.to_bytes().try_into().expect("Fix: one BF16 word")),
        expected
    );
}

/// Locks the connected six-stage topology and final operator-visible lifetime.
#[test]
fn graph_stage_order_is_canonical() {
    let graph = dense_gated_mlp_graph(&DenseGatedMlpSpec {
        batch: 1,
        sequence: 1,
        hidden_dim: 2,
        intermediate_dim: 3,
        norm_eps: 1e-6,
        dtype: DataType::F32,
    })
    .expect("Fix: canonical dense MLP must build");
    assert_eq!(
        graph
            .nodes()
            .iter()
            .map(|node| node.name.as_str())
            .collect::<Vec<_>>(),
        [
            "mlp.norm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.swiglu",
            "mlp.down_proj",
            "mlp.residual"
        ]
    );
    graph.analyze().expect("Fix: dense MLP graph must analyze");
}

/// Proves Qwen3.5-27B production dimensions materialize exact checkpoint counts.
#[test]
fn production_qwen35_dimensions_build_exact_weight_contracts() {
    let graph = dense_gated_mlp_graph(&DenseGatedMlpSpec {
        batch: 1,
        sequence: 1,
        hidden_dim: 5120,
        intermediate_dim: 17408,
        norm_eps: 1e-6,
        dtype: DataType::BF16,
    })
    .expect("Fix: production Qwen dense MLP dimensions must build");
    let gate = graph
        .nodes()
        .iter()
        .find(|node| node.name == "mlp.gate_proj")
        .expect("Fix: gate projection exists");
    let down = graph
        .nodes()
        .iter()
        .find(|node| node.name == "mlp.down_proj")
        .expect("Fix: down projection exists");
    assert_eq!(gate.program.buffers()[1].count, 17408 * 5120);
    assert_eq!(down.program.buffers()[1].count, 5120 * 17408);
}

/// Ensures invalid dtypes, empty dimensions, and hostile products fail closed.
#[test]
fn invalid_dense_mlp_contracts_are_rejected() {
    let base = DenseGatedMlpSpec {
        batch: 1,
        sequence: 1,
        hidden_dim: 2,
        intermediate_dim: 2,
        norm_eps: 1e-6,
        dtype: DataType::F32,
    };
    assert!(dense_gated_mlp_graph(&DenseGatedMlpSpec {
        batch: 0,
        ..base.clone()
    })
    .is_err());
    assert!(dense_gated_mlp_graph(&DenseGatedMlpSpec {
        dtype: DataType::U32,
        ..base.clone()
    })
    .is_err());
    assert!(dense_gated_mlp_graph(&DenseGatedMlpSpec {
        batch: u32::MAX,
        sequence: 2,
        ..base
    })
    .is_err());
}
