//! Chunk-size-64 gated delta prefill execution contracts.

#![forbid(unsafe_code)]

use vyre::ir::DataType;
use vyre_libs::nn::attention::{chunked_gated_delta, recurrent_gated_delta};
use vyre_reference::value::Value;

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode(value: &Value) -> Vec<f32> {
    value
        .to_bytes()
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("Fix: exact f32 word")))
        .collect()
}

fn execute(sequence: u32, chunked: bool, state: &[f32], seed: usize) -> (Vec<f32>, Vec<f32>) {
    let len = sequence as usize;
    let query = (0..len)
        .map(|index| 0.5 + ((index + seed) % 7) as f32 / 8.0)
        .collect::<Vec<_>>();
    let key = (0..len)
        .map(|index| 0.25 + ((index + seed * 3) % 5) as f32 / 6.0)
        .collect::<Vec<_>>();
    let value = (0..len)
        .map(|index| ((index + seed) % 13) as f32 - 4.0)
        .collect::<Vec<_>>();
    let decay = (0..len)
        .map(|index| -0.01 * (1 + (index % 3)) as f32)
        .collect::<Vec<_>>();
    let beta = (0..len)
        .map(|index| -1.5 + (index % 11) as f32 / 4.0)
        .collect::<Vec<_>>();
    let program = if chunked {
        chunked_gated_delta
    } else {
        recurrent_gated_delta
    }(
        "query",
        "key",
        "value",
        "decay",
        "beta",
        "state.in",
        "output",
        "state.out",
        1,
        sequence,
        1,
        1,
        1,
        1,
        0.0,
        DataType::F32,
    )
    .expect("Fix: valid delta fixture must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(&query)),
            Value::from(bytes(&key)),
            Value::from(bytes(&value)),
            Value::from(bytes(&decay)),
            Value::from(bytes(&beta)),
            Value::from(bytes(state)),
            Value::from(vec![0; len * 4]),
            Value::from(vec![0; state.len() * 4]),
        ],
    )
    .expect("Fix: delta schedule must execute");
    assert_eq!(decode(&outputs[0]), state);
    (decode(&outputs[1]), decode(&outputs[2]))
}

fn assert_close(actual: &[f32], expected: &[f32], sequence: u32, label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= 1e-4,
            "{label} mismatch at sequence {sequence}, index {index}: {actual} != {expected}, error={error}"
        );
    }
}

fn assert_schedule_parity(sequence: u32, state: &[f32], seed: usize) {
    let recurrent = execute(sequence, false, state, seed);
    let chunked = execute(sequence, true, state, seed);
    assert_close(&chunked.0, &recurrent.0, sequence, "output");
    assert_close(&chunked.1, &recurrent.1, sequence, "state");
}

/// Proves one complete triangular tile stays within F32-ordering tolerance of token recurrence.
#[test]
fn complete_64_token_tile_matches_recurrent_execution() {
    assert_schedule_parity(64, &[0.0], 1);
}

/// Locks the one-token tile and every nonmultiple final-tile boundary around 64 rows.
#[test]
fn guarded_padding_never_reads_or_updates_beyond_logical_sequence() {
    for sequence in [1, 2, 63, 65, 127, 129] {
        assert_schedule_parity(sequence, &[0.0], sequence as usize);
    }
}

/// Proves a warm incoming matrix state is decayed and continued identically by both schedules.
#[test]
fn warm_state_continuation_matches_recurrent_execution() {
    for sequence in [1, 64, 65, 130] {
        assert_schedule_parity(sequence, &[3.25], 9);
    }
}

/// Prevents later tiles from losing accumulated decay or delta updates on a realistic long prompt.
#[test]
fn long_prompt_crosses_multiple_tiles_without_state_drift() {
    assert_schedule_parity(257, &[-2.0], 17);
}

/// Proves chunk output and returned state are deterministic across repeated execution.
#[test]
fn chunk_schedule_replays_exact_output_and_final_state() {
    let first = execute(65, true, &[0.75], 23);
    let replay = execute(65, true, &[0.75], 23);
    assert_eq!(first, replay);
}

#[allow(clippy::too_many_arguments)]
fn execute_chunk_fixture(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    decay: &[f32],
    beta_logits: &[f32],
    state: &[f32],
    sequence: u32,
    key_dim: u32,
    value_dim: u32,
) -> (Vec<f32>, Vec<f32>) {
    let program = chunked_gated_delta(
        "query",
        "key",
        "value",
        "decay",
        "beta",
        "state.in",
        "output",
        "state.out",
        1,
        sequence,
        1,
        1,
        key_dim,
        value_dim,
        1e-6,
        DataType::F32,
    )
    .expect("Fix: authoritative chunk fixture must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(query)),
            Value::from(bytes(key)),
            Value::from(bytes(value)),
            Value::from(bytes(decay)),
            Value::from(bytes(beta_logits)),
            Value::from(bytes(state)),
            Value::from(vec![0; value.len() * 4]),
            Value::from(vec![0; state.len() * 4]),
        ],
    )
    .expect("Fix: authoritative chunk fixture must execute");
    assert_eq!(decode(&outputs[0]), state);
    (decode(&outputs[1]), decode(&outputs[2]))
}

fn l2_rows(values: &[f32], rows: usize, width: usize) -> Vec<f32> {
    let mut normalized = vec![0.0; values.len()];
    for row in 0..rows {
        let start = row * width;
        let sum = values[start..start + width]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let scale = 1.0 / (sum + 1e-6).sqrt();
        for feature in 0..width {
            normalized[start + feature] = values[start + feature] * scale;
        }
    }
    normalized
}

#[allow(clippy::too_many_arguments)]
fn transformers_chunk_oracle(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    decay: &[f32],
    beta_logits: &[f32],
    initial_state: &[f32],
    sequence: usize,
    key_dim: usize,
    value_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut query = l2_rows(query, sequence, key_dim);
    let key = l2_rows(key, sequence, key_dim);
    let query_scale = 1.0 / (key_dim as f32).sqrt();
    for component in &mut query {
        *component *= query_scale;
    }
    let beta = beta_logits
        .iter()
        .map(|logit| 1.0 / (1.0 + (-logit).exp()))
        .collect::<Vec<_>>();
    let mut state = initial_state.to_vec();
    let mut output = vec![0.0; sequence * value_dim];

    for chunk_start in (0..sequence).step_by(64) {
        let len = (sequence - chunk_start).min(64);
        let mut cumulative = vec![0.0_f32; len];
        for row in 0..len {
            cumulative[row] =
                decay[chunk_start + row] + if row == 0 { 0.0 } else { cumulative[row - 1] };
        }

        let mut inverse = vec![vec![0.0_f32; len]; len];
        for row in 0..len {
            for column in 0..row {
                let dot = (0..key_dim)
                    .map(|feature| {
                        key[(chunk_start + row) * key_dim + feature]
                            * key[(chunk_start + column) * key_dim + feature]
                    })
                    .sum::<f32>();
                inverse[row][column] =
                    -beta[chunk_start + row] * dot * (cumulative[row] - cumulative[column]).exp();
            }
        }
        for row in 1..len {
            let original = inverse[row][..row].to_vec();
            for column in 0..row {
                let correction = (0..row)
                    .map(|inner| original[inner] * inverse[inner][column])
                    .sum::<f32>();
                inverse[row][column] = original[column] + correction;
            }
        }
        for row in 0..len {
            inverse[row][row] = 1.0;
        }

        let mut transformed = vec![vec![0.0_f32; value_dim]; len];
        let mut cumulative_key = vec![vec![0.0_f32; key_dim]; len];
        for row in 0..len {
            for column in 0..=row {
                let source = chunk_start + column;
                let coefficient = inverse[row][column] * beta[source];
                for feature in 0..value_dim {
                    transformed[row][feature] += coefficient * value[source * value_dim + feature];
                }
                for feature in 0..key_dim {
                    cumulative_key[row][feature] +=
                        coefficient * cumulative[column].exp() * key[source * key_dim + feature];
                }
            }
        }

        let mut value_new = transformed;
        for row in 0..len {
            for value_feature in 0..value_dim {
                let state_projection = (0..key_dim)
                    .map(|key_feature| {
                        cumulative_key[row][key_feature]
                            * state[key_feature * value_dim + value_feature]
                    })
                    .sum::<f32>();
                value_new[row][value_feature] -= state_projection;
            }
        }

        for row in 0..len {
            let token = chunk_start + row;
            for value_feature in 0..value_dim {
                let state_term = (0..key_dim)
                    .map(|key_feature| {
                        query[token * key_dim + key_feature]
                            * cumulative[row].exp()
                            * state[key_feature * value_dim + value_feature]
                    })
                    .sum::<f32>();
                let local_term = (0..=row)
                    .map(|column| {
                        let source = chunk_start + column;
                        let dot = (0..key_dim)
                            .map(|key_feature| {
                                query[token * key_dim + key_feature]
                                    * key[source * key_dim + key_feature]
                            })
                            .sum::<f32>();
                        dot * (cumulative[row] - cumulative[column]).exp()
                            * value_new[column][value_feature]
                    })
                    .sum::<f32>();
                output[token * value_dim + value_feature] = state_term + local_term;
            }
        }

        let last_decay = cumulative[len - 1];
        let mut next_state = vec![0.0_f32; state.len()];
        for key_feature in 0..key_dim {
            for value_feature in 0..value_dim {
                let mut next = last_decay.exp() * state[key_feature * value_dim + value_feature];
                for row in 0..len {
                    next += (last_decay - cumulative[row]).exp()
                        * key[(chunk_start + row) * key_dim + key_feature]
                        * value_new[row][value_feature];
                }
                next_state[key_feature * value_dim + value_feature] = next;
            }
        }
        state = next_state;
    }
    (output, state)
}

/// Differentially locks the matrix triangular solve against the Transformers reference formula.
#[test]
fn triangular_matrix_formula_matches_transformers_across_chunk_boundary() {
    let sequence = 65_usize;
    let key_dim = 2_usize;
    let value_dim = 2_usize;
    let query = (0..sequence * key_dim)
        .map(|index| 0.2 + (index % 7) as f32 * 0.13)
        .collect::<Vec<_>>();
    let key = (0..sequence * key_dim)
        .map(|index| -0.4 + (index % 5) as f32 * 0.21)
        .collect::<Vec<_>>();
    let value = (0..sequence * value_dim)
        .map(|index| -1.0 + (index % 11) as f32 * 0.3)
        .collect::<Vec<_>>();
    let decay = (0..sequence)
        .map(|index| -0.01 * (1 + index % 4) as f32)
        .collect::<Vec<_>>();
    let beta = (0..sequence)
        .map(|index| -1.25 + (index % 9) as f32 * 0.25)
        .collect::<Vec<_>>();
    let initial_state = vec![0.2, -0.3, 0.4, 0.1];
    let actual = execute_chunk_fixture(
        &query,
        &key,
        &value,
        &decay,
        &beta,
        &initial_state,
        sequence as u32,
        key_dim as u32,
        value_dim as u32,
    );
    let expected = transformers_chunk_oracle(
        &query,
        &key,
        &value,
        &decay,
        &beta,
        &initial_state,
        sequence,
        key_dim,
        value_dim,
    );
    assert_close(
        &actual.0,
        &expected.0,
        sequence as u32,
        "Transformers output",
    );
    assert_close(
        &actual.1,
        &expected.1,
        sequence as u32,
        "Transformers state",
    );
}

fn bf16_word(value: f32) -> u16 {
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    bits.wrapping_add(bias).wrapping_shr(16) as u16
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16_word(*value).to_le_bytes())
        .collect()
}

fn decode_bf16(value: &Value) -> Vec<u16> {
    value
        .to_bytes()
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().expect("Fix: exact BF16 word")))
        .collect()
}

fn execute_bf16_schedule(sequence: u32, chunked: bool, state: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let len = sequence as usize;
    let query = (0..len)
        .map(|index| 0.5 + (index % 5) as f32 * 0.125)
        .collect::<Vec<_>>();
    let key = (0..len)
        .map(|index| 0.25 + (index % 3) as f32 * 0.25)
        .collect::<Vec<_>>();
    let value = (0..len)
        .map(|index| (index % 9) as f32 - 3.0)
        .collect::<Vec<_>>();
    let decay = vec![-0.015625; len];
    let beta = (0..len)
        .map(|index| -1.0 + (index % 7) as f32 * 0.25)
        .collect::<Vec<_>>();
    let program = if chunked {
        chunked_gated_delta
    } else {
        recurrent_gated_delta
    }(
        "query",
        "key",
        "value",
        "decay",
        "beta",
        "state.in",
        "output",
        "state.out",
        1,
        sequence,
        1,
        1,
        1,
        1,
        0.0,
        DataType::BF16,
    )
    .expect("Fix: BF16 delta schedule must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bf16_bytes(&query)),
            Value::from(bf16_bytes(&key)),
            Value::from(bf16_bytes(&value)),
            Value::from(bf16_bytes(&decay)),
            Value::from(bf16_bytes(&beta)),
            Value::from(bytes(state)),
            Value::from(vec![0; len * 2]),
            Value::from(vec![0; state.len() * 4]),
        ],
    )
    .expect("Fix: BF16 delta schedule must execute");
    assert_eq!(decode(&outputs[0]), state);
    (decode_bf16(&outputs[1]), decode(&outputs[2]))
}

/// Locks source-dtype rounding while triangular reductions and state remain F32.
#[test]
fn bf16_chunk_output_and_f32_state_match_recurrent_schedule() {
    let recurrent = execute_bf16_schedule(65, false, &[0.5]);
    let chunked = execute_bf16_schedule(65, true, &[0.5]);
    assert_eq!(chunked.0, recurrent.0);
    assert_close(&chunked.1, &recurrent.1, 65, "BF16 final state");
}

/// Proves grouped value heads share K/Q rows but retain independent values and matrix states.
#[test]
fn grouped_value_heads_match_recurrent_without_cross_head_state() {
    let sequence = 65_u32;
    let len = sequence as usize;
    let query = vec![1.0; len];
    let key = vec![1.0; len];
    let value = (0..len)
        .flat_map(|index| [index as f32 * 0.01, 2.0 - index as f32 * 0.02])
        .collect::<Vec<_>>();
    let decay = vec![-0.02; len * 2];
    let beta = (0..len)
        .flat_map(|index| [-0.5 + index as f32 * 0.01, 0.75 - index as f32 * 0.005])
        .collect::<Vec<_>>();
    let state = vec![0.25, -0.75];
    let run = |chunked: bool| {
        let program = if chunked {
            chunked_gated_delta
        } else {
            recurrent_gated_delta
        }(
            "query",
            "key",
            "value",
            "decay",
            "beta",
            "state.in",
            "output",
            "state.out",
            1,
            sequence,
            1,
            2,
            1,
            1,
            0.0,
            DataType::F32,
        )
        .expect("Fix: grouped delta schedule must build");
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(bytes(&query)),
                Value::from(bytes(&key)),
                Value::from(bytes(&value)),
                Value::from(bytes(&decay)),
                Value::from(bytes(&beta)),
                Value::from(bytes(&state)),
                Value::from(vec![0; value.len() * 4]),
                Value::from(vec![0; state.len() * 4]),
            ],
        )
        .expect("Fix: grouped delta schedule must execute");
        (decode(&outputs[1]), decode(&outputs[2]))
    };
    let recurrent = run(false);
    let chunked = run(true);
    assert_close(&chunked.0, &recurrent.0, sequence, "grouped output");
    assert_close(&chunked.1, &recurrent.1, sequence, "grouped state");
}
