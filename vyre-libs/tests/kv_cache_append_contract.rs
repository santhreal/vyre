//! Explicit K/V cache successor execution contracts.

#![forbid(unsafe_code)]

mod wire_words;
use wire_words::{f32_bytes as bytes, f32_words_of as decode, kv_cache_append_test_spec as spec};

use vyre::ir::DataType;
use vyre_libs::nn::attention::{kv_cache_append, KvCacheAppendError};
use vyre_reference::value::Value;

#[allow(clippy::too_many_arguments)]
fn execute(
    prior: &[f32],
    chunk: &[f32],
    batch: u32,
    heads: u32,
    capacity: u32,
    chunk_len: u32,
    dim: u32,
    offset: u32,
) -> Vec<f32> {
    let program = kv_cache_append(spec(
        batch,
        heads,
        capacity,
        chunk_len,
        dim,
        offset,
        DataType::F32,
    ))
    .expect("Fix: valid cache append fixture must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(prior)),
            Value::from(bytes(chunk)),
            Value::from(vec![0; prior.len() * 4]),
        ],
    )
    .expect("Fix: cache append must execute");
    assert_eq!(decode(&outputs[0]), prior);
    decode(&outputs[1])
}

/// Proves a middle decode chunk replaces only its exact token interval.
#[test]
fn middle_chunk_preserves_prefix_and_suffix_bytes() {
    assert_eq!(
        execute(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &[100.0, 101.0, 102.0, 103.0],
            1,
            1,
            4,
            2,
            2,
            1,
        ),
        vec![0.0, 1.0, 100.0, 101.0, 102.0, 103.0, 6.0, 7.0]
    );
}

/// Locks independent batch/head rows so one sequence cannot corrupt another cache.
#[test]
fn batch_and_head_rows_receive_only_their_corresponding_chunk() {
    let prior = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    assert_eq!(
        execute(&prior, &[100.0, 200.0, 300.0, 400.0], 2, 2, 3, 1, 1, 2),
        vec![0.0, 1.0, 100.0, 3.0, 4.0, 200.0, 6.0, 7.0, 300.0, 9.0, 10.0, 400.0,]
    );
}

/// Proves prefill replaces the complete initialized cache without retaining stale bytes.
#[test]
fn full_prefill_replaces_every_cache_element() {
    assert_eq!(
        execute(
            &[-1.0; 6],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            1,
            1,
            3,
            3,
            2,
            0
        ),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

/// Ensures empty, overflowing, and out-of-capacity transitions fail before execution.
#[test]
fn invalid_cache_transitions_fail_closed() {
    assert_eq!(
        kv_cache_append(spec(0, 1, 1, 1, 1, 0, DataType::F32)),
        Err(KvCacheAppendError::EmptyShape)
    );
    assert_eq!(
        kv_cache_append(spec(1, 1, 4, 2, 1, 3, DataType::F32)),
        Err(KvCacheAppendError::Range {
            offset: 3,
            chunk_len: 2,
            capacity: 4
        })
    );
    assert_eq!(
        kv_cache_append(spec(u32::MAX, 2, 1, 1, 1, 0, DataType::F32)),
        Err(KvCacheAppendError::ElementCountOverflow)
    );
}
