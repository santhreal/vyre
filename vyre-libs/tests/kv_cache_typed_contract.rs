//! Typed K/V cache successor contracts for checkpoint-native activations.

#![forbid(unsafe_code)]

mod wire_words;
use wire_words::{u16_bytes as bytes, u16_words_of as words};

use vyre::ir::DataType;
use vyre_libs::nn::attention::{kv_cache_append_typed, KvCacheAppendError};
use vyre_reference::value::Value;

#[allow(clippy::too_many_arguments)]
fn execute_bf16(
    prior: &[u16],
    chunk: &[u16],
    batch: u32,
    heads: u32,
    capacity: u32,
    chunk_len: u32,
    dim: u32,
    offset: u32,
) -> Vec<u16> {
    let program = kv_cache_append_typed(
        "prior",
        "chunk",
        "next",
        batch,
        heads,
        capacity,
        chunk_len,
        dim,
        offset,
        DataType::BF16,
    )
    .expect("Fix: valid BF16 cache transition must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(prior)),
            Value::from(bytes(chunk)),
            Value::from(vec![0; prior.len() * size_of::<u16>()]),
        ],
    )
    .expect("Fix: BF16 cache transition must execute");
    assert_eq!(words(&outputs[0]), prior);
    words(&outputs[1])
}

/// Prevents checkpoint BF16 cache words outside the appended interval from being widened or changed.
#[test]
fn bf16_middle_chunk_preserves_exact_prefix_and_suffix_words() {
    let prior = [
        0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0xc040, 0x3f81,
    ];
    let chunk = [0x4100, 0x4110, 0xc100, 0xc110];
    assert_eq!(
        execute_bf16(&prior, &chunk, 1, 1, 4, 2, 2, 1),
        vec![0x3f80, 0x4000, 0x4100, 0x4110, 0xc100, 0xc110, 0xc040, 0x3f81]
    );
}

/// Locks independent batch and head cache rows so one sequence cannot overwrite another sequence's state.
#[test]
fn bf16_batch_and_head_rows_receive_their_own_chunk_words() {
    let prior = (0_u16..12).map(|word| 0x3f00 + word).collect::<Vec<_>>();
    let chunk = [0x4100, 0x4200, 0x4300, 0x4400];
    assert_eq!(
        execute_bf16(&prior, &chunk, 2, 2, 3, 1, 1, 2),
        vec![
            0x3f00, 0x3f01, 0x4100, 0x3f03, 0x3f04, 0x4200, 0x3f06, 0x3f07, 0x4300, 0x3f09, 0x3f0a,
            0x4400,
        ]
    );
}

/// Proves a full BF16 prefill generation removes every stale cache word.
#[test]
fn bf16_full_prefill_replaces_complete_cache_generation() {
    let chunk = [0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0];
    assert_eq!(execute_bf16(&[0xbf80; 6], &chunk, 1, 1, 3, 3, 2, 0), chunk);
}

/// Prevents integer cache storage from entering floating-point attention state.
#[test]
fn integer_cache_dtype_fails_closed() {
    assert_eq!(
        kv_cache_append_typed("prior", "chunk", "next", 1, 1, 1, 1, 1, 0, DataType::U32,),
        Err(KvCacheAppendError::UnsupportedDtype {
            dtype: DataType::U32,
        })
    );
}

/// Prevents offset addition overflow from wrapping a BF16 append into an earlier cache interval.
#[test]
fn bf16_offset_overflow_fails_closed() {
    assert_eq!(
        kv_cache_append_typed(
            "prior",
            "chunk",
            "next",
            1,
            1,
            u32::MAX,
            2,
            1,
            u32::MAX,
            DataType::BF16,
        ),
        Err(KvCacheAppendError::Range {
            offset: u32::MAX,
            chunk_len: 2,
            capacity: u32::MAX,
        })
    );
}
