//! The reference runs behind the succinct rank/select contract suites.
//!
//! Two suites cover the same three builders, and each case used to pack its own
//! buffers, call the interpreter, and decode the answer. That put the ABI in
//! forty places at once: when the interpreter stopped accepting a placeholder
//! for a backend-allocated output, every one of them was a separate edit. The
//! call lives here now, so a suite states inputs and expectations and nothing
//! else.

use vyre_reference::value::Value;

use crate::wire_words::{decode_u32_words, u32_bytes};

/// Build the rank superblock prefix table for `bits`.
pub(crate) fn superblocks(bits: &[u32], words: u32, block_words: u32) -> Vec<u32> {
    let program = vyre_libs::math::succinct::rank1_superblocks("bits", "sb", words, block_words);
    let outputs = vyre_reference::reference_eval(&program, &[Value::from(u32_bytes(bits))])
        .expect("Fix: rank1_superblocks must execute in the reference interpreter.");
    decode_u32_words(&outputs[0].to_bytes())
}

/// Answer `queries` against `bits` and its superblock table.
pub(crate) fn rank_query(
    bits: &[u32],
    superblocks: &[u32],
    queries: &[u32],
    block_words: u32,
) -> Vec<u32> {
    try_rank_query(bits, superblocks, queries, block_words)
        .expect("Fix: rank1_query must execute in the reference interpreter.")
}

/// Answer `queries`, handing back the refusal a trap contract asserts on.
pub(crate) fn try_rank_query(
    bits: &[u32],
    superblocks: &[u32],
    queries: &[u32],
    block_words: u32,
) -> Result<Vec<u32>, vyre_reference::ReferenceError> {
    let program = vyre_libs::math::succinct::rank1_query(
        "bits",
        "sb",
        "q",
        "out",
        bits.len() as u32,
        queries.len() as u32,
        block_words,
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(u32_bytes(bits)),
            Value::from(u32_bytes(superblocks)),
            Value::from(u32_bytes(queries)),
        ],
    )?;
    Ok(decode_u32_words(&outputs[0].to_bytes()))
}

/// Resolve one-based ranks to zero-based bit positions.
pub(crate) fn select_query(bits: &[u32], queries: &[u32]) -> Vec<u32> {
    try_select_query(bits, queries)
        .expect("Fix: select1_query must execute in the reference interpreter.")
}

/// Resolve one-based ranks, handing back the refusal a trap contract asserts on.
pub(crate) fn try_select_query(
    bits: &[u32],
    queries: &[u32],
) -> Result<Vec<u32>, vyre_reference::ReferenceError> {
    let program = vyre_libs::bitset::select::select1_query(
        "bits",
        "q",
        "out",
        bits.len() as u32,
        queries.len() as u32,
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(u32_bytes(bits)),
            Value::from(u32_bytes(queries)),
        ],
    )?;
    Ok(decode_u32_words(&outputs[0].to_bytes()))
}
