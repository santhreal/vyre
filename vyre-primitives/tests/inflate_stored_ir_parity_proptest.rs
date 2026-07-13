//! Tier 3 - Property: differential proptest driving the ACTUAL `decode::inflate_stored` IR (DEFLATE
//! BTYPE=0 stored-block decode) through `reference_eval` vs `inflate_stored_reference_words`. The op
//! had `reference_eval` = 0 in tests/.
//!
//! The kernel parses the 5-word stored-block header (BFINAL/BTYPE byte, LEN u16, NLEN u16), validates
//! `NLEN == !LEN`, then copies `LEN` payload words from `input[5..]` to `output[lane]` and writes the
//! inflated length. The sweep generates VALID stored blocks with random payloads (0..=200 bytes),
//! BFINAL both ways, and trailing padding, asserting `output[0..inflated_len]` == the oracle's decoded
//! data AND `inflated_len` == LEN. A wrong header field offset, a mis-encoded LEN/NLEN check, a
//! payload base-offset slip, or an off-by-one on the copy bound diverges. Deterministic anchor: the
//! canonical `"hello"` stored block from the inventory fixture.
#![cfg(all(feature = "decode", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::decode::inflate::{inflate_stored, inflate_stored_reference_words};

const HEADER_WORDS: usize = 5;

/// Build a valid BTYPE=0 stored-block word stream (one byte per u32 low-8-bits).
fn build_stored_block(payload: &[u8], bfinal: u32, trailing_pad: usize) -> Vec<u32> {
    let len = payload.len() as u16;
    let nlen = !len;
    let mut words = Vec::with_capacity(HEADER_WORDS + payload.len() + trailing_pad);
    words.push(bfinal & 0x1); // BFINAL in bit0, BTYPE=00 in bits1-2
    words.push(u32::from(len & 0xFF));
    words.push(u32::from((len >> 8) & 0xFF));
    words.push(u32::from(nlen & 0xFF));
    words.push(u32::from((nlen >> 8) & 0xFF));
    words.extend(payload.iter().map(|&b| u32::from(b)));
    words.extend(std::iter::repeat(0u32).take(trailing_pad));
    words
}

/// Run the IR; returns (data, inflated_len).
fn run_ir(input_words: &[u32]) -> (Vec<u32>, u32) {
    let input_len = input_words.len() as u32;
    let program = inflate_stored("input", "output", "inflated_len", input_len);
    let pack = |d: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(d));
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(input_words),                     // input (0, RO)
            pack(&vec![0u32; input_len as usize]), // output (1, output)
            pack(&[0u32]),                         // inflated_len (2, RW)
        ],
    )
    .expect("inflate_stored reference evaluation must succeed");
    // RW/output buffers in binding order: output(1) then inflated_len(2).
    let data: Vec<u32> = outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ilen = {
        let b = outputs[1].to_bytes();
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };
    (data, ilen)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1500))]

    #[test]
    fn inflate_stored_ir_matches_oracle(
        payload in prop::collection::vec(any::<u8>(), 0..=200),
        bfinal in 0u32..=1,
        pad in 0usize..=8,
    ) {
        let words = build_stored_block(&payload, bfinal, pad);
        let want = inflate_stored_reference_words(&words)
            .expect("generated stored block must be valid");
        let (data, ilen) = run_ir(&words);

        prop_assert_eq!(ilen, want.inflated_len, "inflated_len");
        prop_assert_eq!(ilen as usize, payload.len());
        prop_assert_eq!(&data[..want.data.len()], &want.data[..], "decoded payload diverges");
    }
}

#[test]
fn inflate_stored_ir_hello_anchor_and_empty() {
    // Canonical "hello" stored block (matches the inventory fixture).
    let words = build_stored_block(b"hello", 1, 0);
    let (data, ilen) = run_ir(&words);
    assert_eq!(ilen, 5);
    let want: Vec<u32> = b"hello".iter().map(|&b| u32::from(b)).collect();
    assert_eq!(&data[..5], &want[..], "hello payload");

    // Empty stored block: LEN=0, nothing copied.
    let empty = build_stored_block(&[], 0, 4);
    let (_data, ilen) = run_ir(&empty);
    assert_eq!(ilen, 0, "empty stored block inflates to zero bytes");
}
