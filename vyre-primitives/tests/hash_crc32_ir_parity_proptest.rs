//! Tier 3 - Property: differential proptest driving the ACTUAL CRC-32 GPU IR (`hash::crc32_program`)
//! through `reference_eval` vs the `hash::crc32` byte oracle.
//!
//! MOTIVATION — real IR gap. `proptest_hash_crc32.rs` and the `sweep_hash_crc32_*` matrices assert
//! `crc32(bytes)` against table/algebraic properties or a second CPU CRC — `grep reference_eval` = 0 in
//! all of them. The shipped GPU bit-loop kernel (`crc32_body`: per-byte 8-iteration LFSR reduction
//! with the reflected `CRC32_POLY`, `CRC32_INIT` seed, final inversion) gets its ONLY randomized-input
//! reference_eval check from the single inventory fixture. A wrong poly constant, a missed final XOR,
//! a byte-order slip, or an off-by-one on the `n`-word loop passes every existing CRC test.
//!
//! This sweep runs the real Program over random byte strings (len 0..=256, one byte per u32 slot in
//! the low 8 bits as the op contracts) and asserts `out[0] == crc32(bytes)` bit-exact, plus the
//! standard CRC check-value `crc32(b"123456789") == 0xCBF43926` as an absolute anchor (so a
//! self-consistent-but-wrong oracle+IR pair cannot both drift).
#![cfg(all(feature = "hash", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::hash::crc32::{crc32, crc32_program};

/// Run the CRC-32 IR over `bytes` (one byte per u32 slot) and return `out[0]`.
fn run_ir(bytes: &[u8]) -> u32 {
    let n = bytes.len() as u32;
    let words: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
    let program = crc32_program("input", "out", n);
    let pack = |data: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(data));
    // An n==0 program declares a zero-count input buffer; hand it an empty slice.
    let outputs = vyre_reference::reference_eval(&program, &[pack(&words), pack(&[0u32])])
        .expect("crc32 reference evaluation must succeed");
    // Output buffer `out` (binding 1) → results[1]? RW/output buffers are returned in binding order;
    // `input` is ReadOnly so only `out` is returned → results[0].
    let b = outputs[0].to_bytes();
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn crc32_ir_matches_byte_oracle(bytes in prop::collection::vec(any::<u8>(), 0..=256)) {
        let got = run_ir(&bytes);
        let want = crc32(&bytes);
        prop_assert_eq!(got, want, "len={} bytes[..8]={:?}", bytes.len(), &bytes[..bytes.len().min(8)]);
    }
}

#[test]
fn crc32_ir_standard_check_value_and_boundaries() {
    // The canonical CRC-32 check value: CRC of "123456789" is 0xCBF43926.
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926, "oracle check value");
    assert_eq!(run_ir(b"123456789"), 0xCBF4_3926, "IR check value");

    // Empty input, single byte, and an all-same run.
    assert_eq!(run_ir(&[]), crc32(&[]));
    assert_eq!(run_ir(&[0x00]), crc32(&[0x00]));
    assert_eq!(run_ir(&[0xFF]), crc32(&[0xFF]));
    let run = vec![0xABu8; 129];
    assert_eq!(
        run_ir(&run),
        crc32(&run),
        "129-byte run crosses the single-workgroup body"
    );
}
