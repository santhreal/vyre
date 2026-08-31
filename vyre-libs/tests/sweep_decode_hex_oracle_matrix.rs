//! Handwritten oracle matrix for ASCII hex decode.
//!
//! Compares `vyre_libs::decode::hex::hex_decode` reference evaluation against an
//! independent nibble-table oracle over hostile even-length inputs.

#![forbid(unsafe_code)]
#![cfg(feature = "decode")]

use vyre_libs::decode::hex::hex_decode;
use vyre_libs::decode::hex::hex_decode_table_ref;
use vyre_reference::value::Value;

const HEX_CASES: u32 = 512;
const ALPHABET: &[u8] = b"0123456789abcdefABCDEFXz*#\n\r\t ";

#[test]
fn hex_decode_oracle_matrix_matches_independent_nibble_table() {
    let mut assertions = 0usize;
    for seed in 0..HEX_CASES {
        let input = hostile_hex_input(seed);
        let actual = run_hex_decode(&input);
        let expected = oracle_hex_decode_packed(&input);
        assert_eq!(
            actual,
            expected,
            "Fix: hex_decode seed={seed} len={} must match the independent oracle.",
            input.len()
        );
        assertions += 1;

        assert_eq!(
            actual.len(),
            input.len() / 2,
            "Fix: hex_decode seed={seed} must emit one byte per nibble pair."
        );
        assertions += 1;
    }
    assert_eq!(assertions, HEX_CASES as usize * 2);
}

fn run_hex_decode(input: &[u8]) -> Vec<u32> {
    assert_eq!(
        input.len() % 2,
        0,
        "hex oracle matrix requires even-length inputs"
    );
    let program = hex_decode("input", "output", input.len() as u32);
    let packed_input: Vec<u32> = input.iter().map(|&byte| u32::from(byte)).collect();
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(vyre_primitives::wire::pack_u32_slice(&packed_input)),
            Value::from(vyre_primitives::wire::pack_u32_slice(hex_decode_table_ref())),
        ],
    )
    .expect("Fix: hex_decode reference_eval must succeed for oracle matrix inputs.");
    vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes())
}

fn oracle_hex_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        *slot = match i as u8 {
            b'0'..=b'9' => u32::from(i as u8 - b'0'),
            b'A'..=b'F' => u32::from(i as u8 - b'A' + 10),
            b'a'..=b'f' => u32::from(i as u8 - b'a' + 10),
            _ => 0,
        };
    }
    table
}

fn oracle_hex_decode_packed(input: &[u8]) -> Vec<u32> {
    vyre_reference::composition_witness::hex_decode_packed_witness(input)
}

fn hostile_hex_input(seed: u32) -> Vec<u8> {
    let pairs = 1 + (seed % 32);
    let mut state = seed ^ 0x48EC_DECD;
    let mut input = Vec::with_capacity(pairs as usize * 2);
    for _ in 0..(pairs * 2) {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        input.push(ALPHABET[(state as usize) % ALPHABET.len()]);
    }
    input
}
