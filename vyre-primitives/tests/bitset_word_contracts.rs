//! Word-level bitset primitive contracts.

use vyre_primitives::bitset::{and::bitset_and, and_not::bitset_and_not};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

#[test]
fn bitset_and_overwrites_dirty_output_words() {
    let program = bitset_and("lhs", "rhs", "out", 2);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(&[0xFF00_FF00, 0xAAAA_5555])),
            Value::from(pack(&[0x0F0F_F0F0, 0xFFFF_0000])),
            Value::from(pack(&[u32::MAX, u32::MAX])),
        ],
    )
    .expect("bitset_and reference evaluation must succeed");

    assert_eq!(
        unpack(&outputs[0].to_bytes()),
        vec![0x0F00_F000, 0xAAAA_0000]
    );
}

#[test]
fn bitset_and_not_overwrites_dirty_output_words() {
    let program = bitset_and_not("lhs", "rhs", "out", 2);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(&[0xFFFF_0000, 0xAAAA_5555])),
            Value::from(pack(&[0x0F0F_F0F0, 0xFFFF_0000])),
            Value::from(pack(&[u32::MAX, u32::MAX])),
        ],
    )
    .expect("bitset_and_not reference evaluation must succeed");

    assert_eq!(
        unpack(&outputs[0].to_bytes()),
        vec![0xF0F0_0000, 0x0000_5555]
    );
}

/// Regression (AUDIT_2026-07-10): the `bitset_test_bit` PROGRAM must agree with
/// `cpu_ref` for EVERY `bit_idx`: including out-of-range indices. Before the
/// bounds gate, the GPU program loaded `buf[bit_idx/32]` unconditionally, so an
/// out-of-range `bit_idx` (word >= words) read past `buf` on the GPU while
/// `cpu_ref` returned 0, a CPU/GPU parity divergence + GPU safety hole (the
/// sibling `bitset_contains` had already been audited for exactly this, F-BSC-01;
/// `bitset_test_bit` had not). This runs the program on the reference backend and
/// asserts it equals `cpu_ref` at in-bounds and out-of-range indices.
#[test]
fn bitset_test_bit_program_matches_cpu_ref_including_out_of_range() {
    use vyre_primitives::bitset::test_bit::{bitset_test_bit, cpu_ref};

    // buf = 2 words (words = 2); bit 34 is set (word 1, bit 2).
    let buf = [0u32, 0b100];
    let words = buf.len() as u32;

    // In-bounds hit (34), in-bounds miss (0), just-past-end (64 -> word 2 == words),
    // and far out-of-range (1024 -> word 32). The last two are the regression:
    // the program must return 0 (matching cpu_ref), not read past `buf`.
    for bit_idx in [34u32, 0, 64, 1024] {
        let program = bitset_test_bit("buf", bit_idx, "out", words);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[Value::from(pack(&buf)), Value::from(pack(&[0]))],
        )
        .expect("bitset_test_bit reference evaluation must succeed");
        assert_eq!(
            unpack(&outputs[0].to_bytes())[0],
            cpu_ref(&buf, bit_idx),
            "program must equal cpu_ref for bit_idx {bit_idx} (out-of-range must yield 0)"
        );
    }
}
