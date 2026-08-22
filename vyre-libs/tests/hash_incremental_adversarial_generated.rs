//! Generated adversarial incremental-hash tests for CRC32, Adler32, and FNV-1a.

use vyre_reference::composition_witness::{
    adler32_finalize_witness as adler32_finalize_state,
    adler32_initial_a_witness as adler32_initial_a_state,
    adler32_initial_b_witness as adler32_initial_b_state,
    adler32_update_byte_witness as adler32_update_byte_state, adler32_witness as adler32,
    crc32_finalize_witness as crc32_finalize_state,
    crc32_initial_state_witness as crc32_initial_state, crc32_table_witness as build_table,
    crc32_update_byte_witness as crc32_update_byte_state, crc32_witness as crc32,
    fnv1a32_initial_state_witness as fnv1a32_initial_state,
    fnv1a32_update_byte_witness as fnv1a32_update_byte, fnv1a32_witness as fnv1a32,
    fnv1a64_initial_state_witness as fnv1a64_initial_state,
    fnv1a64_update_byte_witness as fnv1a64_update_byte, fnv1a64_witness as fnv1a64,
};

fn generated_case(seed: u32) -> Vec<u8> {
    let len = ((seed.wrapping_mul(17) ^ (seed >> 3)) % 257) as usize;
    let mut state = seed ^ 0xA5A5_5A5A;
    let mut bytes = Vec::with_capacity(len);
    for idx in 0..len {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        bytes.push(
            (state
                .wrapping_add(idx as u32)
                .rotate_left((idx % 31) as u32)
                & 0xFF) as u8,
        );
    }
    bytes
}

#[test]
fn incremental_hash_state_helpers_match_slice_hashers_across_generated_adversarial_cases() {
    let crc_table = build_table();
    for seed in 0..4096u32 {
        let bytes = generated_case(seed);

        let mut crc = crc32_initial_state();
        let mut fnv32 = fnv1a32_initial_state();
        let mut fnv64 = fnv1a64_initial_state();
        let mut adler_a = adler32_initial_a_state();
        let mut adler_b = adler32_initial_b_state();

        for &byte in &bytes {
            crc = crc32_update_byte_state(crc, &crc_table, byte);
            fnv32 = fnv1a32_update_byte(fnv32, byte);
            fnv64 = fnv1a64_update_byte(fnv64, byte);
            let adler = adler32_update_byte_state(adler_a, adler_b, byte);
            adler_a = adler.0;
            adler_b = adler.1;
        }

        assert_eq!(
            crc32_finalize_state(crc),
            crc32(&bytes),
            "crc32 seed {seed}"
        );
        assert_eq!(fnv32, fnv1a32(&bytes), "fnv1a32 seed {seed}");
        assert_eq!(fnv64, fnv1a64(&bytes), "fnv1a64 seed {seed}");
        assert_eq!(
            adler32_finalize_state(adler_a, adler_b),
            adler32(&bytes),
            "adler32 seed {seed}"
        );
    }
}
