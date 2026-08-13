//! CUDA adapter for exact-input identity keys.
//!
//! CUDA graph replay and compiled-pipeline materialized output caching share the
//! backend-neutral keying contract from `vyre-driver`; this module keeps the
//! CUDA-private import path stable without forking the BLAKE3 tuple envelope.

pub(crate) use vyre_driver::input_identity::{exact_input_key, ExactInputKey};

#[cfg(test)]
mod tests {
    use super::exact_input_key;

    #[test]
    fn cuda_exact_input_key_preserves_shared_tuple_boundary_contract() {
        let tuple_key = exact_input_key(&[b"ab".as_slice(), b"c".as_slice()])
            .expect("Fix: CUDA tuple exact-input key should fit");
        let concatenated_key = exact_input_key(&[b"abc".as_slice()])
            .expect("Fix: CUDA concatenated exact-input key should fit");
        let empty_separated_key = exact_input_key(&[b"ab".as_slice(), &[], b"c".as_slice()])
            .expect("Fix: CUDA empty-separated exact-input key should fit");

        assert_ne!(tuple_key, concatenated_key);
        assert_ne!(tuple_key, empty_separated_key);
    }

    #[test]
    fn generated_cuda_exact_input_keys_use_shared_collision_filter_contract() {
        for seed in 0_u32..1024 {
            let len = ((seed.wrapping_mul(53) ^ seed.rotate_left(7)) % 128 + 1) as usize;
            let mut bytes = Vec::with_capacity(len);
            let mut state = seed ^ 0xA11C_EE5D;
            for index in 0..len {
                state = state
                    .wrapping_mul(747_796_405)
                    .wrapping_add(2_891_336_453)
                    .rotate_left((index as u32) & 15);
                bytes.push((state >> ((index & 3) * 8)) as u8);
            }
            let mut mutated = bytes.clone();
            mutated[(seed as usize) % len] ^= 1 | ((seed as u8) << 1);

            let original = exact_input_key(&[bytes.as_slice()])
                .expect("Fix: generated CUDA exact-input key should fit");
            let changed = exact_input_key(&[mutated.as_slice()])
                .expect("Fix: generated mutated CUDA exact-input key should fit");

            assert_ne!(original, changed);
        }
    }
}
