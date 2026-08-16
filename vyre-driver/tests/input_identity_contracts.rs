//! Contracts for `vyre_driver::input_identity`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::input_identity::{domain_separated_exact_input_key, exact_input_key};

use vyre_driver::input_identity::{domain_separated_exact_input_key, exact_input_key};

#[test]
fn exact_input_key_separates_tuple_boundaries_for_4096_generated_cases() {
    for seed in 0_u32..4096 {
        let left_len = ((seed.wrapping_mul(17) ^ seed.rotate_left(5)) % 31 + 1) as usize;
        let right_len = ((seed.wrapping_mul(29) ^ seed.rotate_left(9)) % 31 + 1) as usize;
        let mut state = seed ^ 0xC0DA_CAFE;
        let mut left = Vec::with_capacity(left_len);
        let mut right = Vec::with_capacity(right_len);
        for index in 0..left_len {
            state = state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .rotate_left((index as u32) & 15);
            left.push((state ^ seed.rotate_left(index as u32 & 31)) as u8);
        }
        for index in 0..right_len {
            state = state
                .wrapping_mul(22_695_477)
                .wrapping_add(1)
                .rotate_left((index as u32) & 7);
            right.push((state ^ seed.rotate_right(index as u32 & 31)) as u8);
        }
        let mut concatenated = Vec::with_capacity(left_len + right_len);
        concatenated.extend_from_slice(&left);
        concatenated.extend_from_slice(&right);

        let tuple_key = exact_input_key(&[left.as_slice(), right.as_slice()])
            .expect("Fix: generated tuple exact-input key must fit");
        let concatenated_key = exact_input_key(&[concatenated.as_slice()])
            .expect("Fix: generated concatenated exact-input key must fit");
        let empty_separated_key = exact_input_key(&[left.as_slice(), &[], right.as_slice()])
            .expect("Fix: generated empty-separated exact-input key must fit");

        assert_ne!(
            tuple_key, concatenated_key,
            "Fix: exact-input key must length-prefix slots so tuple boundaries cannot alias for generated case {seed}."
        );
        assert_ne!(
            tuple_key, empty_separated_key,
            "Fix: exact-input key must include empty input slots instead of collapsing them for generated case {seed}."
        );
    }
}

#[test]
fn exact_input_key_changes_on_4096_generated_single_byte_mutations() {
    for seed in 0_u32..4096 {
        let len = ((seed.wrapping_mul(37) ^ seed.rotate_left(11)) % 96 + 1) as usize;
        let mut bytes = Vec::with_capacity(len);
        let mut state = seed ^ 0xA5A5_5A5A;
        for index in 0..len {
            state = state
                .wrapping_mul(1_103_515_245)
                .wrapping_add(12_345)
                .rotate_left((index as u32) & 15);
            bytes.push((state >> ((index & 3) * 8)) as u8);
        }
        let mut mutated = bytes.clone();
        let mutation_index = (seed as usize) % len;
        mutated[mutation_index] ^= 0x80 | ((seed as u8) & 0x7f);

        let base_key = exact_input_key(&[bytes.as_slice()])
            .expect("Fix: base generated exact-input key must fit");
        let mutated_key = exact_input_key(&[mutated.as_slice()])
            .expect("Fix: mutated generated exact-input key must fit");

        assert_ne!(
            base_key, mutated_key,
            "Fix: exact-input key must change when one byte changes for generated case {seed}."
        );
    }
}

#[test]
fn domain_separated_exact_input_key_preserves_domain_and_tuple_boundaries() {
    for seed in 0_u32..2048 {
        let left_len = ((seed.wrapping_mul(19) ^ seed.rotate_left(3)) % 48 + 1) as usize;
        let right_len = ((seed.wrapping_mul(41) ^ seed.rotate_left(9)) % 48 + 1) as usize;
        let mut state = seed ^ 0x1DEA_7E5D;
        let mut left = Vec::with_capacity(left_len);
        let mut right = Vec::with_capacity(right_len);
        for index in 0..left_len {
            state = state
                .wrapping_mul(747_796_405)
                .wrapping_add(2_891_336_453)
                .rotate_left((index as u32) & 15);
            left.push((state >> ((index & 3) * 8)) as u8);
        }
        for index in 0..right_len {
            state = state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .rotate_left((index as u32) & 7);
            right.push((state ^ seed.rotate_right(index as u32 & 31)) as u8);
        }
        let mut concatenated = Vec::with_capacity(left_len + right_len);
        concatenated.extend_from_slice(&left);
        concatenated.extend_from_slice(&right);
        let domain_id = u64::from(seed) << 1;
        let feature_key = u64::from(seed.rotate_left(11)) | 1;

        let key = domain_separated_exact_input_key(
            b"generated.cache.domain",
            domain_id,
            feature_key,
            &[left.as_slice(), right.as_slice()],
        )
        .expect("Fix: generated domain-separated exact-input key must fit");
        let different_tag = domain_separated_exact_input_key(
            b"generated.cache.other",
            domain_id,
            feature_key,
            &[left.as_slice(), right.as_slice()],
        )
        .expect("Fix: generated domain tag variation must fit");
        let different_domain = domain_separated_exact_input_key(
            b"generated.cache.domain",
            domain_id ^ 0x55AA,
            feature_key,
            &[left.as_slice(), right.as_slice()],
        )
        .expect("Fix: generated domain id variation must fit");
        let different_feature = domain_separated_exact_input_key(
            b"generated.cache.domain",
            domain_id,
            feature_key.rotate_left(17),
            &[left.as_slice(), right.as_slice()],
        )
        .expect("Fix: generated feature key variation must fit");
        let concatenated_key = domain_separated_exact_input_key(
            b"generated.cache.domain",
            domain_id,
            feature_key,
            &[concatenated.as_slice()],
        )
        .expect("Fix: generated concatenated domain key must fit");

        assert_ne!(key, different_tag);
        assert_ne!(key, different_domain);
        assert_ne!(key, different_feature);
        assert_ne!(key, concatenated_key);
        assert_ne!(
            key,
            exact_input_key(&[left.as_slice(), right.as_slice()])
                .expect("Fix: generated plain exact-input key must fit"),
            "Fix: domain-separated exact-input keys must not alias plain replay keys."
        );
    }
}

#[test]
fn domain_separated_exact_input_key_rejects_empty_domain_tag() {
    let error = domain_separated_exact_input_key(&[], 0, 0, &[b"payload".as_slice()])
        .expect_err("Fix: empty cache domains must be rejected");
    assert!(
        error.to_string().contains("non-empty domain tag"),
        "Fix: empty domain tag diagnostics must explain the rejected cache-domain contract."
    );
}
