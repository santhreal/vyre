//! Shared compare dual-reference machinery.

use crate::dual_impls::common::read_two_words;

#[must_use]
pub(crate) fn binary_direct_predicate(input: &[u8], op: impl FnOnce(u32, u32) -> bool) -> Vec<u8> {
    let Some((left, right)) = read_two_words(input) else {
        return zero_word();
    };
    bool_word(op(left, right))
}

#[must_use]
pub(crate) fn eq_bytes(input: &[u8]) -> Vec<u8> {
    if input.len() < 8 {
        return zero_word();
    }
    bool_word(input[0..4] == input[4..8])
}

#[must_use]
pub(crate) fn lt_bytes(input: &[u8]) -> Vec<u8> {
    if input.len() < 8 {
        return zero_word();
    }
    for byte_index in (0..4).rev() {
        let left = input[byte_index];
        let right = input[byte_index + 4];
        if left != right {
            return bool_word(left < right);
        }
    }
    bool_word(false)
}

fn bool_word(value: bool) -> Vec<u8> {
    u32::from(value).to_le_bytes().to_vec()
}

fn zero_word() -> Vec<u8> {
    vec![0; 4]
}
