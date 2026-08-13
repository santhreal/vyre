//! Comparison references derived by walking the input bytes rather than
//! decoding two words, so they stay independent of the direct `u32` arm of
//! each dual pair.

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
