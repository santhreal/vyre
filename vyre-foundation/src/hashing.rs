//! Deterministic framing helpers for content-addressed hashes.

/// Add one length-delimited label/value field to a BLAKE3 hash.
pub fn update_length_delimited_field(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}
