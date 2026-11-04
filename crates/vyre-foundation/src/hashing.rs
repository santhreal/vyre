//! Deterministic framing helpers for content-addressed hashes.

/// Add one length-delimited label/value field to a BLAKE3 hash.
pub fn update_length_delimited_field(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}
/// Compute a 32-byte BLAKE3 content digest over raw bytes.
#[must_use]
pub fn digest_bytes(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Compute a domain-separated BLAKE3 content digest over raw bytes.
#[must_use]
pub fn domain_digest(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}
