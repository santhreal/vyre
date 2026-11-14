//! Exact-input identity keys shared by replay and materialized-output caches.
//!
//! Backend replay caches need the same collision-resistant,
//! tuple-boundary-preserving identity for borrowed input slice lists. The key is
//! only a hot-path filter: cache users must still retain collision-safe exact
//! byte checks before reusing bytes.

use crate::BackendError;

const DOMAIN_SEPARATED_INPUT_IDENTITY_PREFIX: &[u8] = b"vyre.input-identity.domain.v1";

/// Fixed-width exact-input identity key.
pub type ExactInputKey = [u8; 32];

fn input_identity_count(value: usize, field: &'static str) -> Result<u64, BackendError> {
    u64::try_from(value).map_err(|source| BackendError::InvalidProgram {
        fix: format!(
            "Fix: exact-input key {field} cannot fit u64 while hashing replay inputs: {source}."
        ),
    })
}

fn update_len_prefixed_bytes(
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
    field: &'static str,
) -> Result<(), BackendError> {
    let byte_len = input_identity_count(bytes.len(), field)?;
    hasher.update(&byte_len.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

fn update_input_tuple(hasher: &mut blake3::Hasher, inputs: &[&[u8]]) -> Result<(), BackendError> {
    let input_count = input_identity_count(inputs.len(), "input count")?;
    hasher.update(&input_count.to_le_bytes());
    for input in inputs {
        update_len_prefixed_bytes(hasher, input, "input length")?;
    }
    Ok(())
}

/// Hash a borrowed input tuple with explicit arity and length prefixes.
///
/// # Errors
///
/// Returns [`BackendError`] when the input arity or one input length cannot fit
/// the stable `u64` hash envelope.
pub fn exact_input_key(inputs: &[&[u8]]) -> Result<ExactInputKey, BackendError> {
    let mut hasher = blake3::Hasher::new();
    update_input_tuple(&mut hasher, inputs)?;
    Ok(*hasher.finalize().as_bytes())
}

/// Hash a borrowed input tuple under an explicit cache domain and device salt.
///
/// Use this for resident/static caches that need the same tuple-boundary
/// protection as replay keys, but must not alias across different cache users,
/// logical domains, or backend feature sets.
///
/// # Errors
///
/// Returns [`BackendError`] when the domain tag is empty, the domain tag cannot
/// fit the stable `u64` envelope, or an input arity/length cannot fit.
pub fn domain_separated_exact_input_key(
    domain_tag: &[u8],
    domain_id: u64,
    feature_key: u64,
    inputs: &[&[u8]],
) -> Result<ExactInputKey, BackendError> {
    if domain_tag.is_empty() {
        return Err(BackendError::InvalidProgram {
            fix: "Fix: exact-input domain-separated key requires a non-empty domain tag."
                .to_string(),
        });
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN_SEPARATED_INPUT_IDENTITY_PREFIX);
    update_len_prefixed_bytes(&mut hasher, domain_tag, "domain tag length")?;
    hasher.update(&domain_id.to_le_bytes());
    hasher.update(&feature_key.to_le_bytes());
    update_input_tuple(&mut hasher, inputs)?;
    Ok(*hasher.finalize().as_bytes())
}
