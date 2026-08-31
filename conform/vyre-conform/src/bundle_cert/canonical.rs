//! Turning a corpus into a deterministic byte stream and its hash.
//!
//! Both issue and verify hash the same way, so the canonical form has one
//! owner: witnesses sorted by name, every field length-prefixed, so an empty
//! output cannot collide with an absent one.

use vyre_conform_spec::ConformanceCase;

use super::error::BundleCertError;

/// Canonicalise a corpus into a deterministic input stream + hash.
///
/// Sorts witnesses by `name`, then for each writes
/// `len(name) || name || witness_count || for each input: len || bytes`.
/// A consumer that receives the same witness set in any order
/// produces the same hash.
pub fn canonicalise_corpus(
    corpus: &[ConformanceCase],
) -> Result<(Vec<usize>, [u8; 32]), BundleCertError> {
    let mut sorted_indices: Vec<usize> = (0..corpus.len()).collect();
    sorted_indices.sort_by(|&left, &right| corpus[left].name.cmp(&corpus[right].name));

    // Reject duplicate names *after*
    // the stable sort so the error names one colliding entry exactly
    // once. A deterministic hash of `[dup, dup]` previously passed
    // verification while any downstream index-by-name consumer
    // silently dropped the second entry.
    if let Some(dup) = sorted_indices
        .windows(2)
        .find_map(|pair| (corpus[pair[0]].name == corpus[pair[1]].name).then_some(pair[0]))
    {
        return Err(BundleCertError::DuplicateWitnessName {
            name: corpus[dup].name.clone(),
        });
    }

    let mut hasher = blake3::Hasher::new();
    for idx in &sorted_indices {
        let w = &corpus[*idx];
        hasher.update(&(w.name.len() as u64).to_le_bytes());
        hasher.update(w.name.as_bytes());
        hasher.update(&(w.inputs.len() as u64).to_le_bytes());
        for input in &w.inputs {
            hasher.update(&(input.len() as u64).to_le_bytes());
            hasher.update(input);
        }
    }
    let mut hash = [0u8; 32];
    hash.copy_from_slice(hasher.finalize().as_bytes());
    Ok((sorted_indices, hash))
}

/// Fold one output stream into `hasher`, length-prefixing every buffer.
///
/// The prefix is what keeps an empty buffer distinguishable from an absent
/// one, which plain concatenation would collide.
#[inline]
pub fn hash_output_stream(hasher: &mut blake3::Hasher, stream: &[Vec<u8>]) {
    hasher.update(&(stream.len() as u64).to_le_bytes());
    for buf in stream {
        hasher.update(&(buf.len() as u64).to_le_bytes());
        hasher.update(buf);
    }
}

/// Lowercase hex of a 32-byte digest, the form every cert field carries.
pub fn hex32(bytes: &[u8; 32]) -> String {
    // Previous impl `let _ = write!(&mut
    // out, ...)` silently discarded the Result. String::write_str is
    // infallible today, but swallowing the Result would mask a
    // regression if it ever changed  -  violating the 'never swallow
    // errors' standard. Use hex::encode, which produces the same
    // output and propagates any internal failure as a panic with a
    // meaningful message rather than silently truncating the string.
    hex::encode(bytes)
}
