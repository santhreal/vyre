//! Every start offset the substring search admits indexes the match bitmap.
//!
//! WHY: the bitmap declares one slot per haystack byte, while the set of start
//! offsets a needle can begin at is `haystack_len - needle_len + 1`. The two
//! agree for every needle of at least one byte and diverge for the empty
//! needle, which also matches at `haystack_len`, one slot past the end. The
//! guard admits an offset and the body stores unconditionally, so a divergence
//! is an out-of-bounds store rather than a wrong bit. This sweeps the whole
//! `(haystack_len, needle_len)` rectangle across the boundary instead of the
//! one pair that first showed the defect.
//!
//! What this does not catch: the reference interpreter rejects the store, so
//! the case proves the composition never asks for the slot. A backend that
//! silently absorbs an out-of-range store is a backend contract, covered by
//! the conformance dispatch, not here.

#![cfg(feature = "pattern-substring")]

use vyre_libs_pattern::pattern::substring_search;
use vyre_primitives::wire::pack_u32_slice;

/// Bytes of `text`, one byte per u32 word, in the packing the composition reads.
fn packed(text: &[u8]) -> Vec<u8> {
    pack_u32_slice(&text.iter().copied().map(u32::from).collect::<Vec<_>>())
}

/// The match bitmap the composition claims to compute, evaluated on the host.
fn expected_bitmap(haystack: &[u8], needle: &[u8]) -> Vec<u32> {
    (0..haystack.len())
        .map(|i| u32::from(haystack[i..].starts_with(needle)))
        .collect()
}

fn decode_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

#[test]
fn every_admitted_offset_indexes_the_match_bitmap() {
    const ALPHABET: &[u8] = b"abab";

    for haystack_len in 0u32..=6 {
        for needle_len in 0u32..=haystack_len + 2 {
            let haystack: Vec<u8> = (0..haystack_len as usize)
                .map(|i| ALPHABET[i % ALPHABET.len()])
                .collect();
            let needle: Vec<u8> = (0..needle_len as usize)
                .map(|i| ALPHABET[i % ALPHABET.len()])
                .collect();

            let program =
                substring_search("haystack", "needle", "matches", haystack_len, needle_len);
            let buffers = vec![
                packed(&haystack),
                packed(&needle),
                vec![0u8; haystack.len() * 4],
            ];
            let inputs = vyre_reference::reference_inputs(&program, buffers);

            let outputs = vyre_reference::ReferenceRequest::standard(&program, &inputs)
                .outputs()
                .unwrap_or_else(|error| {
                    panic!(
                        "Fix: substring_search must store inside the match bitmap for a \
                         {haystack_len}-byte haystack and a {needle_len}-byte needle. Bound the \
                         start offset by the bitmap extent before the store: {error:?}"
                    )
                });

            assert_eq!(
                outputs.len(),
                1,
                "Fix: substring_search declares one read-write buffer"
            );
            let got = decode_words(&outputs[0].to_bytes());
            assert_eq!(
                got,
                expected_bitmap(&haystack, &needle),
                "Fix: substring_search marked the wrong offsets for a {haystack_len}-byte \
                 haystack and a {needle_len}-byte needle"
            );
        }
    }
}
