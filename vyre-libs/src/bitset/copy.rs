//! `bitset_copy`  -  per-word bitwise copy (`target = source`).
//!
//! Replaces the `bitset_or_into` "OR-into-zero" idiom that external analyzer was
//! using as a structural copy. Explicit primitive: doc-clear,
//! semantics obvious, kernel one assignment per word. Downstream analyzer's
//! lower_expr's BindingRef arm (and any other "structural copy
//! between two same-shape bitset buffers") consumes this directly.

use vyre_foundation::ir::Program;

use super::binary_word::copy_word_program;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::bitset::copy";

/// Build a Program: `target[w] = source[w]` for `w` in `0..words`.
#[must_use]
pub fn bitset_copy(target: &str, source: &str, words: u32) -> Program {
    copy_word_program(OP_ID, target, source, words)
}

const EXPECTED_BITSET_COPY_OUTPUT_BYTES: [u8; 8] = [0xAD, 0xDE, 0x00, 0x00, 0xEF, 0xBE, 0x00, 0x00];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || bitset_copy("target", "source", 2),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 0]),
                to_bytes(&[0xDEAD, 0xBEEF]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_BITSET_COPY_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_laws(&["idempotent"])
}

#[cfg(test)]
mod tests {

    fn reference_copy(dst: &mut [u32], src: &[u32]) {
        let len = dst.len().min(src.len());
        dst[..len].copy_from_slice(&src[..len]);
    }

    #[test]
    fn reference_copies_word_for_word() {
        let src = vec![0x1234_5678, 0xDEAD_BEEF, 0x0000_FFFF, 0xFFFF_0000];
        let mut dst = vec![0u32; src.len()];
        reference_copy(&mut dst, &src);
        assert_eq!(dst, src);
    }

    #[test]
    fn reference_stops_at_shorter_source() {
        let src = vec![1u32, 2, 3];
        let mut dst = vec![10u32, 20, 30, 40, 50];
        reference_copy(&mut dst, &src);
        assert_eq!(dst, vec![1u32, 2, 3, 40, 50]);
    }
}
