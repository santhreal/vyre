use super::*;

/// Word count to allocate for a source haystack of `haystack_len` bytes.
///
/// A literal length sizes the buffer exactly; a runtime length is not known at
/// build time, so the declaration falls back to one word and the emitted IR
/// carries the real bound.
pub(super) fn haystack_word_count(haystack_len: &Expr, packed_haystack: bool) -> u32 {
    match haystack_len {
        Expr::LitU32(n) => source_haystack_words(*n, packed_haystack),
        _ => 1,
    }
}
