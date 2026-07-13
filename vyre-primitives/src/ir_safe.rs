//! Shared, safety-carrying IR-expression builders.
//!
//! These encode the two bounds-safety idioms the GPU↔CPU parity audit turned on,
//! so the SAFE form has exactly ONE home and cannot drift. Both guard the same
//! hazard: a data-derived index that, unmasked/unclamped, reads past a buffer end 
//! a raw OOB read that the reference interpreter silently zero-fills but a real GPU
//! (CUDA does no bounds-checking) faults or corrupts memory on. `base64::clamp_lookup`
//! once forgot the table mask (a real OOB bug fixed this cycle), which a canonical
//! helper makes impossible to reintroduce.
#![cfg(feature = "vyre-foundation")]

use vyre_foundation::ir::{DataType, Expr};

/// Look up `byte` in a fixed **256-entry** table, masking the index with `& 0xFF`
/// so a value `> 255` can never read past the table.
///
/// This is the canonical form for every `load(table, byte)` where `table` has
/// exactly 256 entries (char-class maps, CRC/decode tables, histogram bins). Call
/// it instead of hand-writing `Expr::load(table, Expr::bitand(byte, 0xFF))`: the
/// mask is then structurally guaranteed at every such lookup.
#[must_use]
pub fn byte_table_lookup(table: &str, byte: Expr) -> Expr {
    Expr::load(table, Expr::bitand(byte, Expr::u32(0xFF)))
}

/// Widen a byte-carrying scalar load to `u32` and look it up in a 256-entry table
/// with the `& 0xFF` mask, the common "load a source byte, classify it" shape
/// (`char_class`, `byte_histogram`) in one call.
#[must_use]
pub fn source_byte_table_lookup(table: &str, source: &str, index: Expr) -> Expr {
    byte_table_lookup(table, Expr::cast(DataType::U32, Expr::load(source, index)))
}

/// Load `buf[idx]` with the index CLAMPED to the buffer's runtime length, so an
/// out-of-range `idx` reads the last valid element instead of past the end.
///
/// The clamp guarantees the LOAD itself is in bounds on EVERY backend (not just
/// those whose driver happens to clamp OOB reads). Callers that must return a
/// sentinel for the out-of-range case wrap this in an `Expr::select` on the same
/// bound, the clamp does not change the returned value for in-range indices; it
/// only removes the illegal access for out-of-range ones. This is the pattern in
/// `graph::path_reconstruct` and `parsing::line_splice_classify`.
#[must_use]
pub fn clamped_load(buf: &str, idx: Expr) -> Expr {
    clamped_load_to(buf, idx, Expr::buf_len(buf))
}

/// Like [`clamped_load`] but clamps `idx` to a CALLER-SUPPLIED `bound` instead of
/// the buffer's full length, for the common case where the logical valid range is
/// tighter than the physical buffer (e.g. `min(buf_len, logical_count)`). An index
/// `>= bound` reads element `bound - 1`; the caller's outer guard supplies the
/// out-of-range VALUE. This is the shared clamp behind `line_splice_classify`'s
/// U8 and U32 neighbor reads.
#[must_use]
pub fn clamped_load_to(buf: &str, idx: Expr, bound: Expr) -> Expr {
    let safe_idx = Expr::select(
        Expr::lt(idx.clone(), bound.clone()),
        idx,
        Expr::saturating_sub(bound, Expr::u32(1)),
    );
    Expr::load(buf, safe_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_lookup_masks_the_index() {
        // The lookup index must be `byte & 0xFF`, never the raw byte, a >255 value
        // folds into the 256-entry table instead of reading past it.
        let expr = byte_table_lookup("table", Expr::u32(0x0141));
        let rendered = format!("{expr:?}");
        assert!(
            rendered.contains("255") || rendered.contains("0xFF") || rendered.contains("Ff"),
            "byte_table_lookup must AND the index with 0xFF; got {rendered}"
        );
    }

    #[test]
    fn clamped_load_bounds_the_index_against_buf_len() {
        // The load index must be a select against BufLen, so it is always in range.
        let expr = clamped_load("buf", Expr::u32(9_999));
        let rendered = format!("{expr:?}");
        assert!(
            rendered.contains("BufLen") || rendered.contains("buf_len"),
            "clamped_load must clamp against the buffer length; got {rendered}"
        );
    }

    #[test]
    fn clamped_load_to_clamps_against_the_caller_bound_not_buf_len() {
        // With an explicit bound, the clamp must select on THAT bound (not BufLen) so
        // a logical range tighter than the physical buffer is honored, the shared
        // clamp behind line_splice_classify's `min(buf_len, byte_count)` neighbor read.
        let expr = clamped_load_to("buf", Expr::u32(9_999), Expr::u32(7));
        let rendered = format!("{expr:?}");
        assert!(
            !rendered.contains("BufLen"),
            "clamped_load_to must clamp against the caller bound, not BufLen; got {rendered}"
        );
        assert!(
            rendered.contains('7') || rendered.contains("Select"),
            "clamped_load_to must select the safe index against the given bound; got {rendered}"
        );
    }
}
