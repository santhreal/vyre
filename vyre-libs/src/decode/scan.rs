//! Shared decode-to-DFA scan bodies.

use crate::builder::TableStateMachineComposer;
use vyre_foundation::ir::{Expr, Node};

/// Build a bounded Aho-Corasick scan body for fused decoders.
///
/// The scanner walks the decoded stream once and writes every accepting state
/// in order. This preserves the existing Aho-Corasick output contract without
/// replaying the prefix independently for every output position.
#[must_use]
pub(crate) fn linear_aho_scan_body(
    input: &str,
    transitions: &str,
    accept: &str,
    matches: &str,
    valid_len: Expr,
) -> Vec<Node> {
    TableStateMachineComposer::new(transitions).linear_scan_body(input, accept, matches, valid_len)
}

/// Build a single-invocation tiled Aho-Corasick body over a caller-supplied
/// byte expression.
///
/// The body keeps DFA state in registers and advances over bounded tiles,
/// alternating the decoded byte through two scalar slots. For decoders that can
/// expose `byte_at(index)` cheaply, this avoids the old decode-buffer readback
/// pass: decode for the next slot and scan for the current slot are fused in one
/// loop nest. The optional `store_decoded` hook preserves the public decoded
/// buffer contract for existing builders.
#[must_use]
pub(crate) fn tiled_decode_aho_scan_body<ByteAt, StoreDecoded>(
    transitions: &str,
    accept: &str,
    matches: &str,
    valid_len: Expr,
    tile_width: u32,
    byte_at: ByteAt,
    store_decoded: StoreDecoded,
) -> Vec<Node>
where
    ByteAt: FnMut(Expr) -> Expr,
    StoreDecoded: FnMut(Expr, Expr) -> Option<Node>,
{
    TableStateMachineComposer::new(transitions).tiled_decode_scan_body(
        accept,
        matches,
        valid_len,
        tile_width,
        byte_at,
        store_decoded,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_decode_scan_uses_tile_count_loop_not_byte_count_gate() {
        let body = tiled_decode_aho_scan_body(
            "transitions",
            "accept",
            "matches",
            Expr::u32(1024),
            8,
            |index| Expr::load("decoded", index),
            |_index, _byte| None,
        );
        let rendered = format!("{body:?}");
        assert!(
            rendered.contains("decode_scan_tile_index"),
            "fused decode-scan must loop over tile indices, not every byte offset"
        );
        assert!(
            rendered.contains("decode_scan_tile_base"),
            "fused decode-scan must derive a tile base from the tile index"
        );
    }
}
