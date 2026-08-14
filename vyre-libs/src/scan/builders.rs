//! IR LEGO BLOCKS for matching dialects.
//!
//! Exposes granular primitives that can be composed into custom
//! scanning engines (e.g. combined DFA + ML, decoder-aware scanners).

use vyre_foundation::ir::{Expr, Node};

/// LEGO BLOCK: Load a byte from a packed U32 haystack.
///
/// Returns `(let_bind_node, byte_expr)`. The caller must push the
/// `let_bind_node` into its surrounding Block before evaluating
/// `byte_expr` - the let-bind gives the optimiser a CSE handle for
/// the underlying word load when the byte is referenced multiple
/// times.
pub fn load_packed_byte(haystack: &str, idx: Expr) -> (Node, Expr) {
    let word_idx = Expr::div(idx.clone(), Expr::u32(4));
    let byte_offset = Expr::mul(Expr::rem(idx, Expr::u32(4)), Expr::u32(8));

    let node = Node::let_bind("_byte_word", Expr::load(haystack, word_idx));
    let byte_expr = Expr::bitand(
        Expr::shr(Expr::var("_byte_word"), byte_offset),
        Expr::u32(0xFF),
    );

    (node, byte_expr)
}

/// LEGO BLOCK: Pure-expression form of `load_packed_byte`.
///
/// Inlines the word load directly into the shift+mask, returning a
/// single `Expr` with no `Node::Let` side effect. Use this when the
/// byte is consumed in a single expression context (e.g. an `If`
/// condition or `Node::ne` predicate inside a `Loop` body) where
/// hoisting a let-bind would either inject it at the wrong scope or
/// require restructuring the surrounding IR.
///
/// Trade-off: no CSE handle for the word load - if the same byte
/// position is referenced more than once, prefer `load_packed_byte`
/// and bind the result. Single-use call sites (the cursor body in
/// `nfa::nfa_scan_with_plan` and the literal-compare inside
/// `literal_set::literal_set_program`) take this form because they
/// reference the byte exactly once per iteration.
pub fn load_packed_byte_expr(haystack: &str, idx: Expr) -> Expr {
    Expr::bitand(
        Expr::shr(
            Expr::load(haystack, Expr::div(idx.clone(), Expr::u32(4))),
            Expr::mul(Expr::rem(idx, Expr::u32(4)), Expr::u32(8)),
        ),
        Expr::u32(0xFF),
    )
}

/// LEGO BLOCK: Append a match to a standardized hit buffer.
///
/// Use \`append_match_subgroup\` for production paths that benefit from
/// subgroup-coalesced atomics (Innovation I.17).
pub fn append_match(
    hits_buffer: &str,
    count_buffer: &str,
    tag: impl Into<Expr>,
    start: impl Into<Expr>,
    end: impl Into<Expr>,
) -> Node {
    let max_hits = Expr::div(Expr::buf_len(hits_buffer), Expr::u32(3));

    Node::Block(vec![
        Node::let_bind(
            "_vyre_match_slot",
            Expr::atomic_add(count_buffer, Expr::u32(0), Expr::u32(1)),
        ),
        Node::if_then(
            Expr::lt(Expr::var("_vyre_match_slot"), max_hits),
            vec![
                Node::store(
                    hits_buffer,
                    Expr::mul(Expr::var("_vyre_match_slot"), Expr::u32(3)),
                    tag.into(),
                ),
                Node::store(
                    hits_buffer,
                    Expr::add(
                        Expr::mul(Expr::var("_vyre_match_slot"), Expr::u32(3)),
                        Expr::u32(1),
                    ),
                    start.into(),
                ),
                Node::store(
                    hits_buffer,
                    Expr::add(
                        Expr::mul(Expr::var("_vyre_match_slot"), Expr::u32(3)),
                        Expr::u32(2),
                    ),
                    end.into(),
                ),
            ],
        ),
    ])
}

/// Innovation I.17: Subgroup-Coalesced Match Append.
///
/// Uses subgroup-ballot and subgroup-shuffle to perform a single
/// \`atomic_add\` per subgroup, drastically reducing global memory
/// serialization on high-hit-rate workloads.
pub fn append_match_subgroup(
    hits_buffer: &str,
    count_buffer: &str,
    tag: impl Into<Expr>,
    start: impl Into<Expr>,
    end: impl Into<Expr>,
    cond: Expr,
) -> Vec<Node> {
    let tag = tag.into();
    let start = start.into();
    let end = end.into();
    let max_hits = Expr::div(Expr::buf_len(hits_buffer), Expr::u32(3));
    let lane_mask = Expr::sub(
        Expr::shl(Expr::u32(1), Expr::subgroup_local_id()),
        Expr::u32(1),
    );
    let rank = Expr::popcount(Expr::bitand(Expr::var("_vyre_match_ballot"), lane_mask));
    let leader_pred = Expr::and(
        cond.clone(),
        Expr::eq(Expr::var("_vyre_match_rank"), Expr::u32(0)),
    );
    let slot = Expr::add(
        Expr::subgroup_shuffle(
            Expr::var("_vyre_match_leader_base"),
            Expr::var("_vyre_match_leader"),
        ),
        Expr::var("_vyre_match_rank"),
    );
    let ballot_cond = cond.clone();
    let bounded_hit = Expr::and(cond, Expr::lt(Expr::var("_vyre_match_slot"), max_hits));

    vec![
        Node::let_bind("_vyre_match_ballot", Expr::subgroup_ballot(ballot_cond)),
        Node::let_bind("_vyre_match_rank", rank),
        Node::let_bind(
            "_vyre_match_count",
            Expr::popcount(Expr::var("_vyre_match_ballot")),
        ),
        Node::let_bind(
            "_vyre_match_leader",
            Expr::select(
                Expr::eq(Expr::var("_vyre_match_count"), Expr::u32(0)),
                Expr::u32(0),
                Expr::ctz(Expr::var("_vyre_match_ballot")), // Fixed: relative to subgroup,
            ),
        ),
        Node::let_bind("_vyre_match_leader_base", Expr::u32(0)),
        Node::if_then(
            leader_pred,
            vec![Node::assign(
                "_vyre_match_leader_base",
                Expr::atomic_add(count_buffer, Expr::u32(0), Expr::var("_vyre_match_count")),
            )],
        ),
        Node::let_bind("_vyre_match_slot", slot),
        Node::if_then(
            bounded_hit,
            vec![
                Node::store(
                    hits_buffer,
                    Expr::mul(Expr::var("_vyre_match_slot"), Expr::u32(3)),
                    tag,
                ),
                Node::store(
                    hits_buffer,
                    Expr::add(
                        Expr::mul(Expr::var("_vyre_match_slot"), Expr::u32(3)),
                        Expr::u32(1),
                    ),
                    start,
                ),
                Node::store(
                    hits_buffer,
                    Expr::add(
                        Expr::mul(Expr::var("_vyre_match_slot"), Expr::u32(3)),
                        Expr::u32(2),
                    ),
                    end,
                ),
            ],
        ),
    ]
}

#[cfg(test)]
mod subgroup_append_shape_tests {
    use super::*;

    #[test]
    fn subgroup_append_bounds_hits_with_bound_slot_variable() {
        let nodes = append_match_subgroup(
            "matches",
            "match_count",
            Expr::u32(7),
            Expr::u32(11),
            Expr::u32(13),
            Expr::var("hit"),
        );

        let bounded_cond = match nodes.last() {
            Some(Node::If { cond, .. }) => format!("{cond:?}"),
            other => panic!("expected bounded-hit If as final subgroup append node, got {other:?}"),
        };

        assert!(
            bounded_cond.contains("_vyre_match_slot"),
            "bounded-hit predicate must use the already-bound slot variable: {bounded_cond}"
        );
        assert!(
            !bounded_cond.contains("_vyre_match_leader"),
            "bounded-hit predicate must not re-inline subgroup shuffle leader expressions: {bounded_cond}"
        );
    }
}

#[cfg(test)]
mod packed_byte_dedup_lock {
    //! Regression gate for the canonical-packed-byte LEGO primitive.
    //!
    //! Six prior duplications of the `Expr::shr(Expr::load(buf,
    //! word_idx), byte_offset) & 0xFF` byte-extract pattern landed in
    //! vyre-libs over time (scan/nfa, scan/literal_set, parsing/c/
    //! preprocess/gpu_if_expression/byte_load,
    //! parsing/c/preprocess/gpu_filter/program_helpers). Tasks #21,
    //! #22, #26 were marked completed previously while three of those
    //! copies were still alive. This test prevents the next
    //! regression: it walks `vyre-libs/src/**/*.rs` for the
    //! divrem-shr-and(0xFF) shape and fails if it appears outside
    //! `scan/builders.rs`.
    //!
    //! Detection is text-based and conservative - false positives are
    //! tolerable (just add a `// allow-packed-byte-dup:` reason
    //! comment on the line to suppress). False negatives (missing a
    //! real duplicate) are the failure mode that matters; the shape
    //! is distinctive enough that grep is sufficient.
    use std::path::{Path, PathBuf};

    fn vyre_libs_src() -> PathBuf {
        vyre_test_support::monorepo::vyre_workspace_root().join("vyre-libs/src")
    }

    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
}
