//! Domain-neutral byte-range ordering predicates.
//!
//! These helpers build IR for relations between tagged byte-range streams.
//! They use the scanner output contract's `counts`, `offsets`, and `lengths`
//! buffer names.

use crate::builder::trip_count::clamped_by_extents;
use vyre_foundation::composition::bounded_index_when;
use vyre_foundation::ir::{Expr, Node};

/// Maximum number of cached positions per tagged range. Matches the
/// source-query dialect scanner-side cap.
pub const MAX_CACHED_POSITIONS: u32 = 256;

/// Maximum logical "depth" used by the same scanner-side convention.
pub const MAX_DEPTH: u32 = 12;

/// Helper to read the element of a packed 2D array laid out as
/// `buffer[id * MAX_CACHED_POSITIONS + index]`.
fn packed_load(buffer: &str, id: Expr, index: Expr) -> Expr {
    Expr::load(
        buffer,
        Expr::add(Expr::mul(id, Expr::u32(MAX_CACHED_POSITIONS)), index),
    )
}

/// Generate a loop block deciding whether any range tagged `left_id`
/// ends at or before some range tagged `right_id` begins.
///
/// Returns `(Vec<Node>, Expr)` where the expression is the boolean
/// result bound to an internal `let` variable named `<res_name>_found`.
///
/// The emitted block assumes the enclosing IR provides three storage
/// buffers: `counts[tag]` (how many ranges carry that tag), and
/// `offsets[tag * MAX_CACHED_POSITIONS + i]` + `lengths[tag * MAX_CACHED_POSITIONS + i]`.
///
/// AUDIT: PHASE5_ASTWALK match_order quadratic  -  replaced nested O(N²)
/// loops with a sweep-line O(N) pass.  The hit positions for each tag
/// are guaranteed to be sorted by ascending offset by the host scanner
/// (see `downstream analyzer::scan::collector::select_hits_for_dispatch`).  Because
/// the inputs are sorted, the predicate `∃ a ∈ A, ∃ b ∈ B : a_end <=
/// b_start` is equivalent to `min_a_end <= max_b_start`.  We compute
/// `min_a_end` with a single linear scan over A and read `max_b_start`
/// directly from the last element of B (the largest offset).  Inner
/// work is O(N) with N ≤ MAX_CACHED_POSITIONS = 256, versus the prior
/// 65 536 iterations per workgroup lane.
#[must_use]
pub fn match_order(left_id: Expr, right_id: Expr, res_name: &str) -> (Vec<Node>, Expr) {
    let mut block = Vec::new();

    let limit_a = Expr::load("counts", left_id.clone());
    let clamped_limit_a = clamped_by_extents(
        Expr::min(limit_a, Expr::u32(MAX_CACHED_POSITIONS)),
        "offsets",
        ["lengths"],
    );

    let limit_b = Expr::load("counts", right_id.clone());
    let clamped_limit_b = clamped_by_extents(
        Expr::min(limit_b, Expr::u32(MAX_CACHED_POSITIONS)),
        "offsets",
        ["lengths"],
    );

    block.push(Node::let_bind(format!("{res_name}_len_a"), clamped_limit_a));
    block.push(Node::let_bind(format!("{res_name}_len_b"), clamped_limit_b));

    // Compute min_a_end across all valid A positions.
    block.push(Node::let_bind(
        format!("{res_name}_min_a_end"),
        Expr::u32(u32::MAX),
    ));

    let scan_a_loop = Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::var(format!("{res_name}_len_a").as_str()),
        vec![
            Node::let_bind(
                "a_start",
                packed_load("offsets", left_id.clone(), Expr::var("i")),
            ),
            Node::let_bind(
                "a_len",
                packed_load("lengths", left_id.clone(), Expr::var("i")),
            ),
            Node::let_bind("a_end", Expr::add(Expr::var("a_start"), Expr::var("a_len"))),
            Node::assign(
                format!("{res_name}_min_a_end"),
                Expr::select(
                    Expr::lt(
                        Expr::var("a_end"),
                        Expr::var(format!("{res_name}_min_a_end")),
                    ),
                    Expr::var("a_end"),
                    Expr::var(format!("{res_name}_min_a_end")),
                ),
            ),
        ],
    );
    block.push(scan_a_loop);

    // B is sorted by offset, so max_b_start is the last valid element. A select
    // evaluates both arms, so an empty B still reads: its length minus one wraps to
    // the largest u32 and the read lands past the packed slab. The index is folded
    // to element zero, which the same select discards.
    let last_b = bounded_index_when(
        Expr::gt(
            Expr::var(format!("{res_name}_len_b").as_str()),
            Expr::u32(0),
        ),
        Expr::sub(
            Expr::var(format!("{res_name}_len_b").as_str()),
            Expr::u32(1),
        ),
    );
    let max_b_start = Expr::select(
        Expr::gt(
            Expr::var(format!("{res_name}_len_b").as_str()),
            Expr::u32(0),
        ),
        packed_load("offsets", right_id.clone(), last_b),
        Expr::u32(0),
    );
    block.push(Node::let_bind(
        format!("{res_name}_max_b_start"),
        max_b_start,
    ));

    // Found iff both sides are non-empty and the earliest-ending A
    // ends at or before the latest-starting B begins.
    let both_non_empty = Expr::and(
        Expr::gt(
            Expr::var(format!("{res_name}_len_a").as_str()),
            Expr::u32(0),
        ),
        Expr::gt(
            Expr::var(format!("{res_name}_len_b").as_str()),
            Expr::u32(0),
        ),
    );
    block.push(Node::let_bind(
        format!("{res_name}_found"),
        Expr::select(
            both_non_empty,
            Expr::select(
                Expr::le(
                    Expr::var(format!("{res_name}_min_a_end").as_str()),
                    Expr::var(format!("{res_name}_max_b_start").as_str()),
                ),
                Expr::u32(1),
                Expr::u32(0),
            ),
            Expr::u32(0),
        ),
    ));

    (block, Expr::var(format!("{res_name}_found")))
}
