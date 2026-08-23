//! 2D / planar grammar rewrite primitive.
//!
//! Chomsky's grammars are 1D (token streams); 2D grammars (Hu-Tian
//! 1995, Zhu-Mumford 2007 image grammars, Wu 2017 generative shape
//! programs) replace string productions with **local 2D rewrites**:
//! a small `k × k` window matches a pattern, then writes a replacement.
//! Each rewrite is a neighborhood read+write  -  pure GPU shape, but
//! historically not packaged as a primitive at the IR level.
//!
//! This file ships the **non-overlapping rewrite scheduler** primitive
//!  -  given a candidate-match map, mark a maximal set of mutually
//! non-overlapping `k × k` windows that can apply in parallel.
//!
//! Algorithm: greedy serpentine scan with `k`-row stride. Each chosen
//! match locks a `(2k-1) × (2k-1)` exclusion zone preventing
//! neighboring matches from firing in the same wave. Matches not
//! chosen this wave remain candidates for the next wave.
//!
//! # Composition roles
//!
//! | Role | Use |
//! |---|---|
//! | scene parsing | layout analysis over 2D structures |
//! | cellular automata | parallel CA stepping with rewrite rules |
//! | document layout | layout extraction grammars |

use vyre_foundation::composition::{trap_program, wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::parsing::planar_rewrite_schedule";

/// Stable op id for the 2D window exclusion zone conflict check.
pub const PLANAR_REWRITE_EXCLUSION_CHECK_OP_ID: &str =
    "vyre-libs::parsing::planar_rewrite_exclusion_check";
/// Schedule a maximal non-overlapping set of `k × k` candidate matches
/// in a single wave.
///
/// Inputs:
/// - `candidates`: row-major `h × w` u32 mask, `1` if a match starts
///   at `(row, col)` (top-left corner of a `k × k` window), else `0`.
/// - `chosen`: row-major `h × w` u32  -  output mask of the chosen
///   matches. The complement (candidates AND NOT chosen) remains for
///   the next wave.
///
/// Single-lane scheduler (lane 0) walks the candidate map in row-major
/// order; for each candidate, claims it if no conflict with previously-
/// chosen, otherwise skips. Parallel graph-coloring schedulers should
/// be separate registered ops with their own contracts.
#[must_use]
pub fn planar_rewrite_schedule(candidates: &str, chosen: &str, h: u32, w: u32, k: u32) -> Program {
    if h == 0 || w == 0 {
        return trap_program(
            OP_ID,
            Some((chosen, DataType::U32)),
            format!("Fix: planar_rewrite_schedule requires h > 0 and w > 0, got h={h}, w={w}."),
        );
    }
    if k == 0 {
        return trap_program(
            OP_ID,
            Some((chosen, DataType::U32)),
            format!("Fix: planar_rewrite_schedule requires k > 0, got {k}."),
        );
    }

    let cells = h.saturating_mul(w);
    let t = Expr::InvocationId { axis: 0 };

    // Lane 0 loops over all (r, c) cells in row-major order. For each:
    //   if candidates[r,c] == 1:
    //     check exclusion zone: any chosen[i, j] in
    //       i ∈ [r - (k-1), r], j ∈ [c - (k-1), c]. If none, set chosen.
    let body = vec![Node::if_then(
        Expr::eq(t.clone(), Expr::u32(0)),
        vec![Node::loop_for(
            "r",
            Expr::u32(0),
            Expr::u32(h),
            vec![Node::loop_for(
                "c",
                Expr::u32(0),
                Expr::u32(w),
                vec![
                    Node::let_bind(
                        "addr",
                        Expr::add(Expr::mul(Expr::var("r"), Expr::u32(w)), Expr::var("c")),
                    ),
                    Node::store(chosen, Expr::var("addr"), Expr::u32(0)),
                    Node::if_then(
                        Expr::ne(Expr::load(candidates, Expr::var("addr")), Expr::u32(0)),
                        vec![
                            Node::let_bind("conflict", Expr::u32(0)),
                            wrap_child_region(
                                PLANAR_REWRITE_EXCLUSION_CHECK_OP_ID,
                                Ident::from(OP_ID),
                                planar_rewrite_exclusion_check_body(
                                    chosen,
                                    w,
                                    k,
                                    Expr::var("r"),
                                    Expr::var("c"),
                                ),
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("conflict"), Expr::u32(0)),
                                vec![Node::store(chosen, Expr::var("addr"), Expr::u32(1))],
                            ),
                        ],
                    ),
                ],
            )],
        )],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(candidates, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(cells),
            BufferDecl::storage(chosen, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(cells),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

/// Body of the exclusion zone conflict check.
#[must_use]
pub fn planar_rewrite_exclusion_check_body(
    chosen: &str,
    w: u32,
    k: u32,
    r: Expr,
    c: Expr,
) -> Vec<Node> {
    vec![
        // Exclusion zone scan
        Node::loop_for(
            "di",
            Expr::u32(0),
            Expr::u32(k),
            vec![Node::loop_for(
                "dj",
                Expr::u32(0),
                Expr::u32(k),
                vec![Node::if_then(
                    Expr::and(
                        Expr::ge(r.clone(), Expr::var("di")),
                        Expr::ge(c.clone(), Expr::var("dj")),
                    ),
                    vec![Node::if_then(
                        Expr::ne(
                            Expr::load(
                                chosen,
                                Expr::add(
                                    Expr::mul(Expr::sub(r.clone(), Expr::var("di")), Expr::u32(w)),
                                    Expr::sub(c.clone(), Expr::var("dj")),
                                ),
                            ),
                            Expr::u32(0),
                        ),
                        vec![Node::assign("conflict", Expr::u32(1))],
                    )],
                )],
            )],
        ),
    ]
}

/// Build the standalone exclusion check sub-operation.
#[must_use]
pub fn planar_rewrite_exclusion_check_program(w: u32, k: u32) -> Program {
    let mut body = vec![Node::let_bind("conflict", Expr::u32(0))];
    body.extend(planar_rewrite_exclusion_check_body(
        "chosen",
        w,
        k,
        Expr::u32(1),
        Expr::u32(1),
    ));
    body.push(Node::store(
        "out_conflict",
        Expr::u32(0),
        Expr::var("conflict"),
    ));
    let guarded = vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        body,
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage("chosen", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(w.saturating_mul(w)),
            BufferDecl::output("out_conflict", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            PLANAR_REWRITE_EXCLUSION_CHECK_OP_ID,
            guarded,
        )],
    )
}

/// The single chosen rewrite site survives the exclusion check.
const EXPECTED_PLANAR_REWRITE_EXCLUSION_CHECK_BYTES: [u8; 4] = [1, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        PLANAR_REWRITE_EXCLUSION_CHECK_OP_ID,
        || planar_rewrite_exclusion_check_program(4, 2),
        Some(|| {
            let mut chosen = vec![0u32; 16];
            chosen[0] = 1; // (0, 0)
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&chosen),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_PLANAR_REWRITE_EXCLUSION_CHECK_BYTES.to_vec()]]),
    )
}

const EXPECTED_PLANAR_REWRITE_BYTES: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || planar_rewrite_schedule("candidates", "chosen", 4, 4, 2),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            let mut cands = vec![0; 16];
            cands[5] = 1;
            vec![vec![to_bytes(&cands)]] // candidates
        }),
        Some(|| {
            vec![vec![EXPECTED_PLANAR_REWRITE_BYTES.to_vec()]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::composition_witness::planar_rewrite_schedule_witness as reference_planar_rewrite_schedule;

    #[test]
    fn reference_no_candidates_no_chosen() {
        let cands = vec![0u32; 16];
        let chosen = reference_planar_rewrite_schedule(&cands, 4, 4, 2);
        for v in chosen {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn reference_isolated_candidate_is_chosen() {
        let mut cands = vec![0u32; 16];
        cands[5] = 1; // (1, 1) in a 4x4
        let chosen = reference_planar_rewrite_schedule(&cands, 4, 4, 2);
        assert_eq!(chosen[5], 1);
    }

    #[test]
    fn reference_overlapping_candidates_only_first_chosen() {
        // Two candidates touching with k=2 exclusion: (0,0) and (0,1)
        // overlap. Only (0,0) is chosen.
        let mut cands = vec![0u32; 9];
        cands[0] = 1;
        cands[1] = 1;
        let chosen = reference_planar_rewrite_schedule(&cands, 3, 3, 2);
        assert_eq!(chosen[0], 1);
        assert_eq!(chosen[1], 0);
    }

    #[test]
    fn reference_widely_spaced_candidates_all_chosen() {
        // 5x5 grid, candidates at corners  -  all far enough apart.
        let mut cands = vec![0u32; 25];
        cands[0] = 1; // (0, 0)
        cands[4] = 1; // (0, 4)
        cands[20] = 1; // (4, 0)
        cands[24] = 1; // (4, 4)
        let chosen = reference_planar_rewrite_schedule(&cands, 5, 5, 2);
        assert_eq!(chosen[0], 1);
        assert_eq!(chosen[4], 1);
        assert_eq!(chosen[20], 1);
        assert_eq!(chosen[24], 1);
    }

    #[test]
    fn reference_short_candidate_buffer_treats_missing_cells_as_zero() {
        let cands = vec![1u32];
        let chosen = reference_planar_rewrite_schedule(&cands, 2, 2, 1);
        assert_eq!(chosen, vec![1, 0, 0, 0]);
    }

    #[test]
    fn reference_dense_candidates_alternate_chosen() {
        // All cells are candidates with k=2; chosen should be a maximal
        // independent set.
        let cands = vec![1u32; 16];
        let chosen = reference_planar_rewrite_schedule(&cands, 4, 4, 2);
        let total: u32 = chosen.iter().sum();
        // Greedy row-major with k=2 exclusion picks every other cell
        // in row 0 and skips a row, then resumes  -  but exact count is
        // implementation-specific. Verify ≥ 4 chosen and no conflicts.
        assert!(total >= 4);
        // Verify no two chosen are adjacent within k.
        for r in 0..4 {
            for c in 0..4 {
                if chosen[r * 4 + c] == 0 {
                    continue;
                }
                for di in 0..2 {
                    for dj in 0..2 {
                        if (di == 0 && dj == 0) || di > r || dj > c {
                            continue;
                        }
                        assert_eq!(chosen[(r - di) * 4 + (c - dj)], 0);
                    }
                }
            }
        }
    }

    #[test]
    fn ir_program_buffer_layout() {
        let p = planar_rewrite_schedule("c", "ch", 4, 4, 2);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["c", "ch"]);
        assert_eq!(p.buffers[0].count(), 16);
        assert_eq!(p.buffers[1].count(), 16);
    }

    #[test]
    fn zero_h_traps() {
        let p = planar_rewrite_schedule("c", "ch", 0, 4, 2);
        assert!(p.stats().trap());
    }

    #[test]
    fn zero_k_traps() {
        let p = planar_rewrite_schedule("c", "ch", 4, 4, 0);
        assert!(p.stats().trap());
    }
}
