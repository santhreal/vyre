//! Tier 2.5 bracket-pair detector  -  bounded-stack scanner over a
//! token-kind buffer.
//!
//! The op uses a parallel per-token matcher when `max_depth >= n`, because
//! the depth cap cannot affect semantics in that case. Depth-capped shards keep
//! the bounded-stack single-lane fallback: overflow opens are deliberately
//! ignored, so that stateful behavior is not replaced by an approximation.
//!
//! Every parser dialect that needs matched-brace detection reaches this one
//! kernel: C, Rust, Go, and Python f-string interpolation.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Stable op id for the Tier 2.5 primitive.
pub const BRACKET_MATCH_OP_ID: &str = "vyre-libs::matching::bracket_match";
/// Workgroup size for the uncapped parallel parser-bracket path.
pub const BRACKET_MATCH_PARALLEL_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Token kind: not a brace.
pub const BRACKET_KIND_OTHER: u32 = 0;
/// Token kind: `{`
pub const BRACKET_KIND_OPEN: u32 = 1;
/// Token kind: `}`
pub const BRACKET_KIND_CLOSE: u32 = 2;
/// Unmatched sentinel written to `match_pairs`.
pub const BRACKET_MATCH_NONE: u32 = u32::MAX;

/// Dispatch grid for [`bracket_match`].
#[must_use]
pub const fn bracket_match_dispatch_grid(n: u32, max_depth: u32) -> [u32; 3] {
    if max_depth < n {
        return [1, 1, 1];
    }
    let blocks = n.div_ceil(BRACKET_MATCH_PARALLEL_WORKGROUP_SIZE[0]);
    if blocks == 0 {
        [1, 1, 1]
    } else {
        [blocks, 1, 1]
    }
}

/// Build a Program that matches brace tokens using a bounded stack.
///
/// `kinds[i]` is `BRACKET_KIND_OTHER`, `BRACKET_KIND_OPEN`, or `BRACKET_KIND_CLOSE`.
/// `stack` is scratch storage with `max_depth` entries.
/// Initializes unmatched entries to [`BRACKET_MATCH_NONE`] and writes bidirectional
/// links for every matched brace pair.
#[must_use]
pub fn bracket_match(
    kinds: &str,
    stack: &str,
    match_pairs: &str,
    n: u32,
    max_depth: u32,
) -> Program {
    if max_depth >= n {
        return bracket_match_parallel(kinds, stack, match_pairs, n, max_depth);
    }
    bracket_match_bounded_stack(kinds, stack, match_pairs, n, max_depth)
}

fn bracket_match_bounded_stack(
    kinds: &str,
    stack: &str,
    match_pairs: &str,
    n: u32,
    max_depth: u32,
) -> Program {
    let body = vec![wrap_anonymous_region(
        BRACKET_MATCH_OP_ID,
        vec![Node::if_then(
            Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
            vec![
                Node::let_bind("depth", Expr::u32(0)),
                Node::loop_for(
                    "i",
                    Expr::u32(0),
                    Expr::u32(n),
                    vec![
                        Node::let_bind("k", Expr::load(kinds, Expr::var("i"))),
                        Node::store(match_pairs, Expr::var("i"), Expr::u32(BRACKET_MATCH_NONE)),
                        Node::if_then_else(
                            Expr::eq(Expr::var("k"), Expr::u32(BRACKET_KIND_OPEN)),
                            vec![Node::if_then(
                                Expr::lt(Expr::var("depth"), Expr::u32(max_depth)),
                                vec![
                                    Node::store(stack, Expr::var("depth"), Expr::var("i")),
                                    Node::assign(
                                        "depth",
                                        Expr::add(Expr::var("depth"), Expr::u32(1)),
                                    ),
                                ],
                            )],
                            vec![Node::if_then(
                                Expr::eq(Expr::var("k"), Expr::u32(BRACKET_KIND_CLOSE)),
                                vec![Node::if_then(
                                    Expr::lt(Expr::u32(0), Expr::var("depth")),
                                    vec![
                                        Node::assign(
                                            "depth",
                                            Expr::sub(Expr::var("depth"), Expr::u32(1)),
                                        ),
                                        Node::let_bind(
                                            "open_idx",
                                            Expr::load(stack, Expr::var("depth")),
                                        ),
                                        Node::store(
                                            match_pairs,
                                            Expr::var("open_idx"),
                                            Expr::var("i"),
                                        ),
                                        Node::store(
                                            match_pairs,
                                            Expr::var("i"),
                                            Expr::var("open_idx"),
                                        ),
                                    ],
                                )],
                            )],
                        ),
                    ],
                ),
            ],
        )],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(kinds, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::read_write(stack, 1, DataType::U32).with_count(max_depth),
            BufferDecl::output(match_pairs, 2, DataType::U32).with_count(n),
        ],
        [1, 1, 1],
        body,
    )
}

fn bracket_match_parallel(
    kinds: &str,
    stack: &str,
    match_pairs: &str,
    n: u32,
    max_depth: u32,
) -> Program {
    let lane = Expr::InvocationId { axis: 0 };
    let lane_body = vec![
        Node::store(match_pairs, lane.clone(), Expr::u32(BRACKET_MATCH_NONE)),
        Node::let_bind("kind_self", Expr::load(kinds, lane.clone())),
        Node::if_then(
            Expr::eq(Expr::var("kind_self"), Expr::u32(BRACKET_KIND_OPEN)),
            vec![
                Node::let_bind("forward_depth", Expr::u32(1)),
                Node::let_bind("forward_active", Expr::u32(1)),
                Node::loop_for(
                    "j",
                    Expr::add(lane.clone(), Expr::u32(1)),
                    Expr::u32(n),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("forward_active"), Expr::u32(1)),
                        vec![
                            Node::let_bind("forward_kind", Expr::load(kinds, Expr::var("j"))),
                            Node::if_then(
                                Expr::eq(Expr::var("forward_kind"), Expr::u32(BRACKET_KIND_OPEN)),
                                vec![Node::assign(
                                    "forward_depth",
                                    Expr::add(Expr::var("forward_depth"), Expr::u32(1)),
                                )],
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("forward_kind"), Expr::u32(BRACKET_KIND_CLOSE)),
                                vec![
                                    Node::assign(
                                        "forward_depth",
                                        Expr::sub(Expr::var("forward_depth"), Expr::u32(1)),
                                    ),
                                    Node::if_then(
                                        Expr::eq(Expr::var("forward_depth"), Expr::u32(0)),
                                        vec![
                                            Node::store(match_pairs, lane.clone(), Expr::var("j")),
                                            Node::assign("forward_active", Expr::u32(0)),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    )],
                ),
            ],
        ),
        Node::if_then(
            Expr::eq(Expr::var("kind_self"), Expr::u32(BRACKET_KIND_CLOSE)),
            vec![
                Node::let_bind("backward_depth", Expr::u32(1)),
                Node::let_bind("backward_active", Expr::u32(1)),
                Node::loop_for(
                    "offset",
                    Expr::u32(1),
                    Expr::add(lane.clone(), Expr::u32(1)),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("backward_active"), Expr::u32(1)),
                        vec![
                            Node::let_bind(
                                "backward_j",
                                Expr::sub(lane.clone(), Expr::var("offset")),
                            ),
                            Node::let_bind(
                                "backward_kind",
                                Expr::load(kinds, Expr::var("backward_j")),
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("backward_kind"), Expr::u32(BRACKET_KIND_CLOSE)),
                                vec![Node::assign(
                                    "backward_depth",
                                    Expr::add(Expr::var("backward_depth"), Expr::u32(1)),
                                )],
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("backward_kind"), Expr::u32(BRACKET_KIND_OPEN)),
                                vec![
                                    Node::assign(
                                        "backward_depth",
                                        Expr::sub(Expr::var("backward_depth"), Expr::u32(1)),
                                    ),
                                    Node::if_then(
                                        Expr::eq(Expr::var("backward_depth"), Expr::u32(0)),
                                        vec![
                                            Node::store(
                                                match_pairs,
                                                lane.clone(),
                                                Expr::var("backward_j"),
                                            ),
                                            Node::assign("backward_active", Expr::u32(0)),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    )],
                ),
            ],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(kinds, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::read_write(stack, 1, DataType::U32).with_count(max_depth),
            BufferDecl::output(match_pairs, 2, DataType::U32).with_count(n),
        ],
        BRACKET_MATCH_PARALLEL_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            BRACKET_MATCH_OP_ID,
            vec![Node::if_then(Expr::lt(lane, Expr::u32(n)), lane_body)],
        )],
    )
}



inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        BRACKET_MATCH_OP_ID,
        || bracket_match("kinds", "stack", "match_pairs", 4, 4),
        Some(|| vec![vec![
            vyre_primitives::wire::pack_u32_slice(&[BRACKET_KIND_OPEN, BRACKET_KIND_OPEN, BRACKET_KIND_CLOSE, BRACKET_KIND_CLOSE]),
            vyre_primitives::wire::pack_u32_slice(&[0, 0, 0, 0]),
            vyre_primitives::wire::pack_u32_slice(&[BRACKET_MATCH_NONE, BRACKET_MATCH_NONE, BRACKET_MATCH_NONE, BRACKET_MATCH_NONE]),
        ]]),
        Some(|| vec![vec![
            vyre_primitives::wire::pack_u32_slice(&[0, 0, 0, 0]),
            vyre_primitives::wire::pack_u32_slice(&[3, 2, 1, 0]),
        ]]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ref_balanced_single_pair() {
        assert_eq!(
            bracket_match_cpu_ref(
                &[BRACKET_KIND_OPEN, BRACKET_KIND_OTHER, BRACKET_KIND_CLOSE],
                3
            ),
            vec![2, BRACKET_MATCH_NONE, 0]
        );
    }

    #[test]
    fn cpu_ref_nested_pairs() {
        assert_eq!(
            bracket_match_cpu_ref(
                &[
                    BRACKET_KIND_OPEN,
                    BRACKET_KIND_OPEN,
                    BRACKET_KIND_CLOSE,
                    BRACKET_KIND_CLOSE
                ],
                4
            ),
            vec![3, 2, 1, 0]
        );
    }

    #[test]
    fn cpu_ref_unbalanced_extra_open() {
        assert_eq!(
            bracket_match_cpu_ref(
                &[BRACKET_KIND_OPEN, BRACKET_KIND_OPEN, BRACKET_KIND_CLOSE],
                3
            ),
            vec![BRACKET_MATCH_NONE, 2, 1]
        );
    }

    #[test]
    fn cpu_ref_unbalanced_extra_close() {
        assert_eq!(
            bracket_match_cpu_ref(
                &[BRACKET_KIND_CLOSE, BRACKET_KIND_OPEN, BRACKET_KIND_CLOSE],
                3
            ),
            vec![BRACKET_MATCH_NONE, 2, 1]
        );
    }

    #[test]
    fn cpu_ref_depth_cap_truncates_extra_opens() {
        assert_eq!(
            bracket_match_cpu_ref(
                &[
                    BRACKET_KIND_OPEN,
                    BRACKET_KIND_OPEN,
                    BRACKET_KIND_OPEN,
                    BRACKET_KIND_CLOSE,
                    BRACKET_KIND_CLOSE,
                    BRACKET_KIND_CLOSE
                ],
                2,
            ),
            vec![4, 3, BRACKET_MATCH_NONE, 1, 0, BRACKET_MATCH_NONE]
        );
    }

    #[test]
    fn cpu_ref_into_reuses_output_and_stack_storage() {
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&[7, 8, 9, 10, 11]);
        let mut stack = Vec::with_capacity(8);
        stack.extend_from_slice(&[99, 100, 101]);
        let out_cap = out.capacity();
        let stack_cap = stack.capacity();

        bracket_match_cpu_ref_into(
            &[
                BRACKET_KIND_OPEN,
                BRACKET_KIND_OTHER,
                BRACKET_KIND_CLOSE,
                BRACKET_KIND_OPEN,
            ],
            4,
            &mut out,
            &mut stack,
        );

        assert_eq!(out, vec![2, BRACKET_MATCH_NONE, 0, BRACKET_MATCH_NONE]);
        assert_eq!(out.capacity(), out_cap);
        assert_eq!(stack.capacity(), stack_cap);
        assert_eq!(
            stack,
            vec![3],
            "Fix: bracket_match_cpu_ref_into must clear stale stack entries before each run and leave only currently-unmatched opens."
        );

        bracket_match_cpu_ref_into(&[BRACKET_KIND_OTHER], 4, &mut out, &mut stack);
        assert_eq!(out, vec![BRACKET_MATCH_NONE]);
        assert!(stack.is_empty());
        assert_eq!(out.capacity(), out_cap);
        assert_eq!(stack.capacity(), stack_cap);
    }

    #[test]
    fn builder_uses_parallel_program_when_depth_covers_tokens() {
        let program = bracket_match("kinds", "stack", "match_pairs", 513, 513);

        assert_eq!(
            program.workgroup_size(),
            BRACKET_MATCH_PARALLEL_WORKGROUP_SIZE
        );
        assert_eq!(bracket_match_dispatch_grid(0, 0), [1, 1, 1]);
        assert_eq!(bracket_match_dispatch_grid(1, 1), [1, 1, 1]);
        assert_eq!(bracket_match_dispatch_grid(256, 256), [1, 1, 1]);
        assert_eq!(bracket_match_dispatch_grid(257, 257), [2, 1, 1]);
        assert_eq!(bracket_match_dispatch_grid(513, 513), [3, 1, 1]);
    }

    #[test]
    fn builder_keeps_bounded_stack_when_depth_cap_can_change_pairs() {
        let program = bracket_match("kinds", "stack", "match_pairs", 513, 64);

        assert_eq!(program.workgroup_size(), [1, 1, 1]);
        assert_eq!(bracket_match_dispatch_grid(513, 64), [1, 1, 1]);
    }

    #[test]
    fn generated_uncapped_cases_match_stack_reference_contract() {
        let mut state = 0xB0A_CE5_u32;
        for case in 0..4096u32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let len = (state % 96) as usize;
            let mut kinds = Vec::with_capacity(len);
            for index in 0..len {
                state = state.rotate_left(5) ^ (index as u32).wrapping_mul(0x9E37_79B9);
                let kind = match state % 5 {
                    0 => BRACKET_KIND_OPEN,
                    1 => BRACKET_KIND_CLOSE,
                    _ => BRACKET_KIND_OTHER,
                };
                kinds.push(kind);
            }

            let expected = bracket_match_cpu_ref(&kinds, kinds.len() as u32);
            for (index, &pair) in expected.iter().enumerate() {
                if pair == BRACKET_MATCH_NONE {
                    continue;
                }
                assert!(
                    pair < kinds.len() as u32,
                    "generated uncapped case {case} pair at {index} must stay in range"
                );
                assert_eq!(
                    expected[pair as usize], index as u32,
                    "generated uncapped case {case} pair symmetry failed at {index}"
                );
            }
        }
    }
}
