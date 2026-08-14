//! ROADMAP A27  -  fission a `Node::Loop` whose body partitions cleanly
//! into two consecutive halves that touch disjoint buffer sets.
//!
//! Op id: `vyre-foundation::optimizer::passes::loop_fission`.
//! Soundness: `Exact` under the conservative buffer-disjoint partition
//! check. If `body = a; b` and `buffers_touched(a) ∩ buffers_touched(b)
//! == ∅` and `b` does not depend on any name bound in `a`, the loop
//! `for i in from..to { a; b }` is observably equivalent to
//! `for i in from..to { a }; for i in from..to { b }` because no
//! cross-iteration or cross-arm dependency exists. Cost direction:
//! `node_count` rises by one Loop wrapper, but the per-arm body
//! becomes vectorizable / tilable / strip-minable in isolation; this
//! is an enabler pass for A29 strip-mine and the SIMD-fan rewrites.
//! Preserves: every analysis. Invalidates: nothing (the loops cover
//! the same iteration space and emit the same observable side
//! effects in the same order).
//!
//! ## Pattern
//!
//! ```text
//! Loop(i, a, b, [s_1, ..., s_k, s_{k+1}, ..., s_n])
//!   where buffers_touched(s_1..s_k) ∩ buffers_touched(s_{k+1}..s_n) == ∅
//!   AND no name bound in s_1..s_k is read by s_{k+1}..s_n
//!   AND no Barrier / IndirectDispatch / AsyncWait sits at the split point
//! → Loop(i, a, b, [s_1, ..., s_k]); Loop(j, a, b, [s_{k+1}, ..., s_n])
//! ```
//!
//! ## Conservatism
//!
//! - `from`/`to` must be `Expr::LitU32` with the same values in both
//!   resulting loops; we copy the original bounds verbatim and freshen
//!   the second loop's induction variable.
//! - The split point is the first index where the prefix and suffix
//!   touch disjoint buffer sets and no name-flow crosses the boundary.
//!   This is a single split  -  multi-way fission falls out by repeated
//!   application of the pass.
//! - Barrier-bearing loops are rejected: a Barrier inside the body
//!   sequences memory across iterations, and splitting it across two
//!   loops changes the observed ordering at the device.
//! - IndirectDispatch / AsyncWait carry queue-level effects whose
//!   relative ordering with the surrounding work cannot be split, so
//!   any presence in the body blocks fission.

use super::{collect_touched_buffers, legality};
use crate::ir::{Expr, Ident, Node, Program};
use crate::optimizer::passes::driver;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};
use rustc_hash::FxHashSet;

/// Fission a `Node::Loop` with a buffer-disjoint partitionable body
/// into two sibling loops covering the same iteration space.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "loop_fission",
    requires = [],
    invalidates = []
)]
/// ABI-preserving loop fission pass for buffer-disjoint partitionable loop bodies.
pub struct LoopFission;

impl LoopFission {
    /// Skip programs without a fissionable Loop.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        driver::analyze_candidates(
            program,
            &[crate::ir::stats::NODE_KIND_LOOP],
            &mut is_fissionable_loop,
        )
    }

    /// Walk the entry tree and split fissionable Loops.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        driver::rewrite_entry_bodies(program, &mut fission_in_body)
    }
}

/// Split every fissionable `Loop` in `body` into a prefix and a suffix loop.
///
/// This is a body rule rather than a node rule because the two loops replace
/// one node as siblings, and a `Block` around them would be new IR that no
/// later pass removes.
fn fission_in_body(body: &[Node]) -> Option<Vec<Node>> {
    let mut out: Vec<Node> = Vec::with_capacity(body.len());
    let mut split_any = false;
    for node in body {
        let Node::Loop {
            var,
            from,
            to,
            body: loop_body,
        } = node
        else {
            out.push(node.clone());
            continue;
        };
        let bounds_ok = matches!(from, Expr::LitU32(_)) && matches!(to, Expr::LitU32(_));
        if !bounds_ok {
            out.push(node.clone());
            continue;
        }
        let Some((prefix, suffix)) = try_partition(loop_body, var) else {
            out.push(node.clone());
            continue;
        };
        split_any = true;
        let fresh_var = freshen(var, loop_body);
        let renamed_suffix: Vec<Node> = suffix
            .into_iter()
            .map(|n| legality::rename_var_in_node(n, var, &fresh_var))
            .collect();
        out.push(Node::Loop {
            var: var.clone(),
            from: from.clone(),
            to: to.clone(),
            body: prefix,
        });
        out.push(Node::Loop {
            var: fresh_var,
            from: from.clone(),
            to: to.clone(),
            body: renamed_suffix,
        });
    }
    split_any.then_some(out)
}

/// True iff `nodes` contains an op whose ordering or cross-thread
/// synchronization semantics fission's intra-thread buffer-disjointness
/// analysis cannot safely reorder: a Barrier, `IndirectDispatch`, async op,
/// cross-thread collective, trap, or opaque node. We only check direct
/// siblings because the partition itself only splits direct siblings.
fn has_barrier_like(nodes: &[Node]) -> bool {
    nodes.iter().any(|n| {
        matches!(
            n,
            Node::Barrier { .. }
                | Node::IndirectDispatch { .. }
                | Node::AsyncWait { .. }
                | Node::AsyncLoad { .. }
                | Node::AsyncStore { .. }
                | Node::Trap { .. }
                | Node::Resume { .. }
                | Node::Opaque(_)
                // Cross-thread collectives synchronize a CommGroup; splitting a
                // loop reorders them relative to surrounding code, which the
                // intra-thread buffer-disjointness check cannot prove safe.
                // loop_unroll / loop_strip_mine already block on these.
                | Node::AllReduce { .. }
                | Node::AllGather { .. }
                | Node::ReduceScatter { .. }
                | Node::Broadcast { .. }
        )
    })
}

/// Partition the body into the largest prefix + non-empty suffix
/// whose touched-buffer sets are disjoint AND whose name-flow does
/// not cross the split. Returns `(prefix, suffix)` if such a split
/// exists with both halves non-empty; `None` otherwise.
fn try_partition(body: &[Node], loop_var: &Ident) -> Option<(Vec<Node>, Vec<Node>)> {
    if body.len() < 2 {
        return None;
    }
    // A barrier-like node, or an effect the touched-buffer summary cannot
    // see, defeats the disjoint-buffer proof fission depends on. Either one
    // must keep the body whole.
    if has_barrier_like(body) || legality::unsummarisable_effect(body) {
        return None;
    }
    for split in 1..body.len() {
        let prefix = &body[..split];
        let suffix = &body[split..];
        if super::buffers_disjoint_with(prefix, suffix, collect_touched_buffers)
            && !legality::bindings_flow_across(prefix, suffix, loop_var)
        {
            return Some((prefix.to_vec(), suffix.to_vec()));
        }
    }
    None
}

/// Pick a name not used as a Let/Assign/Loop var anywhere in `body`.
fn freshen(base: &Ident, body: &[Node]) -> Ident {
    let mut used: FxHashSet<Ident> = FxHashSet::default();
    legality::collect_bound_names(body, &mut used);
    used.insert(base.clone());
    let mut counter = 0u32;
    loop {
        let candidate = Ident::from(format!("{}__fis_{counter}", base.as_str()));
        if !used.contains(&candidate) {
            return candidate;
        }
        counter += 1;
    }
}

fn is_fissionable_loop(node: &Node) -> bool {
    if let Node::Loop {
        var,
        body,
        from,
        to,
    } = node
    {
        if !matches!(from, Expr::LitU32(_)) || !matches!(to, Expr::LitU32(_)) {
            return false;
        }
        try_partition(body, var).is_some()
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, ExprNode, Node};

    vyre_test_support::test_expr_extension! {
        OpaqueReader,
        kind: "test.opaque_buffer_reader",
        identity: "opaque_reader",
        result_type: Some(DataType::U32),
        cse_safe: false,
        fingerprint: 11,
    }

    fn buf(name: &str) -> BufferDecl {
        BufferDecl::storage(name, 0, BufferAccess::ReadWrite, DataType::U32).with_count(8)
    }

    fn program_with_entry(buffers: Vec<BufferDecl>, entry: Vec<Node>) -> Program {
        Program::wrapped(buffers, [1, 1, 1], entry)
    }

    fn count_loops(nodes: &[Node]) -> usize {
        crate::test_ir_inspect::count_nodes(nodes, |node| matches!(node, Node::Loop { .. }))
    }

    /// Positive: a loop body that writes two distinct buffers fissions
    /// into two sibling loops with the same iteration space.
    #[test]
    fn fissions_two_disjoint_stores() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(8),
            body: vec![
                Node::store("a", Expr::var("i"), Expr::u32(1)),
                Node::store("b", Expr::var("i"), Expr::u32(2)),
            ],
        }];
        let program = program_with_entry(vec![buf("a"), buf("b")], entry);
        let result = LoopFission::transform(program);
        assert!(result.changed, "two-buffer-disjoint Loop must fission");
        assert_eq!(
            count_loops(result.program.entry()),
            2,
            "exactly two sibling loops after fission"
        );
    }

    /// Negative: an opaque expression's buffer effects are unknowable, so the
    /// touched-buffer disjointness check cannot see them. `collect_buffers_in_expr`
    /// summarises `Expr::Opaque` as touching no buffers, so a naive split would
    /// declare the halves disjoint and reorder the opaque's hidden buffer
    /// accesses past a sibling store, breaking a possible cross-iteration
    /// dependency. Fission must refuse, like it does for Node-level opaque ops.
    #[test]
    fn keeps_when_body_contains_opaque_expr() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(8),
            body: vec![
                Node::store("a", Expr::var("i"), Expr::opaque(OpaqueReader)),
                Node::store("b", Expr::var("i"), Expr::u32(2)),
            ],
        }];
        let program = program_with_entry(vec![buf("a"), buf("b")], entry);
        let result = LoopFission::transform(program);
        assert!(
            !result.changed,
            "an opaque expression's unknowable buffer effects must block fission",
        );
        assert_eq!(
            count_loops(result.program.entry()),
            1,
            "a loop with an opaque-valued store must not fission",
        );
    }

    /// Negative: shared buffer between halves blocks the fission.
    #[test]
    fn keeps_when_halves_share_a_buffer() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(8),
            body: vec![
                Node::store("a", Expr::var("i"), Expr::u32(1)),
                Node::store("a", Expr::var("i"), Expr::u32(2)),
            ],
        }];
        let program = program_with_entry(vec![buf("a")], entry);
        let result = LoopFission::transform(program);
        assert!(
            !result.changed,
            "shared buffer must block fission  -  alias proof unavailable"
        );
        assert_eq!(count_loops(result.program.entry()), 1);
    }

    /// Negative: a name flow from prefix to suffix blocks the fission.
    #[test]
    fn keeps_when_suffix_reads_prefix_let_name() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(8),
            body: vec![
                Node::let_bind("v", Expr::u32(7)),
                Node::store("a", Expr::var("i"), Expr::var("v")),
                Node::store("b", Expr::var("i"), Expr::var("v")),
            ],
        }];
        let program = program_with_entry(vec![buf("a"), buf("b")], entry);
        let result = LoopFission::transform(program);
        assert!(
            !result.changed,
            "name flow across split point must block fission"
        );
        assert_eq!(count_loops(result.program.entry()), 1);
    }

    /// Negative: a Barrier inside the loop body sequences memory across
    /// iterations and must not be split.
    #[test]
    fn keeps_when_body_contains_barrier() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(8),
            body: vec![
                Node::store("a", Expr::var("i"), Expr::u32(1)),
                Node::barrier_with_ordering(crate::ir::MemoryOrdering::Relaxed),
                Node::store("b", Expr::var("i"), Expr::u32(2)),
            ],
        }];
        let program = program_with_entry(vec![buf("a"), buf("b")], entry);
        let result = LoopFission::transform(program);
        assert!(!result.changed, "Barrier must block fission");
        assert_eq!(count_loops(result.program.entry()), 1);
    }

    /// Negative: a cross-thread collective (AllReduce) in the loop body
    /// synchronizes a CommGroup across threads. Fission's intra-thread
    /// buffer-disjointness check cannot prove that reordering the collective
    /// into a separate loop preserves cross-thread semantics, so it must block
    /// the split, exactly as a Barrier does, and as loop_unroll /
    /// loop_strip_mine already do. (Before the fix, the disjoint `a`/`b`
    /// buffers let fission split this loop, reordering the collective.)
    #[test]
    fn keeps_when_body_contains_collective() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(8),
            body: vec![
                Node::AllReduce {
                    buffer: Ident::from("a"),
                    op: crate::ir::CollectiveOp::Sum,
                    group: crate::ir::CommGroup::WORLD,
                },
                Node::store("b", Expr::var("i"), Expr::u32(2)),
            ],
        }];
        let program = program_with_entry(vec![buf("a"), buf("b")], entry);
        let result = LoopFission::transform(program);
        assert!(
            !result.changed,
            "a cross-thread collective must block fission like a barrier"
        );
        assert_eq!(count_loops(result.program.entry()), 1);
    }

    /// Negative: a single-statement body cannot be fissioned (needs at
    /// least two siblings to split).
    #[test]
    fn keeps_when_body_has_one_statement() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(8),
            body: vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
        }];
        let program = program_with_entry(vec![buf("a")], entry);
        let result = LoopFission::transform(program);
        assert!(!result.changed);
        assert_eq!(count_loops(result.program.entry()), 1);
    }

    /// Negative: runtime upper bound rejects the fission gate (we keep
    /// the bounds-must-be-literal contract symmetrical with A26 fusion
    /// and A29 strip-mine).
    #[test]
    fn keeps_when_upper_bound_is_runtime() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::var("n"),
            body: vec![
                Node::store("a", Expr::var("i"), Expr::u32(1)),
                Node::store("b", Expr::var("i"), Expr::u32(2)),
            ],
        }];
        let program = program_with_entry(vec![buf("a"), buf("b")], entry);
        let result = LoopFission::transform(program);
        assert!(!result.changed);
    }

    /// Positive: a three-arm body (`a`, `b`, `c` writing distinct
    /// buffers) fissions in repeated applications. One pass picks the
    /// earliest cleavable split  -  here the prefix `[a]` versus suffix
    /// `[b; c]`. The resulting `[b; c]` body remains fissionable for a
    /// second pass invocation.
    #[test]
    fn fissions_at_first_cleavable_split() {
        let entry = vec![Node::Loop {
            var: Ident::from("i"),
            from: Expr::u32(0),
            to: Expr::u32(8),
            body: vec![
                Node::store("a", Expr::var("i"), Expr::u32(1)),
                Node::store("b", Expr::var("i"), Expr::u32(2)),
                Node::store("c", Expr::var("i"), Expr::u32(3)),
            ],
        }];
        let program = program_with_entry(vec![buf("a"), buf("b"), buf("c")], entry);
        let first = LoopFission::transform(program);
        assert!(first.changed, "first pass must fission earliest split");
        assert_eq!(
            count_loops(first.program.entry()),
            2,
            "after one fission, two sibling loops exist"
        );
        let second = LoopFission::transform(first.program);
        assert!(
            second.changed,
            "second pass must fission the remaining two-arm loop"
        );
        assert_eq!(
            count_loops(second.program.entry()),
            3,
            "after second fission, three sibling loops exist"
        );
    }

    /// `analyze` short-circuits when no Loop is fissionable.
    #[test]
    fn analyze_skips_program_with_no_loops() {
        let entry = vec![Node::store("a", Expr::u32(0), Expr::u32(1))];
        let program = program_with_entry(vec![buf("a")], entry);
        assert!(matches!(
            crate::optimizer::ProgramPass::analyze(&LoopFission, &program),
            PassAnalysis::SKIP
        ));
    }
}
