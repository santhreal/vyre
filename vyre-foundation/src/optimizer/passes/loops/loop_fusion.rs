//! ROADMAP A26  -  fuse adjacent `Node::Loop` siblings whose bounds
//! match and whose bodies touch disjoint buffer sets.
//!
//! Op id: `vyre-foundation::optimizer::passes::loop_fusion`.
//! Soundness: `Exact` under the conservative buffer-disjointness
//! check. Two loops with identical literal `from..to` ranges, distinct
//! loop variables, and disjoint touched-buffer sets cannot have any
//! cross-loop dependency through memory; fusing them lets the runtime
//! amortise the loop overhead and may unlock further fusion / scratch
//! reuse downstream. Cost direction: monotone-down on `node_count`
//! (one fewer Loop wrapper) and on per-iteration loop overhead.
//! Preserves: every analysis. Invalidates: nothing.
//!
//! ## Rule
//!
//! ```text
//! Node::Loop { var: i, from: LitU32(a), to: LitU32(b), body: [body_a] }
//! Node::Loop { var: j, from: LitU32(a), to: LitU32(b), body: [body_b] }
//!     where buffers_touched(body_a) ∩ buffers_touched(body_b) == ∅
//!     AND body_b uses no name bound inside body_a (other than j itself)
//! →
//! Node::Loop {
//!     var: i,
//!     from: LitU32(a),
//!     to: LitU32(b),
//!     body: [
//!         body_a...,
//!         body_b... (with `j` rewritten to `i`),
//!     ],
//! }
//! ```
//!
//! ## Conservatism
//!
//! - Bounds must be `Expr::LitU32` and structurally equal.
//! - Only adjacent siblings inside the same container body.
//! - Buffer sets must be disjoint  -  any shared buffer would need an
//!   alias / cross-iteration-dependency proof we do not have without
//!   the downstream dataflow analysis.
//! - The second loop's body is rewritten so every `Expr::Var(j)`
//!   becomes `Expr::Var(i)`. A Let in body_a whose name shadows `j`
//!   (or vice versa) blocks the fusion to keep the rewrite local.
//! - The two bodies must not bind a common local name, and body_b must not
//!   bind the fused loop variable `i`: fusion merges both bodies into ONE
//!   scope, so a shared `let` name becomes a duplicate sibling binding (V032)
//!   and a body_b binding of `i` shadows the loop var (V008). The block-scoped
//!   IR pops each loop body's bindings at loop exit, so two sibling loops
//!   binding the same local are legal pre-fusion but collide once merged.
//! - Neither body may `Assign` a scalar the other body reads or assigns: fusion
//!   interleaves the bodies (`A(0); B(0); A(1); B(1); ...`), so a scalar one
//!   loop writes and the other touches is a cross-loop dependency the original
//!   ordering (all of loop_a, then all of loop_b) does not have, and the
//!   interleaving silently changes the observed values. Assign-free bodies (the
//!   common map/transform loops) take the fast path and are unaffected.

use super::{collect_touched_buffers, collect_var_reads, legality};
use crate::ir::{Expr, Ident, Node, Program};
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};
use crate::visit::node_map;
use rustc_hash::FxHashSet;

/// Fuse adjacent `Node::Loop` siblings under the buffer-disjoint
/// conservatism rule.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "loop_fusion",
    requires = [],
    invalidates = [],
    phase = "loop",
    boundary_class = "abi_preserving",
    cost_model_family = "loop"
)]
/// ABI-preserving loop fusion pass for adjacent loops with compatible iteration spaces.
pub struct LoopFusion;

impl LoopFusion {
    /// Skip when no body has a fusable pair. Checks both the
    /// top-level entry vec (transform fuses adjacent siblings there
    /// too) and every nested If/Loop/Block/Region body.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        // Fusion needs at least two adjacent Loops; absent any Loop
        // at all the recursive walk has nothing to find.
        if !program.stats().has_node_loop() {
            return PassAnalysis::SKIP;
        }
        if body_has_fusable_pair(program.entry())
            || program
                .entry()
                .iter()
                .any(|n| node_map::any_descendant(n, &mut has_fusable_pair))
        {
            PassAnalysis::RUN
        } else {
            PassAnalysis::SKIP
        }
    }

    /// Walk the program; fuse every fusable adjacent Loop pair found.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        let mut changed = false;
        let program = program.map_entry(|entry| fuse_in_body(entry, &mut changed));
        PassResult { program, changed }
    }
}

fn fuse_in_body(body: Vec<Node>, changed: &mut bool) -> Vec<Node> {
    let body: Vec<Node> = body.into_iter().map(|n| recurse(n, changed)).collect();
    let mut out: Vec<Node> = Vec::with_capacity(body.len());
    let mut iter = body.into_iter().peekable();
    while let Some(node) = iter.next() {
        let Node::Loop {
            var: var_a,
            from: from_a,
            to: to_a,
            body: body_a,
        } = node
        else {
            out.push(node);
            continue;
        };
        let next_is_fusable = matches!(iter.peek(), Some(Node::Loop { .. }));
        if !next_is_fusable {
            out.push(Node::Loop {
                var: var_a,
                from: from_a,
                to: to_a,
                body: body_a,
            });
            continue;
        }
        let Some(Node::Loop {
            var: var_b,
            from: from_b,
            to: to_b,
            body: body_b,
        }) = iter.next()
        else {
            unreachable!("peek confirmed Loop above");
        };
        if !pair_is_fusable(
            &LoopRef {
                var: &var_a,
                from: &from_a,
                to: &to_a,
                body: &body_a,
            },
            &LoopRef {
                var: &var_b,
                from: &from_b,
                to: &to_b,
                body: &body_b,
            },
        ) {
            // Cannot fuse  -  emit the first loop, push the second back
            // for the next iteration to consider against its successor.
            out.push(Node::Loop {
                var: var_a,
                from: from_a,
                to: to_a,
                body: body_a,
            });
            // We can't actually push back into a Peekable<vec::IntoIter>;
            // emit body_b as-is. Re-fusion across the missed pair will
            // happen on the next pass-scheduler iteration if applicable.
            out.push(Node::Loop {
                var: var_b,
                from: from_b,
                to: to_b,
                body: body_b,
            });
            continue;
        }
        let mut fused = body_a;
        let renamed_body_b: Vec<Node> = body_b
            .into_iter()
            .map(|n| legality::rename_var_in_node(n, &var_b, &var_a))
            .collect();
        fused.extend(renamed_body_b);
        *changed = true;
        out.push(Node::Loop {
            var: var_a,
            from: from_a,
            to: to_a,
            body: fused,
        });
    }
    out
}

fn recurse(node: Node, changed: &mut bool) -> Node {
    let recursed = node_map::map_children(node, &mut |child| recurse(child, changed));
    node_map::map_body(recursed, &mut |body| fuse_in_body(body, changed))
}

fn bounds_match(from_a: &Expr, to_a: &Expr, from_b: &Expr, to_b: &Expr) -> bool {
    matches!(
        (from_a, to_a, from_b, to_b),
        (
            Expr::LitU32(_),
            Expr::LitU32(_),
            Expr::LitU32(_),
            Expr::LitU32(_)
        )
    ) && from_a == from_b
        && to_a == to_b
}

/// One side of a candidate fusion, borrowed from the `Node::Loop` it came
/// from so the transform and the analysis gate ask the same question without
/// cloning either body.
struct LoopRef<'a> {
    var: &'a Ident,
    from: &'a Expr,
    to: &'a Expr,
    body: &'a [Node],
}

/// The complete fusion legality gate. `a` runs entirely before `b` in the
/// original program and fusing interleaves their iterations, so every
/// dependency between the two bodies has to be ruled out first.
///
/// Memory dependence, unsummarisable effects, and the binding-capture hazard
/// come from the shared [`legality`] core. The binding-collision and
/// scalar-dependence checks are fusion-specific: they describe what happens
/// when two scopes merge into one, which fission never does.
///
/// Async, collective, and indirect-dispatch nodes are deliberately allowed.
/// Their buffer operands ARE captured by [`collect_touched_buffers`], so the
/// disjointness test already covers them, and refusing them would needlessly
/// forbid legal fusions of loops whose async or collective ops touch disjoint
/// buffers. Fission refuses them through its own barrier gate because
/// splitting a loop reorders them against the surrounding work; fusion has no
/// such reordering, so the asymmetry is intentional.
fn pair_is_fusable(a: &LoopRef<'_>, b: &LoopRef<'_>) -> bool {
    bounds_match(a.from, a.to, b.from, b.to)
        && a.var != b.var
        && super::buffers_disjoint_with(a.body, b.body, collect_touched_buffers)
        && !legality::unsummarisable_effect(a.body)
        && !legality::unsummarisable_effect(b.body)
        && !legality::bindings_flow_across(a.body, b.body, b.var)
        && !fusion_collides_bindings(a.body, b.body, a.var, b.var)
        && !fusion_has_scalar_dependency(a.body, b.body)
}

/// True iff fusing the two bodies would introduce a duplicate or shadowing
/// binding the validator rejects. Fusion concatenates the bodies into ONE loop
/// scope (`fused = body_a ++ rename(body_b, var_b -> var_a)`), so a name bound
/// by BOTH bodies becomes a duplicate sibling binding (V032), and a body_b
/// binding of the fused loop variable `var_a` shadows it (V008). The block-scoped
/// IR pops each loop body's bindings at loop exit, so two sibling loops binding
/// the same local are legal pre-fusion but collide once merged.
///
/// [`legality::collect_bound_names`] recurses into nested scopes (and counts
/// `Assign` targets and nested loop variables), so this is conservative: it may
/// refuse a fusion whose shared name actually sits in disjoint nested scopes,
/// but it never permits a real collision. This is disjoint from the *capture*
/// hazard guarded by [`legality::bindings_flow_across`] (body_b READING a
/// body_a binding); this is the duplicate-BINDING hazard.
fn fusion_collides_bindings(
    body_a: &[Node],
    body_b: &[Node],
    var_a: &Ident,
    var_b: &Ident,
) -> bool {
    let mut a_lets: FxHashSet<Ident> = FxHashSet::default();
    legality::collect_bound_names(body_a, &mut a_lets);
    let mut b_lets: FxHashSet<Ident> = FxHashSet::default();
    legality::collect_bound_names(body_b, &mut b_lets);
    b_lets.iter().any(|name| {
        // body_b is rewritten var_b -> var_a before splicing, so a body_b
        // binding of var_b lands as var_a in the fused scope.
        let fused_name = if name == var_b { var_a } else { name };
        // Collides with the fused loop variable (shadow, V008) or with a name
        // body_a already binds (duplicate sibling, V032).
        fused_name == var_a || a_lets.contains(fused_name)
    })
}

/// Collect every `Node::Assign` target name in `nodes` (recursively). These are
/// the scalars a body MUTATES (as opposed to merely binds via `Let` or reads).
fn collect_assign_targets(nodes: &[Node], out: &mut FxHashSet<Ident>) {
    for node in nodes {
        match node {
            Node::Assign { name, .. } => {
                out.insert(name.clone());
            }
            Node::If {
                then, otherwise, ..
            } => {
                collect_assign_targets(then, out);
                collect_assign_targets(otherwise, out);
            }
            Node::Loop { body, .. } | Node::Block(body) => collect_assign_targets(body, out),
            Node::Region { body, .. } => collect_assign_targets(body, out),
            _ => {}
        }
    }
}

/// True iff fusing the two bodies would reorder a cross-loop dependency through
/// a shared mutable scalar. `buffers_disjoint` rules out memory dependencies and
/// [`legality::bindings_flow_across`] rules out body_b capturing a body_a
/// binding, but NEITHER covers a scalar that one body WRITES (via `Node::Assign`)
/// and the other body reads or writes. The original program runs loop_a entirely
/// before loop_b; fusing interleaves them, so any such scalar dependency changes
/// the observed values (a silent value miscompile, e.g. body_a reading an outer
/// `s` that body_b overwrites). Conservative and name-based (recurses into
/// nested scopes): if either body assigns a name the other body references,
/// refuse. The early return keeps the common assign-free map/transform loops on
/// the fast path.
fn fusion_has_scalar_dependency(body_a: &[Node], body_b: &[Node]) -> bool {
    let mut writes_a: FxHashSet<Ident> = FxHashSet::default();
    collect_assign_targets(body_a, &mut writes_a);
    let mut writes_b: FxHashSet<Ident> = FxHashSet::default();
    collect_assign_targets(body_b, &mut writes_b);
    if writes_a.is_empty() && writes_b.is_empty() {
        return false; // neither body mutates a scalar -> no scalar dependency
    }
    // refs_x = every name body_x reads OR writes.
    let mut refs_a: FxHashSet<Ident> = FxHashSet::default();
    collect_var_reads(body_a, &mut refs_a);
    refs_a.extend(writes_a.iter().cloned());
    let mut refs_b: FxHashSet<Ident> = FxHashSet::default();
    collect_var_reads(body_b, &mut refs_b);
    refs_b.extend(writes_b.iter().cloned());
    // A scalar written by one body and touched by the other is a cross-loop
    // dependency the interleaving would violate.
    !writes_a.is_disjoint(&refs_b) || !writes_b.is_disjoint(&refs_a)
}

fn has_fusable_pair(node: &Node) -> bool {
    let body: &[Node] = match node {
        Node::If {
            then, otherwise, ..
        } => {
            return body_has_fusable_pair(then) || body_has_fusable_pair(otherwise);
        }
        Node::Loop { body, .. } | Node::Block(body) => body,
        Node::Region { body, .. } => body.as_ref(),
        _ => return false,
    };
    body_has_fusable_pair(body)
}

fn body_has_fusable_pair(body: &[Node]) -> bool {
    body.windows(2).any(|window| {
        let (
            Node::Loop {
                var: var_a,
                from: from_a,
                to: to_a,
                body: body_a,
            },
            Node::Loop {
                var: var_b,
                from: from_b,
                to: to_b,
                body: body_b,
            },
        ) = (&window[0], &window[1])
        else {
            return false;
        };
        pair_is_fusable(
            &LoopRef {
                var: var_a,
                from: from_a,
                to: to_a,
                body: body_a,
            },
            &LoopRef {
                var: var_b,
                from: from_b,
                to: to_b,
                body: body_b,
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, ExprNode, Node, NodeExtension};

    fn buf(name: &str) -> BufferDecl {
        BufferDecl::storage(name, 0, BufferAccess::ReadWrite, DataType::U32).with_count(8)
    }

    fn program(entry: Vec<Node>) -> Program {
        Program::wrapped(vec![buf("a"), buf("b")], [1, 1, 1], entry)
    }

    /// An opaque expression whose real buffer effect (it may read or write ANY
    /// buffer) is invisible to `collect_buffers_in_expr`, which summarises
    /// `Expr::Opaque(_)` as touching no buffers.
    #[derive(Debug)]
    struct OpaqueReader;

    impl ExprNode for OpaqueReader {
        fn extension_kind(&self) -> &'static str {
            "test.fusion.opaque_buffer_reader"
        }
        fn debug_identity(&self) -> &str {
            "opaque_reader"
        }
        fn result_type(&self) -> Option<DataType> {
            Some(DataType::U32)
        }
        fn cse_safe(&self) -> bool {
            false
        }
        fn stable_fingerprint(&self) -> [u8; 32] {
            [13; 32]
        }
        fn validate_extension(&self) -> std::result::Result<(), String> {
            Ok(())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// An opaque statement node whose real buffer effect is invisible to
    /// `collect_touched_buffers`, which summarises `Node::Opaque(_)` as
    /// touching no buffers.
    #[derive(Debug)]
    struct OpaqueWriter;

    impl NodeExtension for OpaqueWriter {
        fn extension_kind(&self) -> &'static str {
            "test.fusion.opaque_buffer_writer"
        }
        fn debug_identity(&self) -> &str {
            "opaque_writer"
        }
        fn stable_fingerprint(&self) -> [u8; 32] {
            [14; 32]
        }
        fn validate_extension(&self) -> std::result::Result<(), String> {
            Ok(())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn does_not_fuse_when_a_body_holds_an_opaque_expr() {
        // body_a's only explicit buffer is `a`, body_b's is `b`, so
        // `buffers_disjoint` reports them disjoint  -  but body_a's stored
        // value is an opaque expression that may read or write `b`. Fusing
        // would interleave that unknowable access with body_b's writes to
        // `b`, breaking a cross-loop dependency the disjointness proof cannot
        // see. The pass must keep the two loops separate.
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("a", Expr::var("i"), Expr::opaque(OpaqueReader))],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("b", Expr::var("j"), Expr::u32(2))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(
            !result.changed,
            "an opaque expression's unknowable buffer effect must block fusion"
        );
        assert_eq!(
            count_loops(&region_body(result.program.entry())),
            2,
            "loops bracketing an opaque-valued store must not fuse"
        );
    }

    #[test]
    fn does_not_fuse_when_a_body_holds_an_opaque_node() {
        // body_a is a single opaque node touching no buffer that
        // `collect_touched_buffers` can see, so it reports `{}`  -  trivially
        // disjoint from body_b's `{b}`. But the opaque node may write `b`;
        // fusing would interleave that hidden write with body_b's stores.
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::opaque(OpaqueWriter)],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("b", Expr::var("j"), Expr::u32(2))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(
            !result.changed,
            "an opaque node's unknowable buffer effect must block fusion"
        );
        assert_eq!(
            count_loops(&region_body(result.program.entry())),
            2,
            "a loop whose body is an opaque node must not fuse with its sibling"
        );
    }

    #[test]
    fn does_not_fuse_when_a_shuffle_lane_loads_the_siblings_buffer() {
        // body_a stores to `a`, but the stored value is a subgroup shuffle
        // whose LANE index is loaded from `b`. `collect_buffers_in_expr`
        // elided `SubgroupShuffle.lane`, so it reported body_a touching only
        // `{a}`  -  disjoint from body_b's `{b}`  -  and the loops fused. But the
        // lane load reads `b`, which body_b writes; interleaving the two loops
        // reorders that read across the writes. The collector must descend into
        // the lane operand so the shared buffer blocks fusion.
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store(
                    "a",
                    Expr::var("i"),
                    Expr::subgroup_shuffle(Expr::u32(5), Expr::load("b", Expr::var("i"))),
                )],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("b", Expr::var("j"), Expr::u32(2))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(
            !result.changed,
            "a buffer load hidden in a shuffle lane must block fusion"
        );
        assert_eq!(
            count_loops(&region_body(result.program.entry())),
            2,
            "loops sharing buffer `b` through a shuffle lane load must not fuse"
        );
    }

    #[test]
    fn does_not_fuse_when_a_shuffle_lane_reads_a_cross_loop_scalar() {
        // The scalar `s` is written every iteration of loop_a and read by
        // loop_b ONLY through a shuffle lane. `fusion_has_scalar_dependency`
        // collects body_b's reads via collect_var_reads -> collect_vars_in_expr,
        // which dropped `SubgroupShuffle.lane`; the read of `s` was therefore
        // invisible and the loops fused. After fusing, body_b observes `s == j`
        // each iteration instead of the value loop_a left behind  -  a silent
        // value miscompile. The lane read must register as a cross-loop scalar
        // dependency that blocks fusion.
        let entry = vec![
            Node::let_bind("s", Expr::u32(0)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![
                    Node::assign("s", Expr::var("i")),
                    Node::store("a", Expr::var("i"), Expr::var("s")),
                ],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store(
                    "b",
                    Expr::var("j"),
                    Expr::subgroup_shuffle(Expr::u32(5), Expr::var("s")),
                )],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(
            !result.changed,
            "a scalar read hidden in a shuffle lane is a cross-loop dependency; the loops must not fuse"
        );
        assert_eq!(
            count_loops(&region_body(result.program.entry())),
            2,
            "loops with a shuffle-lane scalar dependency must stay separate"
        );
    }

    fn region_body(program_entry: &[Node]) -> Vec<Node> {
        for n in program_entry {
            if let Node::Region { body, .. } = n {
                return body.as_ref().clone();
            }
        }
        program_entry.to_vec()
    }

    fn count_loops(nodes: &[Node]) -> usize {
        nodes
            .iter()
            .map(|n| match n {
                Node::Loop { body, .. } => 1 + count_loops(body),
                Node::If {
                    then, otherwise, ..
                } => count_loops(then) + count_loops(otherwise),
                Node::Block(body) => count_loops(body),
                Node::Region { body, .. } => count_loops(body),
                _ => 0,
            })
            .sum()
    }

    #[test]
    fn fuses_two_disjoint_buffer_loops_with_matching_bounds() {
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("b", Expr::var("j"), Expr::u32(2))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(result.changed);
        assert_eq!(
            count_loops(&region_body(result.program.entry())),
            1,
            "two loops fused into one"
        );
    }

    #[test]
    fn does_not_fuse_when_bounds_differ() {
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(16),
                vec![Node::store("b", Expr::var("j"), Expr::u32(2))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(!result.changed);
    }

    #[test]
    fn does_not_fuse_when_buffers_overlap() {
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("a", Expr::var("j"), Expr::u32(2))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(
            !result.changed,
            "shared buffer blocks fusion under disjoint-only conservatism"
        );
    }

    #[test]
    fn does_not_fuse_when_loop_vars_match() {
        // Two loops with the same var name would shadow each other in
        // the fused body; the rename rule rewrites by var name, and a
        // collision could change resolution. Refuse.
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
            ),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("b", Expr::var("i"), Expr::u32(2))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(!result.changed, "same loop var name blocks fusion");
    }

    #[test]
    fn renames_second_loop_var_in_fused_body() {
        // Fused body: `Store("a", i, 1); Store("b", i_renamed_from_j, 2)`.
        // The j-Var inside body_b becomes i.
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("b", Expr::var("j"), Expr::u32(2))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(result.changed);
        let body = region_body(result.program.entry());
        // Invariant: `result.changed` (asserted above) means `fuse_in_body`
        // rewrote the adjacent loop pair, and the only node it ever writes for
        // a fused pair is a `Node::Loop` (see `fuse_in_body`, which pushes the
        // rebuilt `Node::Loop` and skips the consumed second loop). `Node` is
        // one enum for every IR shape, so no type carries "this slot holds the
        // fused loop"; the pass is the guarantee and this is where it is read.
        let Node::Loop { body: fused, .. } = &body[0] else {
            panic!("Fix: fusion reported a change, so body[0] must be the fused Node::Loop");
        };
        assert_eq!(fused.len(), 2);
        if let Node::Store { index, .. } = &fused[1] {
            assert_eq!(
                index,
                &Expr::var("i"),
                "second store's index must be renamed to outer var"
            );
        } else {
            // Invariant: fusion concatenates body_a then body_b verbatim
            // (`fuse_in_body` extends the rebuilt body with both slices), so
            // index 1 is body_b's only node, the `Store("b", j, 2)` built at
            // the top of this test with `j` rewritten to the surviving loop
            // var. Established by the pass, not by any type.
            panic!("Fix: fusion concatenates body_b verbatim, so fused[1] must be body_b's Store");
        }
    }

    #[test]
    fn does_not_fuse_when_body_b_reads_a_let_bound_in_body_a() {
        // body_a binds "tmp"; body_b reads "tmp"  -  fusing would
        // change resolution because body_b has no access to body_a's
        // scope across iterations.
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![
                    Node::let_bind("tmp", Expr::u32(7)),
                    Node::store("a", Expr::var("i"), Expr::var("tmp")),
                ],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("b", Expr::var("j"), Expr::var("tmp"))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(!result.changed, "shared name `tmp` blocks fusion");
    }

    #[test]
    fn does_not_fuse_when_both_bodies_bind_same_local() {
        // Both bodies bind `x` in their own loop scopes (valid pre-fusion).
        // body_b does NOT read `x`, so the capture guard
        // (body_a_let_names_collide_with_b) passes -- but fusing concatenates
        // the two `let x` into one loop scope, a duplicate sibling binding the
        // validator rejects (V032). Refuse. (Oracle-differential proof:
        // tests/loop_fusion_binding_collision.rs.)
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![
                    Node::let_bind("x", Expr::u32(1)),
                    Node::store("a", Expr::var("i"), Expr::var("x")),
                ],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                // binds `x` but never reads it -> capture guard does not fire
                vec![
                    Node::let_bind("x", Expr::u32(2)),
                    Node::store("b", Expr::var("j"), Expr::u32(9)),
                ],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(
            !result.changed,
            "a local name bound by both bodies blocks fusion (duplicate sibling)"
        );
        assert_eq!(
            count_loops(&region_body(result.program.entry())),
            2,
            "both loops must survive unfused"
        );
    }

    #[test]
    fn fuses_when_bodies_bind_distinct_locals() {
        // Distinct local names (`x` vs `y`) cannot collide, so the
        // duplicate-binding guard must NOT block this fusion.
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![
                    Node::let_bind("x", Expr::u32(1)),
                    Node::store("a", Expr::var("i"), Expr::var("x")),
                ],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![
                    Node::let_bind("y", Expr::u32(2)),
                    Node::store("b", Expr::var("j"), Expr::var("y")),
                ],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(
            result.changed,
            "distinct local names must still allow fusion"
        );
        assert_eq!(
            count_loops(&region_body(result.program.entry())),
            1,
            "the two loops fuse into one"
        );
    }

    #[test]
    fn does_not_fuse_when_body_b_writes_a_scalar_body_a_reads() {
        // body_a reads outer scalar `s`; body_b writes it via Assign. The
        // original runs loop_a fully (observing s's pre-loop value) before
        // loop_b; fusing interleaves the read and write, changing observed
        // values -- a silent value miscompile. Refuse. (Oracle-differential
        // proof: tests/loop_fusion_scalar_dependency.rs.)
        let entry = vec![
            Node::let_bind("s", Expr::u32(0)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("a", Expr::var("i"), Expr::var("s"))],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::assign("s", Expr::var("j"))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(
            !result.changed,
            "a cross-loop scalar read/write dependency blocks fusion"
        );
        assert_eq!(
            count_loops(&region_body(result.program.entry())),
            2,
            "both loops survive unfused"
        );
    }

    #[test]
    fn fuses_when_bodies_mutate_independent_scalars() {
        // Each body mutates its OWN distinct outer scalar (acc1 vs acc2); there
        // is no cross-loop dependency, so the scalar-dependency guard must NOT
        // block this fusion.
        let entry = vec![
            Node::let_bind("acc1", Expr::u32(0)),
            Node::let_bind("acc2", Expr::u32(0)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![
                    Node::assign("acc1", Expr::add(Expr::var("acc1"), Expr::var("i"))),
                    Node::store("a", Expr::var("i"), Expr::var("acc1")),
                ],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![
                    Node::assign("acc2", Expr::add(Expr::var("acc2"), Expr::var("j"))),
                    Node::store("b", Expr::var("j"), Expr::var("acc2")),
                ],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(
            result.changed,
            "independent scalar accumulators must still allow fusion"
        );
        assert_eq!(
            count_loops(&region_body(result.program.entry())),
            1,
            "the two loops fuse into one"
        );
    }

    #[test]
    fn analyze_skips_when_no_fusable_pair() {
        let entry = vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(8),
            vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
        )];
        assert_eq!(
            crate::optimizer::ProgramPass::analyze(&LoopFusion, &program(entry)),
            PassAnalysis::SKIP
        );
    }

    #[test]
    fn analyze_runs_when_fusable_pair_exists() {
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("b", Expr::var("j"), Expr::u32(2))],
            ),
        ];
        assert_eq!(
            crate::optimizer::ProgramPass::analyze(&LoopFusion, &program(entry)),
            PassAnalysis::RUN
        );
    }
}
