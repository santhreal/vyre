use super::substitution::{body_writes_loop_var, substitute_node, substitute_nodes};
use crate::ir::{Expr, Node, Program};
use crate::optimizer::rewrite::rewrite_node_slices;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

const MAX_UNROLL_TRIP_COUNT: u32 = 16;
const MAX_UNROLLED_BODY_COST: u32 = 64;

/// Expand loops with small compile-time-known trip counts.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "loop_unroll",
    requires = ["const_fold"],
    invalidates = ["const_fold", "value_numbering", "fusion"],
    phase = "loop",
    boundary_class = "abi_preserving",
    cost_model_family = "loop"
)]
pub struct LoopUnroll;

impl LoopUnroll {
    /// O(1) gate: skip when the program contains no `Node::Loop` at all.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        if !program.stats().has_node_loop() {
            return PassAnalysis::SKIP;
        }
        PassAnalysis::RUN
    }

    /// Replace bounded `from..to` loops with repeated bodies when the trip
    /// count is compile-time-known and small enough to avoid code-size blowup.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        match rewrite_nodes(program.entry()) {
            Cow::Borrowed(_) => PassResult::unchanged(program),
            Cow::Owned(entry) => PassResult {
                program: program.with_rewritten_entry(entry),
                changed: true,
            },
        }
    }
}

fn rewrite_nodes(nodes: &[Node]) -> Cow<'_, [Node]> {
    rewrite_node_slices(nodes, rewrite_node)
}

fn rewrite_node(node: &Node) -> Cow<'_, [Node]> {
    match node {
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            let rewritten_body = rewrite_nodes(body);
            let body_slice = rewritten_body.as_ref();
            if let Some(values) = unroll_values(from, to, body_slice) {
                if body_writes_loop_var(body_slice, var) || body_contains_assign(body_slice) {
                    let rebuilt = rebuild_loop_if_needed(node, rewritten_body);
                    return rebuilt.map_or_else(
                        || Cow::Borrowed(std::slice::from_ref(node)),
                        |n| Cow::Owned(vec![n]),
                    );
                }
                let isolate_iteration_scope = body_declares_locals(body_slice);
                let trip_count = values.len();
                let mut out = Vec::with_capacity(if isolate_iteration_scope {
                    trip_count
                } else {
                    body_slice.len().saturating_mul(trip_count)
                });
                for value in values {
                    let replacement = Expr::u32(value);
                    if isolate_iteration_scope {
                        out.push(Node::block(substitute_nodes(body_slice, var, &replacement)));
                    } else {
                        for item in body_slice {
                            out.push(substitute_node(item, var, &replacement));
                        }
                    }
                }
                Cow::Owned(out)
            } else {
                let rebuilt = rebuild_loop_if_needed(node, rewritten_body);
                rebuilt.map_or_else(
                    || Cow::Borrowed(std::slice::from_ref(node)),
                    |n| Cow::Owned(vec![n]),
                )
            }
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let rewritten_then = rewrite_nodes(then);
            let rewritten_otherwise = rewrite_nodes(otherwise);
            if matches!(
                (&rewritten_then, &rewritten_otherwise),
                (Cow::Borrowed(_), Cow::Borrowed(_))
            ) {
                Cow::Borrowed(std::slice::from_ref(node))
            } else {
                Cow::Owned(vec![Node::if_then_else(
                    cond.clone(),
                    rewritten_then.into_owned(),
                    rewritten_otherwise.into_owned(),
                )])
            }
        }
        Node::Block(body) => match rewrite_nodes(body) {
            Cow::Borrowed(_) => Cow::Borrowed(std::slice::from_ref(node)),
            Cow::Owned(body) => Cow::Owned(vec![Node::block(body)]),
        },
        Node::Region {
            generator,
            source_region,
            body,
        } => match rewrite_nodes(body) {
            Cow::Borrowed(_) => Cow::Borrowed(std::slice::from_ref(node)),
            Cow::Owned(body) => Cow::Owned(vec![Node::Region {
                generator: generator.clone(),
                source_region: source_region.clone(),
                body: Arc::new(body),
            }]),
        },
        _ => Cow::Borrowed(std::slice::from_ref(node)),
    }
}

fn rebuild_loop_if_needed(node: &Node, body: Cow<'_, [Node]>) -> Option<Node> {
    let Node::Loop { var, from, to, .. } = node else {
        return None;
    };
    match body {
        Cow::Borrowed(_) => None,
        Cow::Owned(body) => Some(Node::loop_for(var, from.clone(), to.clone(), body)),
    }
}

fn unroll_values(from: &Expr, to: &Expr, body: &[Node]) -> Option<Range<u32>> {
    let from = literal_u32(from)?;
    let to = literal_u32(to)?;
    let trip_count = to.checked_sub(from)?;
    if trip_count == 0 || trip_count > MAX_UNROLL_TRIP_COUNT {
        return None;
    }
    let body_cost = unroll_body_cost(body)?;
    // A trip-count-1 loop inlines its body EXACTLY once: there is no
    // duplication, so the size-blowup cap (which bounds code growth from copying
    // the body `trip_count` times) does not apply. Gating trip-1 promotion on
    // body cost made loop_unroll non-idempotent (FINDING-OPT-IDEM-1): a body
    // whose cost exceeds the cap is left as a `Loop { from 0, to 1 }` in phase 2,
    // then the phase-3 CSE/DCE cleanup that runs *after* this pass shrinks the
    // body below the cap, so the trip-1 Loop->Block promotion fired on the
    // *second* optimize() but not the first. Lifting the cap for trip_count == 1
    // promotes it on the first pass; the rewrite is strictly cost-monotone
    // (it removes loop control overhead and never grows the body), so the
    // scheduler's cost-monotone gate accepts it. For trip_count > 1 the cap
    // still guards real code-size blowup from duplication.
    if trip_count > 1 && body_cost.saturating_mul(trip_count) > MAX_UNROLLED_BODY_COST {
        return None;
    }
    Some(from..to)
}

fn literal_u32(expr: &Expr) -> Option<u32> {
    match expr {
        Expr::LitU32(value) => Some(*value),
        Expr::LitI32(value) => u32::try_from(*value).ok(),
        _ => None,
    }
}

// `body_writes_loop_var` lives in `super::substitution` (one canonical copy
// shared by every loop pass that reasons about induction-variable stability).
// `body_contains_assign` below is unroll-specific (any Assign at all is unsafe
// to duplicate across unrolled copies, not just an assign to the loop var).

/// True when any statement under `nodes` assigns to a binding. Child
/// enumeration comes from
/// [`child_bodies`](crate::transform::visit::child_bodies) so a future nesting
/// variant cannot hide an assignment from the unroll safety check.
fn body_contains_assign(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| {
        matches!(node, Node::Assign { .. })
            || crate::transform::visit::child_bodies(node)
                .into_iter()
                .any(body_contains_assign)
    })
}

/// True when unrolling `nodes` would produce a duplicate sibling binding.
///
/// The hazard is V032, a duplicate `Let` among siblings in one scope, so what
/// matters is whether a copy of a binding lands beside the original rather than
/// whether a binding exists anywhere below. `If` arms, `Block` bodies and
/// `Loop` bodies each open a scope, so a `Let` inside one is a sibling of the
/// other copies' equivalents, not of the original; descending into `If` and
/// `Block` is therefore stricter than V032 requires and stays only because
/// dropping it would let loops unroll that do not unroll today. `Node::Loop` is
/// excluded and pinned by `does_not_substitute_shadowed_inner_loop_body`.
///
/// `Node::Region` is included because a region body is walked without a fresh
/// scope frame (`validate::nodes`), so on the reading that its bindings are
/// siblings of the surrounding sequence, isolation is required. No test
/// observes a failure without it, so this is conservatism, not a fix.
fn body_declares_locals(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| match node {
        Node::Let { .. } => true,
        Node::If {
            then, otherwise, ..
        } => body_declares_locals(then) || body_declares_locals(otherwise),
        Node::Block(body) => body_declares_locals(body),
        Node::Region { body, .. } => body_declares_locals(body),
        // Any other variant either opens its own scope or holds no statements,
        // so a copy of it cannot introduce a sibling binding at this level.
        _ => false,
    })
}

fn unroll_body_cost(nodes: &[Node]) -> Option<u32> {
    nodes.iter().try_fold(0u32, |acc, node| {
        Some(acc.saturating_add(node_unroll_cost(node)?))
    })
}

fn node_unroll_cost(node: &Node) -> Option<u32> {
    match node {
        Node::Let { value, .. } | Node::Assign { value, .. } => {
            Some(1u32.saturating_add(expr_unroll_cost(value)))
        }
        Node::Store { index, value, .. } => Some(
            2u32.saturating_add(expr_unroll_cost(index))
                .saturating_add(expr_unroll_cost(value)),
        ),
        Node::If {
            cond,
            then,
            otherwise,
        } => Some(
            4u32.saturating_add(expr_unroll_cost(cond))
                .saturating_add(unroll_body_cost(then)?)
                .saturating_add(unroll_body_cost(otherwise)?),
        ),
        Node::Loop { from, to, body, .. } => Some(
            6u32.saturating_add(expr_unroll_cost(from))
                .saturating_add(expr_unroll_cost(to))
                .saturating_add(unroll_body_cost(body)?),
        ),
        Node::Block(body) => unroll_body_cost(body),
        Node::Region { body, .. } => unroll_body_cost(body),
        Node::Return
        | Node::Barrier { .. }
        | Node::IndirectDispatch { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncStore { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::AsyncWait { .. }
        | Node::Trap { .. }
        | Node::Resume { .. }
        | Node::Opaque(_) => None,
    }
}

fn expr_unroll_cost(expr: &Expr) -> u32 {
    let mut cost = 0u32;
    let mut stack: SmallVec<[&Expr; 16]> = SmallVec::new();
    stack.push(expr);
    while let Some(expr) = stack.pop() {
        cost = cost.saturating_add(1);
        crate::optimizer::rewrite::push_expr_children(expr, &mut stack);
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType};
    use crate::optimizer::passes::algebraic::const_fold::ConstFold;
    use crate::optimizer::{PassScheduler, ProgramPassKind};

    #[test]
    fn analyze_skips_program_with_no_loop() {
        let program = Program::wrapped(Vec::new(), [1, 1, 1], vec![Node::Return]);
        match crate::optimizer::ProgramPass::analyze(&LoopUnroll, &program) {
            PassAnalysis::SKIP => {}
            other => panic!("expected SKIP for loop-free program, got {other:?}"),
        }
    }

    #[test]
    fn unrolls_small_u32_loop_and_substitutes_index() {
        let program = Program::wrapped(
            vec![BufferDecl::read_write("out", 0, DataType::U32)],
            [1, 1, 1],
            vec![Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(3),
                vec![Node::store(
                    "out",
                    Expr::var("i"),
                    Expr::add(Expr::var("i"), Expr::u32(1)),
                )],
            )],
        );

        let optimized = PassScheduler::with_passes(vec![
            ProgramPassKind::new(ConstFold),
            ProgramPassKind::new(LoopUnroll),
        ])
        .run(program)
        .expect("Fix: loop unroll should converge");

        let body = crate::test_region_body::region_body(&optimized);
        assert_eq!(body.len(), 3);
        for (index, node) in body.iter().enumerate() {
            assert!(matches!(
                node,
                Node::Store {
                    index: Expr::LitU32(i),
                    value: Expr::LitU32(v),
                    ..
                } if *i == index as u32 && *v == index as u32 + 1
            ));
        }
    }

    #[test]
    fn keeps_large_loop_bounded() {
        fn large_loop_program() -> Program {
            Program::wrapped(
                Vec::new(),
                [1, 1, 1],
                vec![Node::loop_for(
                    "i",
                    Expr::u32(0),
                    Expr::u32(MAX_UNROLL_TRIP_COUNT + 1),
                    vec![Node::let_bind("x", Expr::var("i"))],
                )],
            )
        }

        let program = large_loop_program();
        let expected = large_loop_program();
        let optimized = LoopUnroll::transform(program).program;
        assert_eq!(optimized, expected);
    }

    #[test]
    fn unrolls_tiny_loop_above_old_trip_limit_when_cost_is_small() {
        let program = Program::wrapped(
            vec![BufferDecl::read_write("out", 0, DataType::U32)],
            [1, 1, 1],
            vec![Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(12),
                vec![Node::store("out", Expr::var("i"), Expr::u32(1))],
            )],
        );

        let optimized = LoopUnroll::transform(program).program;
        let body = crate::test_region_body::region_body(&optimized);
        assert_eq!(body.len(), 12);
        assert!(matches!(
            &body[11],
            Node::Store {
                index: Expr::LitU32(11),
                ..
            }
        ));
    }

    #[test]
    fn keeps_small_trip_loop_when_body_cost_would_bloat_ir() {
        let expensive_value = (0..20).fold(Expr::var("x"), |acc, n| Expr::add(acc, Expr::u32(n)));
        let program = Program::wrapped(
            Vec::new(),
            [1, 1, 1],
            vec![Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(4),
                vec![Node::let_bind("x", expensive_value)],
            )],
        );

        let result = LoopUnroll::transform(program);
        assert!(!result.changed);
        let body = crate::test_region_body::region_body(&result.program);

        assert!(matches!(&body[0], Node::Loop { .. }));
    }

    #[test]
    fn keeps_loop_with_barrier_as_control_boundary() {
        let program = Program::wrapped(
            Vec::new(),
            [1, 1, 1],
            vec![Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(2),
                vec![Node::barrier()],
            )],
        );

        let result = LoopUnroll::transform(program);
        assert!(!result.changed);
        let body = crate::test_region_body::region_body(&result.program);
        assert!(matches!(&body[0], Node::Loop { .. }));
    }

    #[test]
    fn does_not_substitute_shadowed_inner_loop_body() {
        let program = Program::wrapped(
            Vec::new(),
            [1, 1, 1],
            vec![Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(2),
                vec![Node::loop_for(
                    "i",
                    Expr::var("i"),
                    Expr::u32(4),
                    vec![Node::let_bind("x", Expr::var("i"))],
                )],
            )],
        );

        let optimized = LoopUnroll::transform(program).program;
        let body = crate::test_region_body::region_body(&optimized);
        assert_eq!(body.len(), 2);
        assert!(matches!(
            &body[0],
            Node::Loop {
                from: Expr::LitU32(0),
                body,
                ..
            } if matches!(&body[0], Node::Let { value: Expr::Var(name), .. } if name == "i")
        ));
    }
}
