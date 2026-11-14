//! software pipelining.
//!
//! 2-stage Load-then-Store narrow slice shipped here. Detect the
//! tight pattern:
//!
//! ```text
//! Loop(i, LitU32(lo), LitU32(hi), [
//!   Let(x, Load(buf_in, Var(i))),
//!   Store(buf_out, Var(i), expr_using_x),
//! ])
//!     where lo, hi are literals AND hi - lo >= 2
//!     AND `buf_in` != `buf_out` (distinct named buffers  -  A12 alias proof)
//!     AND `expr_using_x` reads `Var(x)` (so the Store depends on the Load)
//!     AND `expr_using_x` is observably free of side effects beyond `Var(x)`
//! ```
//!
//! Rewrite to overlap the load of iteration `i+1` with the
//! compute / store of iteration `i`:
//!
//! ```text
//! Let(__sp_x_pipe, Load(buf_in, LitU32(lo)));   // prologue: load 0th iteration
//! Loop(i, lo, hi - 1, [
//!   Let(__sp_x_next, Load(buf_in, Var(i) + LitU32(1))),  // prefetch next
//!   Store(buf_out, Var(i), expr_using_x[x := __sp_x_pipe]),  // process current
//!   Assign(__sp_x_pipe, Var(__sp_x_next)),  // shuffle
//! ]);
//! Store(buf_out, LitU32(hi - 1),
//!       expr_using_x[x := __sp_x_pipe]);  // epilogue: last iteration
//! ```
//!
//! Op id: `vyre-foundation::optimizer::passes::loop_software_pipeline`.
//! Soundness: `Exact`. Each iteration `i ∈ [lo, hi)` produces the
//! same (load, compute, store) triple in both forms; the rewrite
//! shifts the load forward by one iteration without changing what
//! gets stored or in what order. The A12 alias proof on
//! `buf_in != buf_out` guarantees the prefetched load doesn't read
//! a value the previous iteration's store invalidated.
//!
//! Cost direction: monotone-down on per-iteration latency (the
//! load latency is hidden behind the previous iteration's compute);
//! `node_count` rises by ~3 (prologue Let + Assign + epilogue
//! Store) but the per-iteration body shrinks by zero  -  net
//! constant overhead amortised over `hi - lo` iterations.
//!
//! ## Conservatism
//!
//! - Both bounds must be `Expr::LitU32` literals; trip count must
//!   be `>= 2` (otherwise prologue/epilogue degenerate).
//! - Index expression in both Load and Store must be exactly
//!   `Var(loop_var)`. Affine indexing (e.g. `Var(i) * stride`) is
//!   the next refinement and lands beside this row.
//! - The Store value expression must read `Var(name)` where `name`
//!   is the Let-bound symbol from the Load. No other reads of
//!   `name` in the body, no reassignment, no nested capture.
//! - `buf_in` and `buf_out` must be provably-distinct
//!   (`ProgramFacts::buffers_provably_distinct` from A12). If they
//!   alias, the prefetch could read a value the previous iteration
//!   just overwrote.
//! - The Store value expression must be observably free apart from
//!   the `Var(name)` read  -  no Load (other than via name), no
//!   Atomic, no Call, no Opaque, no Subgroup.

use crate::ir::{BinOp, Expr, Ident, Node, Program};
use crate::optimizer::program_soa::ProgramFacts;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};
use crate::visit;

/// 2-stage Load-then-Store software pipeline pass.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "loop_software_pipeline",
    requires = ["const_fold"],
    invalidates = ["loop_unroll", "loop_strip_mine"],
    phase = "loop",
    boundary_class = "abi_preserving",
    cost_model_family = "loop"
)]
pub struct LoopSoftwarePipeline;

impl LoopSoftwarePipeline {
    /// Skip programs without a candidate loop.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        // Pipelining requires a Loop. Without one, the ProgramFacts
        // build (full SoA walk) and the recursive pipelinable check
        // would both come up empty.
        if !program.stats().has_node_loop() {
            return PassAnalysis::SKIP;
        }
        let facts = ProgramFacts::build_cached(program);
        PassAnalysis::run_if(
            program
                .entry()
                .iter()
                .any(|n| node_has_pipelinable_loop(n, &facts)),
        )
    }

    /// Walk the entry tree; rewrite every pipelinable Loop into
    /// prologue + steady-state + epilogue.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        if !Self::analyze_impl(&program).should_run {
            return PassResult::unchanged(program);
        }
        let facts = ProgramFacts::build_cached(&program);
        let mut changed = false;
        let program = program.map_entry(|entry| {
            entry
                .into_iter()
                .flat_map(|n| rewrite_node(n, &facts, &mut changed))
                .collect()
        });
        PassResult { program, changed }
    }
}

fn rewrite_body(body: Vec<Node>, facts: &ProgramFacts, changed: &mut bool) -> Vec<Node> {
    body.into_iter()
        .flat_map(|n| rewrite_node(n, facts, changed))
        .collect()
}

fn rewrite_node(node: Node, facts: &ProgramFacts, changed: &mut bool) -> Vec<Node> {
    match node {
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            if let Some(plan) = analyse_pipelinable(&var, &from, &to, &body, facts) {
                *changed = true;
                return apply_pipeline(&plan);
            }
            vec![Node::Loop {
                var,
                from,
                to,
                body: rewrite_body(body, facts, changed),
            }]
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => vec![Node::If {
            cond,
            then: rewrite_body(then, facts, changed),
            otherwise: rewrite_body(otherwise, facts, changed),
        }],
        Node::Block(body) => vec![Node::Block(rewrite_body(body, facts, changed))],
        other => vec![visit::node_map::map_body(other, &mut |body| {
            rewrite_body(body, facts, changed)
        })],
    }
}

struct PipelinePlan {
    loop_var: Ident,
    lo: u32,
    hi: u32,
    pipe_name: Ident,
    next_name: Ident,
    let_name: Ident,
    buf_in: Ident,
    buf_out: Ident,
    store_value_template: Expr,
}

fn analyse_pipelinable(
    var: &Ident,
    from: &Expr,
    to: &Expr,
    body: &[Node],
    facts: &ProgramFacts,
) -> Option<PipelinePlan> {
    let (lo, hi) = match (from, to) {
        (Expr::LitU32(lo), Expr::LitU32(hi)) if hi.checked_sub(*lo).is_some_and(|n| n >= 2) => {
            (*lo, *hi)
        }
        _ => return None,
    };
    if body.len() != 2 {
        return None;
    }
    let (let_name, buf_in) = match &body[0] {
        Node::Let {
            name,
            value: Expr::Load { buffer, index },
        } => match index.as_ref() {
            Expr::Var(idx_var) if idx_var == var => (name.clone(), buffer.clone()),
            _ => return None,
        },
        _ => return None,
    };
    let (buf_out, store_value) = match &body[1] {
        Node::Store {
            buffer,
            index: Expr::Var(idx_var),
            value,
        } if idx_var == var => (buffer.clone(), value.clone()),
        _ => return None,
    };
    if buf_in == buf_out {
        return None;
    }
    if !facts.buffers_provably_distinct(buf_in.as_str(), buf_out.as_str()) {
        return None;
    }
    if !expr_reads_only(&store_value, &let_name) {
        return None;
    }
    let pipe_name = Ident::from(format!("__sp_{}_pipe", let_name.as_str()));
    let next_name = Ident::from(format!("__sp_{}_next", let_name.as_str()));
    Some(PipelinePlan {
        loop_var: var.clone(),
        lo,
        hi,
        pipe_name,
        next_name,
        let_name,
        buf_in,
        buf_out,
        store_value_template: store_value,
    })
}

fn apply_pipeline(plan: &PipelinePlan) -> Vec<Node> {
    let prologue = Node::let_bind(
        plan.pipe_name.clone(),
        Expr::Load {
            buffer: plan.buf_in.clone(),
            index: Box::new(Expr::u32(plan.lo)),
        },
    );
    let prefetch = Node::let_bind(
        plan.next_name.clone(),
        Expr::Load {
            buffer: plan.buf_in.clone(),
            index: Box::new(Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(Expr::Var(plan.loop_var.clone())),
                right: Box::new(Expr::u32(1)),
            }),
        },
    );
    let pipe_value = substitute_var(
        plan.store_value_template.clone(),
        &plan.let_name,
        &plan.pipe_name,
    );
    let store_current = Node::Store {
        buffer: plan.buf_out.clone(),
        index: Expr::Var(plan.loop_var.clone()),
        value: pipe_value.clone(),
    };
    let shuffle = Node::Assign {
        name: plan.pipe_name.clone(),
        value: Expr::Var(plan.next_name.clone()),
    };
    let steady = Node::Loop {
        var: plan.loop_var.clone(),
        from: Expr::u32(plan.lo),
        to: Expr::u32(plan.hi - 1),
        body: vec![prefetch, store_current, shuffle],
    };
    let epilogue = Node::Store {
        buffer: plan.buf_out.clone(),
        index: Expr::u32(plan.hi - 1),
        value: pipe_value,
    };
    // The epilogue is the last iteration (`i == hi - 1`) hoisted OUT of the
    // steady loop, so the loop variable is no longer in scope at this point.
    // The store value may legitimately reference it (e.g. `buf_out[i] = x + i`),
    // so substitute the loop var with the concrete last-iteration index. The
    // steady-state store keeps `Var(loop_var)` (still in scope inside the loop).
    // Reuses the canonical induction-var substitution rather than a local copy.
    let epilogue = crate::transform::subst::substitute_node(
        &epilogue,
        &plan.loop_var,
        &Expr::u32(plan.hi - 1),
    );
    vec![prologue, steady, epilogue]
}

/// True iff `expr` reads `Var(name)` at least once AND has no
/// observable side effect beyond that read (no Load except via
/// `name`, no Atomic, no Call, no Opaque, no Subgroup).
fn expr_reads_only(expr: &Expr, name: &Ident) -> bool {
    let mut reads_name = false;
    let observable = expr_visit_check(expr, name, &mut reads_name);
    observable && reads_name
}

fn expr_visit_check(expr: &Expr, name: &Ident, reads_name: &mut bool) -> bool {
    match expr {
        Expr::Var(n) => {
            if n == name {
                *reads_name = true;
            }
            true
        }
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::BufferRef { .. }
        | Expr::BufLen { .. }
        | Expr::InvocationId { .. }
        | Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. } => true,
        Expr::BinOp { left, right, .. } => {
            expr_visit_check(left, name, reads_name) && expr_visit_check(right, name, reads_name)
        }
        Expr::UnOp { operand, .. } => expr_visit_check(operand, name, reads_name),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            expr_visit_check(cond, name, reads_name)
                && expr_visit_check(true_val, name, reads_name)
                && expr_visit_check(false_val, name, reads_name)
        }
        Expr::Cast { value, .. } => expr_visit_check(value, name, reads_name),
        Expr::Fma { a, b, c } => {
            expr_visit_check(a, name, reads_name)
                && expr_visit_check(b, name, reads_name)
                && expr_visit_check(c, name, reads_name)
        }
        Expr::Load { .. }
        | Expr::Atomic { .. }
        | Expr::Call { .. }
        | Expr::Opaque(_)
        | Expr::SubgroupBallot { .. }
        | Expr::SubgroupShuffle { .. }
        | Expr::SubgroupReduce { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize => false,
    }
}

fn substitute_var(expr: Expr, from: &Ident, to: &Ident) -> Expr {
    super::rename_var_in_expr(expr, from, to)
}

fn node_has_pipelinable_loop(node: &Node, facts: &ProgramFacts) -> bool {
    match node {
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            analyse_pipelinable(var, from, to, body, facts).is_some()
                || body.iter().any(|n| node_has_pipelinable_loop(n, facts))
        }
        Node::If {
            then, otherwise, ..
        } => {
            then.iter().any(|n| node_has_pipelinable_loop(n, facts))
                || otherwise
                    .iter()
                    .any(|n| node_has_pipelinable_loop(n, facts))
        }
        Node::Block(body) => body.iter().any(|n| node_has_pipelinable_loop(n, facts)),
        Node::Region { body, .. } => body.iter().any(|n| node_has_pipelinable_loop(n, facts)),
        // A variant this match does not name may still carry child bodies.
        // `child_bodies` owns which slots exist, so a loop nested under a new
        // variant is found instead of skipped.
        _ => visit::child_bodies(node)
            .into_iter()
            .flatten()
            .any(|n| node_has_pipelinable_loop(n, facts)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node};

    fn ro(name: &str) -> BufferDecl {
        BufferDecl::storage(name, 0, BufferAccess::ReadOnly, DataType::U32).with_count(16)
    }

    fn rw(name: &str, binding: u32) -> BufferDecl {
        BufferDecl::storage(name, binding, BufferAccess::ReadWrite, DataType::U32).with_count(16)
    }

    fn program(buffers: Vec<BufferDecl>, entry: Vec<Node>) -> Program {
        Program::wrapped(buffers, [1, 1, 1], entry)
    }

    fn load_store_loop(
        from: Expr,
        to: Expr,
        load_buf: &str,
        load_index: Expr,
        extra_stmt: Option<Node>,
        store_value: Expr,
    ) -> Vec<Node> {
        let mut body = vec![Node::let_bind(
            "x",
            Expr::Load {
                buffer: Ident::from(load_buf),
                index: Box::new(load_index),
            },
        )];
        if let Some(stmt) = extra_stmt {
            body.push(stmt);
        }
        body.push(Node::store("rw", Expr::var("i"), store_value));
        vec![Node::Loop {
            var: Ident::from("i"),
            from,
            to,
            body,
        }]
    }

    fn add_one_x() -> Expr {
        Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(Expr::var("x")),
            right: Box::new(Expr::u32(1)),
        }
    }

    fn pipelinable_loop(lo: u32, hi: u32) -> Vec<Node> {
        load_store_loop(
            Expr::u32(lo),
            Expr::u32(hi),
            "ro",
            Expr::var("i"),
            None,
            add_one_x(),
        )
    }

    fn count_loops_and_stores(nodes: &[Node]) -> (usize, usize) {
        let mut loops = 0;
        let mut stores = 0;
        for n in nodes {
            if matches!(n, Node::Loop { .. }) {
                loops += 1;
            } else if matches!(n, Node::Store { .. }) {
                stores += 1;
            }
            for body in visit::child_bodies(n) {
                let (l, s) = count_loops_and_stores(body);
                loops += l;
                stores += s;
            }
        }
        (loops, stores)
    }

    fn assert_unchanged(buffers: Vec<BufferDecl>, entry: Vec<Node>) {
        let result = LoopSoftwarePipeline::transform(program(buffers, entry));
        assert!(!result.changed);
    }

    #[test]
    fn pipelines_simple_load_store_loop() {
        let prog = program(vec![ro("ro"), rw("rw", 1)], pipelinable_loop(0, 8));
        let result = LoopSoftwarePipeline::transform(prog);
        assert!(result.changed, "Load-then-Store loop must pipeline");
        let (loops, stores) = count_loops_and_stores(result.program.entry());
        assert_eq!(loops, 1, "exactly one steady-state Loop after pipelining");
        assert_eq!(stores, 2);
    }

    #[test]
    fn keeps_loop_with_trip_count_one() {
        assert_unchanged(vec![ro("ro"), rw("rw", 1)], pipelinable_loop(0, 1));
    }

    #[test]
    fn keeps_loop_when_buffers_alias() {
        let entry = load_store_loop(
            Expr::u32(0),
            Expr::u32(8),
            "rw",
            Expr::var("i"),
            None,
            add_one_x(),
        );
        assert_unchanged(vec![rw("rw", 1)], entry);
    }

    #[test]
    fn keeps_loop_with_non_var_index() {
        let bad_idx = Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(Expr::var("i")),
            right: Box::new(Expr::u32(1)),
        };
        let entry = load_store_loop(
            Expr::u32(0),
            Expr::u32(8),
            "ro",
            bad_idx,
            None,
            Expr::var("x"),
        );
        assert_unchanged(vec![ro("ro"), rw("rw", 1)], entry);
    }

    #[test]
    fn keeps_loop_with_three_body_stmts() {
        let entry = load_store_loop(
            Expr::u32(0),
            Expr::u32(8),
            "ro",
            Expr::var("i"),
            Some(Node::let_bind("y", Expr::u32(1))),
            Expr::var("x"),
        );
        assert_unchanged(vec![ro("ro"), rw("rw", 1)], entry);
    }

    #[test]
    fn keeps_loop_when_store_does_not_use_load() {
        let entry = load_store_loop(
            Expr::u32(0),
            Expr::u32(8),
            "ro",
            Expr::var("i"),
            None,
            Expr::u32(99),
        );
        assert_unchanged(vec![ro("ro"), rw("rw", 1)], entry);
    }

    #[test]
    fn keeps_loop_when_store_value_has_other_load() {
        let other_load = Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(Expr::var("x")),
            right: Box::new(Expr::Load {
                buffer: Ident::from("ro"),
                index: Box::new(Expr::u32(0)),
            }),
        };
        let entry = load_store_loop(
            Expr::u32(0),
            Expr::u32(8),
            "ro",
            Expr::var("i"),
            None,
            other_load,
        );
        assert_unchanged(vec![ro("ro"), rw("rw", 1)], entry);
    }

    #[test]
    fn keeps_loop_with_runtime_bounds() {
        let entry = load_store_loop(
            Expr::u32(0),
            Expr::var("n"),
            "ro",
            Expr::var("i"),
            None,
            Expr::var("x"),
        );
        assert_unchanged(vec![ro("ro"), rw("rw", 1)], entry);
    }

    #[test]
    fn analyze_skips_program_without_pipelinable_loop() {
        let entry = vec![Node::store("rw", Expr::u32(0), Expr::u32(1))];
        let prog = program(vec![rw("rw", 1)], entry);
        match crate::optimizer::ProgramPass::analyze(&LoopSoftwarePipeline, &prog) {
            PassAnalysis::SKIP => {}
            other => panic!("expected SKIP, got {other:?}"),
        }
    }

    #[test]
    fn keeps_loop_when_pipelinability_check_would_overflow() {
        assert_unchanged(
            vec![ro("ro"), rw("rw", 1)],
            pipelinable_loop(u32::MAX, u32::MAX),
        );
    }
}
