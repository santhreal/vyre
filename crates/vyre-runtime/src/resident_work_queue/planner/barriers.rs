//! Runtime megakernel barrier elision for independent arm chains.
//!
//! Foundation coalesces adjacent barriers. This pass handles the runtime
//! composition case: `Block/Region, Barrier, Block/Region` sequences emitted
//! while stitching megakernel arms. A barrier is removed only when both
//! neighboring arms have known buffer effects and no same-buffer read/write or
//! write/write dependency crosses the barrier.

use smallvec::SmallVec;
use vyre_foundation::ir::{Expr, Ident, Node, Program};
use vyre_foundation::transform::rewrite_walk::{rewrite_body, NodeRewrite};
use vyre_foundation::visit::{any_descendant, child_bodies};

/// Report returned by [`elide_value_flow_barriers`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BarrierElisionReport {
    /// Number of `Node::Barrier` values removed.
    pub removed: usize,
}

/// Remove barriers between independent megakernel arms.
///
/// The rewrite is intentionally conservative. It only removes a barrier when
/// the previous and next sibling are explicit arm containers (`Block` or
/// `Region`) and their recursively collected buffer effects cannot conflict.
///
/// INFALLIBLE by construction (Law 10): every working buffer is sized by the
/// program's IR node count, kernel STRUCTURE (the fused arms + scan loop), NOT
/// input/catalog/data-scaled, so it is bounded and reserved with
/// `Vec::with_capacity`, exactly like the sibling `rule_catalog` host build.
/// There is therefore no fallible-staging error to swallow, so the pass ALWAYS
/// elides; the previous `try_*` + `Err(_) => fallback` silently shipped the
/// un-elided (slower) program on a staging-reserve failure, which this removes.
#[must_use]
pub fn elide_value_flow_barriers(program: Program) -> (Program, BarrierElisionReport) {
    let mut report = BarrierElisionReport::default();
    let rewritten = match elided_scope(program.entry(), &mut report) {
        Some(entry) => program.with_rewritten_entry(entry),
        None => program,
    };
    (rewritten, report)
}

fn nodes_have_barrier(nodes: &[Node]) -> bool {
    nodes.iter().any(node_has_barrier)
}

/// True when `node` or anything under it is a barrier.
///
/// Delegates to [`any_descendant`], which enumerates children through
/// `vyre_foundation::visit::child_bodies`, the one exhaustive owner
/// of "which `Node` variants contain other nodes". This function used to run its
/// own `match node` over `If`, `Loop`, `Block`, and `Region` ending in
/// `_ => false`. That listed every nesting variant that exists today and would
/// classify the next one as a leaf, which turns the whole elision pass into a
/// no-op for any program using it: `elide_value_flow_barriers` and
/// `rewrite_nodes` both return early when this reports no barrier.
fn node_has_barrier(node: &Node) -> bool {
    any_descendant(node, &mut |candidate| {
        matches!(candidate, Node::Barrier { .. })
    })
}

/// The barrier-elision policy for one scope, driven by the rewrite owner.
///
/// It answers only the body positions. Operands carry no barrier, so leaving
/// them alone is what keeps an untouched subtree from being rebuilt.
struct ElideBarriers<'a> {
    report: &'a mut BarrierElisionReport,
}

impl NodeRewrite for ElideBarriers<'_> {
    fn operand(&mut self, _expr: &Expr) -> Option<Expr> {
        None
    }

    fn body(&mut self, _parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        elided_scope(body, self.report)
    }
}

/// Elide inside every nested body of `nodes`, then between its own siblings.
///
/// Returns `None` when no barrier was removed anywhere under `nodes`, so an
/// unchanged scope is never rebuilt and the caller keeps the original slice.
///
/// Which positions hold nested bodies is `rewrite_walk::rewrite_node`'s
/// answer, not this file's. That match is exhaustive with no catch-all, so a
/// `Node` variant added with a body fails to compile there and is descended
/// into here on the commit that declares it. The hand-written match this
/// replaced ended in `other => other`, which classified every variant it had
/// not been told about as childless: a barrier nested in a new variant would
/// have stayed unexamined, and an unexamined barrier that should have been
/// kept is a race.
fn elided_scope(nodes: &[Node], report: &mut BarrierElisionReport) -> Option<Vec<Node>> {
    if !nodes_have_barrier(nodes) {
        return None;
    }
    let before = report.removed;
    let descended = {
        let mut policy = ElideBarriers { report };
        rewrite_body(nodes, &mut policy)
    };
    let elided = elide_barrier_siblings(descended.unwrap_or_else(|| nodes.to_vec()), report);
    (report.removed != before).then_some(elided)
}

fn elide_barrier_siblings(nodes: Vec<Node>, report: &mut BarrierElisionReport) -> Vec<Node> {
    let mut out = Vec::with_capacity(nodes.len());
    let mut iter = nodes.into_iter().peekable();
    while let Some(node) = iter.next() {
        if matches!(&node, Node::Barrier { .. }) {
            if let (Some(left), Some(right)) = (out.last(), iter.peek()) {
                if is_runtime_arm(left)
                    && is_runtime_arm(right)
                    && arms_are_independent(left, right)
                {
                    report.removed += 1;
                    continue;
                }
            }
        }
        out.push(node);
    }
    out
}

fn is_runtime_arm(node: &Node) -> bool {
    matches!(node, Node::Block(_) | Node::Region { .. })
}

fn arms_are_independent(left: &Node, right: &Node) -> bool {
    let mut left_access = AccessSet::default();
    let mut right_access = AccessSet::default();
    collect_node_access(left, &mut left_access);
    collect_node_access(right, &mut right_access);
    !left_access.unknown && !right_access.unknown && !left_access.conflicts_with(&right_access)
}

#[derive(Debug, Default)]
struct AccessSet<'a> {
    reads: SmallVec<[&'a Ident; 8]>,
    writes: SmallVec<[&'a Ident; 8]>,
    unknown: bool,
}

impl<'a> AccessSet<'a> {
    fn read(&mut self, buffer: &'a Ident) {
        push_unique(&mut self.reads, buffer);
    }

    fn write(&mut self, buffer: &'a Ident) {
        push_unique(&mut self.writes, buffer);
    }

    fn read_write(&mut self, buffer: &'a Ident) {
        self.read(buffer);
        self.write(buffer);
    }

    fn conflicts_with(&self, other: &Self) -> bool {
        intersects(&self.writes, &other.reads)
            || intersects(&self.reads, &other.writes)
            || intersects(&self.writes, &other.writes)
    }
}

fn push_unique<'a>(set: &mut SmallVec<[&'a Ident; 8]>, value: &'a Ident) {
    if !set.iter().any(|existing| *existing == value) {
        set.push(value);
    }
}

fn intersects(left: &[&Ident], right: &[&Ident]) -> bool {
    if left.len() <= right.len() {
        left.iter()
            .any(|value| right.iter().any(|other| other == value))
    } else {
        right
            .iter()
            .any(|value| left.iter().any(|other| other == value))
    }
}

/// Buffer effects of `node` and everything under it.
///
/// The match answers one question only: which buffers this node itself touches
/// through its own operands. Descent is `visit::child_bodies`, the exhaustive
/// owner of which variants hold bodies, so a `Node` variant added with a body
/// is walked here on the commit that declares it.
///
/// The catch-all is therefore a decision about effects and not a claim about
/// structure: an unenumerated variant may read or write any buffer, so it marks
/// the arm unknown and `arms_are_independent` refuses to elide anything across
/// it. A future variant added without touching this file costs a missed
/// elision, never an elided barrier that was load-bearing. The collectives
/// reach it today for that reason.
fn collect_node_access<'a>(node: &'a Node, out: &mut AccessSet<'a>) {
    match node {
        Node::Let { value, .. } | Node::Assign { value, .. } => collect_expr_access(value, out),
        Node::Store {
            buffer,
            index,
            value,
        } => {
            out.write(buffer);
            collect_expr_access(index, out);
            collect_expr_access(value, out);
        }
        Node::If { cond, .. } => collect_expr_access(cond, out),
        Node::Loop { from, to, .. } => {
            collect_expr_access(from, out);
            collect_expr_access(to, out);
        }
        Node::IndirectDispatch { count_buffer, .. } => out.read(count_buffer),
        Node::AsyncLoad {
            source: s,
            destination: d,
            offset: off,
            size: sz,
            ..
        }
        | Node::AsyncStore {
            source: s,
            destination: d,
            offset: off,
            size: sz,
            ..
        } => {
            out.read(s);
            out.write(d);
            collect_expr_access(off, out);
            collect_expr_access(sz, out);
        }
        Node::AsyncWait { .. }
        | Node::Return
        | Node::Barrier { .. }
        | Node::Resume { .. }
        | Node::Block(_)
        | Node::Region { .. } => {}
        Node::Trap { address, .. } => {
            collect_expr_access(address, out);
            out.unknown = true;
        }
        Node::Opaque(_) => out.unknown = true,
        _ => out.unknown = true,
    }
    for body in child_bodies(node) {
        collect_nodes_access(body, out);
    }
}

fn collect_nodes_access<'a>(nodes: &'a [Node], out: &mut AccessSet<'a>) {
    for node in nodes {
        collect_node_access(node, out);
    }
}

fn collect_expr_access<'a>(expr: &'a Expr, out: &mut AccessSet<'a>) {
    match expr {
        Expr::Load { buffer, index } => {
            out.read(buffer);
            collect_expr_access(index, out);
        }
        Expr::BufLen { buffer } => out.read(buffer),
        Expr::BinOp { left, right, .. } => {
            collect_expr_access(left, out);
            collect_expr_access(right, out);
        }
        Expr::UnOp { operand, .. } => collect_expr_access(operand, out),
        Expr::Call { args, .. } => {
            for arg in args {
                collect_expr_access(arg, out);
            }
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            collect_expr_access(cond, out);
            collect_expr_access(true_val, out);
            collect_expr_access(false_val, out);
        }
        Expr::Cast { value, .. } => collect_expr_access(value, out),
        Expr::Fma { a, b, c } => {
            collect_expr_access(a, out);
            collect_expr_access(b, out);
            collect_expr_access(c, out);
        }
        Expr::SubgroupBallot { cond } => collect_expr_access(cond, out),
        Expr::SubgroupShuffle { value, lane } => {
            collect_expr_access(value, out);
            collect_expr_access(lane, out);
        }
        Expr::SubgroupReduce { value, .. } => collect_expr_access(value, out),
        Expr::Atomic {
            buffer,
            index,
            expected,
            value,
            ..
        } => {
            out.read_write(buffer);
            collect_expr_access(index, out);
            if let Some(expected) = expected {
                collect_expr_access(expected, out);
            }
            collect_expr_access(value, out);
        }
        Expr::Opaque(_) => out.unknown = true,
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize => {}
        _ => out.unknown = true,
    }
}

// Inline: `vyre_runtime::resident_work_queue::planner::barriers` is `private`, so no integration
// test can reach what this suite exercises.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType};

    use super::*;

    fn buffer(name: &str, binding: u32) -> BufferDecl {
        BufferDecl::storage(name, binding, BufferAccess::ReadWrite, DataType::U32)
    }

    /// Nodes anywhere in `nodes` and its nested bodies that satisfy `pred`.
    ///
    /// Descent comes from `vyre_foundation::visit::for_each_node`,
    /// the single owner of which node variants nest, rather than from a match
    /// here that would silently treat a new nesting variant as a leaf.
    fn count_matching(nodes: &[Node], mut pred: impl FnMut(&Node) -> bool) -> usize {
        let mut count = 0;
        vyre_foundation::visit::for_each_node(nodes, |node| {
            if pred(node) {
                count += 1;
            }
        });
        count
    }

    fn barrier_count(nodes: &[Node]) -> usize {
        count_matching(nodes, |node| matches!(node, Node::Barrier { .. }))
    }

    fn store_count(nodes: &[Node]) -> usize {
        count_matching(nodes, |node| matches!(node, Node::Store { .. }))
    }

    #[test]
    fn removes_barrier_between_disjoint_runtime_arms() {
        let program = Program::wrapped(
            vec![buffer("a", 0), buffer("b", 1)],
            [64, 1, 1],
            vec![
                Node::Block(vec![Node::store("a", Expr::u32(0), Expr::u32(1))]),
                Node::barrier(),
                Node::Block(vec![Node::store("b", Expr::u32(0), Expr::u32(2))]),
            ],
        );

        let (rewritten, report) = elide_value_flow_barriers(program);

        assert_eq!(report.removed, 1);
        assert_eq!(barrier_count(rewritten.entry()), 0);
    }

    /// Law 10 / infallibility lock: a program with SEVERAL barriers between
    /// pairwise-disjoint runtime arms must have EVERY such barrier elided in one
    /// pass. The pass is infallible (its working buffers are sized by the bounded
    /// IR node count, reserved with `Vec::with_capacity`), so it can never bail to
    /// the old `Err(_) => fallback` that silently shipped the un-elided program
    /// with these barriers still present. Three barriers between four disjoint
    /// arms must all go (removed == 3, zero barriers left).
    #[test]
    fn elides_every_barrier_across_many_disjoint_arms_in_one_pass() {
        let program = Program::wrapped(
            vec![
                buffer("a", 0),
                buffer("b", 1),
                buffer("c", 2),
                buffer("d", 3),
            ],
            [64, 1, 1],
            vec![
                Node::Block(vec![Node::store("a", Expr::u32(0), Expr::u32(1))]),
                Node::barrier(),
                Node::Block(vec![Node::store("b", Expr::u32(0), Expr::u32(2))]),
                Node::barrier(),
                Node::Block(vec![Node::store("c", Expr::u32(0), Expr::u32(3))]),
                Node::barrier(),
                Node::Block(vec![Node::store("d", Expr::u32(0), Expr::u32(4))]),
            ],
        );

        let (rewritten, report) = elide_value_flow_barriers(program);

        assert_eq!(
            report.removed, 3,
            "all three disjoint-arm barriers must be elided"
        );
        assert_eq!(barrier_count(rewritten.entry()), 0);
        // All four independent store arms must survive the rewrite (no arm dropped
        // while elliding barriers, regardless of how `Program::wrapped` nests them).
        assert_eq!(
            store_count(rewritten.entry()),
            4,
            "all four independent store arms must survive the rewrite"
        );
    }

    #[test]
    fn no_barrier_program_is_returned_without_rewrite() {
        let program = Program::wrapped(
            vec![buffer("a", 0)],
            [64, 1, 1],
            vec![Node::Block(vec![Node::store(
                "a",
                Expr::u32(0),
                Expr::u32(1),
            )])],
        );
        let expected = program.clone();

        let (rewritten, report) = elide_value_flow_barriers(program);

        assert_eq!(report.removed, 0);
        assert_eq!(
            rewritten.fingerprint(),
            expected.fingerprint(),
            "Fix: barrier-free megakernel plans must avoid structural rewrites."
        );
    }

    #[test]
    fn keeps_barrier_when_next_arm_reads_previous_write() {
        let program = Program::wrapped(
            vec![buffer("a", 0)],
            [64, 1, 1],
            vec![
                Node::Block(vec![Node::store("a", Expr::u32(0), Expr::u32(1))]),
                Node::barrier(),
                Node::Block(vec![Node::let_bind("x", Expr::load("a", Expr::u32(0)))]),
            ],
        );

        let (rewritten, report) = elide_value_flow_barriers(program);

        assert_eq!(report.removed, 0);
        assert_eq!(barrier_count(rewritten.entry()), 1);
    }

    #[test]
    fn keeps_barrier_around_unknown_opaque_arm() {
        let program = Program::wrapped(
            vec![buffer("a", 0), buffer("b", 1)],
            [64, 1, 1],
            vec![
                Node::Block(vec![Node::Opaque(Arc::new(TestOpaqueNode))]),
                Node::barrier(),
                Node::Block(vec![Node::store("b", Expr::u32(0), Expr::u32(2))]),
            ],
        );

        let (rewritten, report) = elide_value_flow_barriers(program);

        assert_eq!(report.removed, 0);
        assert_eq!(barrier_count(rewritten.entry()), 1);
    }

    use vyre_foundation::ir::NodeExtension;

    vyre_test_support::test_node_extension! {
        TestOpaqueNode,
        kind: "test.opaque",
        identity: "test.opaque",
        fingerprint: 7,
    }
}
