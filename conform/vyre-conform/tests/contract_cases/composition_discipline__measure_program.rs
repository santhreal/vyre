// Composition discipline CI gates.
//
// These tests enforce the "After Effects" compositional architecture:
//
// 1. **No monoliths**  -  every registered op must stay under a complexity
//    budget. If it exceeds the threshold, the author must split the op
//    into smaller, reusable compositions.
//
// 2. **No reimplementation**  -  if an op's IR contains a subgraph that
//    structurally matches another registered op, the author must call
//    that op via `Expr::Call` instead of inlining its logic.
//
// Together these gates enforce a composition ratchet: the op catalog
// grows organically, and every new composition automatically benefits
// every pipeline that calls it.

#[path = "composition_discipline__every_op_is_under_complexity_budget.rs"]
mod composition_discipline_every_op_is_under_complexity_budget;

use std::collections::BTreeSet;
use std::sync::LazyLock;

use vyre::ir::{Expr, Node, Program};

// ───────────────────────────────────────────────────────────────────
// Complexity measurement
// ───────────────────────────────────────────────────────────────────

/// Complexity stats for a single registered op.
#[derive(Debug, Clone, Copy)]
struct Complexity {
    /// Total number of IR statement nodes (recursive).
    total_nodes: usize,
    /// Total number of IR expression nodes (recursive).
    total_exprs: usize,
    /// Maximum nesting depth of control-flow nodes (If / Loop).
    max_depth: usize,
    /// Number of Loop nodes.
    loop_count: usize,
}

fn measure_program(program: &Program) -> Complexity {
    let mut stats = Complexity {
        total_nodes: 0,
        total_exprs: 0,
        max_depth: 0,
        loop_count: 0,
    };
    for node in program.entry() {
        measure_node(node, 0, &mut stats);
    }
    stats
}

fn measure_node(node: &Node, depth: usize, stats: &mut Complexity) {
    stats.total_nodes += 1;
    stats.max_depth = stats.max_depth.max(depth);
    match node {
        Node::Let { value, .. } => {
            count_expr(value, stats);
        }
        Node::Assign { value, .. } => {
            count_expr(value, stats);
        }
        Node::Store { index, value, .. } => {
            count_expr(index, stats);
            count_expr(value, stats);
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            count_expr(cond, stats);
            for n in then {
                measure_node(n, depth + 1, stats);
            }
            for n in otherwise {
                measure_node(n, depth + 1, stats);
            }
        }
        Node::Loop { from, to, body, .. } => {
            stats.loop_count += 1;
            count_expr(from, stats);
            count_expr(to, stats);
            for n in body {
                measure_node(n, depth + 1, stats);
            }
        }
        Node::Block(nodes) => {
            for n in nodes {
                measure_node(n, depth, stats);
            }
        }
        Node::Region {
            generator,
            source_region,
            body,
        } => {
            if is_child_composition(
                generator.as_str(),
                source_region.as_ref().map(|r| r.as_str()),
            ) {
                return;
            }
            for n in body.iter() {
                measure_node(n, depth, stats);
            }
        }
        Node::Return
        | Node::Barrier {
            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
        }
        | Node::IndirectDispatch { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncWait { .. }
        | Node::Opaque(_) => {}
        // Non-exhaustive variants are leaf nodes for this structural budget.
        _ => {}
    }
}

fn count_expr(expr: &Expr, stats: &mut Complexity) {
    stats.total_exprs += 1;
    match expr {
        Expr::BinOp { left, right, .. } => {
            count_expr(left, stats);
            count_expr(right, stats);
        }
        Expr::UnOp { operand, .. } => {
            count_expr(operand, stats);
        }
        Expr::Load { index, .. } => {
            count_expr(index, stats);
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            count_expr(cond, stats);
            count_expr(true_val, stats);
            count_expr(false_val, stats);
        }
        Expr::Cast { value, .. } => {
            count_expr(value, stats);
        }
        Expr::Fma { a, b, c } => {
            count_expr(a, stats);
            count_expr(b, stats);
            count_expr(c, stats);
        }
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            count_expr(index, stats);
            if let Some(exp) = expected {
                count_expr(exp, stats);
            }
            count_expr(value, stats);
        }
        Expr::SubgroupBallot { cond } => {
            count_expr(cond, stats);
        }
        Expr::SubgroupShuffle { value, lane } => {
            count_expr(value, stats);
            count_expr(lane, stats);
        }
        Expr::SubgroupReduce { value, .. } => {
            count_expr(value, stats);
        }
        Expr::Call { args, .. } => {
            for arg in args {
                count_expr(arg, stats);
            }
        }
        // Leaf expressions
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::BufLen { .. }
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::Opaque(_) => {}
        // Non-exhaustive variants are leaf exprs for this structural budget.
        _ => {}
    }
}

// ───────────────────────────────────────────────────────────────────
// Structural fingerprinting for subsumption detection
// ───────────────────────────────────────────────────────────────────

/// Hash the structural "shape" of a program's entry nodes, ignoring local
/// binding names while preserving literal values, buffer roles, op targets,
/// and child-composition bodies. Two ops with isomorphic control flow and
/// identical semantic constants produce the same fingerprint.
fn structural_fingerprint(program: &Program) -> u64 {
    let mut hasher = 0u64;
    for node in program.entry() {
        hash_node(node, &mut hasher);
    }
    hasher
}

fn hash_node(node: &Node, h: &mut u64) {
    match node {
        Node::Let { value, .. } => {
            mix(h, 1);
            hash_expr(value, h);
        }
        Node::Assign { value, .. } => {
            mix(h, 2);
            hash_expr(value, h);
        }
        Node::Store {
            buffer,
            index,
            value,
        } => {
            mix(h, 3);
            hash_str(buffer.as_str(), h);
            hash_expr(index, h);
            hash_expr(value, h);
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            mix(h, 4);
            hash_expr(cond, h);
            mix(h, then.len() as u64);
            for n in then {
                hash_node(n, h);
            }
            mix(h, otherwise.len() as u64);
            for n in otherwise {
                hash_node(n, h);
            }
        }
        Node::Loop { from, to, body, .. } => {
            mix(h, 5);
            hash_expr(from, h);
            hash_expr(to, h);
            mix(h, body.len() as u64);
            for n in body {
                hash_node(n, h);
            }
        }
        Node::Block(nodes) => {
            mix(h, 6);
            for n in nodes {
                hash_node(n, h);
            }
        }
        Node::Region {
            generator,
            source_region,
            body,
        } => {
            mix(h, 7);
            if is_child_composition(
                generator.as_str(),
                source_region.as_ref().map(|r| r.as_str()),
            ) {
                hash_str(generator.as_str(), h);
            }
            for n in body.iter() {
                hash_node(n, h);
            }
        }
        Node::Return => mix(h, 10),
        Node::Barrier {
            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
        } => mix(h, 11),
        Node::IndirectDispatch { .. } => mix(h, 12),
        Node::AsyncLoad { .. } => mix(h, 13),
        Node::AsyncWait { .. } => mix(h, 14),
        Node::Opaque(_) => mix(h, 15),
        _ => mix(h, 16),
    }
}

fn hash_expr(expr: &Expr, h: &mut u64) {
    match expr {
        Expr::LitU32(value) => {
            mix(h, 100);
            mix(h, u64::from(*value));
        }
        Expr::LitI32(value) => {
            mix(h, 101);
            mix(h, *value as u32 as u64);
        }
        Expr::LitF32(value) => {
            mix(h, 102);
            mix(h, u64::from(value.to_bits()));
        }
        Expr::LitBool(value) => {
            mix(h, 103);
            mix(h, u64::from(*value));
        }
        Expr::Var(_) => mix(h, 104),
        Expr::Load { buffer, index } => {
            mix(h, 105);
            for byte in buffer.as_str().bytes() {
                mix(h, byte as u64);
            }
            hash_expr(index, h);
        }
        Expr::BufLen { buffer } => {
            mix(h, 106);
            for byte in buffer.as_str().bytes() {
                mix(h, byte as u64);
            }
        }
        Expr::InvocationId { .. } => mix(h, 107),
        Expr::WorkgroupId { .. } => mix(h, 108),
        Expr::LocalId { .. } => mix(h, 109),
        Expr::BinOp { op, left, right } => {
            mix(h, 110);
            // Hash the full operator name to distinguish Add from Mul etc.
            for byte in format!("{op:?}").bytes() {
                mix(h, byte as u64);
            }
            hash_expr(left, h);
            hash_expr(right, h);
        }
        Expr::UnOp { op, operand } => {
            mix(h, 111);
            for byte in format!("{op:?}").bytes() {
                mix(h, byte as u64);
            }
            hash_expr(operand, h);
        }
        Expr::Call { op_id, args } => {
            // CRITIQUE_CONFORM_2026-04-23 H7: hashing only the
            // discriminant + arity collapsed every `Expr::Call` with
            // the same arg count into a single fingerprint. An
            // attacker could trivially craft a call to op `b` whose
            // structural hash matched a call to op `a`, bypassing the
            // cross-namespace subsumption gate. Recurse into every
            // arg and mix the op_id bytes so distinct calls produce
            // distinct fingerprints.
            mix(h, 112);
            for b in op_id.as_bytes() {
                mix(h, u64::from(*b));
            }
            mix(h, args.len() as u64);
            for arg in args {
                hash_expr(arg, h);
            }
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            mix(h, 113);
            hash_expr(cond, h);
            hash_expr(true_val, h);
            hash_expr(false_val, h);
        }
        Expr::Cast { target, value } => {
            mix(h, 114);
            for byte in format!("{target:?}").bytes() {
                mix(h, byte as u64);
            }
            hash_expr(value, h);
        }
        Expr::Fma { a, b, c } => {
            mix(h, 115);
            hash_expr(a, h);
            hash_expr(b, h);
            hash_expr(c, h);
        }
        Expr::Atomic {
            op,
            index,
            expected,
            value,
            ..
        } => {
            mix(h, 116);
            for byte in format!("{op:?}").bytes() {
                mix(h, byte as u64);
            }
            hash_expr(index, h);
            if let Some(exp) = expected {
                hash_expr(exp, h);
            }
            hash_expr(value, h);
        }
        Expr::SubgroupBallot { cond } => {
            mix(h, 117);
            hash_expr(cond, h);
        }
        Expr::SubgroupShuffle { value, lane } => {
            mix(h, 118);
            hash_expr(value, h);
            hash_expr(lane, h);
        }
        Expr::SubgroupReduce { value, .. } => {
            mix(h, 119);
            hash_expr(value, h);
        }
        Expr::Opaque(_) => mix(h, 199),
        // Future-proof: unknown variants get a unique tag.
        _ => mix(h, 200),
    }
}

/// Every operation id the registry carries, which is the only vocabulary a
/// region may name to be treated as a call to somebody else's operation.
///
/// Every tier counts, not just the library one. A composition is free to
/// delegate to a foundation or intrinsic operation, and charging it for that
/// body would push the author back toward inlining the very thing the budget
/// wants factored out. `Unknown` is excluded because it means the id matched
/// no accepted namespace, which is not a registration.
static REGISTERED_OP_IDS: LazyLock<BTreeSet<&'static str>> = LazyLock::new(|| {
    vyre_foundation::operation::OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier != vyre_foundation::operation::OperationTier::Unknown)
        .map(|entry| entry.id)
        .collect()
});

/// True when a region delegates to another registered operation.
///
/// The exemption exists so a caller is not charged for a building block that
/// carries its own budget row. That makes the generator the deciding field:
/// it names the operation being invoked, while `source_region` only records
/// the parent that reparented the region. Composition stamps `source_region`
/// onto every entry region it moves, anonymous ones included, so reading it
/// as consent let any operation drop under any cap by wrapping its own body
/// in a region with an invented name, leaving the program byte-identical.
///
/// A generator earns the exemption only by naming an operation the catalog
/// actually registers. `source_region` is still required, because a root
/// region names its own operation and must never exempt that operation's
/// entire body from its own budget.
fn is_child_composition(generator: &str, source_region: Option<&str>) -> bool {
    source_region.is_some()
        && !vyre_foundation::composition::is_anonymous_generator(generator)
        && REGISTERED_OP_IDS.contains(generator)
}

fn hash_str(value: &str, h: &mut u64) {
    for byte in value.as_bytes() {
        mix(h, u64::from(*byte));
    }
}

/// FNV-1a–style mixer.
fn mix(h: &mut u64, v: u64) {
    *h ^= v;
    *h = h.wrapping_mul(0x100000001b3);
}


// ───────────────────────────────────────────────────────────────────
// Gate integrity: the budget exemption must not be self-serve
// ───────────────────────────────────────────────────────────────────

/// A loop nest with enough shape to register on every budget axis.
fn budgeted_body() -> Vec<Node> {
    vec![Node::Loop {
        var: vyre::ir::Ident::from("i"),
        from: Expr::u32(0),
        to: Expr::u32(8),
        body: vec![Node::Let {
            name: vyre::ir::Ident::from("t"),
            value: Expr::u32(1),
        }],
    }]
}

fn measure_entry(entry: Vec<Node>) -> Complexity {
    measure_program(&Program::wrapped(Vec::new(), [1, 1, 1], entry))
}

/// WHY: the exemption exists so a region delegating to another registered op
/// is measured under that op's own budget row instead of twice. It said only
/// `source_region.is_some()`, and composition stamps `source_region` onto
/// every region it reparents, so any op could drop under any cap by wrapping
/// its body in a region with an invented generator. The program is then
/// byte-identical and only the attribution moved, which is a falsified gate
/// rather than a fix. Complexity must be insensitive to a wrapper that names
/// no registered operation.
///
/// Does not catch: a wrapper that names a *real* registered id while holding
/// a body that operation does not contain. Nothing checks that today. The
/// subsumption gate compares whole operations pairwise by fingerprint, so it
/// never inspects whether a region's body matches the operation it names.
/// Closing that needs a body-to-operation comparison this budget does not do.
#[test]
fn an_unregistered_child_region_does_not_hide_complexity() {
    let plain = measure_entry(budgeted_body());
    let wrapped = measure_entry(vec![vyre_foundation::composition::wrap_child_region(
        "vyre-libs::graph::dominator_frontier::pred_dominance",
        vyre::ir::Ident::from("graph.dominator_frontier"),
        budgeted_body(),
    )]);

    assert!(
        plain.loop_count > 0 && plain.total_nodes > 2,
        "Fix: the fixture must carry real complexity, measured {plain:?}",
    );
    assert_eq!(
        wrapped.loop_count, plain.loop_count,
        "an unregistered wrapper hid {} of {} loops",
        plain.loop_count - wrapped.loop_count,
        plain.loop_count,
    );
    // Both programs carry exactly one enclosing region over the same body:
    // `Program::wrapped` adds a root region to the bare nodes, and leaves the
    // already region-chained entry alone. So an honest measure counts the
    // same nodes either way.
    assert_eq!(
        wrapped.total_nodes, plain.total_nodes,
        "an unregistered wrapper must not erase the body it encloses",
    );
    assert_eq!(
        wrapped.max_depth, plain.max_depth,
        "an unregistered wrapper must not hide the nesting it encloses",
    );
}

/// WHY: the tightened predicate must still exempt genuine reuse, or every op
/// that calls a registered building block pays for it twice and the gate
/// pushes authors back toward monoliths.
#[test]
fn a_registered_child_region_is_still_exempt() {
    let registered = vyre_libs::operation_catalog::all_entries()
        .next()
        .expect("Fix: the operation catalog must register at least one op");

    let wrapped = measure_entry(vec![vyre_foundation::composition::wrap_child_region(
        registered.id,
        vyre::ir::Ident::from("some.calling.op"),
        budgeted_body(),
    )]);

    assert_eq!(
        wrapped.loop_count, 0,
        "delegating to registered op `{}` must not charge its loops twice",
        registered.id,
    );
}

/// WHY: an anonymous generator is minted by composition itself for a body
/// that has no operation behind it. It carries a `source_region` like any
/// other reparented region, so it must be counted, not exempted.
#[test]
fn an_anonymous_child_region_does_not_hide_complexity() {
    for prefix in vyre_foundation::composition::ANONYMOUS_GENERATOR_PREFIXES {
        let wrapped = measure_entry(vec![vyre_foundation::composition::wrap_child_region(
            &format!("{prefix}graph.dominator_frontier"),
            vyre::ir::Ident::from("graph.dominator_frontier"),
            budgeted_body(),
        )]);
        assert_eq!(
            wrapped.loop_count, 1,
            "generator prefix `{prefix}` names no operation and must be counted",
        );
    }
}