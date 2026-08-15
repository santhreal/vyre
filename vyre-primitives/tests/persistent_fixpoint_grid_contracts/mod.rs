use vyre_foundation::ir::{AtomicOp, Expr, MemoryOrdering, Node, Program};
use vyre_foundation::visit::any_descendant;

use vyre_primitives::fixpoint::persistent_fixpoint::{
    cpu_ref, persistent_fixpoint, persistent_fixpoint_grid, OP_ID_GRID,
    PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
};
use vyre_reference::value::Value;

/// Lane index every emitted body is indexed by.
fn lane() -> Expr {
    Expr::InvocationId { axis: 0 }
}

/// The top-level wave list: the body of the single generator `Region`
/// `Program::wrapped` puts at the entry.
fn wave_nodes(program: &Program) -> &[Node] {
    match program.entry() {
        [Node::Region {
            generator, body, ..
        }] => {
            assert_eq!(
                generator.as_str(),
                OP_ID_GRID,
                "the grid builder must attribute its region to its own op id"
            );
            body
        }
        other => panic!("expected exactly one generator Region at the entry, got {other:?}"),
    }
}

/// Count every node in the whole program tree matching `pred`.
fn count_nodes<P>(program: &Program, mut pred: P) -> usize
where
    P: FnMut(&Node) -> bool,
{
    let mut total = 0usize;
    for node in program.entry() {
        // `any_descendant` short-circuits on `true`, so the predicate
        // always answers `false` and the walk becomes an exhaustive
        // preorder visit that tallies instead of searching.
        let _ = any_descendant(node, &mut |candidate| {
            if pred(candidate) {
                total += 1;
            }
            false
        });
    }
    total
}

/// Every `atomic_or` index applied to `buffer`, in visitation order.
///
/// The builder emits the flag set as `Node::Let { value: Expr::Atomic { .. } }`,
/// so matching the `Let` value directly is exact for this program shape.
fn atomic_or_indices(program: &Program, buffer: &str) -> Vec<u32> {
    let mut indices = Vec::new();
    for node in program.entry() {
        let _ = any_descendant(node, &mut |candidate| {
            if let Node::Let {
                value:
                    Expr::Atomic {
                        op: AtomicOp::Or,
                        buffer: target,
                        index,
                        ..
                    },
                ..
            } = candidate
            {
                if target.as_str() == buffer {
                    match index.as_ref() {
                        Expr::LitU32(word) => indices.push(*word),
                        other => panic!("flag index must be a literal word, got {other:?}"),
                    }
                }
            }
            false
        });
    }
    indices
}

/// Elementwise `next[t] = current[t]`: a fixpoint on the first wave.
fn identity_body(words: u32) -> Vec<Node> {
    vec![Node::if_then(
        Expr::lt(lane(), Expr::u32(words)),
        vec![Node::store("next", lane(), Expr::load("current", lane()))],
    )]
}

/// Elementwise `next[t] = current[t] | mask`: one changing wave, then flat.
fn or_const_body(words: u32, mask: u32) -> Vec<Node> {
    vec![Node::if_then(
        Expr::lt(lane(), Expr::u32(words)),
        vec![Node::store(
            "next",
            lane(),
            Expr::bitor(Expr::load("current", lane()), Expr::u32(mask)),
        )],
    )]
}

/// Elementwise `next[t] = current[t] | (current[t] << 8)`: several
/// changing waves before the fixpoint, within one word.
fn shift_or_body(words: u32) -> Vec<Node> {
    vec![Node::if_then(
        Expr::lt(lane(), Expr::u32(words)),
        vec![Node::store(
            "next",
            lane(),
            Expr::bitor(
                Expr::load("current", lane()),
                Expr::shl(Expr::load("current", lane()), Expr::u32(8)),
            ),
        )],
    )]
}

/// `next[0] = current[0]`, `next[t] = current[t] | current[t - 1]`: a
/// carry that walks ACROSS words, so lane `t` reads a word lane `t - 1`
/// wrote in the previous wave. This is the shape that actually depends on
/// the inter-wave barrier; without it the carry skips or doubles.
fn carry_body(words: u32) -> Vec<Node> {
    vec![Node::if_then(
        Expr::lt(lane(), Expr::u32(words)),
        vec![
            Node::if_then(
                Expr::eq(lane(), Expr::u32(0)),
                vec![Node::store("next", lane(), Expr::load("current", lane()))],
            ),
            // `current[t - 1]` is only ever evaluated for `t > 0`, so the
            // index never underflows into an out-of-bounds read.
            Node::if_then(
                Expr::gt(lane(), Expr::u32(0)),
                vec![Node::store(
                    "next",
                    lane(),
                    Expr::bitor(
                        Expr::load("current", lane()),
                        Expr::load("current", Expr::sub(lane(), Expr::u32(1))),
                    ),
                )],
            ),
        ],
    )]
}

fn pack(words: &[u32]) -> Value {
    Value::from(vyre_primitives::wire::pack_u32_slice(words))
}

fn unpack(value: &Value) -> Vec<u32> {
    value
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Dispatch outcome: the converged state in `current` and the raw
/// `changed` buffer.
struct Run {
    state: Vec<u32>,
    changed: Vec<u32>,
}

/// Workgroup and invocation STEP ORDER. Reversing it is identical for
/// any race-free program and diverges only where groups race, so a
/// forward-versus-reversed comparison is a deterministic race probe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Order {
    Forward,
    Reversed,
}

fn eval(program: &Program, inputs: &[Value], order: Order) -> Vec<Value> {
    match order {
        Order::Forward => vyre_reference::reference_eval(program, inputs),
        Order::Reversed => vyre_reference::reference_eval_lane_reversed(program, inputs),
    }
    .expect("reference evaluation must succeed")
}

fn run_grid_ordered(body: Vec<Node>, seed: &[u32], max_iterations: u32, order: Order) -> Run {
    let words = u32::try_from(seed.len()).expect("seed length must fit u32");
    let program =
        persistent_fixpoint_grid(body, "current", "next", "changed", words, max_iterations);
    let outputs = eval(
        &program,
        &[
            pack(seed),
            pack(&vec![0u32; seed.len()]),
            // Zero-filled, per the contract: the primitive never clears it.
            pack(&vec![0u32; max_iterations.max(1) as usize]),
        ],
        order,
    );
    Run {
        state: unpack(&outputs[0]),
        changed: unpack(&outputs[2]),
    }
}

fn run_workgroup_ordered(body: Vec<Node>, seed: &[u32], max_iterations: u32, order: Order) -> Run {
    let words = u32::try_from(seed.len()).expect("seed length must fit u32");
    let program = persistent_fixpoint(body, "current", "next", "changed", words, max_iterations);
    let outputs = eval(
        &program,
        &[pack(seed), pack(&vec![0u32; seed.len()]), pack(&[0u32])],
        order,
    );
    Run {
        state: unpack(&outputs[0]),
        changed: unpack(&outputs[2]),
    }
}

fn run_grid(body: Vec<Node>, seed: &[u32], max_iterations: u32) -> Run {
    run_grid_ordered(body, seed, max_iterations, Order::Forward)
}

fn run_workgroup(body: Vec<Node>, seed: &[u32], max_iterations: u32) -> Run {
    run_workgroup_ordered(body, seed, max_iterations, Order::Forward)
}

/// Iterations ENTERED, decoded from the per-iteration flag array: the
/// waves run as a prefix of ones, and the wave that reads zero is the one
/// that returns, so it counts too.
fn passes_from_flags(changed: &[u32], max_iterations: u32) -> u32 {
    changed
        .iter()
        .position(|word| *word == 0)
        .map_or(max_iterations, |first_zero| {
            u32::try_from(first_zero).expect("flag index must fit u32") + 1
        })
}

mod host_orchestration;
mod parity_and_races;
mod structure;
