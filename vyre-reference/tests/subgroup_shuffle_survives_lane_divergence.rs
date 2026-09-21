//! A subgroup shuffle inside a loop resolves every peer lane, whatever a
//! divergent block earlier in that loop cost one lane.
//!
//! The interpreter steps lanes round-robin, one node per lane per round, and
//! captures every lane's locals once at the start of a round so a collective
//! can read its peers. A branch whose condition is not lane-uniform gives one
//! lane more nodes to run than the others, so the lanes drift apart by that
//! many rounds and stay apart across the loop back-edge. A local bound inside
//! the loop body is unbound when the body's scope pops, so once the drift is
//! not a multiple of the body length, a lane that has popped the previous
//! iteration and not yet reached the binding again has no entry for it, and a
//! peer that reads it through `SubgroupShuffle` fails on a name the program
//! does bind.
//!
//! The drift is what decides it, so the divergent block is swept over one to
//! eight extra nodes: a single width lands on one alignment and passes while
//! the defect is intact.
#![cfg(feature = "subgroup-ops")]

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

const LANES: u32 = 32;
const ITERATIONS: u32 = 6;

/// `out[lane] = subgroupShuffle(v, 0)` from inside a loop, where `v` is bound
/// per iteration and lane one runs `extra_nodes` more nodes than every other
/// lane on each pass.
fn drifting_shuffle_program(extra_nodes: u32) -> Program {
    let lane = Expr::InvocationId { axis: 0 };
    let divergent = (0..extra_nodes)
        .map(|_| Node::store("out", lane.clone(), Expr::var("peer")))
        .collect::<Vec<_>>();
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(LANES)],
        [LANES, 1, 1],
        vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(ITERATIONS),
            vec![
                Node::let_bind("v", Expr::add(Expr::u32(1), Expr::var("i"))),
                Node::let_bind(
                    "peer",
                    Expr::SubgroupShuffle {
                        value: Box::new(Expr::var("v")),
                        lane: Box::new(Expr::u32(0)),
                    },
                ),
                Node::store("out", lane.clone(), Expr::var("peer")),
                Node::if_then(Expr::eq(lane.clone(), Expr::u32(1)), divergent),
            ],
        )],
    )
}

/// WHY: the reference interpreter is the parity oracle for every registered
/// composition, so a shuffle it cannot evaluate is reported as a defect in the
/// program under test. Lane drift across a loop back-edge made it refuse a
/// local the program binds on every iteration.
///
/// Does not catch a shuffle whose value is bound in a scope no lane reaches at
/// all; that is a real missing binding, not drift.
#[test]
fn a_shuffle_in_a_loop_resolves_every_peer_under_lane_drift() {
    for extra_nodes in 1..=8u32 {
        let program = drifting_shuffle_program(extra_nodes);
        let outputs = vyre_reference::ReferenceRequest::standard(&program, &[])
            .outputs()
            .unwrap_or_else(|error| {
                panic!(
                    "Fix: a shuffle of a loop-scoped local must resolve every peer lane; \
                     {extra_nodes} extra nodes on one lane made the oracle refuse it: {error:?}"
                )
            });
        let bytes = outputs
            .first()
            .expect("the program declares an out buffer")
            .to_bytes();
        let produced = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect::<Vec<_>>();
        assert_eq!(
            produced,
            vec![ITERATIONS; LANES as usize],
            "Fix: every lane must read lane zero's last-iteration value; {extra_nodes} \
             extra nodes on one lane changed the shuffled result"
        );
    }
}
