//! A subgroup collective resolves its peers by LANE INDEX, never by step position.
//!
//! The interpreter captures every lane's locals once per round-robin sweep so a
//! collective can read its peers. `subgroup_slice` carves the subgroup window out of
//! that vector POSITIONALLY, and the shuffle addresses a lane inside the window by
//! `linear_local_index % subgroup_width`, so position in the vector IS lane identity.
//! The vector was captured in STEP order, which a non-forward schedule permutes, so
//! every collective read the wrong lanes under a permuted schedule: a ballot returned
//! the mask of a different subgroup, a shuffle sourced a different lane.
//!
//! That is a change in RESULT, not in scheduling. A ballot and a shuffle are defined
//! over lane identity, so their outputs must be identical under every legal schedule,
//! and the whole point of the reversed/rotated schedules is to expose a program whose
//! result is schedule-dependent. An oracle that is itself schedule-dependent reports
//! the interpreter's own defect as a defect in the program under test.
//!
//! Coverage: reversed AND rotated step orders. Reversal is a symmetric permutation, so
//! an implementation that maps lane identity onto step position can be repaired into
//! reversal symmetry and stay wrong; the rotations pin lane identity for real. Each
//! case is compared against a hand-computed oracle as well as across schedules, so an
//! interpreter that is consistently wrong in every schedule still fails.
#![cfg(feature = "subgroup-ops")]

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::value::Value;

/// One 64-lane workgroup, so the dispatch spans two 32-lane subgroups and a window
/// resolved by step position lands on the neighbouring subgroup rather than merely
/// permuting its own.
const LANES: u32 = 64;
const SUBGROUP_WIDTH: u32 = 32;

/// Rotations exercised alongside forward and reversed order: inside a subgroup, across
/// its boundary, and past the whole workgroup.
const ROTATIONS: [u32; 4] = [1, 7, 31, 33];

fn pack(words: &[u32]) -> Value {
    Value::from(
        words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
}

fn unpack(values: &[Value]) -> Vec<u32> {
    let bytes = values
        .first()
        .expect("the program declares one output buffer")
        .to_bytes();
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Run one program under forward, reversed and every rotated step order and return the
/// labelled outputs, forward first.
fn under_every_step_order(program: &Program, inputs: &[Value]) -> Vec<(String, Vec<u32>)> {
    let mut runs = vec![(
        "forward".to_string(),
        unpack(
            &vyre_reference::reference_eval(program, inputs)
                .expect("forward evaluation must succeed"),
        ),
    )];
    runs.push((
        "reversed".to_string(),
        unpack(
            &vyre_reference::reference_eval_lane_reversed(program, inputs)
                .expect("reversed evaluation must succeed"),
        ),
    ));
    for by in ROTATIONS {
        runs.push((
            format!("rotated by {by}"),
            unpack(
                &vyre_reference::reference_eval_lane_rotated(program, inputs, by)
                    .expect("rotated evaluation must succeed"),
            ),
        ));
    }
    runs
}

fn assert_every_step_order_matches(
    program: &Program,
    inputs: &[Value],
    expected: &[u32],
    what: &str,
) {
    for (label, produced) in under_every_step_order(program, inputs) {
        assert_eq!(
            produced, expected,
            "Fix: {what} disagreed with its lane-indexed oracle in {label} step order. A \
             subgroup collective is defined over lane identity, so the interpreter must \
             resolve a lane's peers by lane index, not by the position the schedule \
             happens to step them in."
        );
    }
}

/// `out[i] = subgroupBallot(cond[i] == 1)`, no guards: every lane is in bounds.
fn ballot_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("cond", 0, BufferAccess::ReadOnly, DataType::U32).with_count(LANES),
            BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
        ],
        [LANES, 1, 1],
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::let_bind(
                "mask",
                Expr::SubgroupBallot {
                    cond: Box::new(Expr::eq(Expr::load("cond", Expr::var("idx")), Expr::u32(1))),
                },
            ),
            Node::store("out", Expr::var("idx"), Expr::var("mask")),
        ],
    )
}

/// `out[i] = subgroupShuffle(values[i], lanes[i])`, no guards.
fn shuffle_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("values", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(LANES),
            BufferDecl::storage("lanes", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(LANES),
            BufferDecl::output("out", 2, DataType::U32).with_count(LANES),
        ],
        [LANES, 1, 1],
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::let_bind(
                "shuffled",
                Expr::SubgroupShuffle {
                    value: Box::new(Expr::load("values", Expr::var("idx"))),
                    lane: Box::new(Expr::load("lanes", Expr::var("idx"))),
                },
            ),
            Node::store("out", Expr::var("idx"), Expr::var("shuffled")),
        ],
    )
}

/// The predicate is true only on the first eight lanes, so the two subgroups carry
/// DIFFERENT masks and no rotation or reversal of the lane window reproduces either.
#[test]
fn ballot_reports_its_own_subgroups_predicate_in_every_step_order() {
    let cond: Vec<u32> = (0..LANES).map(|lane| u32::from(lane < 8)).collect();
    let expected: Vec<u32> = (0..LANES)
        .map(|lane| {
            let start = (lane / SUBGROUP_WIDTH) * SUBGROUP_WIDTH;
            (0..SUBGROUP_WIDTH)
                .filter(|bit| cond[(start + bit) as usize] == 1)
                .fold(0u32, |mask, bit| mask | (1u32 << bit))
        })
        .collect();
    assert_eq!(
        (expected[0], expected[63]),
        (0x0000_00FF, 0),
        "the fixture must give the two subgroups different masks, or the test cannot \
         tell a lane window apart from its neighbour"
    );

    assert_every_step_order_matches(
        &ballot_program(),
        &[pack(&cond), pack(&vec![0u32; LANES as usize])],
        &expected,
        "subgroup ballot",
    );
}

/// Every lane reads its right-hand neighbour inside its own subgroup, wrapping at the
/// subgroup boundary. A rotated or reversed lane window shifts the whole answer.
#[test]
fn shuffle_sources_the_requested_lane_in_every_step_order() {
    let values: Vec<u32> = (0..LANES).map(|lane| lane * 10 + 1).collect();
    let lanes: Vec<u32> = (0..LANES).map(|lane| (lane + 1) % SUBGROUP_WIDTH).collect();
    let expected: Vec<u32> = (0..LANES)
        .map(|lane| {
            let start = (lane / SUBGROUP_WIDTH) * SUBGROUP_WIDTH;
            values[(start + lanes[lane as usize]) as usize]
        })
        .collect();
    assert_eq!(
        (expected[0], expected[31], expected[32]),
        (values[1], values[0], values[33]),
        "the fixture must wrap inside each subgroup, or the test cannot tell a \
         subgroup-relative source lane from a global one"
    );

    assert_every_step_order_matches(
        &shuffle_program(),
        &[
            pack(&values),
            pack(&lanes),
            pack(&vec![0u32; LANES as usize]),
        ],
        &expected,
        "subgroup shuffle",
    );
}
