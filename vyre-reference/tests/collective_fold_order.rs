//! A subgroup reduction's lane fold order belongs to the explored schedule.
//!
//! # WHY
//!
//! A subgroup reduction states no lane order. Hardware picks one, and a
//! shuffle-down tree is the common pick, so any single fold order is one
//! answer out of several a conforming device may produce. The interpreter
//! folded ascending lane index under every explored order, which made the
//! oracle certify that one association as the expected output: a device that
//! reduced in a tree was graded wrong for being right, and a program whose
//! result genuinely depends on the association was certified as though it had
//! one answer.
//!
//! Race exploration permutes the workgroup list and the invocation list to
//! surface order dependence. The reduction fold was the one ordered step it
//! did not reach, which is why `oracle_race_exploration` covers lane,
//! workgroup and device scope and covers subgroup scope nowhere.
//!
//! # What these hold
//!
//! The operator set comes from `SubgroupReduceOp::ALL` at run time, so an
//! operator added tomorrow is judged without editing this file.
//! `SubgroupReduceOp::is_f32_order_independent` is the only claim about which
//! operators may disagree, and both directions are checked: an operator it
//! calls order-independent must agree across every explored order, and an
//! operator it calls order-dependent must be reachable as a disagreement. A
//! new operator has no answer recorded there, so the spec crate's own match
//! fails to compile rather than this file assuming either way.
//!
//! What this does not catch: the tie between `-0.0` and `+0.0` under `Min`
//! and `Max`, which is a property of the tie-break rather than of the
//! association, and a device that folds in an order no `LaneOrder` names.
#![cfg(feature = "subgroup-ops")]

use crate::lane_collectives;

use lane_collectives::{lane_program, shuffle_values_by};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, SubgroupReduceOp,
};
use vyre_reference::value::Value;
use vyre_reference::{DeterministicSchedulePolicy, ReferenceRequest};

/// One subgroup's worth of lanes, so every lane folds over the same window.
const LANES: u32 = 32;

/// Rotations exercised alongside forward and reversed order.
const ROTATIONS: [u32; 3] = [1, 7, 31];

/// Every explored step order, forward first.
fn every_policy() -> Vec<(String, DeterministicSchedulePolicy)> {
    let mut policies = vec![
        ("forward".to_string(), DeterministicSchedulePolicy::Forward),
        (
            "reversed".to_string(),
            DeterministicSchedulePolicy::LaneReversed,
        ),
        (
            "bounded interleaving".to_string(),
            DeterministicSchedulePolicy::BoundedInterleaving,
        ),
    ];
    policies.extend(ROTATIONS.into_iter().map(|by| {
        (
            format!("rotated by {by}"),
            DeterministicSchedulePolicy::LaneRotated(by),
        )
    }));
    policies
}

/// `out[i] = subgroupReduce(op, in[i])` over one subgroup.
///
/// Every lane stores the reduction to its own slot, so the output bytes are
/// the reduction itself rather than a race between lanes.
fn reducing_program(op: SubgroupReduceOp, ty: DataType) -> Program {
    lane_program(
        LANES,
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, ty.clone()).with_count(LANES),
            BufferDecl::output("out", 1, ty).with_count(LANES),
        ],
        vec![
            Node::let_bind(
                "folded",
                Expr::SubgroupReduce {
                    op,
                    value: Box::new(Expr::load("in", Expr::var("idx"))),
                },
            ),
            Node::store("out", Expr::var("idx"), Expr::var("folded")),
        ],
    )
}

fn pack_u32(words: &[u32]) -> Value {
    Value::from(vyre_primitives::wire::pack_u32_slice(words))
}

fn pack_f32(lanes: &[f32]) -> Value {
    Value::from(vyre_primitives::wire::pack_f32_slice(lanes))
}

/// Raw output bytes of the single declared output buffer.
fn output_bytes(
    program: &Program,
    inputs: &[Value],
    policy: DeterministicSchedulePolicy,
) -> Vec<u8> {
    ReferenceRequest::standard(program, inputs)
        .with_schedule_policy(policy)
        .outputs()
        .expect("the reducing program evaluates under every explored policy")
        .first()
        .expect("the program declares one output buffer")
        .to_bytes()
        .to_vec()
}

/// Integer lanes with bits set across the whole word, so a bitwise, a
/// min/max and a wrapping arithmetic reduction all produce distinct values.
fn integer_lanes() -> Vec<u32> {
    (0..LANES)
        .map(|lane| 0x0F0F_0F0F ^ lane.wrapping_mul(0x9E37_79B9))
        .collect()
}

/// f32 lanes whose fold result differs between ascending and descending lane
/// order under `op`, or `None` for an operator whose association holds.
///
/// A witness has to be built per operator: one input cannot expose two
/// different roundings. Both below are exact, not tuned to a host.
///
/// `Add`: `1e9` is exactly representable and its ULP is 64, so adding `8.0`
/// to it rounds straight back down and the 31 small lanes vanish one at a
/// time. Folded together first they reach 248, which is past half an ULP, so
/// the descending order keeps them and the two orders differ by one ULP.
///
/// `Mul`: `1e30` squared is past `f32::MAX` and `1e-30` squared is past the
/// smallest subnormal, so the ascending order saturates to infinity on its
/// second step and the descending order collapses to zero before it ever
/// reaches a large lane.
fn f32_association_witness(op: SubgroupReduceOp) -> Option<Vec<f32>> {
    let lanes = LANES as usize;
    match op {
        SubgroupReduceOp::Add => {
            let mut values = vec![8.0_f32; lanes];
            values[0] = 1.0e9;
            Some(values)
        }
        SubgroupReduceOp::Mul => {
            let mut values = vec![1.0e-30_f32; lanes];
            values[0] = 1.0e30;
            values[1] = 1.0e30;
            Some(values)
        }
        _ => None,
    }
}

/// f32 lanes for the operators whose association holds, where the magnitudes
/// still span enough range that a reassociating fold would be visible.
fn spread_lanes() -> Vec<f32> {
    (0..LANES)
        .map(|lane| {
            let magnitude = 1.0e-6_f32 * 10.0_f32.powi(i32::try_from(lane % 12).unwrap_or(0));
            if lane % 2 == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect()
}

/// An operator the spec calls order-independent agrees under every order.
///
/// WHY: permuting the fold must not invent a disagreement. This is the
/// direction that keeps the oracle from rejecting a correct program. Every
/// integer operator is covered, bitwise included, plus the f32 operators the
/// spec records as associative.
#[test]
fn an_order_independent_reduction_agrees_under_every_explored_order() {
    let integers = pack_u32(&integer_lanes());
    let floats = pack_f32(&spread_lanes());
    for op in SubgroupReduceOp::ALL {
        // Integer reduction is order-independent for every operator.
        let program = reducing_program(op, DataType::U32);
        let inputs = [integers.clone()];
        let baseline = output_bytes(&program, &inputs, DeterministicSchedulePolicy::Forward);
        for (label, policy) in every_policy() {
            assert_eq!(
                output_bytes(&program, &inputs, policy),
                baseline,
                "the u32 {op:?} reduction disagreed between forward and {label} step order. \
                 Every integer reduction is associative and commutative, so no fold order may \
                 change it."
            );
        }

        // f32 only where the spec records that every order agrees.
        if op.is_f32_order_independent() != Some(true) {
            continue;
        }
        let program = reducing_program(op, DataType::F32);
        let inputs = [floats.clone()];
        let baseline = output_bytes(&program, &inputs, DeterministicSchedulePolicy::Forward);
        for (label, policy) in every_policy() {
            assert_eq!(
                output_bytes(&program, &inputs, policy),
                baseline,
                "the f32 {op:?} reduction disagreed between forward and {label} step order, and \
                 is_f32_order_independent records it as order-independent. Fix: correct the \
                 record, or the fold, so the two agree."
            );
        }
    }
}

/// An operator the spec calls order-dependent is reached by the exploration.
///
/// WHY: this is the regression. Before the fold followed the explored
/// schedule, every order produced the ascending-lane association, so an f32
/// `Add` or `Mul` reduction agreed with itself under every policy and the
/// oracle certified one association as the expected output. The assertion is
/// that some explored order disagrees with forward, which is what makes the
/// dependence visible to a caller instead of certified.
#[test]
fn an_order_dependent_reduction_disagrees_under_some_explored_order() {
    let dependent: Vec<SubgroupReduceOp> = SubgroupReduceOp::ALL
        .into_iter()
        .filter(|op| op.is_f32_order_independent() == Some(false))
        .collect();
    assert!(
        !dependent.is_empty(),
        "no operator is recorded as order-dependent over f32, so this proof holds nothing. f32 \
         addition and multiplication are not associative."
    );
    for op in dependent {
        let witness = f32_association_witness(op).unwrap_or_else(|| {
            panic!(
                "is_f32_order_independent records {op:?} as order-dependent over f32 and no \
                 witness input exposes it, so nothing here proves the exploration reaches it. \
                 Fix: add lanes to f32_association_witness whose fold differs between ascending \
                 and descending order under {op:?}."
            )
        });
        let program = reducing_program(op, DataType::F32);
        let inputs = [pack_f32(&witness)];
        let baseline = output_bytes(&program, &inputs, DeterministicSchedulePolicy::Forward);
        let disagreed = every_policy()
            .into_iter()
            .any(|(_, policy)| output_bytes(&program, &inputs, policy) != baseline);
        assert!(
            disagreed,
            "every explored order produced the same f32 {op:?} reduction, so the fold order does \
             not follow the explored schedule and the oracle is certifying one association out \
             of the several a device may produce."
        );
    }
}

/// A lane-addressed collective is unchanged by every explored order.
///
/// WHY: the fold permutation must stay on the reduction. Permuting a gather a
/// ballot or a shuffle reads would move a ballot bit or a shuffle source,
/// which changes what the program means rather than which association it
/// takes. Each of the three gathers is addressed here: lane `i` requests
/// source lane `LANES - 1 - i`, so permuting the selectors changes which lane
/// each one asks for; the values come from a buffer indexed by lane, so
/// permuting them changes what is found there; and the ballot predicate is
/// true only below lane 8, so permuting the predicates changes the mask.
#[test]
fn a_lane_addressed_collective_is_unchanged_by_every_explored_order() {
    let program = lane_program(
        LANES,
        vec![
            BufferDecl::storage("values", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(LANES),
            BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
        ],
        vec![
            shuffle_values_by(
                "shuffled",
                Expr::sub(Expr::u32(LANES - 1), Expr::var("idx")),
            ),
            Node::let_bind(
                "mask",
                Expr::SubgroupBallot {
                    cond: Box::new(Expr::lt(Expr::var("idx"), Expr::u32(8))),
                },
            ),
            Node::store(
                "out",
                Expr::var("idx"),
                Expr::add(Expr::var("shuffled"), Expr::var("mask")),
            ),
        ],
    );
    let inputs = [pack_u32(&integer_lanes())];
    let baseline = output_bytes(&program, &inputs, DeterministicSchedulePolicy::Forward);
    for (label, policy) in every_policy() {
        assert_eq!(
            output_bytes(&program, &inputs, policy),
            baseline,
            "a shuffle and a ballot addressed by lane identity changed under {label} step order, \
             so the fold permutation reached a gather that addresses lanes rather than the \
             reduction fold it belongs to."
        );
    }
}
