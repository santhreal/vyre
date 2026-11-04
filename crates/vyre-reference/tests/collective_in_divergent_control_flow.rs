//! A rendezvous reached from a branch whose condition is not uniform over that
//! rendezvous's own scope is rejected, not answered.
//!
//! The interpreter releases a lane holding for its peers once every live lane
//! has arrived, and a lane that took the other branch and retired is not live.
//! A rendezvous under a divergent branch therefore resolved over whichever
//! lanes were still running, and the oracle issued that partial result as the
//! expected output. No target defines that value: the lanes that skipped the
//! branch contribute on one and are undefined on another.
//!
//! The scope is the construct's own. A barrier synchronizes the workgroup, so
//! its condition must agree across every lane of the workgroup. A subgroup
//! collective reads only its own subgroup, so its condition must agree across
//! that subgroup and may differ between subgroups: a workgroup whose subgroups
//! each own one output, with the tail subgroups masked off, is the standard
//! shape and is legal.
#![cfg(feature = "subgroup-ops")]

use vyre_foundation::ir::{BufferDecl, DataType, Expr, MemoryOrdering, Node, Program};
use vyre_reference::ReferenceRequest;

const LANES: u32 = 4;

/// Lanes in one simulated subgroup.
const SUBGROUP_WIDTH: u32 = 32;

/// Lanes of the workgroup that splits on a subgroup boundary: two subgroups.
const SPLIT_LANES: u32 = SUBGROUP_WIDTH * 2;

/// Where in the `If` the collective sits.
///
/// The arm matters because a body the lane did not take is still a body whose
/// rendezvous the lanes that took it will wait at. The nesting matters because
/// the search has to reach through the statements between the branch and the
/// collective, which is where a non-recursive check passes every direct case
/// and misses every real program.
#[derive(Clone, Copy, Debug)]
struct Placement {
    /// True to put the collective in `otherwise` rather than `then`.
    otherwise: bool,
    /// True to wrap it in a loop inside that arm.
    nested: bool,
}

impl Placement {
    /// Every arm and nesting combination.
    const ALL: [Self; 4] = [
        Self {
            otherwise: false,
            nested: false,
        },
        Self {
            otherwise: false,
            nested: true,
        },
        Self {
            otherwise: true,
            nested: false,
        },
        Self {
            otherwise: true,
            nested: true,
        },
    ];
}

/// `if <cond> { .. } else { .. }` with `out[lane] = <collective>` at
/// `placement`.
///
/// `uniform` selects a condition every lane agrees on rather than one that
/// splits the workgroup. The other arm stores a constant, so both arms write
/// the output and only the collective distinguishes them.
fn guarded_collective(collective: Expr, uniform: bool, placement: Placement) -> Program {
    let lane = Expr::InvocationId { axis: 0 };
    let cond = if uniform {
        Expr::lt(lane.clone(), Expr::u32(LANES))
    } else {
        Expr::lt(lane.clone(), Expr::u32(2))
    };
    let mut collecting = vec![Node::store("out", lane.clone(), collective)];
    if placement.nested {
        collecting = vec![Node::loop_for("i", Expr::u32(0), Expr::u32(1), collecting)];
    }
    let plain = vec![Node::store("out", lane, Expr::u32(0))];
    let (then, otherwise) = if placement.otherwise {
        (plain, collecting)
    } else {
        (collecting, plain)
    };
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(LANES)],
        [LANES, 1, 1],
        vec![Node::if_then_else(cond, then, otherwise)],
    )
}

/// Every collective expression the interpreter treats as a peer rendezvous.
///
/// Built from the constructors rather than a literal list of variants, so a
/// collective added to the IR is added here by the same edit that teaches the
/// interpreter to hold on it.
fn collectives() -> Vec<(&'static str, Expr)> {
    let value = Expr::InvocationId { axis: 0 };
    vec![
        ("subgroup_add", Expr::subgroup_add(value.clone())),
        (
            "subgroup_shuffle",
            Expr::SubgroupShuffle {
                value: Box::new(value.clone()),
                lane: Box::new(Expr::u32(0)),
            },
        ),
        (
            "subgroup_ballot",
            Expr::SubgroupBallot {
                cond: Box::new(Expr::lt(value, Expr::u32(2))),
            },
        ),
    ]
}

/// WHY: the oracle answered a divergent collective with a reduction over the
/// lanes that happened to still be running. An oracle that issues a value no
/// target defines is worse than one that refuses: every backend that computes
/// something else is then reported as the defect.
///
/// Covers each collective expression, because the release path is shared and a
/// check written for one of them is a check written for none.
///
/// Does not catch divergence introduced by an early return or by a loop trip
/// count that differs across lanes.
#[test]
fn a_collective_under_a_divergent_branch_is_refused() {
    for (name, collective) in collectives() {
        for place in Placement::ALL {
            let program = guarded_collective(collective.clone(), false, place);
            let error = ReferenceRequest::standard(&program, &[])
                .outputs()
                .expect_err(&format!(
                    "Fix: `{name}` at {place:?} in a branch half the workgroup skips has no \
                     defined value; the oracle must refuse it instead of reducing over the \
                     surviving lanes"
                ));
            let text = format!("{error}");
            assert!(
                text.contains("uniform-control-flow"),
                "Fix: the refusal must name the rule it enforces; `{name}` at {place:?} \
                 reported {text}"
            );
            assert!(
                text.contains("subgroup collective"),
                "Fix: the refusal must name the construct that diverged, not the barrier rule; \
                 `{name}` at {place:?} reported {text}"
            );
        }
    }
}

/// WHY: the check above passes for a rule that refuses every guarded
/// collective. The condition being non-uniform is what decides it.
#[test]
fn a_collective_under_a_uniform_branch_is_accepted() {
    for (name, collective) in collectives() {
        for place in Placement::ALL {
            let program = guarded_collective(collective.clone(), true, place);
            let outputs = ReferenceRequest::standard(&program, &[])
                .outputs()
                .unwrap_or_else(|error| {
                    panic!(
                        "Fix: every lane enters this branch, so `{name}` at {place:?} is a \
                         legal rendezvous: {error:?}"
                    )
                });
            assert_eq!(
                outputs
                    .first()
                    .expect("the program declares an out buffer")
                    .to_bytes()
                    .len(),
                LANES as usize * 4,
                "Fix: `{name}` at {place:?} must write one word per lane"
            );
        }
    }
}

/// WHY: the rule is about a rendezvous, not about any operand under a branch.
/// Refusing a divergent branch that computes nothing collective would reject
/// ordinary lane-local control flow, which is most of every program.
#[test]
fn a_divergent_branch_without_a_rendezvous_is_accepted() {
    let lane = Expr::InvocationId { axis: 0 };
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(LANES)],
        [LANES, 1, 1],
        vec![Node::if_then(
            Expr::lt(lane.clone(), Expr::u32(2)),
            vec![Node::store("out", lane, Expr::u32(7))],
        )],
    );

    ReferenceRequest::standard(&program, &[])
        .outputs()
        .expect("Fix: a divergent branch with no peer rendezvous in it is a legal program");
}

/// WHY: a subgroup collective reads only its own subgroup, so holding it to
/// workgroup uniformity refuses the standard shape where each subgroup owns
/// one output and a tail workgroup masks the subgroups with no output to
/// compute. That refusal reached production: the quantized grouped-affine
/// linear layer is exactly this shape, and the oracle rejected it as a
/// divergent collective.
///
/// The condition here splits the workgroup on a subgroup boundary, so every
/// lane of a subgroup agrees and the subgroups disagree.
///
/// Does not catch a collective whose subgroup width differs from the
/// simulator's, which no program can currently state.
#[test]
fn a_collective_diverging_only_between_subgroups_is_accepted() {
    for (name, collective) in collectives() {
        for place in Placement::ALL {
            let program = subgroup_split_rendezvous(
                vec![Node::store(
                    "out",
                    Expr::InvocationId { axis: 0 },
                    collective.clone(),
                )],
                place,
            );
            ReferenceRequest::standard(&program, &[])
                .outputs()
                .unwrap_or_else(|e| {
                    panic!(
                        "Fix: `{name}` at {place:?} under a condition every lane of its subgroup \
                         agrees on reads only its own subgroup and must be accepted: {e}"
                    )
                });
        }
    }
}

/// WHY: the acceptance above must not be a rule that accepts every guarded
/// rendezvous. A barrier synchronizes the whole workgroup, so the same branch
/// that is legal for a collective is a deadlock for a barrier and must still
/// be refused, naming the barrier and the workgroup rather than the subgroup.
///
/// The refusal arrives from the IR validator rather than from the
/// interpreter's dynamic rule, because a barrier under a non-uniform branch is
/// visible in the program text and is rejected before a lane runs. What is
/// asserted is therefore the refusal and what it names, not which owner
/// produced it: an owner that stops refusing while the other still does leaves
/// this test green, which the divergent-collective case above covers for the
/// dynamic rule.
#[test]
fn a_barrier_diverging_only_between_subgroups_is_refused() {
    for place in Placement::ALL {
        let program = subgroup_split_rendezvous(
            vec![
                Node::Barrier {
                    ordering: MemoryOrdering::SeqCst,
                },
                Node::store("out", Expr::InvocationId { axis: 0 }, Expr::u32(1)),
            ],
            place,
        );
        let error = ReferenceRequest::standard(&program, &[])
            .outputs()
            .expect_err(&format!(
                "Fix: a barrier at {place:?} that only some subgroups reach holds the lanes that \
                 did reach it for peers that never arrive, and must be refused"
            ));
        let text = format!("{error}");
        assert!(
            text.to_lowercase().contains("barrier"),
            "Fix: the refusal must name the construct that cannot be reached by part of the \
             workgroup; at {place:?} it reported {text}"
        );
        assert!(
            text.contains("workgroup"),
            "Fix: a barrier's scope is the workgroup, and the refusal must say so rather than \
             report the subgroup a collective would; at {place:?} it reported {text}"
        );
    }
}

/// `if lane < SUBGROUP_WIDTH { <body> } else { out[lane] = 0 }` over a
/// workgroup of two subgroups.
///
/// Every lane of a subgroup agrees on the condition and the two subgroups
/// disagree, which is the one shape that separates a subgroup-scoped rule from
/// a workgroup-scoped one. `placement` varies the arm and the nesting for the
/// same reason it does in [`guarded_collective`].
fn subgroup_split_rendezvous(body: Vec<Node>, placement: Placement) -> Program {
    let lane = Expr::InvocationId { axis: 0 };
    let mut rendezvous = body;
    if placement.nested {
        rendezvous = vec![Node::loop_for("i", Expr::u32(0), Expr::u32(1), rendezvous)];
    }
    let plain = vec![Node::store("out", lane.clone(), Expr::u32(0))];
    let (then, otherwise) = if placement.otherwise {
        (plain, rendezvous)
    } else {
        (rendezvous, plain)
    };
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(SPLIT_LANES)],
        [SPLIT_LANES, 1, 1],
        vec![Node::if_then_else(
            Expr::lt(lane, Expr::u32(SUBGROUP_WIDTH)),
            then,
            otherwise,
        )],
    )
}
