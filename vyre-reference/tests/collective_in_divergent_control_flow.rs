//! A subgroup collective reached from a branch whose condition is not
//! workgroup-uniform is rejected, not answered.
//!
//! The interpreter releases a lane holding for its peers once every live lane
//! has arrived, and a lane that took the other branch and retired is not live.
//! A collective under a divergent branch therefore resolved over whichever
//! lanes were still running, and the oracle issued that partial reduction as
//! the expected output. No target defines that value: the lanes that skipped
//! the branch contribute on one and are undefined on another.
//!
//! The rule already existed for `Barrier`. A collective is the same
//! rendezvous, so it carries the same rule.
#![cfg(feature = "subgroup-ops")]

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::ReferenceRequest;

const LANES: u32 = 4;

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
