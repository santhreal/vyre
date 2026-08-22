//! A select never computes both directions of an unsigned difference.
//!
//! WHY: both arms of `Expr::Select` evaluate. A stage that writes the two
//! directions of a difference as the two arms therefore wraps one of them in
//! every lane, whatever the condition decides. A correct evaluator discards the
//! wrapped arm, so the reference agrees and the shape survives review, while a
//! backend shader compiler receives a subtraction whose result is unreachable
//! in the branch that consumes it. That cost the visual filter chain its
//! identity on a production adapter: the two textually identical channel
//! comparisons resolved differently within one invocation and the saturate
//! stage added the distance it was supposed to subtract. The divergence
//! appeared only after five unrelated pipelines had been compiled and
//! dispatched in the same process, so no single-operation run could see it.
//!
//! The population is every operation this build registers, read at run time, so
//! a new composition or primitive witness with this shape turns the suite red.
//! Write the magnitude once with `max`/`min` and scale that.
//!
//! What this does not prove: that an unmirrored subtraction stays in range.
//! It rejects the shape whose intermediate is guaranteed to leave it.

use vyre_foundation::ir::{BinOp, Expr, Program};
use vyre_foundation::operation::OperationRegistry;
use vyre_foundation::visit::{for_each_subexpr, walk_exprs};

/// Every `left - right` pair anywhere inside `expr`.
fn difference_pairs(expr: &Expr) -> Vec<(&Expr, &Expr)> {
    let mut pairs = Vec::new();
    for_each_subexpr(expr, &mut |sub| {
        if let Expr::BinOp {
            op: BinOp::Sub,
            left,
            right,
        } = sub
        {
            pairs.push((left.as_ref(), right.as_ref()));
        }
    });
    pairs
}

/// True when `expr` is a select whose arms subtract the same pair of operands
/// in opposite orders.
fn selects_mirrored_difference(expr: &Expr) -> bool {
    let Expr::Select {
        true_val,
        false_val,
        ..
    } = expr
    else {
        return false;
    };
    let taken = difference_pairs(true_val);
    if taken.is_empty() {
        return false;
    }
    difference_pairs(false_val)
        .iter()
        .any(|(other_left, other_right)| {
            taken
                .iter()
                .any(|(left, right)| left == other_right && right == other_left)
        })
}

fn mirrored_difference_selects(program: &Program) -> bool {
    let mut found = false;
    walk_exprs(program, |expr| {
        if selects_mirrored_difference(expr) {
            found = true;
        }
    });
    found
}

#[test]
fn a_mirrored_difference_in_two_arms_is_detected_and_a_magnitude_is_not() {
    let mirrored = Expr::select(
        Expr::ge(Expr::var("channel"), Expr::var("luma")),
        Expr::sub(Expr::var("channel"), Expr::var("luma")),
        Expr::sub(Expr::var("luma"), Expr::var("channel")),
    );
    assert!(
        selects_mirrored_difference(&mirrored),
        "Fix: the detector must see a difference and its reverse across the two arms"
    );

    let nested = Expr::select(
        Expr::ge(Expr::var("channel"), Expr::var("luma")),
        Expr::add(
            Expr::u32(1),
            Expr::sub(Expr::var("channel"), Expr::var("luma")),
        ),
        Expr::mul(
            Expr::sub(Expr::var("luma"), Expr::var("channel")),
            Expr::u32(2),
        ),
    );
    assert!(
        selects_mirrored_difference(&nested),
        "Fix: the detector must descend into each arm, not only match an arm's root"
    );

    let magnitude = Expr::select(
        Expr::ge(Expr::var("luma"), Expr::var("down")),
        Expr::sub(Expr::var("luma"), Expr::var("down")),
        Expr::u32(0),
    );
    assert!(
        !selects_mirrored_difference(&magnitude),
        "Fix: a clamped single-direction difference is the correct shape and must stay silent"
    );

    let unrelated = Expr::select(
        Expr::ge(Expr::var("a"), Expr::var("b")),
        Expr::sub(Expr::var("a"), Expr::var("b")),
        Expr::sub(Expr::var("c"), Expr::var("d")),
    );
    assert!(
        !selects_mirrored_difference(&unrelated),
        "Fix: two unrelated differences are not a mirrored pair"
    );
}

#[test]
fn no_registered_program_selects_between_both_directions_of_a_difference() {
    let mut examined = 0usize;
    let mut offenders = Vec::new();
    for entry in OperationRegistry::global().iter() {
        let Some(program) = entry.program() else {
            continue;
        };
        examined += 1;
        if mirrored_difference_selects(&program) {
            offenders.push(entry.id);
        }
    }

    assert!(
        examined > 0,
        "Fix: this build must register at least one operation, or this contract proves nothing"
    );
    assert!(
        offenders.is_empty(),
        "{} of {} registered programs evaluate a difference and its reverse as the two arms of one select, so one arm wraps in every lane: {}. Fix: bind the magnitude once as max(a, b) - min(a, b) and scale that.",
        offenders.len(),
        examined,
        offenders.join(", ")
    );
}
