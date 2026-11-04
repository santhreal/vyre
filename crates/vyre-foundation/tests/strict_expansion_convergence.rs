//! Does a strict-IEEE expansion reach an optimizer fixpoint?
//!
//! `expand_strict_transcendentals` replaces one native transcendental with a
//! few hundred exactly specified operations in a single expression tree. Whether
//! the release pass pipeline converges on that shape is a property of the
//! passes, so it is proved here rather than behind a device feature. A pipeline
//! that does not converge fails every strict dispatch before emission with
//! `optimizer did not reach a fixpoint`, which is how the defect first showed
//! up: on a Metal adapter, in the five row 136 parity proptests.
//!
//! The failure message carries the per-iteration trace, because "it oscillates"
//! and "it makes progress slower than the cap" need different repairs and the
//! error alone cannot tell them apart.

use std::collections::HashMap;

use vyre_foundation::fp_expansion::expand_strict_transcendentals;
use vyre_foundation::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};
use vyre_foundation::optimizer::{
    fingerprint_program, optimize, registered_passes_for_profile, AdapterCaps, OptimizerProfile,
};
use vyre_foundation::visit::for_each_expr;

/// The five operators the strict expansion owns.
const EXPANDED: [UnOp; 5] = [UnOp::Sin, UnOp::Cos, UnOp::Sqrt, UnOp::Exp, UnOp::Log];

/// `out[i] = op(in[i])`, the shape the parity proptests dispatch.
fn unary_program(op: &UnOp) -> Program {
    let index = Expr::gid_x();
    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(1),
            BufferDecl::output("out", 1, DataType::F32).with_count(1),
        ],
        [64, 1, 1],
        vec![Node::if_then(
            Expr::lt(index.clone(), Expr::u32(1)),
            vec![Node::store(
                "out",
                index.clone(),
                Expr::UnOp {
                    op: op.clone(),
                    operand: Box::new(Expr::load("in", index)),
                },
            )],
        )],
    )
}

fn expanded_program(op: &UnOp) -> Program {
    expand_strict_transcendentals(&unary_program(op))
        .expect("Fix: the five row 136 operators must all have expansions")
        .expect("Fix: a program containing one of them must be rewritten")
}

/// Where two structurally different programs first disagree, as a window on
/// their debug encodings.
///
/// A pass that reports a rewrite the canonical fingerprint cannot see has
/// traded one of the three things that identity normalizes: declaration order,
/// commutative operand order, or a transparent block. Naming which one is the
/// difference between reading the repair off the failure and guessing it.
fn divergence(before: &Program, after: &Program) -> String {
    let (before, after) = (format!("{before:?}"), format!("{after:?}"));
    let (before, after): (Vec<char>, Vec<char>) =
        (before.chars().collect(), after.chars().collect());
    let at = before
        .iter()
        .zip(&after)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| before.len().min(after.len()));
    let from = at.saturating_sub(60);
    let window = |text: &[char]| -> String { text.iter().skip(from).take(180).collect::<String>() };
    format!("at {at}: {:?} -> {:?}", window(&before), window(&after))
}

/// One line per iteration: which passes rewrote the program, and what the
/// rewrite did to its identity.
///
/// This replays the scheduled pass order without the scheduler's post-condition
/// gates, so a cycle it reports is a cycle in the rewrites themselves. A pass
/// is a mover when the program it returns differs structurally, which is the
/// scheduler's own test. The fingerprint is recorded next to it because the two
/// disagree in exactly one interesting case: a rewrite that only reorders
/// commutative operands changes the program and not its normalized identity,
/// and a pair of passes trading such a swap never converges while looking
/// identical from the outside.
fn trace(program: &Program, iterations: usize) -> String {
    let passes = registered_passes_for_profile(OptimizerProfile::Release)
        .expect("Fix: the release profile must schedule");
    let caps = AdapterCaps::conservative();
    let mut current = program.clone();
    let mut seen: HashMap<u64, usize> = HashMap::new();
    let mut lines = String::new();
    for iteration in 0..iterations {
        let mut movers = Vec::new();
        for pass in &passes {
            if !pass.analyze(&current).should_run {
                continue;
            }
            let before = current.clone();
            let result = pass.batch_apply(current, &caps);
            if result.program != before {
                let same_identity =
                    fingerprint_program(&result.program) == fingerprint_program(&before);
                movers.push(format!(
                    "{}{}",
                    pass.pass_id(),
                    if same_identity {
                        format!(
                            " (same fingerprint, {})",
                            divergence(&before, &result.program)
                        )
                    } else {
                        String::new()
                    }
                ));
            }
            current = result.program;
        }
        let fingerprint = fingerprint_program(&current);
        let repeat = seen.insert(fingerprint, iteration);
        lines.push_str(&format!(
            "  iteration {iteration}: {fingerprint:#018x} moved by {movers:?}{}\n",
            match repeat {
                Some(first) => format!(" (same fingerprint as iteration {first})"),
                None => String::new(),
            }
        ));
        if movers.is_empty() {
            lines.push_str("  converged\n");
            break;
        }
    }
    lines
}

/// Every strict expansion optimizes to a fixed point.
///
/// The five operators are taken from one list so an expansion added without a
/// convergence answer turns this red rather than waiting for a device host.
#[test]
fn every_strict_expansion_reaches_an_optimizer_fixpoint() {
    for op in &EXPANDED {
        let program = expanded_program(op);
        assert!(
            optimize(program.clone()).is_ok(),
            "Fix: the strict expansion of {op:?} does not converge under the release \
             profile, so every strict-mode dispatch of it fails before emission.\n{}",
            trace(&program, 12)
        );
    }
}

/// The expansion itself is a fixed point of the pipeline it feeds.
///
/// Convergence alone would also be satisfied by a pipeline that rewrites the
/// expansion into something the device cannot emit bit-identically. The
/// optimized program must still be free of the constructs the expansion exists
/// to avoid: a fused multiply-add and a division.
#[test]
fn optimizing_a_strict_expansion_introduces_no_fma_and_no_division() {
    for op in &EXPANDED {
        let optimized = optimize(expanded_program(op))
            .unwrap_or_else(|error| panic!("Fix: {op:?} must converge: {error}"));
        let mut fused = 0usize;
        let mut divisions = 0usize;
        for_each_expr(optimized.entry(), |expr| match expr {
            Expr::Fma { .. } => fused += 1,
            Expr::BinOp { op: BinOp::Div, .. } => divisions += 1,
            _ => {}
        });
        assert_eq!(
            (fused, divisions),
            (0, 0),
            "Fix: optimizing the strict expansion of {op:?} reintroduced {fused} fused \
             multiply-add(s) and {divisions} division(s); both round differently from the \
             two-step form the expansion is built out of."
        );
    }
}
