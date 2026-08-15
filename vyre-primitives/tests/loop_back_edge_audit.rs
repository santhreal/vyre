//! Every shipped program whose loop body synchronizes must validate.
//!
//! `V055` refuses an invocation returning from a synchronizing loop body after
//! that body's last barrier: one invocation takes the back edge and writes while
//! a sibling has not yet reached the exit, the sibling leaves, and the data it
//! owned freezes partway through. Nothing hangs, so the symptom is a wrong
//! answer at some rate rather than a stall, which is why the first instance in
//! this tree was found from a downstream flake and not from a failing test.
//!
//! Four programs have been caught with that shape so far, all built the same
//! way (clear a flag, synchronize, step, exit as the LAST node of the body).
//! Three were in this crate or its callers. Every one of them was found because
//! something ELSE went red: a nondeterministic consumer, or 95 dispatch failures
//! in another crate after an unrelated lowering change made a latent exit live.
//! None was found by asking the question directly.
//!
//! This is that question, asked directly and cheaply. It builds each shipped
//! program in this crate whose body contains both a loop and a barrier and
//! validates it on the host, with no GPU and no dispatch, so a fifth instance
//! surfaces here by name instead of as a rate of wrong answers somewhere
//! downstream. The audit is deliberately about the SHIPPED builders, not about
//! constructed bodies: the rule's own unit tests already cover synthetic shapes,
//! and what those cannot tell you is whether a real caller ships one.
//!
//! Result on the day it was written, since an audit that does not say what it
//! found is just a test: all thirteen shipped programs validate, at five
//! iteration budgets each, so there is no fifth instance among them. That is a
//! measured absence over THIS set, not a proof about the crate: a builder whose
//! file does not contain both `Node::loop_for` and `Node::barrier` was never a
//! candidate and is not covered. Of the thirteen, exactly four put a barrier
//! inside a loop body and so are governed by the rule at all, and the tests below
//! pin which four by name.
//!
//! Keeping this honest as the crate grows: the set below is the set of builders
//! reachable from files that contain both `Node::loop_for` and `Node::barrier`.
//! A new synchronizing loop in a new file is NOT audited until it is added here,
//! so add it when you write one. `assorted_iteration_counts` exists because the
//! shape can depend on the iteration budget, and a builder that clamps or
//! specializes at 0 or 1 can be legal at one budget and not another.
#![forbid(unsafe_code)]

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::validate::validate;
use vyre_primitives::fixpoint::persistent_fixpoint::{
    persistent_fixpoint, persistent_fixpoint_grid,
};
use vyre_primitives::graph::persistent_bfs::{
    persistent_bfs, persistent_bfs_batch, persistent_bfs_batch_with_density,
    persistent_bfs_with_density,
};
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::parsing::ast_cse_structural_hash::ast_cse_structural_hash_program;
use vyre_primitives::reduce::workgroup_tree::{
    workgroup_max_f32, workgroup_max_u32, workgroup_min_f32, workgroup_min_u32, workgroup_sum_f32,
    workgroup_sum_u32,
};

/// A transfer body for the fixpoint builders: OR each word of `current` into
/// `next` and flag growth, which is the monotone shape those builders exist for.
fn transfer_body() -> Vec<Node> {
    vec![Node::if_then(
        Expr::lt(Expr::InvocationId { axis: 0 }, Expr::u32(4)),
        vec![Node::store(
            "next",
            Expr::InvocationId { axis: 0 },
            Expr::load("current", Expr::InvocationId { axis: 0 }),
        )],
    )]
}

/// Every shipped synchronizing-loop program, with a label for the failure
/// message. Built at one representative iteration budget.
fn synchronizing_programs(max_iters: u32) -> Vec<(&'static str, Program)> {
    let shape = ProgramGraphShape::new(64, 256);
    vec![
        (
            "persistent_bfs",
            persistent_bfs(shape, "frontier_in", "frontier_out", u32::MAX, max_iters),
        ),
        (
            "persistent_bfs_with_density",
            persistent_bfs_with_density(
                shape,
                "frontier_in",
                "frontier_out",
                "density_active",
                u32::MAX,
                max_iters,
            ),
        ),
        (
            "persistent_bfs_batch",
            persistent_bfs_batch(
                shape,
                "frontier_in",
                "frontier_out",
                "changed",
                "converged",
                4,
                u32::MAX,
                max_iters,
            ),
        ),
        (
            "persistent_bfs_batch_with_density",
            persistent_bfs_batch_with_density(
                shape,
                "frontier_in",
                "frontier_out",
                "changed",
                "converged",
                "density_active",
                4,
                u32::MAX,
                max_iters,
            ),
        ),
        (
            "persistent_fixpoint",
            persistent_fixpoint(transfer_body(), "current", "next", "changed", 4, max_iters),
        ),
        (
            "persistent_fixpoint_grid",
            persistent_fixpoint_grid(transfer_body(), "current", "next", "changed", 4, max_iters),
        ),
        (
            "workgroup_sum_f32",
            workgroup_sum_f32("values", "out", 1024, 256),
        ),
        (
            "workgroup_sum_u32",
            workgroup_sum_u32("values", "out", 1024, 256),
        ),
        (
            "workgroup_max_f32",
            workgroup_max_f32("values", "out", 1024, 256),
        ),
        (
            "workgroup_max_u32",
            workgroup_max_u32("values", "out", 1024, 256),
        ),
        (
            "workgroup_min_f32",
            workgroup_min_f32("values", "out", 1024, 256),
        ),
        (
            "workgroup_min_u32",
            workgroup_min_u32("values", "out", 1024, 256),
        ),
        (
            "ast_cse_structural_hash_program",
            ast_cse_structural_hash_program(64, 128),
        ),
    ]
}

/// Barriers anywhere under `nodes`.
fn has_barrier(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| match node {
        Node::Barrier { .. } => true,
        Node::If {
            then, otherwise, ..
        } => has_barrier(then) || has_barrier(otherwise),
        Node::Loop { body, .. } => has_barrier(body),
        Node::Block(body) => has_barrier(body),
        Node::Region { body, .. } => has_barrier(body),
        _ => false,
    })
}

/// Loop bodies anywhere under `nodes` that contain a barrier, which is exactly
/// the population `V055` governs.
fn synchronizing_loop_bodies<'a>(nodes: &'a [Node], found: &mut Vec<&'a [Node]>) {
    for node in nodes {
        match node {
            Node::Loop { body, .. } => {
                if has_barrier(body) {
                    found.push(body.as_slice());
                }
                synchronizing_loop_bodies(body, found);
            }
            Node::If {
                then, otherwise, ..
            } => {
                synchronizing_loop_bodies(then, found);
                synchronizing_loop_bodies(otherwise, found);
            }
            Node::Block(body) => synchronizing_loop_bodies(body, found),
            Node::Region { body, .. } => synchronizing_loop_bodies(body, found),
            _ => {}
        }
    }
}

fn messages(program: &Program) -> Vec<String> {
    validate(program)
        .iter()
        .map(|error| error.message().to_string())
        .collect()
}

/// Locks out: any shipped synchronizing-loop program in this crate being
/// invalid, V055 or otherwise.
///
/// The direct form of the question that three separate investigations reached
/// only through a downstream symptom. Asserts the whole error list per program,
/// so a violation names both the builder and the rule.
#[test]
fn every_shipped_synchronizing_loop_program_validates() {
    let mut broken: Vec<(&str, Vec<String>)> = Vec::new();
    for (label, program) in synchronizing_programs(8) {
        let found = messages(&program);
        if !found.is_empty() {
            broken.push((label, found));
        }
    }
    assert_eq!(
        broken,
        Vec::new(),
        "shipped programs must validate; each entry is (builder, errors)"
    );
}

/// Locks out: a violation that only appears at a particular iteration budget.
///
/// Several of these builders clamp or specialize on the budget, so a body that
/// is legal at 8 iterations is not automatically legal at 0, 1, or a large
/// count. 0 and 1 are the boundary a caller reaches with an empty or
/// single-element input.
#[test]
fn assorted_iteration_counts_stay_valid() {
    for max_iters in [0_u32, 1, 2, 7, 4096] {
        for (label, program) in synchronizing_programs(max_iters) {
            assert_eq!(
                messages(&program),
                Vec::<String>::new(),
                "{label} must validate at max_iters {max_iters}"
            );
        }
    }
}

/// Locks out: this audit silently auditing nothing, and records which shipped
/// programs `V055` actually governs.
///
/// The two tests above pass trivially if the listed programs stop having
/// synchronizing loops, whether a builder was rewritten or the list was gutted.
/// So the population is pinned BY NAME at what it measurably is, rather than by
/// a count I guessed: exactly four of the thirteen shipped programs put a
/// barrier inside a loop body, which is the only shape this rule can refuse.
///
/// The four are worth knowing because two obvious candidates are NOT among them,
/// OBSERVED by walking the built programs: plain `persistent_bfs` has no barrier
/// inside its loop body, and `persistent_fixpoint_grid` contains no loop node at
/// all. The six workgroup reductions and the AST hash program synchronize outside
/// a loop.
///
/// `persistent_fixpoint_grid` is the one to understand before extending anything
/// here, because it looks like a blind spot and is not one. It is exempt because
/// there is no back edge to guard: it UNROLLS into `max_iterations` top-level
/// waves separated by `GridSync` barriers instead of emitting one wave inside a
/// loop, and only its sibling `persistent_fixpoint` in the same file has a loop at
/// all (verified: one `Node::loop_for` in that file). Do not "simplify" that
/// unroll into a loop to bring it under this rule. Per `ExactnessRegression`, who
/// owns that file, the monotonic-counter grid barrier computes its release target
/// at EMIT time from a static barrier index, so a loop emits one instance with one
/// fixed target: iteration 0 releases correctly and every later iteration finds
/// the counter already past the target and becomes a SILENT no-op, leaving the
/// batch unsynchronized past its first wave with every test still passing. The
/// hazard is written up under "NEVER fold the grid form's waves back into a loop"
/// in `exatok/src/gpu_loop.rs`. So this exemption is load bearing in the
/// OPPOSITE direction from V055: the shape this rule governs is the shape that
/// program must never take.
///
/// A new synchronizing loop anywhere in this crate makes this fail, which is the
/// point: it forces a deliberate decision about the new body instead of letting
/// it join the audit silently or escape it entirely.
#[test]
fn exactly_the_known_programs_contain_synchronizing_loops() {
    let mut with_sync_loop: Vec<&str> = Vec::new();
    for (label, program) in synchronizing_programs(8) {
        let mut bodies: Vec<&[Node]> = Vec::new();
        synchronizing_loop_bodies(program.entry(), &mut bodies);
        if !bodies.is_empty() {
            with_sync_loop.push(label);
        }
    }
    with_sync_loop.sort_unstable();
    assert_eq!(
        with_sync_loop,
        vec![
            "persistent_bfs_batch",
            "persistent_bfs_batch_with_density",
            "persistent_bfs_with_density",
            "persistent_fixpoint",
        ],
        "the set of programs governed by the back-edge rule changed"
    );
}

/// Records which synchronizing bodies are exit-proof and which are only exit-FREE,
/// because the difference decides what happens to the next person who adds an
/// early exit to one.
///
/// A body ending in an unconditional barrier is exit-proof: an exit added
/// anywhere inside it stays ordered against the back edge, no thought required.
/// A body ending in anything else is merely exit-free, legal only because nobody
/// has added an exit yet. Both are legal today (all four validate above, with
/// zero returns between them, OBSERVED).
///
/// Measured, so this is a record and not an aspiration: two of the four are
/// exit-proof (`persistent_bfs_batch`, whose body ends with its second barrier,
/// and `persistent_fixpoint`, which ends with the barrier added when this rule
/// first caught it). The two density variants end with a conditional instead, so
/// they are exit-free only. That is a robustness gap and NOT a defect: an exit
/// added there is refused by `V055` at validation, loudly, rather than returning
/// wrong answers, so the rule is the safety net that makes the gap survivable.
/// It is recorded rather than fixed because closing it costs a real barrier per
/// iteration in a shipped program with no defect to justify it.
///
/// This fails if a body moves between the two groups, in either direction. Losing
/// exit-proofing is the direction that matters, and gaining it should be a
/// deliberate, priced change rather than a silent one.
#[test]
fn exit_proof_and_exit_free_bodies_are_where_they_were_measured() {
    let mut exit_proof: Vec<String> = Vec::new();
    let mut exit_free: Vec<String> = Vec::new();
    for (label, program) in synchronizing_programs(8) {
        let mut bodies: Vec<&[Node]> = Vec::new();
        synchronizing_loop_bodies(program.entry(), &mut bodies);
        for (index, body) in bodies.iter().enumerate() {
            let entry = format!("{label} body {index}");
            if matches!(body.last(), Some(Node::Barrier { .. })) {
                exit_proof.push(entry);
            } else {
                exit_free.push(entry);
            }
        }
    }
    exit_proof.sort_unstable();
    exit_free.sort_unstable();
    assert_eq!(
        exit_proof,
        vec![
            "persistent_bfs_batch body 0".to_string(),
            "persistent_fixpoint body 0".to_string(),
        ],
        "the set of bodies ending in an unconditional barrier changed"
    );
    assert_eq!(
        exit_free,
        vec![
            "persistent_bfs_batch_with_density body 0".to_string(),
            "persistent_bfs_with_density body 0".to_string(),
        ],
        "the set of bodies that are exit-free but not exit-proof changed"
    );
}

/// Locks out: a synchronizing body carrying an early exit without the trailing
/// barrier that makes the exit legal. This is the rule itself, restated locally.
///
/// The invariant, and the whole point of the audit: a governed body with an exit
/// MUST end with an unconditional barrier. Stated this way it covers both
/// directions at once, so it keeps holding as bodies gain and lose exits, where a
/// fixed list of exit counts would just need editing.
///
/// Measured today, and the split is instructive. `persistent_fixpoint` holds ONE
/// exit and IS exit-proof: it is the program this rule first caught, and its
/// trailing barrier is the repair. The other three bodies hold zero exits, so
/// the two that are not exit-proof are legal by having nothing to order.
///
/// The oracle for "legal" here is written out rather than imported from the
/// validator, so this fails if the validator's own notion of the shape drifts.
/// The counts are asserted too, so a body silently gaining an exit is visible
/// even while it stays legal.
#[test]
fn a_body_with_an_early_exit_ends_with_a_barrier() {
    let mut exits: Vec<(String, usize, bool)> = Vec::new();
    for (label, program) in synchronizing_programs(8) {
        let mut bodies: Vec<&[Node]> = Vec::new();
        synchronizing_loop_bodies(program.entry(), &mut bodies);
        for (index, body) in bodies.iter().enumerate() {
            let returns = returns_at_any_depth(body);
            let exit_proof = matches!(body.last(), Some(Node::Barrier { .. }));
            assert!(
                returns == 0 || exit_proof,
                "{label} body {index} holds {returns} early exit(s) and does NOT \
                 end with an unconditional barrier, so an invocation can leave \
                 after the body's last barrier while its siblings take the back \
                 edge"
            );
            exits.push((format!("{label} body {index}"), returns, exit_proof));
        }
    }
    exits.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(
        exits,
        vec![
            ("persistent_bfs_batch body 0".to_string(), 0, true),
            (
                "persistent_bfs_batch_with_density body 0".to_string(),
                0,
                false
            ),
            ("persistent_bfs_with_density body 0".to_string(), 0, false),
            ("persistent_fixpoint body 0".to_string(), 1, true),
        ],
        "measured (body, early exits, ends with barrier) changed"
    );
}

/// `Return` nodes at any nesting depth.
fn returns_at_any_depth(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|node| match node {
            Node::Return => 1,
            Node::If {
                then, otherwise, ..
            } => returns_at_any_depth(then) + returns_at_any_depth(otherwise),
            Node::Loop { body, .. } => returns_at_any_depth(body),
            Node::Block(body) => returns_at_any_depth(body),
            Node::Region { body, .. } => returns_at_any_depth(body),
            _ => 0,
        })
        .sum()
}
