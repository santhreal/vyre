//! The scheduler's post-condition gate certificates are cached across passes.
//!
//! What is cached: the four gate certificates (cost, effect row, linear-type
//! violations, shape-predicate violations) for the program the scheduler is
//! holding. What invalidates it: the fingerprint of the program the entry was
//! derived from no longer matching the program a pass is about to be judged
//! against. Every store records the fingerprint of the program its certificates
//! describe, so an entry can go stale but can never be served.
//!
//! The class this closes: any cache keyed on a pass's own `changed` flag rather
//! than on the program. A `ProgramPass` may return a rewritten program while
//! reporting no change, and the scheduler's cost gate then compares the next
//! rewrite against the certificates of a program that no longer exists. That
//! makes the gate fail OPEN, which is the wrong direction for a post-condition
//! check: a cost-up rewrite the gate exists to revert gets accepted instead.
//!
//! Both fixtures below drive that exact sequence through both scheduler entry
//! points, since `run` and `run_with_metrics` carry the cache independently.
//!
//! What these do not catch: a gate whose certificate is not a pure function of
//! the program. All four are, and a new gate that is not cannot be cached this
//! way at all.

use super::*;

/// Program whose single root region holds two stores, with room for a third.
fn two_store_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(3)],
        [1, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::u32(42)),
            Node::store("out", Expr::u32(1), Expr::u32(43)),
        ],
    )
}

/// Replace the body of the program's single root region.
fn with_region_body(program: &Program, body: Vec<Node>) -> Program {
    let entry: Vec<Node> = program
        .entry()
        .iter()
        .map(|node| match node {
            Node::Region {
                generator,
                source_region,
                ..
            } => Node::Region {
                generator: generator.clone(),
                source_region: source_region.clone(),
                body: Arc::new(body.clone()),
            },
            other => other.clone(),
        })
        .collect();
    program.with_rewritten_entry(entry)
}

/// Body of the program's single root region.
fn region_body(program: &Program) -> Vec<Node> {
    match program.entry() {
        [Node::Region { body, .. }] => body.as_ref().clone(),
        entry => panic!("Fix: fixture must keep one root region, got {entry:?}"),
    }
}

/// Drops the trailing store while reporting that it changed nothing.
///
/// A pass is allowed to be wrong about its own `changed` flag, and the
/// scheduler already re-checks the claim structurally. What it must not do is
/// keep judging later rewrites against the program this one replaced.
#[derive(Debug)]
struct SilentShrinkPass {
    metadata: PassMetadata,
}

impl crate::optimizer::private::Sealed for SilentShrinkPass {}

impl ProgramPass for SilentShrinkPass {
    fn metadata(&self) -> PassMetadata {
        self.metadata
    }

    fn analyze(&self, _program: &Program) -> PassAnalysis {
        PassAnalysis::RUN
    }

    fn transform(&self, program: Program) -> PassResult {
        let body = region_body(&program);
        // Matched narrowly on the authored body so the pass fires exactly once
        // per run. A guard that also matched the appended body would turn a
        // stale certificate into non-convergence, which is red for the right
        // reason but says less than the reverted-decision assertion.
        let is_authored_body = match body.as_slice() {
            [Node::Store { .. }, Node::Store { index, .. }] => *index == Expr::u32(1),
            _ => false,
        };
        if is_authored_body {
            let shrunk = with_region_body(&program, body[..1].to_vec());
            // Deliberately false: this is the claim the cache must not trust.
            return PassResult {
                program: shrunk,
                changed: false,
            };
        }
        PassResult::unchanged(program)
    }

    fn fingerprint(&self, _program: &Program) -> u64 {
        0
    }
}

/// Appends a third store to the root region.
///
/// The result is cost-identical to the original two-store program on every
/// certificate dimension, and one node above the program `SilentShrinkPass`
/// actually left behind. So a fresh certificate reverts it and a stale one
/// accepts it, which is exactly the difference the cache must not erase.
#[derive(Debug)]
struct StoreAppendPass {
    metadata: PassMetadata,
}

impl crate::optimizer::private::Sealed for StoreAppendPass {}

impl ProgramPass for StoreAppendPass {
    fn metadata(&self) -> PassMetadata {
        self.metadata
    }

    fn analyze(&self, _program: &Program) -> PassAnalysis {
        PassAnalysis::RUN
    }

    fn transform(&self, program: Program) -> PassResult {
        let mut body = region_body(&program);
        if body.len() != 1 {
            return PassResult::unchanged(program);
        }
        body.push(Node::store("out", Expr::u32(2), Expr::u32(44)));
        PassResult {
            program: with_region_body(&program, body),
            changed: true,
        }
    }

    fn fingerprint(&self, _program: &Program) -> u64 {
        0
    }
}

/// Shrink first, then append, with the cost gate on.
fn stale_cache_scheduler() -> PassScheduler {
    PassScheduler::with_passes(vec![
        ProgramPassKind::new(SilentShrinkPass {
            metadata: PassMetadata::new("silent_shrink", &[], &[]),
        }),
        ProgramPassKind::new(StoreAppendPass {
            metadata: PassMetadata::new("store_append", &["silent_shrink"], &[]),
        }),
    ])
    .with_cost_monotone_enforcement(true)
}

#[test]
fn cost_gate_judges_against_the_program_a_silent_rewrite_left_behind() {
    let report = stale_cache_scheduler()
        .run_with_metrics(two_store_program())
        .expect("Fix: scheduler must converge on the silent-shrink fixture");

    let append = report
        .passes
        .iter()
        .find(|metric| metric.pass == "store_append" && metric.ran)
        .expect("Fix: store_append must run");
    assert_eq!(
        append.decision,
        PassRunDecision::CostReverted,
        "store_append raises node_count over the program silent_shrink actually left behind, so the cost gate must revert it. Accepting it means the gate compared against the certificates of a program that no longer exists."
    );
    assert!(
        !append.changed,
        "a reverted rewrite must not be reported as changed"
    );
    assert_eq!(
        region_body(&report.program),
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
        "the surviving program must be what silent_shrink left, with the reverted store absent"
    );
}

#[test]
fn cost_gate_reverts_the_same_rewrite_on_the_flag_only_path() {
    let program = stale_cache_scheduler()
        .run(two_store_program())
        .expect("Fix: scheduler must converge on the silent-shrink fixture");

    let body = region_body(&program);
    assert_eq!(
        body,
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
        "the flag-only path must revert the appended store too and land on the same program as the metrics path, but it kept: {body:?}"
    );
}

#[test]
fn repeated_runs_of_one_scheduler_agree() {
    // A fact cache is per-`run` state, so nothing a run derives may outlive it.
    // This is not a staleness detector: a cache that serves the wrong facts
    // serves them identically on both runs, which is what the two tests above
    // are for. What it catches is the other failure mode of adding a cache,
    // which is scoping it to the scheduler instead of to the run, so the second
    // caller inherits the first caller's entries and decides differently.
    let scheduler = stale_cache_scheduler();
    let first = scheduler
        .run_with_metrics(two_store_program())
        .expect("Fix: first run must converge");
    let second = scheduler
        .run_with_metrics(two_store_program())
        .expect("Fix: second run must converge");
    assert_eq!(first.program, second.program);
    let decisions = |report: &OptimizerRunReport| -> Vec<(&'static str, PassRunDecision)> {
        report
            .passes
            .iter()
            .map(|metric| (metric.pass, metric.decision))
            .collect()
    };
    assert_eq!(decisions(&first), decisions(&second));
}
