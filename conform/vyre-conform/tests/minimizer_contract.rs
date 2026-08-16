//! Contract tests for the counterexample minimizer.
//!
//! Invariants: monotonic shrinking, termination, and convergence.

use vyre_conform::CounterexampleMinimizer;

#[test]
fn shrink_is_monotonic_in_value() {
    // For the u32 shrinker, "monotonic" means the minimized value is
    // always <= the original failing witness.
    let original = 1_000_000u32;
    let min = CounterexampleMinimizer::shrink_u32(original, |v| v >= 42);
    assert!(
        min <= original,
        "shrunk value {min} must be <= original {original}"
    );
}

#[test]
fn shrink_is_monotonic_for_all_fail() {
    // When every value fails, the shrinker bottoms out at 0.
    let original = 500u32;
    let min = CounterexampleMinimizer::shrink_u32(original, |_| true);
    assert!(
        min <= original,
        "shrunk value {min} must be <= original {original}"
    );
    assert_eq!(min, 0);
}

#[test]
fn shrink_terminates_on_large_input() {
    // O(log n) termination: even for u32::MAX the shrinker must finish
    // instantly (the binary search loop executes at most 32 iterations).
    let start = std::time::Instant::now();
    let min = CounterexampleMinimizer::shrink_u32(u32::MAX, |v| v >= 1);
    let elapsed = start.elapsed();
    assert_eq!(min, 1);
    assert!(
        elapsed.as_millis() < 100,
        "shrinker must terminate in < 100 ms, took {:?}",
        elapsed
    );
}

#[test]
fn shrink_terminates_at_boundary() {
    let start = std::time::Instant::now();
    let min = CounterexampleMinimizer::shrink_u32(1_000_000, |v| v >= 999_999);
    let elapsed = start.elapsed();
    assert_eq!(min, 999_999);
    assert!(
        elapsed.as_millis() < 100,
        "shrinker must terminate in < 100 ms, took {:?}",
        elapsed
    );
}

#[test]
fn shrink_converges_when_minimal() {
    // Once the counterexample cannot be shrunk further, repeated calls
    // with the same minimized value must return the same value.
    let predicate = |v: u32| v == 7;
    let first = CounterexampleMinimizer::shrink_u32(7, predicate);
    assert_eq!(first, 7);

    // Shrinking the already-minimal value again must stay at 7.
    let second = CounterexampleMinimizer::shrink_u32(first, predicate);
    assert_eq!(
        second, 7,
        "convergence failed: second shrink changed the value"
    );
}

#[test]
fn shrink_converges_after_multi_step_reduction() {
    // Start large, shrink to boundary, then verify idempotence.
    let predicate = |v: u32| v >= 100;
    let reduced = CounterexampleMinimizer::shrink_u32(1_000_000, predicate);
    assert_eq!(reduced, 100);

    let again = CounterexampleMinimizer::shrink_u32(reduced, predicate);
    assert_eq!(again, 100, "convergence failed after multi-step reduction");
}

#[test]
fn shrink_converges_to_zero_for_universal_predicate() {
    // When every value satisfies the predicate, the minimizer bottoms
    // out at 0 and stays there.
    let predicate = |_| true;
    let reduced = CounterexampleMinimizer::shrink_u32(10_000, predicate);
    assert_eq!(reduced, 0);

    let again = CounterexampleMinimizer::shrink_u32(reduced, predicate);
    assert_eq!(again, 0, "convergence failed at zero boundary");
}

#[test]
fn shrinks_to_boundary() {
    let min = CounterexampleMinimizer::shrink_u32(1_000_000, |v| v >= 42);
    assert_eq!(min, 42);
}

#[test]
fn shrinks_to_zero_when_all_fail() {
    let min = CounterexampleMinimizer::shrink_u32(500, |_| true);
    assert_eq!(min, 0);
}

#[test]
fn shrinks_minimal_input_stays_same() {
    let min = CounterexampleMinimizer::shrink_u32(7, |v| v == 7);
    assert_eq!(min, 7);
}

#[test]
fn shrink_program_removes_dead_nodes_and_unused_buffers() {
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
    use vyre_foundation::validate::validate;

    // Construct a program with extra dead nodes, unused buffers, and large workgroup size.
    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in_buf", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("unused_buf", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("out_buf", 2, DataType::U32).with_count(1),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("dead1", Expr::u32(9999)),
            Node::let_bind("dead2", Expr::wrapping_add(Expr::u32(10), Expr::u32(20))),
            Node::Store {
                buffer: "out_buf".into(),
                index: Expr::u32(0),
                value: Expr::wrapping_add(
                    Expr::load("in_buf", Expr::u32(0)),
                    Expr::u32(42),
                ),
            },
            Node::let_bind("dead3", Expr::u32(12345)),
        ],
    );

    assert!(validate(&prog).is_empty(), "initial program must be valid");

    // Predicate: requires a store to "out_buf" referencing "in_buf"
    let predicate = |p: &Program| {
        format!("{p:?}").contains("out_buf") && format!("{p:?}").contains("in_buf")
    };

    let minimized = CounterexampleMinimizer::shrink_program(&prog, predicate);

    // Validated at the end
    assert!(validate(&minimized).is_empty(), "minimized program must remain valid");
    // Unused buffer removed
    assert!(!minimized.buffers().iter().any(|b| b.name() == "unused_buf"));
    // Workgroup size reduced
    assert_eq!(minimized.workgroup_size, [1, 1, 1]);
}

#[test]
fn shrink_program_is_deterministic() {
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

    let prog = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [16, 1, 1],
        vec![
            Node::let_bind("a", Expr::u32(100)),
            Node::let_bind("b", Expr::u32(200)),
            Node::Store {
                buffer: "out".into(),
                index: Expr::u32(0),
                value: Expr::wrapping_add(Expr::var("a"), Expr::var("b")),
            },
        ],
    );

    let predicate = |p: &Program| {
        format!("{p:?}").contains("out")
    };

    let run1 = CounterexampleMinimizer::shrink_program(&prog, predicate);
    let run2 = CounterexampleMinimizer::shrink_program(&prog, predicate);

    assert_eq!(format!("{:?}", run1.entry()), format!("{:?}", run2.entry()));
    assert_eq!(run1.buffers().len(), run2.buffers().len());
    assert_eq!(run1.workgroup_size, run2.workgroup_size);
}
