//! WHY: closes the class "the oracle resolves a cross-lane conflict
//! deterministically and certifies the result as if the device would agree".
//!
//! A single-threaded interpreter steps one lane at a time, so two lanes that
//! plain-store the same slot always leave the same winner here while a device
//! leaves the winner driver-defined. Comparing outputs across step orders
//! catches only the subset whose winner is observable in the outputs; a
//! round-robin schedule that happens to complete every write before any read
//! produces identical bytes under every order and hides the conflict entirely.
//! These cases pin both halves of `ReferenceRequest::explore_races`: shadow
//! memory reporting an unsynchronized conflict inside one order, and an output
//! comparison reporting a disagreement between two orders. Each hazardous
//! program is paired with the synchronized program it differs from by one
//! node, so a detector that reports everything fails as loudly as one that
//! reports nothing.
//!
//! What these cases do not catch: the exploration is bounded to
//! `MAX_RACE_EXPLORATION_ORDERS` deterministic permutations of the workgroup
//! and lane lists, not the full interleaving space of a real device. A
//! conflict that only manifests when two lanes are stepped between two
//! statements of a third lane is outside the set. They also say nothing about
//! weak-memory reordering within one lane, which the interpreter does not
//! model at all.

use std::collections::BTreeSet;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, MemoryOrdering, MemoryScope, Node, Program,
    StorageDomain,
};
use vyre_reference::value::Value;
use vyre_reference::{
    MemoryAccessKind, RaceExplorationReport, RaceFinding, ReferenceBudget, ReferenceError,
    ReferenceRequest, ShadowMemory, MAX_RACE_EXPLORATION_ORDERS,
};

/// Memory visibility scopes the closed contract declares.
///
/// Stated here rather than read from the enum so that adding a scope turns
/// this suite red and forces a decision about whether the new scope separates
/// two accesses. The loops below iterate `MemoryScope::ALL`, so the new scope
/// is exercised the moment the count is updated.
const DECLARED_MEMORY_SCOPES: usize = 6;

/// Orderings the closed model declares valid for a barrier.
///
/// Same contract as [`DECLARED_MEMORY_SCOPES`]: a new barrier-valid ordering
/// turns this suite red rather than being silently treated like the ones
/// already covered.
const DECLARED_BARRIER_ORDERINGS: usize = 5;

fn u32_words(value: &Value) -> Vec<u32> {
    value
        .to_bytes()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

fn rw(name: &str, binding: u32, count: u32) -> BufferDecl {
    BufferDecl::storage(name, binding, BufferAccess::ReadWrite, DataType::U32).with_count(count)
}

fn zeros(count: u32) -> Value {
    Value::from(vec![0u8; count as usize * 4].as_slice())
}

fn explore(program: &Program, inputs: &[Value], grid: [u32; 3]) -> (RaceExplorationReport, usize) {
    let request = ReferenceRequest::standard(program, inputs).with_grid(grid);
    let declared = request.declared_race_exploration_orders();
    let report = request
        .explore_races()
        .expect("a well-formed dispatch must explore without a structured refusal");
    assert_eq!(
        report.orders_explored, declared,
        "the exploration must run exactly the number of orders the request declares"
    );
    assert!(
        declared <= MAX_RACE_EXPLORATION_ORDERS,
        "a declared order count of {declared} exceeds the stated ceiling"
    );
    assert!(
        report.steps_executed > 0,
        "the exploration must charge the work it performed against the budget"
    );
    (report, declared)
}

/// The distinct `(buffer, index)` locations the exploration reported an
/// unsynchronized conflict on, across every explored order.
fn conflicting_locations(report: &RaceExplorationReport) -> BTreeSet<(&str, u64)> {
    report
        .findings
        .iter()
        .filter_map(|finding| match finding {
            RaceFinding::UnsynchronizedAccess { buffer, index, .. } => {
                Some((buffer.as_str(), *index))
            }
            RaceFinding::ScheduleDisagreement { .. } => None,
        })
        .collect()
}

fn locations(pairs: &[(&'static str, u64)]) -> BTreeSet<(&'static str, u64)> {
    pairs.iter().copied().collect()
}

fn disagreement_count(report: &RaceExplorationReport) -> usize {
    report
        .findings
        .iter()
        .filter(|finding| matches!(finding, RaceFinding::ScheduleDisagreement { .. }))
        .count()
}

/// Two workgroups store the same slot. The winner is whichever workgroup the
/// dispatch stepped last.
fn cross_workgroup_shared_slot() -> Program {
    Program::wrapped(
        vec![rw("out", 0, 1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::workgroup_x())],
    )
}

/// The same dispatch with each workgroup owning its own slot.
fn cross_workgroup_disjoint_slots() -> Program {
    Program::wrapped(
        vec![rw("out", 0, 2)],
        [1, 1, 1],
        vec![Node::store("out", Expr::workgroup_x(), Expr::u32(7))],
    )
}

/// Lanes publish to scratch and then read a neighbour's slot. `barrier` states
/// the ordering that stands between the two phases, or nothing at all.
fn neighbour_exchange(barrier: Option<MemoryOrdering>, lanes: u32) -> Program {
    let mut nodes = vec![Node::store(
        "scratch",
        Expr::local_x(),
        Expr::add(Expr::local_x(), Expr::u32(10)),
    )];
    if let Some(ordering) = barrier {
        nodes.push(Node::barrier_with_ordering(ordering));
    }
    nodes.push(Node::store(
        "out",
        Expr::local_x(),
        Expr::load(
            "scratch",
            Expr::rem(Expr::add(Expr::local_x(), Expr::u32(1)), Expr::u32(lanes)),
        ),
    ));
    Program::wrapped(
        vec![rw("scratch", 0, lanes), rw("out", 1, lanes)],
        [lanes, 1, 1],
        nodes,
    )
}

/// Workgroups publish to scratch and then read another workgroup's slot.
/// Nothing but a whole-grid fence orders two workgroups.
fn cross_workgroup_exchange(fenced: bool) -> Program {
    let mut nodes = vec![Node::store(
        "scratch",
        Expr::workgroup_x(),
        Expr::add(Expr::workgroup_x(), Expr::u32(1)),
    )];
    if fenced {
        nodes.push(Node::barrier_with_ordering(MemoryOrdering::GridSync));
    }
    nodes.push(Node::store(
        "out",
        Expr::workgroup_x(),
        Expr::load(
            "scratch",
            Expr::rem(Expr::add(Expr::workgroup_x(), Expr::u32(1)), Expr::u32(2)),
        ),
    ));
    Program::wrapped(vec![rw("scratch", 0, 2), rw("out", 1, 2)], [1, 1, 1], nodes)
}

#[test]
fn two_workgroups_writing_one_slot_are_reported() {
    let program = cross_workgroup_shared_slot();
    let (report, declared) = explore(&program, &[zeros(1)], [2, 1, 1]);

    assert_eq!(
        declared, 2,
        "a one-lane workgroup declares the forward order and the workgroup-reversed order"
    );
    assert_eq!(
        conflicting_locations(&report),
        locations(&[("out", 0)]),
        "the conflicting store on `out` must be reported, got {:?}",
        report.findings
    );
    assert_eq!(
        disagreement_count(&report),
        1,
        "reversing the workgroup axis must change the last writer, got {:?}",
        report.findings
    );
}

#[test]
fn two_workgroups_writing_their_own_slot_are_clean() {
    let program = cross_workgroup_disjoint_slots();
    let (report, _) = explore(&program, &[zeros(2)], [2, 1, 1]);

    assert!(
        report.is_race_free(),
        "disjoint output slots carry no hazard, got {:?}",
        report.findings
    );
}

#[test]
fn a_missing_barrier_is_reported_even_when_every_order_agrees() {
    // The defect this case exists for: the round-robin executor advances every
    // lane one statement per round, so each lane's publish completes before any
    // lane's read regardless of the step order. Every explored order therefore
    // produces identical bytes, and an output comparison alone reports nothing.
    let program = neighbour_exchange(None, 4);
    let (report, declared) = explore(&program, &[zeros(4), zeros(4)], [1, 1, 1]);

    assert_eq!(
        declared, MAX_RACE_EXPLORATION_ORDERS,
        "a four-lane workgroup declares the full explored set"
    );
    assert_eq!(
        disagreement_count(&report),
        0,
        "this program's outputs are order-invariant, so the case proves the other detector: {:?}",
        report.findings
    );
    assert_eq!(
        conflicting_locations(&report),
        locations(&[
            ("scratch", 0),
            ("scratch", 1),
            ("scratch", 2),
            ("scratch", 3)
        ]),
        "every lane's publish is read by a peer with nothing between them, got {:?}",
        report.findings
    );
}

#[test]
fn the_same_program_with_a_barrier_is_clean() {
    let program = neighbour_exchange(Some(MemoryOrdering::SeqCst), 4);
    let (report, _) = explore(&program, &[zeros(4), zeros(4)], [1, 1, 1]);

    assert!(
        report.is_race_free(),
        "one barrier between the publish and the read closes the conflict, got {:?}",
        report.findings
    );

    let outputs = ReferenceRequest::standard(&program, &[zeros(4), zeros(4)])
        .with_grid([1, 1, 1])
        .outputs()
        .expect("the synchronized program must evaluate");
    assert_eq!(
        u32_words(&outputs[1]),
        vec![11, 12, 13, 10],
        "the synchronized program must still compute the neighbour exchange"
    );
}

#[test]
fn every_barrier_ordering_separates_the_phases_it_stands_between() {
    let barrier_orderings: Vec<MemoryOrdering> = MemoryOrdering::ALL
        .into_iter()
        .filter(|ordering| ordering.is_valid_for_barrier() && !ordering.requires_grid_sync())
        .collect();
    assert_eq!(
        barrier_orderings.len(),
        DECLARED_BARRIER_ORDERINGS - 1,
        "a new workgroup barrier ordering needs a recorded decision, got {barrier_orderings:?}"
    );
    assert_eq!(
        MemoryOrdering::ALL
            .into_iter()
            .filter(|ordering| ordering.is_valid_for_barrier())
            .count(),
        DECLARED_BARRIER_ORDERINGS,
        "the barrier-valid set changed"
    );

    for ordering in barrier_orderings {
        let program = neighbour_exchange(Some(ordering), 4);
        let (report, _) = explore(&program, &[zeros(4), zeros(4)], [1, 1, 1]);
        assert!(
            report.is_race_free(),
            "a {ordering:?} barrier must separate the publish from the read, got {:?}",
            report.findings
        );
    }
}

#[test]
fn a_workgroup_barrier_does_not_order_two_workgroups() {
    // A barrier is a workgroup rendezvous. Counting it as separation across
    // workgroups would report every cross-workgroup conflict as safe whenever
    // the two workgroups had executed the same number of barriers, which is the
    // usual case.
    let program = cross_workgroup_exchange(false);
    let (report, _) = explore(&program, &[zeros(2), zeros(2)], [2, 1, 1]);

    assert_eq!(
        conflicting_locations(&report),
        locations(&[("scratch", 0), ("scratch", 1)]),
        "one workgroup reading another's publish with no fence must be reported, got {:?}",
        report.findings
    );
}

#[test]
fn a_grid_fence_orders_two_workgroups() {
    let program = cross_workgroup_exchange(true);
    let (report, _) = explore(&program, &[zeros(2), zeros(2)], [2, 1, 1]);

    assert!(
        report.is_race_free(),
        "a whole-grid fence separates the publish from the read, got {:?}",
        report.findings
    );

    let outputs = ReferenceRequest::standard(&program, &[zeros(2), zeros(2)])
        .with_grid([2, 1, 1])
        .outputs()
        .expect("the fenced program must evaluate");
    assert_eq!(
        u32_words(&outputs[1]),
        vec![2, 1],
        "the fenced program must still compute the cross-workgroup exchange"
    );
}

#[test]
fn a_commutative_atomic_carries_no_hazard() {
    let program = Program::wrapped(
        vec![rw("counter", 0, 1)],
        [4, 1, 1],
        vec![Node::let_bind(
            "prev",
            Expr::atomic_add("counter", Expr::u32(0), Expr::u32(1)),
        )],
    );
    let (report, _) = explore(&program, &[zeros(1)], [1, 1, 1]);

    assert!(
        report.is_race_free(),
        "four lanes incrementing one counter atomically agree in every order, got {:?}",
        report.findings
    );

    let outputs = ReferenceRequest::standard(&program, &[zeros(1)])
        .with_grid([1, 1, 1])
        .outputs()
        .expect("the atomic program must evaluate");
    assert_eq!(u32_words(&outputs[0]), vec![4]);
}

#[test]
fn an_order_dependent_atomic_is_reported_as_a_disagreement_not_a_conflict() {
    // Two atomics on one location never conflict: the operation itself orders
    // them. An exchange is still order-dependent, so the hazard shows up in the
    // output comparison rather than in shadow memory. A detector that reported
    // only one of the two classes would miss this program entirely.
    let program = Program::wrapped(
        vec![rw("slot", 0, 1)],
        [4, 1, 1],
        vec![Node::let_bind(
            "prev",
            Expr::atomic_exchange("slot", Expr::u32(0), Expr::local_x()),
        )],
    );
    let (report, _) = explore(&program, &[zeros(1)], [1, 1, 1]);

    assert!(
        conflicting_locations(&report).is_empty(),
        "an atomic pair is ordered by the atomic operation, got {:?}",
        report.findings
    );
    assert!(
        disagreement_count(&report) > 0,
        "the last exchange wins, so two step orders must disagree, got {:?}",
        report.findings
    );
}

#[test]
fn a_conflict_is_reported_at_every_declared_memory_scope() {
    assert_eq!(
        MemoryScope::ALL.len(),
        DECLARED_MEMORY_SCOPES,
        "a new memory scope needs a recorded decision about what separates two accesses"
    );

    for scope in MemoryScope::ALL {
        let mut shadow = ShadowMemory::new();
        assert!(
            shadow
                .record_access(
                    "buf",
                    0,
                    [0, 0, 0],
                    MemoryAccessKind::Write,
                    scope,
                    StorageDomain::DeviceGlobal,
                )
                .is_none(),
            "the first access to a location cannot conflict with anything ({scope:?})"
        );
        let finding = shadow.record_access(
            "buf",
            0,
            [1, 0, 0],
            MemoryAccessKind::Write,
            scope,
            StorageDomain::DeviceGlobal,
        );
        assert!(
            matches!(finding, Some(RaceFinding::UnsynchronizedAccess { .. })),
            "a same-phase write-write pair must be reported at {scope:?}, got {finding:?}"
        );
    }
}

#[test]
fn a_conflict_is_reported_at_every_declared_storage_domain() {
    for domain in StorageDomain::ALL {
        let mut shadow = ShadowMemory::new();
        let _ = shadow.record_access(
            "buf",
            0,
            [0, 0, 0],
            MemoryAccessKind::Write,
            MemoryScope::Workgroup,
            domain,
        );
        let finding = shadow.record_access(
            "buf",
            0,
            [1, 0, 0],
            MemoryAccessKind::Read,
            MemoryScope::Workgroup,
            domain,
        );
        assert!(
            matches!(finding, Some(RaceFinding::UnsynchronizedAccess { .. })),
            "a same-phase write-read pair must be reported in {domain:?}, got {finding:?}"
        );
    }
}

#[test]
fn the_whole_exploration_is_charged_against_one_budget() {
    let program = neighbour_exchange(Some(MemoryOrdering::SeqCst), 4);
    let inputs = [zeros(4), zeros(4)];

    let generous = ReferenceRequest::standard(&program, &inputs)
        .with_grid([1, 1, 1])
        .explore_races()
        .expect("the standard budget must cover this dispatch");

    let refusal = ReferenceRequest::new(&program, &inputs, ReferenceBudget::bounded(4))
        .with_grid([1, 1, 1])
        .explore_races()
        .expect_err("a work ceiling below the exploration's cost must refuse it");
    assert_eq!(
        refusal.error_class(),
        vyre_reference::ReferenceErrorClass::BudgetExhaustion,
        "the refusal must name budget exhaustion, got {refusal:?}"
    );

    // One budget covers every order, so the charge is the sum across the whole
    // exploration rather than the cost of the widest single order.
    let single = ReferenceRequest::standard(&program, &inputs)
        .with_grid([1, 1, 1])
        .outputs_and_steps()
        .expect("the standard budget must cover one order")
        .1;
    assert!(
        generous.steps_executed > single,
        "an exploration of {} orders must charge more than one order ({} vs {single})",
        generous.orders_explored,
        generous.steps_executed
    );
}

#[test]
fn a_structured_refusal_ends_the_exploration_rather_than_becoming_a_finding() {
    // A race report is a verdict about a program the oracle could evaluate. A
    // fault it cannot evaluate past is a different answer and keeps its own
    // error class, so no caller can read an out-of-bounds dispatch as a
    // race-free one.
    let program = Program::wrapped(
        vec![rw("out", 0, 2)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::add(Expr::gid_x(), Expr::u32(64)),
            Expr::u32(1),
        )],
    );
    let error: ReferenceError = ReferenceRequest::standard(&program, &[zeros(2)])
        .with_grid([1, 1, 1])
        .explore_races()
        .expect_err("an out-of-bounds store must refuse under strict exploration");
    assert_eq!(
        error.error_class(),
        vyre_reference::ReferenceErrorClass::OutOfBoundsAccess,
        "the refusal must keep its own class, got {error:?}"
    );
}
