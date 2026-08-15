//! The one assertion of the persistent-fixpoint routing contract.
//!
//! Every op that drives a convergence loop through
//! [`routed_persistent_fixpoint`](super::persistent_fixpoint::routed_persistent_fixpoint)
//! inherits the same four obligations, and they are obligations of the ROUTING,
//! not of the op:
//!
//! 1. At a dispatch span of one workgroup the op keeps the compact single
//!    `changed` word and emits no grid fence, because forcing a cooperative
//!    launch on a dispatch that fits one group costs residency for nothing.
//! 2. One lane past that width it switches to the per-iteration flag, one word
//!    per iteration, and fences the grid. Below the switch
//!    `persistent_fixpoint`'s clear and its sets are ordered only by a
//!    workgroup-scoped barrier, so above one group a clear can erase a set and
//!    the group whose set was erased returns early with unconverged state.
//! 3. The per-iteration flag is exactly `max_iterations` words wide, at every
//!    iteration budget, and matches what `persistent_fixpoint_grid` declares for
//!    itself. An op that re-declares the buffer narrower writes out of bounds on
//!    iteration 1.
//! 4. The threshold is the DISPATCH SPAN, the widest declared buffer, not the
//!    ping-pong state width. `dispatch_element_count_for_program` spans the
//!    largest binding once a program carries atomics, and both harnesses carry
//!    an `atomic_or`, so a wide edge list or kernel matrix makes a launch
//!    multi-workgroup while the state still fits one group.
//!
//! Each routed op used to carry its own copy of those four assertions, and the
//! copies had already drifted in what they accepted: one pinned the fence count
//! to a literal 8 for a four-iteration build while the other pinned the same
//! literal for a different extent, so neither said out loud that the count is
//! two fences per wave. A rule asserted once per op is a rule that can be
//! weakened for one op.

use vyre_foundation::ir::Program;

use super::persistent_fixpoint::{
    count_grid_sync, declared_words, persistent_fixpoint_grid, required_workgroups,
    PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
};

/// Grid fences one wave of [`persistent_fixpoint_grid`] emits: one after the
/// transfer step and one after the compare.
const FENCES_PER_WAVE: usize = 2;

/// A routed convergence op, described by what it builds rather than by its name.
///
/// The two builders take an iteration budget and return the op built at a
/// dispatch span on either side of one workgroup width. Which extent produces
/// that span is the op's business: a scaling vector, a node count, an edge list
/// and a kernel matrix all reach it differently, and that is exactly the
/// knowledge the contract must not restate.
pub(crate) struct RoutedFixpointOp<'a> {
    /// Op name, quoted in every failure message so a red run names the member.
    pub name: &'a str,
    /// Convergence-flag buffer name the op declares.
    pub changed: &'a str,
    /// Builds the op at a dispatch span of exactly one workgroup width.
    pub at_one_workgroup: &'a dyn Fn(u32) -> Program,
    /// Builds the op at a dispatch span past one workgroup width.
    pub past_one_workgroup: &'a dyn Fn(u32) -> Program,
    /// Builds the bare grid harness the op's `changed` declaration is compared
    /// against, over the op's own ping-pong names and state width.
    pub grid_harness: &'a dyn Fn(u32) -> Program,
}

/// Iteration budgets the per-iteration flag width is checked at.
///
/// 1 is the degenerate wave count, 2 the smallest that can be off by one, and 64
/// is past any plausible hardcoded cap.
const ITERATION_BUDGETS: [u32; 4] = [1, 2, 8, 64];

/// Assert `op` obeys every routing obligation in the module docs.
///
/// # Panics
///
/// On any violation, naming `op.name` and the obligation.
pub(crate) fn assert_routes_on_dispatch_span(op: &RoutedFixpointOp<'_>) {
    let width = PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];

    let single = (op.at_one_workgroup)(8);
    assert_eq!(
        single.workgroup_size(),
        PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
        "Fix: {} must declare the harness workgroup size it routes against, so the routing threshold and the emission cannot drift.",
        op.name
    );
    assert_eq!(
        required_workgroups(&single),
        1,
        "Fix: {}'s single-workgroup fixture must span exactly one workgroup at width {width}; it is the fixture that is wrong, not the routing.",
        op.name
    );
    assert_eq!(
        declared_words(&single, op.changed),
        1,
        "Fix: {} must keep the compact one-word convergence flag at one workgroup.",
        op.name
    );
    assert_eq!(
        count_grid_sync(single.entry()),
        0,
        "Fix: {} must not force a cooperative grid launch on a dispatch that fits one workgroup.",
        op.name
    );

    let grid = (op.past_one_workgroup)(8);
    assert!(
        required_workgroups(&grid) >= 2,
        "Fix: {}'s past-one-workgroup fixture must span more than one workgroup; it is the fixture that is wrong, not the routing.",
        op.name
    );
    assert_eq!(
        declared_words(&grid, op.changed),
        8,
        "Fix: {} must switch to the per-iteration convergence-word protocol one lane past one workgroup, not stay on one shared cleared word.",
        op.name
    );
    assert!(
        count_grid_sync(grid.entry()) > 0,
        "Fix: {} must grid-synchronize a multi-workgroup dispatch whichever declared buffer widened it.",
        op.name
    );

    for max_iterations in ITERATION_BUDGETS {
        let program = (op.past_one_workgroup)(max_iterations);
        assert_eq!(
            declared_words(&program, op.changed),
            max_iterations,
            "Fix: {}'s grid route needs one convergence word per iteration; {max_iterations} iterations need {max_iterations} words.",
            op.name
        );
        assert_eq!(
            declared_words(&program, op.changed),
            declared_words(&(op.grid_harness)(max_iterations), op.changed),
            "Fix: {}'s own `changed` declaration must match persistent_fixpoint_grid's.",
            op.name
        );
        assert_eq!(
            count_grid_sync(program.entry()),
            FENCES_PER_WAVE * max_iterations as usize,
            "Fix: {} must fence each of its {max_iterations} waves twice, once after the transfer step and once after the compare.",
            op.name
        );
    }
}

/// The bare grid harness over `state`, for an op whose `changed` width is the
/// only thing being compared.
///
/// Saves each caller from restating the harness's positional parameter order,
/// which is the argument list a transposition would hide in.
pub(crate) fn bare_grid_harness(
    current: &str,
    next: &str,
    changed: &str,
    words: u32,
    max_iterations: u32,
) -> Program {
    persistent_fixpoint_grid(Vec::new(), current, next, changed, words, max_iterations)
}
