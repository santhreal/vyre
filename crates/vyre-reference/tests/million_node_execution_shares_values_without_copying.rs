//! Executing a program far larger than the fixture corpus, and sharing the
//! values it binds rather than copying them.
//!
//! # The class closed here
//!
//! An oracle that can only refuse a large program is not an oracle for one. The
//! statement-node ceiling `vyre_foundation` applies is sized for what a backend
//! compiles into one fused bundle, and a host evaluator that materializes the
//! program itself has different bounds: its memory ceiling and its work ceiling.
//! Applying the compile-time ceiling unconditionally meant the only answer the
//! oracle had for a million-node program was a refusal, so nothing graded a
//! device against one.
//!
//! Three defects stand between an opt-in ceiling and an execution that
//! finishes. The first is a per-value copy: an environment that hands out a
//! deep copy of every value it holds turns one bound local into one copy per
//! read, which is invisible at the scale of a fixture and quadratic at the
//! scale of a fused bundle. The second is a per-scope allocation: a lane that
//! takes a fresh region for every body it enters, and returns it on the way
//! out, pays the allocator once per scope for the whole run rather than
//! reusing the region the previous scope retired. The third is eager work for
//! a result the run discards: an evaluation that fingerprints its program
//! serializes every node of it, and it did so twice, once to have a refusal
//! name ready and once to fill a certificate field that a caller reading
//! output bytes throws away.
//!
//! Five properties are held here. The default ceiling still refuses, so the
//! opt-in is an opt-in. The stated ceiling executes, to the output the program's
//! own statements fix, inside an asserted wall clock and an asserted memory
//! bound. A value shared out of the environment stands on the payload that was
//! bound, not a copy of it. A lane that enters and leaves a scope a hundred
//! thousand times allocates a bounded number of times rather than one per
//! scope. A program that crosses the work ceiling is still named in the
//! refusal, so the name moved off the run rather than out of the product.
//!
//! # What it does not catch
//!
//! The walk cost of a million nodes of a variant this shape does not use. The
//! program here is a flat store chain, so what it proves is breadth: one
//! statement kind, no nesting, one invocation.
//!
//! Peak memory attributable to the evaluation alone. Process resident size is
//! sampled while the run is in flight, so it carries whatever else the test
//! binary holds at that moment. It is a ceiling on the run, not an attribution.
//! Where process resident size cannot be read the peak bound is not asserted
//! and the retained-byte bound is the only memory bound the case states.
//!
//! Sharing through a path no statement in this shape takes: an operation call's
//! argument binding and a subgroup snapshot both share values, and neither is
//! exercised by a store chain.

use std::sync::Arc;
use std::time::Duration;

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::value::Value;
use vyre_reference::{ReferenceBudget, ReferenceErrorClass, ReferenceRequest};
use vyre_test_support::alloc_probe::Region;

use crate::program_scale::{
    count_nodes, flat_chain, flat_chain_output, run_within_deadline, Verdict, MILLION_NODES,
};

/// Statement nodes the million-node execution admits.
///
/// Twice the shape's node count, so the ceiling is stated by the case rather
/// than met exactly by accident: a ceiling equal to the node count would pass
/// against an off-by-one in either direction.
const ADMITTED_PROGRAM_NODES: usize = 2 * MILLION_NODES;

/// Wall clock the million-node execution may take.
///
/// Measured, not chosen: 0.36 seconds under the test profile, which builds
/// `vyre-reference` and `vyre-foundation` at opt-level 3. The ceiling stands at
/// twelve seconds so a loaded host does not fail the run,
/// and below the runner's own 45 second deadline so a lost work bound is
/// reported as this ceiling rather than as a timeout.
const EXECUTION_WALL_CLOCK_CEILING: Duration = Duration::from_secs(12);

/// Heap bytes the million-node execution may still hold when it returns.
///
/// The program declares one `u32` output buffer, so what the evaluation
/// retains is that buffer and the output `Value` wrapping it. The ceiling is
/// four kilobytes: enough for the output and the error-free return path, and
/// far below anything that would indicate the interpreter kept per-node state
/// alive past the run.
const EXECUTION_RETAINED_BYTE_CEILING: usize = 4 * 1024;

/// Heap bytes the million-node execution may request in total.
///
/// Measured on the evaluating thread alone, so this is the memory bound that
/// does not move when the rest of the suite runs alongside it. Measured, not
/// chosen: 183 MB, which is the admission pass's per-node cost over a million
/// nodes. The ceiling stands at 384 MB, above the measurement and far under
/// what the two program serializations the run used to make would reach.
const EXECUTION_ALLOCATED_BYTE_CEILING: usize = 384 * 1000 * 1000;

/// Process resident growth the million-node execution may reach.
///
/// Process-wide, so it carries every other case the test binary is running at
/// the same time and cannot be tightened to the evaluation's own footprint.
/// The ceiling stands at the 1 GiB `ReferenceBudget::max_memory_bytes` default
/// the run also carries, which makes it a statement the run holds whatever
/// else the process is doing. The attributable bound is
/// [`EXECUTION_ALLOCATED_BYTE_CEILING`].
const EXECUTION_PEAK_RESIDENT_CEILING: usize = 1024 * 1024 * 1024;

/// Scope entries the arena proof drives through one lane.
const ARENA_SCOPE_ENTRIES: u32 = 100_000;

/// Allocations the arena proof admits for [`ARENA_SCOPE_ENTRIES`] scopes.
///
/// The arena reuses the region a closing scope retires and holds the name
/// storage that scope bound, so entering a scope takes no allocation once the
/// high-water mark is reached. What remains is per-evaluation setup and the
/// growth steps of the interpreter's own vectors, which is why the ceiling is
/// a constant rather than zero. Measured, not chosen: 34 allocations for a
/// hundred thousand entries, against 200,046 before the arena held the names.
/// The ceiling stands at seven times the measurement and three orders of
/// magnitude below one allocation per scope.
const ARENA_ALLOCATION_CEILING: usize = 256;

/// Extra allocations the larger measurement may take over the smaller one.
///
/// The two runs differ by a factor of ten in scope entries. An arena that
/// streams answers both with the same count, so the slack covers only the
/// growth steps of a vector that happens to cross a power of two between the
/// two trip counts. A region taken per scope would differ by 180,000.
const ARENA_ENTRY_SCALING_SLACK: usize = 16;

/// A million-node program refuses under the default program-size ceiling.
///
/// # The class closed here
///
/// The ceiling became a field, and a field can be read as `0`, as `usize::MAX`,
/// or not at all. A default that admits everything turns the opt-in into a
/// removal of the bound, which is what the refusal path other suites mutate
/// against depends on not happening. This is the refusing side of the same
/// field the case below raises.
///
/// # What it does not catch
///
/// Whether the refusal names the ceiling it crossed, which the message
/// assertion below covers, or whether a smaller program is admitted, which the
/// rest of the suite covers by evaluating one.
#[test]
fn a_million_node_program_refuses_under_the_default_program_size_ceiling() {
    let program = flat_chain(MILLION_NODES);
    let nodes = count_nodes(program.entry());
    assert!(
        nodes > ReferenceBudget::DEFAULT_MAX_PROGRAM_NODES,
        "the shape walks to {nodes} nodes, inside the {} node default ceiling, so a refusal here \
         would prove nothing. Fix: build a program larger than the default.",
        ReferenceBudget::DEFAULT_MAX_PROGRAM_NODES
    );

    let error = ReferenceRequest::standard(&program, &[])
        .outputs()
        .expect_err("a program past the default program-size ceiling must be refused");
    assert_eq!(
        error.error_class(),
        ReferenceErrorClass::BudgetExhaustion,
        "a program refused for its size crossed a resource ceiling. Message: {}",
        error.message()
    );
    let expected = format!(
        "node count exceeds limit {}",
        ReferenceBudget::DEFAULT_MAX_PROGRAM_NODES
    );
    assert!(
        error.message().contains(&expected),
        "the refusal does not state the ceiling it crossed. Expected to contain `{expected}`, \
         got: {}",
        error.message()
    );
}

/// A million-node program executes to its declared output under a stated
/// program-size ceiling, inside an asserted wall clock and memory bound.
///
/// # The class closed here
///
/// This is the property the opt-in exists for: a program of a million IR nodes
/// produces the value its own statements fix, and produces it in bounded time
/// and bounded memory. Asserting the output alone would pass against a run that
/// took an hour; asserting the time alone would pass against a run that
/// answered the wrong value. Both ceilings are asserted, and the wall clock is
/// enforced by a deadline on a separate thread, so a run that loses a bound
/// fails with text rather than stalling the suite.
///
/// The expected output is derived from the generator, not recorded from a run,
/// so a change to the interpreter that alters the answer fails here instead of
/// being absorbed into a new golden value.
///
/// # What it does not catch
///
/// Whether the run charged a sensible number of steps. The work ceiling is
/// armed generously here so nothing but the program-size ceiling and the
/// wall clock decide the outcome.
#[test]
fn a_million_node_program_executes_to_its_declared_output_within_stated_bounds() {
    let program = flat_chain(MILLION_NODES);
    let nodes = count_nodes(program.entry());
    assert!(
        nodes >= MILLION_NODES,
        "the program walks to {nodes} nodes, under the {MILLION_NODES} this proof requires. Fix: \
         build the program the generator claims to build."
    );
    let expected = flat_chain_output(MILLION_NODES);

    let run = run_within_deadline(
        program,
        ReferenceBudget {
            max_program_nodes: ADMITTED_PROGRAM_NODES,
            ..ReferenceBudget::with_work_ceiling(64_000_000)
        },
        "million-node-execution",
    );

    let outputs = match run.verdict {
        Verdict::Outputs(outputs) => outputs,
        Verdict::Refused(class, message) => panic!(
            "a {nodes} node program was refused under a {ADMITTED_PROGRAM_NODES} node ceiling \
             with class {class:?}. `ReferenceBudget::max_program_nodes` is not read by the \
             admission pass. Message: {message}"
        ),
    };
    assert_eq!(
        outputs.len(),
        1,
        "the program declares one output buffer and evaluated to {}",
        outputs.len()
    );
    assert_eq!(
        outputs[0].to_bytes(),
        expected.to_le_bytes().to_vec(),
        "the million-node chain stores {expected} last, so that is the value `out[0]` carries. A \
         different value means statements were skipped, reordered, or coalesced."
    );

    println!(
        "million-node execution: {nodes} nodes in {:?}, {} bytes allocated, retained {} bytes, \
         peak resident growth {:?} bytes",
        run.elapsed, run.allocated_bytes, run.retained_bytes, run.peak_resident_bytes
    );
    assert!(
        run.elapsed <= EXECUTION_WALL_CLOCK_CEILING,
        "the million-node execution took {:?}, past the {EXECUTION_WALL_CLOCK_CEILING:?} ceiling \
         this case states. Fix: restore the per-node walk cost, or restate the ceiling from a \
         fresh measurement.",
        run.elapsed
    );
    assert!(
        run.allocated_bytes <= EXECUTION_ALLOCATED_BYTE_CEILING,
        "the million-node execution requested {} bytes, past the \
         {EXECUTION_ALLOCATED_BYTE_CEILING} byte ceiling this case states. Fix: restore the \
         per-node allocation cost, or restate the ceiling from a fresh measurement.",
        run.allocated_bytes
    );
    assert!(
        run.retained_bytes <= EXECUTION_RETAINED_BYTE_CEILING,
        "the million-node execution still held {} bytes when it returned, past the \
         {EXECUTION_RETAINED_BYTE_CEILING} byte ceiling. The evaluation returns one output \
         buffer; anything beyond that is per-node state the run kept alive.",
        run.retained_bytes
    );
    if let Some(peak) = run.peak_resident_bytes {
        assert!(
            peak <= EXECUTION_PEAK_RESIDENT_CEILING,
            "the million-node execution grew process resident size by {peak} bytes, past the \
             {EXECUTION_PEAK_RESIDENT_CEILING} byte ceiling this case states."
        );
    }
}

/// Sharing a value out of the environment stands on the bound payload rather
/// than a copy of it.
///
/// # The class closed here
///
/// Every read of a local, every argument bound into an operation call, and
/// every subgroup snapshot shares a value. A payload that is copied on share
/// makes each of those cost the payload's size, and equality cannot tell the
/// two apart: a copy compares equal to the value it copied, so a suite that
/// asserts values would stay green while the interpreter did quadratic work.
/// `Value::shares_payload_with` compares the allocation instead, which is the
/// only observation that separates a share from a copy.
///
/// Both payload-carrying variants are covered, because the representation is
/// what carries the property: a variant that holds an owned payload copies on
/// share no matter what the call sites do.
///
/// # What it does not catch
///
/// A payload copied somewhere between two shares. This states that a share is
/// a share; the allocation case below states that the evaluator's hot paths
/// take no allocation per scope.
#[test]
fn sharing_a_value_stands_on_the_bound_payload_rather_than_a_copy() {
    let elements: Arc<[Value]> = (0u32..64).map(Value::U32).collect();
    let bound = Value::Array(Arc::clone(&elements));
    let shared = bound.clone();
    assert!(
        bound.shares_payload_with(&shared),
        "cloning an array value must share its payload. A clone that allocates makes every read \
         of a local cost the payload's size."
    );
    let rebuilt = Value::Array(elements.iter().cloned().collect());
    assert_eq!(
        bound.to_bytes(),
        rebuilt.to_bytes(),
        "a rebuilt array must encode to the same bytes, or this case is not separating what a \
         value is from where it lives"
    );
    assert!(
        !bound.shares_payload_with(&rebuilt),
        "a rebuilt array is a second allocation and must not report as shared, or \
         `shares_payload_with` answers `true` for everything and proves nothing"
    );

    let bytes: Arc<[u8]> = Arc::from(vec![7u8; 256]);
    let bound_bytes = Value::Bytes(Arc::clone(&bytes));
    assert!(
        bound_bytes.shares_payload_with(&bound_bytes.clone()),
        "cloning a bytes value must share its payload"
    );
    assert!(
        !bound_bytes.shares_payload_with(&Value::Bytes(Arc::from(bytes.to_vec()))),
        "a rebuilt bytes value is a second allocation and must not report as shared"
    );

    assert!(
        !Value::U32(1).shares_payload_with(&Value::U32(1)),
        "a scalar carries no payload, so no two scalars share one"
    );
    assert!(
        !bound.shares_payload_with(&bound_bytes),
        "two different variants never share one payload"
    );
}

/// Entering and leaving a scope reuses the region the previous scope retired.
///
/// # The class closed here
///
/// A lane's local environment used to take a fresh region for the names each
/// scope bound and drop it on the way out, so a program that entered a hundred
/// thousand scopes paid the allocator a hundred thousand times for regions it
/// had already had and given back. The arena keeps one region per lane and
/// streams through it: a closing scope truncates back to its mark and the next
/// scope writes over the same slots.
///
/// The proof is the allocation count of a loop whose body is a block, measured
/// across a hundred thousand entries. An arena that streams reaches its
/// high-water mark once, so the count is a constant; an arena that takes a
/// region per scope scales with the entries and crosses the ceiling by two
/// orders of magnitude.
///
/// # What it does not catch
///
/// Capacity returned to the allocator. The arena holds the largest region it
/// has needed until the lane is dropped, which is the point of the reuse and
/// is stated in `ScopeArena`'s own documentation.
///
/// Allocation the oracle makes off the measuring thread, and gross byte
/// traffic, which the per-node ceiling in `oracle_bounds` bounds instead.
#[test]
fn entering_a_scope_reuses_the_region_the_previous_scope_retired() {
    // One evaluation before the measurements, so lazily initialized state
    // shared by every evaluation is charged to neither run.
    drop(ReferenceRequest::standard(&scope_cycling_program(4), &[]).outputs());

    let few = allocations_entering(ARENA_SCOPE_ENTRIES / 10);
    let many = allocations_entering(ARENA_SCOPE_ENTRIES);
    println!(
        "{} scope entries: {few} allocations; {ARENA_SCOPE_ENTRIES} scope entries: {many} \
         allocations",
        ARENA_SCOPE_ENTRIES / 10
    );

    assert!(
        many <= ARENA_ALLOCATION_CEILING,
        "entering {ARENA_SCOPE_ENTRIES} scopes took {many} allocations, past the \
         {ARENA_ALLOCATION_CEILING} this case admits. A scope that allocates its own region, or \
         a binding that allocates its own name, charges the allocator once per entry instead of \
         reusing what the previous scope retired."
    );
    assert!(
        many <= few + ARENA_ENTRY_SCALING_SLACK,
        "ten times the scope entries took {many} allocations against {few}, a rise past the \
         {ARENA_ENTRY_SCALING_SLACK} this case admits. Allocation that tracks the entry count is \
         a region or a name taken per scope rather than reused."
    );
}

/// Allocations one evaluation of `scope_cycling_program(entries)` takes.
///
/// The evaluation is asserted to have produced outputs, so a measurement can
/// never cover a refusal path that never entered a scope.
fn allocations_entering(entries: u32) -> usize {
    let program = scope_cycling_program(entries);
    let region = Region::new();
    let outputs = ReferenceRequest::new(
        &program,
        &[],
        ReferenceBudget::with_work_ceiling(64_000_000),
    )
    .outputs();
    let change = region.change();

    assert!(
        outputs.is_ok(),
        "the scope-cycling program must evaluate, or the measurement covers a refusal path \
         rather than the arena: {outputs:?}"
    );
    change.allocations
}

/// A program whose loop body is a block binding one local, run `entries` times.
///
/// A block is one scope entry and the `Let` inside it is one name bound in that
/// scope, so the loop drives the arena through `entries` open-and-close cycles
/// with a name in each. The store keeps the body from being an empty scope the
/// interpreter could skip.
fn scope_cycling_program(entries: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::loop_(
            "i",
            Expr::u32(0),
            Expr::u32(entries),
            vec![Node::block(vec![
                Node::let_bind("step", Expr::var("i")),
                Node::store("out", Expr::u32(0), Expr::var("step")),
            ])],
        )],
    )
}

/// A program that crosses the work ceiling is named in the refusal.
///
/// # The class closed here
///
/// The refusal names the program so a caller reading a failed comparison knows
/// which dispatch reached the ceiling. That name used to be produced when the
/// budget was armed, before the first statement ran, which charged every
/// evaluation a full serialization of its program for a string that only a
/// refusal reads. Moving the work to the refusal is only correct if the
/// refusal still carries the name, and a fingerprint prefix is the only name a
/// program that declares no entry operation id has.
///
/// # What it does not catch
///
/// What arming costs, which the per-node allocation ceiling in `oracle_bounds`
/// bounds instead.
///
/// A refusal of a program that declares an entry operation id, which is named
/// from the declaration and never fingerprinted.
#[test]
fn a_program_that_crosses_the_work_ceiling_is_named_in_the_refusal() {
    let program = flat_chain(64);
    assert!(
        program.entry_op_id.is_none(),
        "the shape must declare no entry operation id, or the refusal is named from the \
         declaration and this case proves nothing about the fingerprint path"
    );

    let error = ReferenceRequest::new(&program, &[], ReferenceBudget::bounded(4))
        .outputs()
        .expect_err("a program past a four step ceiling must be refused");
    let source = error
        .step_ceiling_source()
        .expect("a work-ceiling refusal carries its structured payload");
    assert_eq!(
        source.ceiling,
        4,
        "the refusal reports the ceiling it was armed with. Message: {}",
        error.message()
    );
    assert!(
        source.program.starts_with("0x")
            && source.program.len() == 18
            && source.program[2..].chars().all(|c| c.is_ascii_hexdigit()),
        "an unnamed program is named by a sixteen digit fingerprint prefix, got `{}`. Fix: name \
         the refusal at the boundary that still holds the program.",
        source.program
    );
    assert!(
        error.message().contains(&source.program),
        "the refusal message carries the same name as its payload. Message: {}",
        error.message()
    );
}
