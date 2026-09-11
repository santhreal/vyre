//! Resource bounds of the reference oracle, proved at the boundary of each one.
//!
//! # The class closed here
//!
//! The oracle materializes a whole dispatch on the host and walks the program
//! for every invocation, so every ceiling in
//! [`vyre_reference::ReferenceBudget`] stands between a hostile or merely large
//! program and an outcome a caller cannot route on: a stack overflow, an
//! allocation failure, or a run that never returns. A ceiling that is declared
//! but not read is the same as no ceiling, and the failure is silent because
//! the ordinary corpus never reaches any of these numbers.
//!
//! Four bounds are held here. Program size, measured on a program with more
//! than a million IR nodes. Per-node allocation, measured as a difference
//! between two sizes so fixed setup cancels. Frame depth, at both sides of
//! [`vyre_reference::ReferenceBudget::max_recursion_depth`]. Allocation, at
//! both sides of [`vyre_reference::ReferenceBudget::max_memory_bytes`].
//!
//! Each depth and memory case runs at three distinct limits. One limit proves
//! that one number happens to work; three prove the field is read, because a
//! constant baked into the interpreter cannot match all three.
//!
//! # What it does not catch
//!
//! Peak resident memory. The allocation probe counts gross heap traffic on the
//! measuring thread, so a run that allocates and frees the same buffer a
//! million times is charged a million times and a run that holds one large
//! buffer is charged once. The ceiling here bounds traffic per node, not the
//! high-water mark.
//!
//! Allocation the oracle does not make through the Rust global allocator, and
//! allocation on any thread other than the one under measurement.
//!
//! The bounds of a program built from an operation registry rather than from
//! IR constructors. The shapes here are synthetic, so a registry entry whose
//! lowering allocates per element is outside what these numbers say.

use std::sync::mpsc::{channel, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use vyre_alloc_probe::{Region, ThreadAlloc};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::visit::child_bodies;
use vyre_reference::{ReferenceBudget, ReferenceErrorClass, ReferenceRequest};

/// Charges every allocation to the thread that made it.
///
/// `cargo test` runs cases on a thread pool, so a process-wide counter would
/// charge the per-node measurement for every allocation every other case in
/// this binary made between its two snapshots.
#[global_allocator]
static GLOBAL: ThreadAlloc = ThreadAlloc;

/// Wall-clock ceiling one bounded evaluation in this file may take.
///
/// Measured, not chosen: the million-node refusal below arrives in 4.2 seconds
/// under the unoptimized test profile, for both the flat and the loop-wrapped
/// shape. The ceiling stands an order of magnitude above that so a loaded host
/// does not fail the run, and far enough below an unbounded evaluation that a
/// lost bound is a failed assertion rather than a stalled suite: the
/// loop-wrapped program declares two billion statements, which the interpreter
/// would grind through for hours if the program-size ceiling stopped refusing
/// it.
const EVALUATION_DEADLINE: Duration = Duration::from_secs(45);

/// IR nodes the program-size proof builds.
const MILLION_NODES: usize = 1_000_000;

/// Bytes the oracle allocates per IR node, measured.
///
/// 887.3 bytes per node, as the difference between a 20,001-node and an
/// 80,001-node evaluation divided by the 60,000-node delta, under the
/// unoptimized test profile. The figure is stable across that span because
/// fixed per-evaluation setup cancels in the difference.
const MEASURED_BYTES_PER_NODE: f64 = 887.3;

/// Bytes per IR node the oracle may allocate.
///
/// [`MEASURED_BYTES_PER_NODE`] plus a third, which absorbs allocator size-class
/// rounding and the growth steps of the interpreter's own vectors without
/// admitting a per-node scratch buffer.
const BYTES_PER_NODE_CEILING: f64 = 1200.0;

/// What one bounded evaluation ended as.
///
/// Reduced to what the bound contract states, and to types that cross a thread
/// boundary: the evaluation produced outputs, or it refused with a class and a
/// message.
#[derive(Debug)]
enum Verdict {
    /// The evaluation produced this many output buffers.
    Outputs(usize),
    /// The evaluation refused with this class and message.
    Refused(ReferenceErrorClass, String),
}

/// IR nodes reachable from `nodes`, counted by walking the built program.
///
/// The generator's loop bound states what it intended to build. This states
/// what it built, including whatever `Program::wrapped` adds around the body.
fn count_nodes(nodes: &[Node]) -> usize {
    let mut pending: Vec<&[Node]> = vec![nodes];
    let mut total = 0usize;
    while let Some(body) = pending.pop() {
        total += body.len();
        for node in body {
            for child in child_bodies(node) {
                if !child.is_empty() {
                    pending.push(child);
                }
            }
        }
    }
    total
}

/// Frame depth a lane reaches at the deepest point of `nodes`.
///
/// Counts what the interpreter counts when it pushes a frame: the entry node
/// list is frame one, and each nested body a lane enters adds one. A loop
/// pushes a second frame for its own iteration state, so this is the depth the
/// budget's static pre-check reads and a lower bound on the depth a lane
/// reaches while running.
///
/// Iterative over an explicit worklist, because the nesting this measures is
/// exactly the nesting a recursive walk cannot survive.
fn static_frame_depth(nodes: &[Node]) -> usize {
    let mut deepest = 1usize;
    let mut pending: Vec<(&[Node], usize)> = vec![(nodes, 1)];
    while let Some((body, depth)) = pending.pop() {
        deepest = deepest.max(depth);
        for node in body {
            for child in child_bodies(node) {
                if !child.is_empty() {
                    pending.push((child, depth + 1));
                }
            }
        }
    }
    deepest
}

/// A program whose entry body is `statements` unnested stores.
fn flat_chain(statements: usize) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        store_chain(statements),
    )
}

/// A program whose `statements` stores sit inside one loop of `trips` trips.
///
/// The node count is the same as [`flat_chain`], and the work the program
/// declares is `trips` times larger. That separates the two bounds: the size
/// ceiling refuses this program before anything runs, and without that ceiling
/// the declared work is what the run would have to execute.
fn looped_chain(statements: usize, trips: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::loop_(
            "outer",
            Expr::u32(0),
            Expr::u32(trips),
            store_chain(statements),
        )],
    )
}

/// `statements` stores into the single-element `out` buffer.
fn store_chain(statements: usize) -> Vec<Node> {
    (0..statements)
        .map(|index| {
            Node::store(
                "out",
                Expr::u32(0),
                Expr::u32(u32::try_from(index % 1000).unwrap_or(0)),
            )
        })
        .collect()
}

/// A program whose single store sits under `depth` nested blocks.
///
/// A block adds one frame, so the frame depth a lane reaches equals the static
/// nesting and the budget's static pre-check is what refuses a limit below it.
fn nested_blocks(depth: usize) -> Program {
    let mut body = vec![Node::store("out", Expr::u32(0), Expr::u32(7))];
    for _ in 0..depth {
        body = vec![Node::block(body)];
    }
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        body,
    )
}

/// A program whose single store sits under `depth` nested loops.
///
/// A loop adds two frames, one for its body and one for its iteration state,
/// so the depth a lane reaches is twice the static nesting. A limit between
/// the two passes the static pre-check and has to be refused while the lane
/// runs, which is the only shape that exercises the per-frame check on its
/// own.
fn nested_loops(depth: usize) -> Program {
    let mut body = vec![Node::store("out", Expr::u32(0), Expr::u32(7))];
    for level in 0..depth {
        body = vec![Node::loop_(
            format!("i{level}"),
            Expr::u32(0),
            Expr::u32(2),
            body,
        )];
    }
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        body,
    )
}

/// A program declaring one output buffer of `elements` `u32` values.
fn sized_output(elements: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(elements)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    )
}

/// Evaluate `program` under `budget` on its own thread and refuse to wait past
/// [`EVALUATION_DEADLINE`].
///
/// The deadline is the point of the call. A bound that stops being enforced
/// turns the run into a stalled suite with no failure text, so the evaluation
/// runs where the assertion can outlive it and the timeout is reported as the
/// termination failure it is.
///
/// # Panics
/// Panics when the evaluation has not terminated within the deadline, and
/// resumes the evaluation's own panic when it had one.
fn verdict_within_deadline(program: Program, budget: ReferenceBudget, label: &str) -> Verdict {
    let (sender, receiver) = channel();
    let started = Instant::now();
    let worker = thread::Builder::new()
        .name(format!("oracle-bounds-{label}"))
        .spawn(move || {
            let verdict = match ReferenceRequest::new(&program, &[], budget).outputs() {
                Ok(outputs) => Verdict::Outputs(outputs.len()),
                Err(error) => {
                    Verdict::Refused(error.error_class(), error.message().to_string())
                }
            };
            drop(sender.send(verdict));
        })
        .expect("spawning the evaluation thread");
    match receiver.recv_timeout(EVALUATION_DEADLINE) {
        Ok(verdict) => {
            let elapsed = started.elapsed();
            worker.join().expect("joining the evaluation thread");
            assert!(
                elapsed <= EVALUATION_DEADLINE,
                "the {label} evaluation took {elapsed:?}, past the {EVALUATION_DEADLINE:?} \
                 ceiling this test states"
            );
            verdict
        }
        Err(RecvTimeoutError::Timeout) => panic!(
            "the {label} evaluation had not terminated {EVALUATION_DEADLINE:?} after it started. \
             The oracle bounds a program by its node count, its declared work, its allocation, \
             and its frame depth; a run that passes this deadline has lost one of them and is \
             unbounded rather than slow."
        ),
        Err(RecvTimeoutError::Disconnected) => {
            let panic = worker
                .join()
                .expect_err("the evaluation thread dropped its sender without panicking");
            std::panic::resume_unwind(panic)
        }
    }
}

/// Assert that `verdict` is one of the only two outcomes a bounded evaluation
/// may end in.
fn assert_bounded_outcome(verdict: &Verdict, label: &str) {
    match verdict {
        Verdict::Outputs(count) => assert_eq!(
            *count, 1,
            "the {label} program declares one output buffer and evaluated to {count}"
        ),
        Verdict::Refused(ReferenceErrorClass::BudgetExhaustion, _) => {}
        Verdict::Refused(class, message) => panic!(
            "the {label} evaluation refused with class {class:?}, not `BudgetExhaustion`. A \
             program refused for its size crossed a resource ceiling, and any other class tells \
             a caller something else went wrong. Message: {message}"
        ),
    }
}

/// A program with more than a million IR nodes ends in outputs or in a
/// `BudgetExhaustion` refusal, within a stated wall-clock ceiling.
///
/// # The class closed here
///
/// An oracle that walks a program per invocation has three ways to answer a
/// program far larger than its corpus that are all worse than a refusal: a
/// stack overflow in a recursive walk, an allocation failure, and a run that
/// never returns. None of the three is an outcome a caller can route on, and
/// two of them end the host process rather than the request. The contract is
/// that a program of any size ends in a value or in a structured refusal that
/// names a ceiling, and that it ends at all.
///
/// Both shapes carry the same node count and differ in declared work. The flat
/// chain declares a million statements, which the armed work ceiling admits,
/// so nothing but the size ceiling can refuse it. The loop-wrapped chain
/// declares two billion, so the deadline is load-bearing: without the size
/// ceiling that run does not finish.
///
/// # What it does not catch
///
/// A program built from a million nodes of a variant these two shapes do not
/// use. Node count is the ceiling that fires here, and it is counted the same
/// way for every variant, but the walk cost per node is not.
///
/// A stack overflow deeper than the 64-level nesting validation admits. Both
/// shapes here are shallow, so the walk cost they prove is breadth, not depth.
#[test]
fn a_million_node_program_ends_in_a_bounded_refusal() {
    for (label, program) in [
        ("flat", flat_chain(MILLION_NODES)),
        ("loop-wrapped", looped_chain(MILLION_NODES, 2000)),
    ] {
        let nodes = count_nodes(program.entry());
        assert!(
            nodes >= MILLION_NODES,
            "the {label} program walks to {nodes} nodes, under the {MILLION_NODES} this proof \
             requires. Fix: build the program the generator claims to build."
        );
        let verdict =
            verdict_within_deadline(program, ReferenceBudget::with_work_ceiling(64_000_000), label);
        assert_bounded_outcome(&verdict, label);
    }
}

/// The oracle allocates under a measured ceiling per IR node.
///
/// # The class closed here
///
/// A per-node allocation that grows with the program is invisible in the
/// fixture corpus and fatal at scale: a scratch buffer taken per node costs
/// nothing at fifty nodes and exhausts the host at a hundred thousand. The
/// figure is measured as a difference between two program sizes so that fixed
/// per-evaluation setup, which does not scale with the program, cancels
/// instead of being amortized into the number.
///
/// # What it does not catch
///
/// Peak resident memory, and anything the oracle allocates off the measuring
/// thread. Gross traffic is what the probe counts, so a run that churns one
/// buffer is charged for every turn of the churn.
///
/// A per-node cost that is constant in node count and large in some other
/// dimension, such as buffer element count or grid size. Both programs here
/// declare one element and one invocation.
#[test]
fn the_oracle_allocates_under_a_measured_ceiling_per_ir_node() {
    // One evaluation before the first measurement, so lazily initialized state
    // shared by every evaluation is charged to neither of the two.
    drop(ReferenceRequest::standard(&flat_chain(64), &[]).outputs());

    let small = flat_chain(20_000);
    let small_nodes = count_nodes(small.entry());
    let small_region = Region::new();
    let small_outputs = ReferenceRequest::standard(&small, &[]).outputs();
    let small_change = small_region.change();

    let large = flat_chain(80_000);
    let large_nodes = count_nodes(large.entry());
    let large_region = Region::new();
    let large_outputs = ReferenceRequest::standard(&large, &[]).outputs();
    let large_change = large_region.change();

    assert!(
        small_outputs.is_ok() && large_outputs.is_ok(),
        "both measured programs must evaluate, or the difference measures a refusal path rather \
         than an evaluation. Small: {small_outputs:?}. Large: {large_outputs:?}"
    );

    let node_delta = large_nodes - small_nodes;
    let byte_delta = large_change
        .bytes_allocated
        .checked_sub(small_change.bytes_allocated)
        .expect("the larger program must allocate at least as much as the smaller one");
    let per_node = byte_delta as f64 / node_delta as f64;

    assert!(
        per_node < BYTES_PER_NODE_CEILING,
        "the oracle allocates {per_node:.1} bytes per IR node, past the \
         {BYTES_PER_NODE_CEILING:.1} byte ceiling. Measured at {MEASURED_BYTES_PER_NODE:.1} \
         bytes across {node_delta} nodes: {} bytes for {small_nodes} nodes and {} bytes for \
         {large_nodes}. Fix: stop taking a per-node allocation, or restate the ceiling from a \
         fresh measurement.",
        small_change.bytes_allocated,
        large_change.bytes_allocated
    );
}

/// `max_recursion_depth` refuses a program one frame past the limit and admits
/// one exactly at it, at three distinct limits.
///
/// # The class closed here
///
/// A frame ceiling that is declared and not read lets a deeply nested program
/// exhaust the host stack, which ends the process rather than the request. Two
/// checks enforce it and each can be lost on its own: a static pre-check that
/// reads the program's nesting before any walk, and a per-frame check while a
/// lane runs. Block nesting reaches the same depth statically and dynamically,
/// so it holds the pre-check. Loop nesting reaches twice its static depth,
/// because a loop pushes a frame for its iteration state, so a limit at the
/// static depth passes the pre-check and only the per-frame check can refuse
/// it.
///
/// Three limits per shape, all distinct, so a constant baked into the
/// interpreter cannot pass: it would have to equal all three.
///
/// # What it does not catch
///
/// Nesting past the 64 levels IR validation admits, which is refused for its
/// nesting rather than against this budget. The limits proved here are 6, 18,
/// and 50 frames for blocks and 6, 10, and 18 for loops, all inside that.
///
/// Frame depth reached through an operation call rather than through block or
/// loop nesting.
#[test]
fn the_frame_ceiling_refuses_one_frame_past_the_limit_and_admits_the_limit() {
    let mut block_limits = Vec::new();
    for depth in [4usize, 16, 48] {
        let program = nested_blocks(depth);
        let limit = static_frame_depth(program.entry());
        block_limits.push(limit);
        assert_admits(
            &program,
            ReferenceBudget::new(u64::MAX, 1 << 30, limit),
            "block",
            limit,
        );
        assert_refuses_depth(
            &program,
            "block",
            limit - 1,
            &format!("the program's static body nesting reaches frame depth {limit}"),
        );
    }
    assert_distinct(&block_limits, "block nesting");

    let mut loop_limits = Vec::new();
    for depth in [2usize, 4, 8] {
        let program = nested_loops(depth);
        let statically = static_frame_depth(program.entry());
        let limit = 2 * depth + 2;
        loop_limits.push(limit);
        assert!(
            statically < limit,
            "a loop shape whose static nesting {statically} already reaches the running depth \
             {limit} does not exercise the per-frame check"
        );
        assert_admits(
            &program,
            ReferenceBudget::new(u64::MAX, 1 << 30, limit),
            "loop",
            limit,
        );
        assert_refuses_depth(
            &program,
            "loop",
            limit - 1,
            &format!(
                "entering a nested body at frame depth {limit} passes the {} frame ceiling",
                limit - 1
            ),
        );
        // At the static nesting the pre-check is satisfied and the lane still
        // runs past the ceiling, so this side is held by the per-frame check
        // alone.
        assert_refuses_depth(
            &program,
            "loop",
            statically,
            &format!(
                "entering a nested body at frame depth {} passes the {statically} frame ceiling",
                statically + 1
            ),
        );
    }
    assert_distinct(&loop_limits, "loop nesting");
}

/// `max_memory_bytes` refuses a program one byte past the limit and admits one
/// exactly at it, at three distinct limits.
///
/// # The class closed here
///
/// The oracle materializes every declared buffer of a whole dispatch on the
/// host, so a program that declares more memory than the host has ends in an
/// allocation failure that takes the process with it unless the ceiling is
/// read. Proving only the refusing side would pass against a ceiling stuck at
/// zero, and proving only the admitting side would pass against no ceiling at
/// all, so both sides of each limit are asserted.
///
/// Three limits, all distinct, so a constant baked into the interpreter cannot
/// pass.
///
/// # What it does not catch
///
/// Memory the oracle allocates that is not a declared buffer: interpreter
/// bookkeeping, the node arena, and per-invocation state are outside this
/// ceiling. `the_oracle_allocates_under_a_measured_ceiling_per_ir_node` bounds
/// that traffic instead.
///
/// The workgroup memory charge, which shares this ceiling but is refused by
/// its own smaller fixed bound first.
#[test]
fn the_allocation_ceiling_refuses_one_byte_past_the_limit_and_admits_the_limit() {
    let mut limits = Vec::new();
    for elements in [4u32, 1024, 65_536] {
        let program = sized_output(elements);
        let declared = elements as usize * size_of::<u32>();
        limits.push(declared);

        let admitted = ReferenceRequest::new(
            &program,
            &[],
            ReferenceBudget::new(u64::MAX, declared, 1024),
        )
        .outputs();
        assert!(
            admitted.is_ok(),
            "a program declaring {declared} bytes was refused against a {declared} byte ceiling: \
             {admitted:?}. The ceiling bounds what an evaluation may allocate, so a program that \
             fits it exactly is inside the bound."
        );

        let error = ReferenceRequest::new(
            &program,
            &[],
            ReferenceBudget::new(u64::MAX, declared - 1, 1024),
        )
        .outputs()
        .expect_err("a program declaring one byte more than the ceiling must be refused");
        assert_eq!(
            error.error_class(),
            ReferenceErrorClass::BudgetExhaustion,
            "a program past the allocation ceiling refused with the wrong class. Message: {}",
            error.message()
        );
        let expected = format!(
            "reaches {declared} bytes, past the {} byte allocation ceiling",
            declared - 1
        );
        assert!(
            error.message().contains(&expected),
            "the refusal does not state the ceiling it crossed. Expected to contain \
             `{expected}`, got: {}",
            error.message()
        );
    }
    assert_distinct(&limits, "allocation");
}

/// Assert `program`, whose nesting is `shape`, evaluates under `budget`, whose
/// frame ceiling is `limit`.
fn assert_admits(program: &Program, budget: ReferenceBudget, shape: &str, limit: usize) {
    let outcome = ReferenceRequest::new(program, &[], budget).outputs();
    assert!(
        outcome.is_ok(),
        "a {shape}-nested program reaching frame depth {limit} was refused against a {limit} \
         frame ceiling: {outcome:?}. The ceiling bounds the depth a lane may reach, so a program \
         that reaches it exactly is inside the bound."
    );
}

/// Assert `program`, whose nesting is `shape`, is refused with
/// `BudgetExhaustion` under a frame ceiling of `limit`, and that the refusal
/// states `expected`.
fn assert_refuses_depth(program: &Program, shape: &str, limit: usize, expected: &str) {
    let outcome = ReferenceRequest::new(
        program,
        &[],
        ReferenceBudget::new(u64::MAX, 1 << 30, limit),
    )
    .outputs();
    let error = match outcome {
        Ok(outputs) => panic!(
            "a {shape}-nested program past a {limit} frame ceiling evaluated to {} output \
             buffer(s) instead of being refused. `ReferenceBudget::max_recursion_depth` is not \
             read.",
            outputs.len()
        ),
        Err(error) => error,
    };
    assert_eq!(
        error.error_class(),
        ReferenceErrorClass::BudgetExhaustion,
        "a {shape}-nested program past the {limit} frame ceiling refused with the wrong class. \
         Message: {}",
        error.message()
    );
    assert!(
        error.message().contains(expected),
        "the {shape}-nested refusal does not state the depth it crossed. Expected to contain \
         `{expected}`, got: {}",
        error.message()
    );
}

/// Assert every limit in `limits` differs from the others.
///
/// A proof that runs three times against one limit proves one number works.
/// Distinct limits are what make it a proof that the field is read.
fn assert_distinct(limits: &[usize], shape: &str) {
    let mut sorted = limits.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        limits.len(),
        "the {shape} proof ran against {limits:?}, which are not three distinct limits. A repeated \
         limit proves one number works rather than that the budget field is read."
    );
}

