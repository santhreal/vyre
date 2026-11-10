//! Program shapes and the bounded runner every program-size proof shares.
//!
//! Two suites state a claim about what the oracle does with a program far
//! larger than its fixture corpus: one that the default ceilings refuse it,
//! and one that an explicit ceiling executes it. Both need the same million-node
//! shapes and the same way of running an evaluation where a lost bound shows up
//! as a failed assertion rather than a suite that never finishes, so the shapes
//! and the runner are defined once here.
//!
//! The runner reports what the evaluation cost as well as what it produced,
//! because a size claim is a claim about time and memory and not only about the
//! output bytes.

use std::sync::mpsc::{channel, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use vyre_alloc_probe::Region;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::visit::child_bodies;
use vyre_reference::value::Value;
use vyre_reference::{ReferenceBudget, ReferenceErrorClass, ReferenceRequest};

/// Wall-clock ceiling one bounded evaluation may take.
///
/// Measured, not chosen: under the test profile, which builds `vyre-reference`
/// and `vyre-foundation` at opt-level 3, the million-node refusal arrives in
/// 0.26 seconds for the flat shape and 0.31 seconds for the loop-wrapped shape,
/// and the million-node execution completes in 0.36 seconds. The ceiling stands
/// two orders of magnitude above that so a loaded or instrumented host does not
/// fail the run, and far enough below an unbounded evaluation that a lost bound
/// is a failed assertion rather than a stalled suite: the loop-wrapped program
/// declares two billion statements, which the interpreter would grind through
/// for hours if the program-size ceiling stopped refusing it.
pub const EVALUATION_DEADLINE: Duration = Duration::from_secs(45);

/// IR nodes the program-size proofs build.
pub const MILLION_NODES: usize = 1_000_000;

/// How often the runner samples process resident size while an evaluation runs.
const RESIDENT_SAMPLE_INTERVAL: Duration = Duration::from_millis(20);

/// What one bounded evaluation ended as.
///
/// Reduced to what the bound contract states, and to types that cross a thread
/// boundary: the evaluation produced outputs, or it refused with a class and a
/// message.
#[derive(Debug)]
pub enum Verdict {
    /// The evaluation produced these output buffers.
    Outputs(Vec<Value>),
    /// The evaluation refused with this class and message.
    Refused(ReferenceErrorClass, String),
}

/// One bounded evaluation: what it ended as, and what it cost.
#[derive(Debug)]
pub struct BoundedRun {
    /// What the evaluation ended as.
    pub verdict: Verdict,
    /// Wall clock from spawning the evaluation to receiving its verdict.
    pub elapsed: Duration,
    /// Heap bytes the evaluation still held when it returned its verdict,
    /// measured on the evaluation's own thread as allocated minus deallocated.
    pub retained_bytes: usize,
    /// Heap bytes the evaluation requested in total, measured on its own
    /// thread.
    ///
    /// Gross traffic, so a run that takes and frees the same buffer is charged
    /// for every turn. It is attributable in a way process resident size is
    /// not: the counters are thread-local, so a suite running other cases
    /// alongside this one does not move the number.
    pub allocated_bytes: usize,
    /// Peak process resident size observed while the evaluation ran, above the
    /// reading taken before it started.
    ///
    /// `None` where process resident size cannot be read, which leaves the
    /// retained-byte bound as the only memory bound the run states.
    pub peak_resident_bytes: Option<usize>,
}

/// IR nodes reachable from `nodes`, counted by walking the built program.
///
/// The generator's loop bound states what it intended to build. This states
/// what it built, including whatever `Program::wrapped` adds around the body.
pub fn count_nodes(nodes: &[Node]) -> usize {
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

/// A program whose entry body is `statements` unnested stores.
///
/// Every store writes `out[0]`, so the program's declared output is the value
/// the last store wrote and [`flat_chain_output`] states it without running
/// anything.
pub fn flat_chain(statements: usize) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        store_chain(statements),
    )
}

/// The `u32` a [`flat_chain`] of `statements` stores leaves in `out[0]`.
///
/// Derived from the generator rather than recorded from a run, so the expected
/// output is a statement about the program and not about the interpreter.
#[must_use]
pub fn flat_chain_output(statements: usize) -> u32 {
    let last = statements.saturating_sub(1);
    u32::try_from(last % 1000).unwrap_or(0)
}

/// A program whose `statements` stores sit inside one loop of `trips` trips.
///
/// The node count is the same as [`flat_chain`], and the work the program
/// declares is `trips` times larger. That separates the two bounds: the size
/// ceiling refuses this program before anything runs, and without that ceiling
/// the declared work is what the run would have to execute.
pub fn looped_chain(statements: usize, trips: u32) -> Program {
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

/// Process resident size in bytes, or `None` where it cannot be read.
fn resident_bytes() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kilobytes = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))?
        .split_whitespace()
        .next()?
        .parse::<usize>()
        .ok()?;
    Some(kilobytes * 1024)
}

/// Evaluate `program` under `budget` on its own thread and refuse to wait past
/// [`EVALUATION_DEADLINE`].
///
/// The deadline is the point of the call. A bound that stops being enforced
/// turns the run into a stalled suite with no failure text, so the evaluation
/// runs where the assertion can outlive it and the timeout is reported as the
/// termination failure it is.
///
/// Resident size is sampled from the waiting thread rather than the evaluating
/// one, because the peak of a run is inside the run and a thread cannot report
/// its own high-water mark after the fact. The sample is process-wide, so it
/// carries whatever else the test binary holds; it is a ceiling on the run, not
/// an attribution of every byte to it.
///
/// # Panics
/// Panics when the evaluation has not terminated within the deadline, and
/// resumes the evaluation's own panic when it had one.
pub fn run_within_deadline(program: Program, budget: ReferenceBudget, label: &str) -> BoundedRun {
    let (sender, receiver) = channel();
    let baseline_resident = resident_bytes();
    let started = Instant::now();
    let worker = thread::Builder::new()
        .name(format!("program-scale-{label}"))
        .spawn(move || {
            let region = Region::new();
            let verdict = match ReferenceRequest::new(&program, &[], budget).outputs() {
                Ok(outputs) => Verdict::Outputs(outputs),
                Err(error) => Verdict::Refused(error.error_class(), error.message().to_string()),
            };
            let change = region.change();
            let retained = usize::try_from(change.net_bytes()).unwrap_or(0);
            drop(sender.send((verdict, retained, change.bytes_allocated)));
        })
        .expect("spawning the evaluation thread");

    let mut peak_resident = baseline_resident;
    loop {
        match receiver.recv_timeout(RESIDENT_SAMPLE_INTERVAL) {
            Ok((verdict, retained_bytes, allocated_bytes)) => {
                let elapsed = started.elapsed();
                worker.join().expect("joining the evaluation thread");
                assert!(
                    elapsed <= EVALUATION_DEADLINE,
                    "the {label} evaluation took {elapsed:?}, past the {EVALUATION_DEADLINE:?} \
                     ceiling this runner states"
                );
                return BoundedRun {
                    verdict,
                    elapsed,
                    retained_bytes,
                    allocated_bytes,
                    peak_resident_bytes: peak_resident
                        .zip(baseline_resident)
                        .map(|(peak, baseline)| peak.saturating_sub(baseline)),
                };
            }
            Err(RecvTimeoutError::Timeout) => {
                if let (Some(peak), Some(sample)) = (peak_resident.as_mut(), resident_bytes()) {
                    *peak = (*peak).max(sample);
                }
                assert!(
                    started.elapsed() <= EVALUATION_DEADLINE,
                    "the {label} evaluation had not terminated {EVALUATION_DEADLINE:?} after it \
                     started. The oracle bounds a program by its node count, its declared work, \
                     its allocation, and its frame depth; a run that passes this deadline has lost \
                     one of them and is unbounded rather than slow."
                );
            }
            Err(RecvTimeoutError::Disconnected) => {
                let panic = worker
                    .join()
                    .expect_err("the evaluation thread dropped its sender without panicking");
                std::panic::resume_unwind(panic)
            }
        }
    }
}

/// Assert that `verdict` is one of the only two outcomes a bounded evaluation
/// may end in.
pub fn assert_bounded_outcome(verdict: &Verdict, label: &str) {
    match verdict {
        Verdict::Outputs(outputs) => assert_eq!(
            outputs.len(),
            1,
            "the {label} program declares one output buffer and evaluated to {}",
            outputs.len()
        ),
        Verdict::Refused(ReferenceErrorClass::BudgetExhaustion, _) => {}
        Verdict::Refused(class, message) => panic!(
            "the {label} evaluation refused with class {class:?}, not `BudgetExhaustion`. A \
             program refused for its size crossed a resource ceiling, and any other class tells \
             a caller something else went wrong. Message: {message}"
        ),
    }
}
