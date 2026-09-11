//! The canonical reference evaluator and its entry points.
//!
//! Every entry point here resolves the same way: normalize the submitted
//! program to the top-level `Region` model, arm the work budget, and interpret
//! the program through [`hashmap::run_hashmap_reference`]. There is one
//! evaluator, so a node has one meaning for the whole crate.

pub(crate) mod async_transfer;
pub(crate) mod call;
pub(crate) mod expr_cast;
pub(crate) mod hashmap;
pub(crate) mod node_tree;
/// Thread-local arithmetic-IR-op counting for roofline / complexity analysis.
pub mod op_count;
/// One-expression entry point into the canonical evaluator.
pub(crate) mod single_expr;
/// Work ceiling that gives the interpreter a termination contract.
pub mod step_budget;
pub(crate) mod tile;
pub(crate) mod typed_ops;

use std::borrow::Cow;
use vyre_foundation::ir::{Node, Program};

use crate::value::Value;

pub(crate) fn axis_value(values: [u32; 3], axis: u8) -> Result<Value, crate::ReferenceError> {
    (axis < 3)
        .then(|| Value::U32(values[axis as usize]))
        .ok_or_else(|| {
            crate::ReferenceError::incomplete_dispatch_semantics(format!(
                "invocation/workgroup ID axis {axis} out of range. Fix: use 0, 1, or 2."
            ))
        })
}

/// If the program satisfies the public top-level-Region model, return a
/// byte-identical clone. If not, the usual case is
/// `optimizer::passes::cleanup::region_inline_engine` having flattened a Category-A wrapper;
/// in that case [`Program::reconcile_runnable_top_level`] matches
/// `Program::wrapped` again. When the first entry node is a `Store` (or the
/// entry is empty), we do **not** auto-wrap: those programs must still use
/// `Program::wrapped` explicitly, matching `region_gate` negative tests.
pub(crate) fn program_for_interpreter(
    program: &Program,
) -> Result<Cow<'_, Program>, crate::ReferenceError> {
    let normalized = if let Some(message) = program.top_level_region_violation() {
        if program.entry().is_empty() {
            return Err(crate::ReferenceError::new(format!(
                "reference interpreter requires a top-level Region-wrapped Program: {message}"
            )));
        }
        if matches!(program.entry().first(), Some(Node::Store { .. })) {
            return Err(crate::ReferenceError::new(format!(
                "reference interpreter requires a top-level Region-wrapped Program: {message}"
            )));
        }
        Cow::Owned(program.clone().reconcile_runnable_top_level())
    } else {
        Cow::Borrowed(program)
    };
    Ok(normalized)
}

/// Deterministic step orders one schedule policy explores, in the order it
/// explores them.
///
/// The match has no catch-all arm, so a new policy states its own exploration
/// rather than borrowing the previous variant's. `BoundedInterleaving` used to
/// borrow `Forward` here, which made every parity result under that policy a
/// claim about a schedule the oracle never ran.
fn explored_step_orders(
    policy: crate::request::DeterministicSchedulePolicy,
    program: &Program,
) -> Vec<hashmap::LaneOrder> {
    match policy {
        crate::request::DeterministicSchedulePolicy::Forward => vec![hashmap::LaneOrder::Forward],
        crate::request::DeterministicSchedulePolicy::LaneReversed => {
            vec![hashmap::LaneOrder::Reversed]
        }
        crate::request::DeterministicSchedulePolicy::LaneRotated(by) => {
            vec![hashmap::LaneOrder::Rotated(by)]
        }
        crate::request::DeterministicSchedulePolicy::BoundedInterleaving => {
            bounded_interleaving_orders(program)
        }
    }
}

/// Step orders a bounded interleaving exploration covers.
///
/// Forward, reversed, and the rotations that move at least one lane without
/// repeating the forward order, capped at
/// [`MAX_BOUNDED_INTERLEAVINGS`]. The rotation count comes from the workgroup
/// extent the program declares, so a one-lane workgroup explores one schedule
/// and a wide one explores the cap rather than a number chosen here.
fn bounded_interleaving_orders(program: &Program) -> Vec<hashmap::LaneOrder> {
    let [sx, sy, sz] = program.workgroup_size();
    let lanes = [sx, sy, sz].iter().copied().fold(1u32, u32::saturating_mul);
    let mut orders = vec![hashmap::LaneOrder::Forward];
    if lanes <= 1 {
        return orders;
    }
    orders.push(hashmap::LaneOrder::Reversed);
    for by in 1..lanes {
        if orders.len() >= MAX_BOUNDED_INTERLEAVINGS {
            break;
        }
        orders.push(hashmap::LaneOrder::Rotated(by));
    }
    orders
}

/// Schedules one bounded interleaving exploration runs at most.
///
/// The exploration is bounded so the oracle keeps a termination contract: the
/// work budget covers every schedule together, and a wide workgroup would
/// otherwise multiply one evaluation by its lane count.
const MAX_BOUNDED_INTERLEAVINGS: usize = 4;

/// Step orders a bounded race exploration covers, in the order it runs them.
///
/// Forward is the baseline every other order is compared against.
/// `WorkgroupReversed` moves the workgroup axis alone, which is the only order
/// in the set that separates a cross-workgroup conflict from an intra-workgroup
/// one: every other order permutes both axes together, so a conflict whose two
/// writers sit in different workgroups keeps the same last writer once the two
/// permutations cancel. Reversed and the rotations then move the lane axis, the
/// rotations asymmetrically so a defect that maps lane identity onto step
/// position cannot survive by symmetry.
///
/// The count is a function of the workgroup extent the program declares, so a
/// caller can state the exact number of orders an exploration will run before
/// it runs, and a one-lane workgroup does not pay for rotations that permute
/// nothing.
pub(crate) fn race_exploration_orders(program: &Program) -> Vec<hashmap::LaneOrder> {
    let [sx, sy, sz] = program.workgroup_size();
    let lanes = [sx, sy, sz].iter().copied().fold(1u32, u32::saturating_mul);
    let mut orders = vec![
        hashmap::LaneOrder::Forward,
        hashmap::LaneOrder::WorkgroupReversed,
    ];
    if lanes <= 1 {
        return orders;
    }
    orders.push(hashmap::LaneOrder::Reversed);
    for by in 1..lanes {
        if orders.len() >= MAX_RACE_EXPLORATION_ORDERS {
            break;
        }
        orders.push(hashmap::LaneOrder::Rotated(by));
    }
    orders
}

/// Step orders one bounded race exploration runs at most.
///
/// The exploration is bounded so the oracle keeps a termination contract: one
/// work budget covers every order together, and a wide workgroup would
/// otherwise multiply one evaluation by its lane count.
pub(crate) const MAX_RACE_EXPLORATION_ORDERS: usize = 6;

/// Run `runnable` once per explored step order and return the outputs every
/// order agreed on.
///
/// Two orders that disagree mean the program's result depends on the order the
/// lanes were stepped in, which a device leaves driver-defined. The oracle has
/// no single answer to certify in that case, so it names both schedules and the
/// output that differs.
fn run_explored_orders(
    runnable: &Program,
    request: &crate::request::ReferenceRequest<'_>,
    orders: &[hashmap::LaneOrder],
) -> Result<Vec<Value>, crate::ReferenceError> {
    let min_dispatch = request.workload_envelope.min_dispatch_elements.unwrap_or(0);
    let mut agreed: Option<(hashmap::LaneOrder, Vec<Value>)> = None;
    for &order in orders {
        let outputs = hashmap::run_hashmap_reference(
            runnable,
            &request.resource_abi.inputs,
            min_dispatch,
            order,
            request.workload_envelope.workgroup_grid,
        )?;
        match &agreed {
            None => agreed = Some((order, outputs)),
            Some((first_order, first_outputs)) => {
                if let Some(index) = first_difference(first_outputs, &outputs) {
                    return Err(crate::ReferenceError::incomplete_dispatch_semantics(format!(
                        "schedule exploration disagreed: output {index} differs between step order \
                         {first_order:?} and {order:?}. Fix: give every shared output slot a single \
                         writer, or write it through a commutative atomic, so the program's result \
                         does not depend on the order the lanes were stepped in."
                    )));
                }
            }
        }
    }
    agreed.map(|(_, outputs)| outputs).ok_or_else(|| {
        crate::ReferenceError::incomplete_dispatch_semantics(
            "the schedule policy explored no step order. Fix: state a policy that names at least \
             one deterministic step order.",
        )
    })
}

/// Index of the first output two schedules disagree on.
fn first_difference(left: &[Value], right: &[Value]) -> Option<usize> {
    if left.len() != right.len() {
        return Some(left.len().min(right.len()));
    }
    left.iter()
        .zip(right)
        .position(|(left, right)| left.to_bytes() != right.to_bytes())
}

pub(crate) fn run_with_request(
    request: &crate::request::ReferenceRequest<'_>,
) -> Result<(Vec<Value>, u64), crate::ReferenceError> {
    crate::oob::reset_oob_report();
    let _strictness = crate::oob::enter_strictness(true);
    let runnable = program_for_interpreter(request.program)?;
    let budget = step_budget::arm_with_budget(&runnable, request.budget)?;
    let orders = explored_step_orders(request.schedule_policy, &runnable);
    let outputs = run_explored_orders(&runnable, request, &orders)?;
    let steps = step_budget::charged();
    drop(budget);
    Ok((outputs, steps))
}

pub(crate) fn run_permissive_with_request(
    request: &crate::request::ReferenceRequest<'_>,
) -> Result<(Vec<Value>, u64, crate::oob::OobReport), crate::ReferenceError> {
    crate::oob::reset_oob_report();
    let _strictness = crate::oob::enter_strictness(false);
    let runnable = program_for_interpreter(request.program)?;
    let budget = step_budget::arm_with_budget(&runnable, request.budget)?;
    let orders = explored_step_orders(request.schedule_policy, &runnable);
    let outputs = run_explored_orders(&runnable, request, &orders)?;
    let steps = step_budget::charged();
    let oob = crate::oob::oob_report();
    drop(budget);
    Ok((outputs, steps, oob))
}

/// Execute one request once per explored step order with race tracking on, and
/// report every hazard the exploration found.
///
/// The exploration reports rather than refuses: a racing program yields a
/// report naming each conflict, so a caller sees every hazard in the dispatch
/// instead of the first one. A fault the interpreter cannot continue past
/// (out-of-bounds access under strict mode, budget exhaustion, a malformed
/// program) still ends the exploration with that error.
///
/// One budget covers the whole exploration, armed once before the first order
/// and read after the last, so N orders cannot spend N times the declared work
/// ceiling.
pub(crate) fn explore_races_with_request(
    request: &crate::request::ReferenceRequest<'_>,
) -> Result<crate::interleaving::RaceExplorationReport, crate::ReferenceError> {
    crate::oob::reset_oob_report();
    let _strictness = crate::oob::enter_strictness(true);
    let runnable = program_for_interpreter(request.program)?;
    let budget = step_budget::arm_with_budget(&runnable, request.budget)?;
    let _tracking = crate::interleaving::enter_race_tracking();
    let orders = race_exploration_orders(&runnable);
    let min_dispatch = request.workload_envelope.min_dispatch_elements.unwrap_or(0);
    let mut findings: Vec<crate::interleaving::RaceFinding> = Vec::new();
    let mut baseline: Option<(hashmap::LaneOrder, Vec<Value>)> = None;
    let mut orders_explored = 0usize;
    for &order in &orders {
        crate::interleaving::begin_explored_order();
        let outputs = hashmap::run_hashmap_reference(
            &runnable,
            &request.resource_abi.inputs,
            min_dispatch,
            order,
            request.workload_envelope.workgroup_grid,
        )?;
        orders_explored += 1;
        for finding in crate::interleaving::take_race_findings() {
            if !findings.contains(&finding) {
                findings.push(finding);
            }
        }
        match &baseline {
            None => baseline = Some((order, outputs)),
            Some((first_order, first_outputs)) => {
                if let Some(output_index) = first_difference(first_outputs, &outputs) {
                    let finding = crate::interleaving::RaceFinding::ScheduleDisagreement {
                        first_order: format!("{first_order:?}"),
                        second_order: format!("{order:?}"),
                        output_index,
                    };
                    if !findings.contains(&finding) {
                        findings.push(finding);
                    }
                }
            }
        }
    }
    let steps_executed = step_budget::charged();
    drop(budget);
    Ok(crate::interleaving::RaceExplorationReport {
        orders_explored,
        findings,
        steps_executed,
    })
}

/// The interpreter's output ABI, single-homed: [`is_reference_output`] is the exact
/// predicate the evaluator uses to collect the buffers it returns, and
/// [`output_index`] locates a named output by that predicate. Re-exported so test
/// harnesses never hand-roll (and drift from) the selection.
pub use hashmap::{is_reference_input, is_reference_output, output_index};

/// Project a declaration-order buffer list onto the interpreter's input ABI.
///
/// A request takes one `Value` per [`is_reference_input`] buffer and
/// nothing for a backend-allocated output, which is what a device artifact
/// enforces. A harness that walks `Program::buffers()` naturally produces the
/// longer declaration-order list instead, with a zeroed stand-in per output.
/// This is the one translation between the two, so a harness never re-derives
/// it and drifts.
///
/// A list already sized to the reference inputs passes through. Anything that
/// is neither shape is handed over unchanged: the interpreter names the buffer
/// it is missing a `Value` for, which is a better diagnostic than a count, and
/// leaving the refusal with the ABI's owner keeps this a translation rather
/// than a second validator with its own opinion.
#[must_use]
pub fn reference_inputs(program: &Program, buffers: Vec<Vec<u8>>) -> Vec<Value> {
    let declared = program
        .buffers()
        .iter()
        .filter(|decl| decl.access() != vyre_foundation::ir::BufferAccess::Workgroup)
        .count();
    let logical = program
        .buffers()
        .iter()
        .filter(|decl| is_reference_input(decl))
        .count();
    if buffers.len() != declared || declared == logical {
        return buffers.into_iter().map(Value::from).collect();
    }
    program
        .buffers()
        .iter()
        .filter(|decl| decl.access() != vyre_foundation::ir::BufferAccess::Workgroup)
        .zip(buffers)
        .filter(|(decl, _)| is_reference_input(decl))
        .map(|(_, bytes)| Value::from(bytes))
        .collect()
}

/// The reference input list does not match what the program declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceInputMismatch {
    /// Reference inputs the program declares.
    pub expected: usize,
    /// Buffers the caller supplied.
    pub received: usize,
    /// First declared reference input with no supplied buffer, when the caller
    /// supplied too few.
    pub missing: Option<String>,
}

impl std::fmt::Display for ReferenceInputMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.missing {
            Some(name) => write!(
                f,
                "missing an input buffer for `{name}`: the program declares {} reference input(s) and the caller supplied {}",
                self.expected, self.received
            ),
            None => write!(
                f,
                "{} extra input buffer(s): the program declares {} reference input(s) and the caller supplied {}",
                self.received.saturating_sub(self.expected),
                self.expected,
                self.received
            ),
        }
    }
}

/// One `Value` per [`is_reference_input`] buffer, taken from `inputs` in
/// declaration order.
///
/// A caller holding borrowed bytes reads this rather than walking
/// `Program::buffers()` itself. Two callers walked it, each spelling the
/// selection out as `access() != Workgroup && !is_backend_allocated_output()`,
/// which is the drifted form [`is_reference_input`] documents: it admits a
/// `Shared` buffer, a `Persistent` buffer, and a non-read-write
/// `pipeline_live_out`, none of which a device stages from the host. A program
/// declaring one of those consumed an input the device never asks for, so every
/// later buffer read the value before it.
pub fn reference_input_values(
    program: &Program,
    inputs: &[&[u8]],
) -> Result<Vec<Value>, ReferenceInputMismatch> {
    let expected = program
        .buffers()
        .iter()
        .filter(|decl| is_reference_input(decl))
        .count();
    if expected != inputs.len() {
        let missing = program
            .buffers()
            .iter()
            .filter(|decl| is_reference_input(decl))
            .nth(inputs.len())
            .map(|decl| decl.name().to_string());
        return Err(ReferenceInputMismatch {
            expected,
            received: inputs.len(),
            missing,
        });
    }
    Ok(inputs.iter().copied().map(Value::from).collect())
}

// Inline: reaches the crate-private normalization the public entry points share.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};

    #[test]
    fn the_oracle_dispatches_singleton_atomic_flags_across_dynamic_byte_input() {
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("bytes_in", 0, BufferAccess::ReadOnly, DataType::U8)
                    .with_count(0),
                BufferDecl::storage("flag", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(1),
            ],
            [256, 1, 1],
            vec![
                Node::let_bind("i", Expr::InvocationId { axis: 0 }),
                Node::if_then(
                    Expr::lt(Expr::var("i"), Expr::buf_len("bytes_in")),
                    vec![Node::if_then(
                        Expr::ne(
                            Expr::cast(DataType::U32, Expr::load("bytes_in", Expr::var("i"))),
                            Expr::u32(0),
                        ),
                        vec![Node::let_bind(
                            "flag_old",
                            Expr::atomic_or("flag", Expr::u32(0), Expr::u32(1)),
                        )],
                    )],
                ),
            ],
        );
        let mut bytes = vec![0u8; 4097];
        bytes[4096] = 1;

        let inputs = [Value::from(bytes), Value::from(vec![0u8; 4])];
        let outputs = crate::ReferenceRequest::standard(&program, &inputs)
            .outputs()
            .expect("Fix: reference interpreter should execute singleton atomic flag scans.");
        let flag = outputs[0].to_bytes();

        assert_eq!(u32::from_le_bytes([flag[0], flag[1], flag[2], flag[3]]), 1);
    }

    /// A byte-scan program whose haystack is PACKED (4 bytes/u32) has fewer buffer
    /// elements than the invocations it needs, one per byte. Buffer-shape grid
    /// inference therefore UNDER-covers it, silently skipping high positions. This
    /// is exactly the region-presence CPU-ref under-fire that the GPU did not have.
    /// A dispatch element floor lets the caller pass the true byte grid so
    /// the interpreter covers what the real dispatch would, no silent
    /// under-coverage (Law 10). This locks both halves: the default under-covers,
    /// the floor covers.
    #[test]
    fn dispatch_floor_covers_packed_byte_scan_that_buffer_inference_under_covers() {
        // 1024 packed words == 4096 bytes; the marker is at byte 4095 (word 1023),
        // reachable only if the grid runs 4096 invocations, not the 1024 the
        // packed buffer's element count implies.
        const PACKED_WORDS: u32 = 1024;
        const BYTE_LEN: u32 = PACKED_WORDS * 4; // 4096
        const MARKER_POS: u32 = BYTE_LEN - 1; // 4095
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("packed", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(PACKED_WORDS),
                BufferDecl::storage("byte_len", 1, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(1),
                BufferDecl::storage("flag", 2, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(1),
            ],
            [256, 1, 1],
            vec![
                Node::let_bind("i", Expr::InvocationId { axis: 0 }),
                Node::if_then(
                    Expr::lt(Expr::var("i"), Expr::load("byte_len", Expr::u32(0))),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("i"), Expr::u32(MARKER_POS)),
                        vec![
                            // Read the packed word for this byte so `packed` is a
                            // genuine input (its 1024 elements are what buffer-shape
                            // inference would cap the grid at).
                            Node::let_bind(
                                "word",
                                Expr::load("packed", Expr::div(Expr::var("i"), Expr::u32(4))),
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var("word"), Expr::u32(0)),
                                vec![Node::let_bind(
                                    "flag_old",
                                    Expr::atomic_or("flag", Expr::u32(0), Expr::u32(1)),
                                )],
                            ),
                        ],
                    )],
                ),
            ],
        );
        let read_flag = |outputs: &[Value]| {
            let bytes = outputs[0].to_bytes();
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        };
        let make_inputs = || {
            vec![
                Value::from(vec![0u8; PACKED_WORDS as usize * 4]),
                Value::from(BYTE_LEN.to_le_bytes().to_vec()),
                Value::from(vec![0u8; 4]),
            ]
        };

        // Default grid: buffer-shape inference caps at the packed buffer's 1024
        // elements, so byte 4095 is never visited, the flag stays clear. This is
        // the SILENT under-coverage the region-presence gate hit.
        let default_inputs = make_inputs();
        let under = crate::ReferenceRequest::standard(&program, &default_inputs)
            .outputs()
            .expect("Fix: interpreter runs the packed byte-scan");
        assert_eq!(
            read_flag(&under),
            0,
            "buffer-shape grid inference must under-cover this packed byte-scan (documents the hole)"
        );

        // Floor = true byte length: the interpreter now covers byte 4095 and the
        // marker is found (parity with what the real dispatch config produces).
        let floored_inputs = make_inputs();
        let covered = crate::ReferenceRequest::standard(&program, &floored_inputs)
            .with_min_dispatch_elements(BYTE_LEN)
            .outputs()
            .expect("Fix: interpreter runs the packed byte-scan with an explicit grid floor");
        assert_eq!(
            read_flag(&covered),
            1,
            "an explicit grid floor of haystack_len must cover every byte position"
        );
    }
}
