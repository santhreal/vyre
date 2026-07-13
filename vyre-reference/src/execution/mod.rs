//! Generic reference interpreter entry points.
//!
//! The stable statement-IR [`reference_eval`] entry point remains delegated to
//! the existing invocation simulator until `Program` stores graph nodes
//! directly.

pub(crate) mod call;
pub mod expr;
pub(crate) mod expr_cast;
pub(crate) mod hashmap;
pub mod node;
pub(crate) mod node_tree;
/// Thread-local arithmetic-IR-op counting for roofline / complexity analysis.
pub mod op_count;
pub mod sequential;
pub(crate) mod typed_ops;

use std::borrow::Cow;

use rustc_hash::FxHashMap;
use vyre::ir::{InterpCtx, Node, NodeId, NodeStorage, Program, Value as IrValue};

use crate::value::Value;

/// If the program satisfies the public top-level-Region model, return a
/// byte-identical clone. If not, the usual case is
/// `optimizer::passes::cleanup::region_inline_engine` having flattened a Category-A wrapper;
/// in that case [`Program::reconcile_runnable_top_level`] matches
/// `Program::wrapped` again. When the first entry node is a `Store` (or the
/// entry is empty), we do **not** auto-wrap: those programs must still use
/// `Program::wrapped` explicitly, matching `region_gate` negative tests.
pub(crate) fn program_for_interpreter(program: &Program) -> Result<Cow<'_, Program>, vyre::Error> {
    let normalized = if let Some(message) = program.top_level_region_violation() {
        if program.entry().is_empty() {
            return Err(vyre::Error::interp(format!(
                "reference interpreter requires a top-level Region-wrapped Program: {message}"
            )));
        }
        if matches!(program.entry().first(), Some(Node::Store { .. })) {
            return Err(vyre::Error::interp(format!(
                "reference interpreter requires a top-level Region-wrapped Program: {message}"
            )));
        }
        Cow::Owned(program.clone().reconcile_runnable_top_level())
    } else {
        Cow::Borrowed(program)
    };
    match vyre_foundation::transform::collectives::lower_single_rank_collectives(
        normalized.as_ref(),
    ) {
        Ok(Some(lowered)) => Ok(Cow::Owned(lowered)),
        Ok(None) => Ok(normalized),
        Err(error) => Err(vyre::Error::interp(error.to_string())),
    }
}

/// The interpreter's output ABI, single-homed: [`is_reference_output`] is the exact
/// predicate `reference_eval` uses to collect the buffers it returns, and
/// [`output_index`] locates a named output by that predicate. Re-exported so test
/// harnesses never hand-roll (and drift from) the selection.
pub use hashmap::{is_reference_output, output_index};

/// Execute a vyre IR program on the pure Rust reference interpreter.
///
/// The current public [`Program`] model is statement-oriented, so this stable
/// entry point delegates to the statement evaluator. Graph-shaped extension
/// nodes use [`run_storage_graph`].
pub fn reference_eval(program: &Program, inputs: &[Value]) -> Result<Vec<Value>, vyre::Error> {
    run_arena_reference(program, inputs)
}

/// [`reference_eval`] plus an [`OobReport`](crate::oob::OobReport) of every
/// out-of-bounds access the interpreter silently absorbed during the run.
///
/// The interpreter DEFINES OOB loads as zero-fill and OOB stores as a no-op so
/// its output stays deterministic, but that silent absorption is exactly what
/// masks a GPU/CPU parity hazard: an IR program with an ungated data-derived index
/// "works" here yet a real GPU (CUDA does no bounds-checking) reads garbage or
/// corrupts memory. Use this to assert a program NEVER relies on that masking: a
/// correctly bounds-gated program handles an out-of-contract index with explicit
/// control flow, so it records `OobReport::total() == 0` even on hostile input. A
/// nonzero total means the IR indexed past a buffer end and needs an explicit gate
/// (the class of fix applied to ziftsieve/base64/sketch/simplicial).
///
/// The tally is per-thread and reset at the start of this call, so it measures
/// exactly this run.
///
/// # Errors
/// Same as [`reference_eval`].
pub fn reference_eval_oob_report(
    program: &Program,
    inputs: &[Value],
) -> Result<(Vec<Value>, crate::oob::OobReport), vyre::Error> {
    crate::oob::reset_oob_report();
    let outputs = reference_eval(program, inputs)?;
    Ok((outputs, crate::oob::oob_report()))
}

/// [`reference_eval_with_dispatch`] plus an [`OobReport`](crate::oob::OobReport).
///
/// The grid floor lets a caller deliberately OVER-FIRE the dispatch (more lanes
/// than the buffer-inferred grid) to probe whether a primitive's per-lane guard
/// actually protects the extra lanes. A guard written as `Expr::and(t < n, load(buf,
/// t))` does NOT, the data-flow AND evaluates the load for `t >= n`, an OOB read
/// (the ssa_dominance_scan bug). Running a valid fixture at an inflated grid and
/// asserting `OobReport::total() == 0` catches that whole class registry-wide.
///
/// # Errors
/// Same as [`reference_eval_with_dispatch`].
pub fn reference_eval_with_dispatch_oob_report(
    program: &Program,
    inputs: &[Value],
    min_dispatch_elements: u32,
) -> Result<(Vec<Value>, crate::oob::OobReport), vyre::Error> {
    crate::oob::reset_oob_report();
    let outputs = reference_eval_with_dispatch(program, inputs, min_dispatch_elements)?;
    Ok((outputs, crate::oob::oob_report()))
}

/// [`reference_eval`] with an explicit grid floor.
///
/// The reference interpreter infers its dispatch grid from buffer SHAPES, which
/// cannot express the per-invocation count of a byte-scan program (the haystack
/// is packed 4 bytes/u32 and the scan length is a runtime value). Pass the true
/// grid, e.g. `haystack_len` for a one-lane-per-byte scan, so the interpreter
/// covers exactly what the real dispatch config would; otherwise high positions
/// are silently skipped (the CPU-ref oracle under-fires while the GPU is correct).
/// `min_dispatch_elements` is a FLOOR: the interpreter still runs at least the
/// buffer-inferred grid, so passing `0` is identical to [`reference_eval`].
///
/// # Errors
/// Same as [`reference_eval`].
pub fn reference_eval_with_dispatch(
    program: &Program,
    inputs: &[Value],
    min_dispatch_elements: u32,
) -> Result<Vec<Value>, vyre::Error> {
    run_arena_reference_with_dispatch(program, inputs, min_dispatch_elements)
}

/// Execute using the statement-IR reference evaluator.
pub fn run_arena_reference(program: &Program, inputs: &[Value]) -> Result<Vec<Value>, vyre::Error> {
    run_arena_reference_with_dispatch(program, inputs, 0)
}

/// [`run_arena_reference`] with an explicit grid floor (see
/// [`reference_eval_with_dispatch`]).
///
/// # Errors
/// Same as [`run_arena_reference`].
pub fn run_arena_reference_with_dispatch(
    program: &Program,
    inputs: &[Value],
    min_dispatch_elements: u32,
) -> Result<Vec<Value>, vyre::Error> {
    let program = program_for_interpreter(program)?;
    hashmap::run_hashmap_reference(
        &program,
        inputs,
        min_dispatch_elements,
        hashmap::LaneOrder::Forward,
    )
}

/// Execute a program with the workgroup/invocation STEP ORDER reversed.
///
/// The result is identical to [`reference_eval`] for any RACE-FREE program (every
/// output slot is written by exactly one lane, or shared slots are touched only by
/// commutative atomics). It DIFFERS only when a non-atomic cross-lane write-write
/// race exists, two lanes plain-`store` the same slot, because the GPU leaves the
/// winner driver-defined while the single-threaded reference otherwise resolves it
/// deterministically (last stepped lane wins). Comparing this against
/// [`reference_eval`] therefore surfaces a hidden race the same way a real GPU would
/// nondeterministically diverge.
///
/// # Errors
/// Same as [`reference_eval`].
pub fn reference_eval_lane_reversed(
    program: &Program,
    inputs: &[Value],
) -> Result<Vec<Value>, vyre::Error> {
    let program = program_for_interpreter(program)?;
    hashmap::run_hashmap_reference(&program, inputs, 0, hashmap::LaneOrder::Reversed)
}

/// Differential oracle retained for tests during the generic interpreter transition.
#[cfg(test)]
pub fn eval_hashmap_reference(
    program: &Program,
    inputs: &[Value],
) -> Result<Vec<Value>, vyre::Error> {
    run_arena_reference(program, inputs)
}

/// Interpret a compact [`NodeStorage`] graph and return output node values.
pub fn run_storage_graph(
    nodes: &[(NodeId, NodeStorage)],
    outputs: &[NodeId],
) -> Result<Vec<IrValue>, vyre::Error> {
    let mut graph = FxHashMap::with_capacity_and_hasher(nodes.len(), Default::default());
    for (id, node) in nodes {
        if graph.insert(*id, node).is_some() {
            return Err(duplicate_node_error(*id));
        }
    }
    let mut ctx = InterpCtx::default();
    let mut states = FxHashMap::with_capacity_and_hasher(graph.len(), Default::default());

    for output in outputs {
        eval_storage_node(*output, &graph, &mut ctx, &mut states)?;
    }

    outputs
        .iter()
        .map(|id| ctx.get(*id).map_err(interp_error))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Done,
}

fn eval_storage_node(
    id: NodeId,
    graph: &FxHashMap<NodeId, &NodeStorage>,
    ctx: &mut InterpCtx,
    states: &mut FxHashMap<NodeId, VisitState>,
) -> Result<(), vyre::Error> {
    match states.get(&id).copied() {
        Some(VisitState::Done) => return Ok(()),
        Some(VisitState::Visiting) => return Err(cycle_error(id)),
        None => {}
    }

    let node = *graph.get(&id).ok_or_else(|| missing_node_error(id))?;
    states.insert(id, VisitState::Visiting);
    let inputs = node.input_ids();
    for input in &inputs {
        eval_storage_node(*input, graph, ctx, states)?;
    }
    ctx.set_operands(inputs);
    let value = node.interpret(ctx).map_err(interp_error)?;
    ctx.set(id, value);
    states.insert(id, VisitState::Done);
    Ok(())
}

fn interp_error(error: vyre::ir::EvalError) -> vyre::Error {
    vyre::Error::interp(error.to_string())
}

fn missing_node_error(id: NodeId) -> vyre::Error {
    vyre::Error::interp(format!(
        "graph references missing node {}. Fix: include every dependency in the interpreter input graph.",
        id.0
    ))
}

fn cycle_error(id: NodeId) -> vyre::Error {
    vyre::Error::interp(format!(
        "graph contains a dependency cycle at node {}. Fix: submit an acyclic dataflow graph.",
        id.0
    ))
}

fn duplicate_node_error(id: NodeId) -> vyre::Error {
    vyre::Error::interp(format!(
        "graph contains duplicate node {}. Fix: submit exactly one storage record for each NodeId before reference execution.",
        id.0
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, NodeStorage};

    #[test]
    fn reference_eval_dispatches_singleton_atomic_flags_across_dynamic_byte_input() {
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

        let outputs = reference_eval(&program, &[Value::from(bytes), Value::from(vec![0u8; 4])])
            .expect("Fix: reference interpreter should execute singleton atomic flag scans.");
        let flag = outputs[0].to_bytes();

        assert_eq!(u32::from_le_bytes([flag[0], flag[1], flag[2], flag[3]]), 1);
    }

    /// A byte-scan program whose haystack is PACKED (4 bytes/u32) has fewer buffer
    /// elements than the invocations it needs, one per byte. Buffer-shape grid
    /// inference therefore UNDER-covers it, silently skipping high positions. This
    /// is exactly the region-presence CPU-ref under-fire that the GPU did not have.
    /// `reference_eval_with_dispatch` lets the caller pass the true byte grid so
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
        let under = reference_eval(&program, &make_inputs())
            .expect("Fix: interpreter runs the packed byte-scan");
        assert_eq!(
            read_flag(&under),
            0,
            "buffer-shape grid inference must under-cover this packed byte-scan (documents the hole)"
        );

        // Floor = true byte length: the interpreter now covers byte 4095 and the
        // marker is found (parity with what the real dispatch config produces).
        let covered = reference_eval_with_dispatch(&program, &make_inputs(), BYTE_LEN)
            .expect("Fix: interpreter runs the packed byte-scan with an explicit grid floor");
        assert_eq!(
            read_flag(&covered),
            1,
            "an explicit grid floor of haystack_len must cover every byte position"
        );
    }

    #[test]
    fn generic_storage_graph_matches_recursive_oracle_for_10k_programs() {
        let mut rng = 0x9e37_79b9_u64;
        for case in 0..10_000 {
            let graph = random_graph(&mut rng, case);
            let output = graph.last().expect("Fix: generated graph is non-empty").0;
            let expected =
                recursive_value(output, &graph).expect("Fix: recursive oracle evaluates");
            let actual = run_storage_graph(&graph, &[output])
                .expect("Fix: generic graph interpreter evaluates")[0];
            assert_eq!(actual, expected, "case {case}");
        }
    }

    fn random_graph(rng: &mut u64, case: u32) -> Vec<(NodeId, NodeStorage)> {
        let len = 2 + (next(rng) as usize % 31);
        let mut graph = Vec::with_capacity(len);
        graph.push((NodeId(0), NodeStorage::LitU32(case)));
        graph.push((NodeId(1), NodeStorage::LitU32(next(rng))));
        for index in 2..len {
            let left = NodeId(next(rng) % index as u32);
            let right = NodeId(next(rng) % index as u32);
            let op = match next(rng) % 5 {
                0 => BinOp::Add,
                1 => BinOp::Sub,
                2 => BinOp::Mul,
                3 => BinOp::BitXor,
                _ => BinOp::BitAnd,
            };
            graph.push((NodeId(index as u32), NodeStorage::BinOp { op, left, right }));
        }
        graph
    }

    fn recursive_value(
        id: NodeId,
        graph: &[(NodeId, NodeStorage)],
    ) -> Result<IrValue, vyre::Error> {
        let node = graph
            .iter()
            .find(|(node_id, _)| *node_id == id)
            .map(|(_, node)| node)
            .ok_or_else(|| missing_node_error(id))?;
        match node {
            NodeStorage::LitU32(value) => Ok(IrValue::U32(*value)),
            NodeStorage::BinOp { op, left, right } => {
                let left = expect_u32(recursive_value(*left, graph)?)?;
                let right = expect_u32(recursive_value(*right, graph)?)?;
                let value = match op {
                    BinOp::Add => left.wrapping_add(right),
                    BinOp::Sub => left.wrapping_sub(right),
                    BinOp::Mul => left.wrapping_mul(right),
                    BinOp::BitXor => left ^ right,
                    BinOp::BitAnd => left & right,
                    _ => {
                        return Err(vyre::Error::interp(
                            "recursive parity oracle received unsupported op. Fix: keep test generation within the oracle domain.",
                        ));
                    }
                };
                Ok(IrValue::U32(value))
            }
            _ => Err(vyre::Error::interp(
                "recursive parity oracle received unsupported node. Fix: keep test generation within the oracle domain.",
            )),
        }
    }

    fn expect_u32(value: IrValue) -> Result<u32, vyre::Error> {
        match value {
            IrValue::U32(value) => Ok(value),
            other => Err(vyre::Error::interp(format!(
                "recursive parity oracle expected u32, got {other:?}. Fix: keep generated graphs scalar-u32 only."
            ))),
        }
    }

    fn next(rng: &mut u64) -> u32 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*rng >> 32) as u32
    }
}
