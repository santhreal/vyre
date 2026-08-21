//! Host-split dispatch: every segment launched as its own kernel, with live
//! buffers rotated between launches and the sequence looped to a fixpoint.

use std::collections::HashMap;

use vyre_foundation::ir::{Ident, Program};

use super::contains_grid_sync;
use super::live_buffers::{
    borrowed_grid_sync_inputs_by_name, collect_final_named_outputs, owned_accumulators_equal,
    refresh_named_outputs, snapshot_owned_accumulators, GridSyncInput,
};
use super::segment_buffers::{
    original_input_names, original_output_names, plan_host_grid_sync_segments,
    PlannedGridSyncSegment,
};
use super::{
    elapsed_wall_ns, grid_sync_segment_error, reject_empty_grid_sync_split,
    reserve_grid_sync_hash_map, reserve_grid_sync_vec,
};
use crate::backend::{
    BackendError, DispatchConfig, OutputBuffers, TimedDispatchResult, VyreBackend,
};

/// Universal dispatch helper that satisfies `Node::Barrier { ordering:
/// GridSync }` on any backend by splitting at the barrier and running
/// each segment as its own kernel launch.
///
/// Backends with native cooperative-launch grid sync (advertised via
/// [`VyreBackend::supports_grid_sync`]) bypass the split  -  the
/// program is dispatched once. Backends without it route here so the
/// kernel-launch boundary becomes the grid-level fence: every prior
/// write is globally visible to subsequent launches.
///
/// # Inputs
/// `inputs` matches the input slice the caller would have passed to
/// `dispatch_borrowed`. After each segment, the helper refreshes
/// every ReadWrite buffer's slot from the segment's readback so the
/// next segment sees the prior writes.
///
/// # Errors
/// Propagates any `BackendError` raised by `dispatch_borrowed` on a
/// segment, prefixed with the segment index for diagnosability.
pub fn dispatch_with_grid_sync_split(
    backend: &dyn VyreBackend,
    program: &Program,
    inputs: &[&[u8]],
    config: &DispatchConfig,
) -> Result<Vec<Vec<u8>>, BackendError> {
    let mut outputs = Vec::new();
    reserve_grid_sync_vec(
        &mut outputs,
        program.output_buffer_indices().len().max(1),
        "grid-sync final outputs",
    )?;
    dispatch_with_grid_sync_split_into(backend, program, inputs, config, &mut outputs)?;
    Ok(outputs)
}

/// Timed variant of [`dispatch_with_grid_sync_split`].
///
/// # Errors
/// Propagates any [`BackendError`] raised by a segment dispatch.
pub fn dispatch_with_grid_sync_split_timed(
    backend: &dyn VyreBackend,
    program: &Program,
    inputs: &[&[u8]],
    config: &DispatchConfig,
) -> Result<TimedDispatchResult, BackendError> {
    let started = std::time::Instant::now();
    let outputs = dispatch_with_grid_sync_split(backend, program, inputs, config)?;
    Ok(TimedDispatchResult::host_timed(
        outputs,
        elapsed_wall_ns(started)?,
    ))
}

fn seed_backend_allocated_segment_inputs<'a>(
    program: &Program,
    segments: &[PlannedGridSyncSegment],
    current_inputs: &mut HashMap<Ident, GridSyncInput<'a>>,
) -> Result<(), BackendError> {
    for name in segments
        .iter()
        .flat_map(|segment| segment.input_names.iter())
    {
        if current_inputs.contains_key(name) {
            continue;
        }
        let Some(buffer) = program.buffer(name.as_str()) else {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: grid-sync segment references undeclared input `{name}`. Rebuild the split from a Program whose buffer table covers every expression dependency."
                ),
            });
        };
        if !buffer.is_backend_allocated_output() {
            continue;
        }
        let static_len =
            buffer
                .static_byte_len()
                .map_err(|error| BackendError::InvalidProgram {
                    fix: format!("Fix: cannot seed grid-sync output `{name}`: {error}"),
                })?;
        let byte_len = static_len
            .or_else(|| buffer.output_byte_range().map(|range| range.end))
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: grid-sync output `{name}` is read-modify-written before its first split output but has no static byte size. Declare a count or output byte range."
                ),
            })?;
        let mut zeroed = Vec::new();
        reserve_grid_sync_vec(
            &mut zeroed,
            byte_len,
            "grid-sync backend-allocated output seed",
        )?;
        zeroed.resize(byte_len, 0);
        current_inputs.insert(name.clone(), GridSyncInput::Owned(zeroed));
    }
    Ok(())
}

/// Variant of [`dispatch_with_grid_sync_split`] that writes final outputs into
/// caller-owned storage.
///
/// # Errors
/// Propagates any `BackendError` raised by a segment dispatch.
fn dispatch_grid_sync_split_generic<D>(
    program: &Program,
    inputs: &[&[u8]],
    config: &DispatchConfig,
    outputs: &mut OutputBuffers,
    mut dispatch_segment: D,
) -> Result<(), BackendError>
where
    D: FnMut(&Program, &[&[u8]], &DispatchConfig, &mut OutputBuffers) -> Result<(), BackendError>,
{
    // These are the explicit non-native grid-sync routes (host split /
    // resident fixpoint). They split unconditionally when the program carries a
    // grid-sync barrier: native cooperative launch has a residency ceiling, so
    // `supports_grid_sync()` no longer implies "this program runs natively".
    // The orchestrator (or the registry's `should_split_grid_sync`) decides
    // native-vs-split per program; once here, always split.
    if !contains_grid_sync(program) {
        return dispatch_segment(program, inputs, config, outputs);
    }
    let segments = plan_host_grid_sync_segments(program)?;
    reject_empty_grid_sync_split(&segments)?;
    crate::observability::record_grid_sync_split(segments.len());
    // Build a mutable input set we rotate between segments. ReadOnly
    // inputs stay borrowed from the caller for the whole split; only
    // ReadWrite buffers become owned after a segment produces updated
    // bytes. The previous implementation cloned every input before
    // the first launch, which turned large read-only buffers into a
    // host-memory copy on the slow path.
    let initial_input_names = original_input_names(program)?;
    if inputs.len() != initial_input_names.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: grid-sync split expected {} initial input buffer(s) but received {}. Rebuild the dispatch inputs from the Program buffer declarations before splitting.",
                initial_input_names.len(),
                inputs.len()
            ),
        });
    }
    let mut current_inputs: HashMap<Ident, GridSyncInput<'_>> = HashMap::new();
    reserve_grid_sync_hash_map(
        &mut current_inputs,
        program.buffers().len(),
        "grid-sync rotating input map",
    )?;
    for (name, bytes) in initial_input_names.into_iter().zip(inputs.iter().copied()) {
        current_inputs.insert(name, GridSyncInput::Borrowed(bytes));
    }
    seed_backend_allocated_segment_inputs(program, &segments, &mut current_inputs)?;
    let mut segment_outputs = Vec::new();
    reserve_grid_sync_vec(
        &mut segment_outputs,
        outputs.capacity().max(1),
        "grid-sync intermediate outputs",
    )?;
    let final_output_names = original_output_names(program)?;

    // Honor the program's fixpoint contract across the split. The
    // non-split dispatch path (`dispatch_borrowed`) re-runs the WHOLE
    // program `fixpoint_iterations` times with persistent ReadWrite
    // buffers, so a program authored as a fixpoint closure converges
    // a multi-hop reachability/dataflow closure is exactly this shape: a
    // `seed (acc |= source) → hop (acc' = step(acc)) → merge (acc |= acc')`
    // body whose accumulator grows by ONE dataflow hop per whole-program
    // pass, relying on the dispatcher to iterate it to a fixpoint.
    //
    // GridSync barriers split that body across segments, so ONE pass over
    // the segment sequence advances the accumulator by exactly one hop.
    // Re-running an individual SEGMENT N times (the previous behavior:
    // `config` with its fixpoint count reached each segment) does NOT
    // converge, re-launching the isolated `hop` segment recomputes the
    // same frontier from an unchanged `acc`. The whole SEQUENCE must be
    // looped instead, with each segment run once per pass. Net device work
    // is identical (sequence_len × iterations launches either way); only
    // the nesting order changes, which is what makes the closure converge.
    // A flow that needs k hops through k-1 intermediate variables (the
    // dominant launch-rule shape: `q = src; sink(q)`) silently returned an
    // empty frontier under the old single-pass split (recall=0).
    let iterations =
        crate::fixpoint_iterations::resolve_fixpoint_iterations(config, "grid-sync split")?;
    let mut segment_config = config.clone();
    segment_config.fixpoint_iterations = Some(1);

    // Adaptive convergence: `iterations` is an UPPER bound (the worst-case hop
    // depth, one hop per whole-sequence pass). The segment sequence is a
    // deterministic function of its live buffers, so once a full pass leaves
    // every evolving (Owned) accumulator unchanged the closure has reached a
    // fixpoint, every remaining pass would re-dispatch the entire segment
    // sequence (hundreds of launches on a large fused program) for zero new
    // dataflow. Stop as soon as two consecutive passes produce the same state.
    let mut prev_state: Option<HashMap<Ident, Vec<u8>>> = None;
    for _ in 0..iterations {
        for (segment_idx, segment) in segments.iter().enumerate() {
            let borrowed = borrowed_grid_sync_inputs_by_name(segment, &current_inputs)?;
            dispatch_segment(
                &segment.program,
                borrowed.as_slice(),
                &segment_config,
                &mut segment_outputs,
            )
            .map_err(|error| grid_sync_segment_error(error, segment_idx, segments.len()))?;
            drop(borrowed);
            refresh_named_outputs(segment, &mut segment_outputs, &mut current_inputs)?;
        }
        if let Some(prev) = &prev_state {
            if owned_accumulators_equal(prev, &current_inputs) {
                break;
            }
        }
        prev_state = Some(snapshot_owned_accumulators(&current_inputs));
    }
    collect_final_named_outputs(&final_output_names, &mut current_inputs, outputs)?;
    Ok(())
}

/// Split a grid-sync program at its barriers and dispatch every segment through
/// `backend`, looping the segment sequence to a fixpoint.
///
/// This is the `&dyn VyreBackend` entry; the split, refresh, and adaptive
/// convergence logic lives in the internal `dispatch_grid_sync_split_generic`, shared with
/// the closure entry [`dispatch_with_grid_sync_split_via_into`].
///
/// # Errors
/// Propagates any [`BackendError`] from splitting or a segment dispatch,
/// prefixed with the segment index.
pub fn dispatch_with_grid_sync_split_into(
    backend: &dyn VyreBackend,
    program: &Program,
    inputs: &[&[u8]],
    config: &DispatchConfig,
    outputs: &mut OutputBuffers,
) -> Result<(), BackendError> {
    dispatch_grid_sync_split_generic(program, inputs, config, outputs, |p, i, c, o| {
        backend.dispatch_borrowed_into(p, i, c, o)
    })
}

/// Closure-driven counterpart of [`dispatch_with_grid_sync_split_into`] for
/// callers that hold an opaque single-launch dispatch closure instead of a
/// `&dyn VyreBackend`.
///
/// This is the entry a host-loop fixpoint solver (an IFDS or dataflow solve)
/// uses to move its convergence loop onto the device without taking a backend
/// handle: it plugs any backend, reference or device, as a
/// `Fn(&Program, &[&[u8]], Option<[u32; 3]>, &mut Vec<Vec<u8>>) -> Result<(),
/// String>` closure. The closure receives each segment's program, its rotated
/// inputs, the whole-grid workgroup count (`config.grid_override`), and a
/// per-segment output slot to fill in the segment program's output order. The
/// split, refresh, and convergence logic is the SAME code as the backend entry
/// (both call the internal `dispatch_grid_sync_split_generic`), so the two paths converge
/// to identical output.
///
/// # Errors
/// Propagates any error the closure returns (wrapped through
/// [`BackendError::new`]) and any structural split error, prefixed with the
/// segment index.
pub fn dispatch_with_grid_sync_split_via_into<F>(
    program: &Program,
    inputs: &[&[u8]],
    config: &DispatchConfig,
    dispatch: &F,
    outputs: &mut OutputBuffers,
) -> Result<(), BackendError>
where
    F: Fn(&Program, &[&[u8]], Option<[u32; 3]>, &mut Vec<Vec<u8>>) -> Result<(), String>,
{
    dispatch_grid_sync_split_generic(program, inputs, config, outputs, |p, i, c, o| {
        dispatch(p, i, c.grid_override, o).map_err(BackendError::new)
    })
}

/// Allocating wrapper over [`dispatch_with_grid_sync_split_via_into`].
///
/// # Errors
/// Propagates any error from [`dispatch_with_grid_sync_split_via_into`].
pub fn dispatch_with_grid_sync_split_via<F>(
    program: &Program,
    inputs: &[&[u8]],
    config: &DispatchConfig,
    dispatch: &F,
) -> Result<Vec<Vec<u8>>, BackendError>
where
    F: Fn(&Program, &[&[u8]], Option<[u32; 3]>, &mut Vec<Vec<u8>>) -> Result<(), String>,
{
    let mut outputs = Vec::new();
    reserve_grid_sync_vec(
        &mut outputs,
        program.output_buffer_indices().len().max(1),
        "grid-sync via final outputs",
    )?;
    dispatch_with_grid_sync_split_via_into(program, inputs, config, dispatch, &mut outputs)?;
    Ok(outputs)
}

// Inline: covers `dispatch_grid_sync_split_generic`, which no integration test can name.
#[path = "host_dispatch_tests.rs"]
#[cfg(test)]
mod host_dispatch_tests;
