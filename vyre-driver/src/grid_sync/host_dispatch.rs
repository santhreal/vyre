//! Host-split dispatch: every segment launched as its own kernel, with live
//! buffers rotated between launches and the sequence looped to a fixpoint.

use std::collections::HashMap;

use vyre_foundation::ir::{Ident, Program};

use super::barrier_split::contains_grid_sync;
use super::live_buffers::{
    borrowed_grid_sync_inputs_by_name, collect_final_named_outputs, owned_accumulator_fingerprint,
    refresh_named_outputs, GridSyncInput,
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
    Ok(TimedDispatchResult {
        outputs,
        wall_ns: elapsed_wall_ns(started)?,
        device_ns: None,
        enqueue_ns: None,
        wait_ns: None,
    })
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
    let mut prev_fingerprint: Option<u64> = None;
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
        let fingerprint = owned_accumulator_fingerprint(&current_inputs);
        if prev_fingerprint == Some(fingerprint) {
            break;
        }
        prev_fingerprint = Some(fingerprint);
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
/// handle: it plugs any backend (CPU reference, CUDA, wgpu) as a
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_sync::barrier_split::entry_sequence;
    use crate::grid_sync::segment_buffers::{
        segment_buffer_consumes_input, segment_buffer_produces_output, segment_output_names,
    };
    use crate::grid_sync::test_programs::{
        apply_out_stores, cross_segment_store_program, grid_sync_chain,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vyre_foundation::ir::MemoryOrdering;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};

    struct ReuseCheckingBackend {
        calls: AtomicUsize,
        final_outputs_addr: usize,
        final_slot_addr: usize,
    }

    impl crate::backend::sealed::Sealed for ReuseCheckingBackend {}

    impl VyreBackend for ReuseCheckingBackend {
        fn id(&self) -> &'static str {
            "grid-sync-reuse-checking"
        }

        fn dispatch_borrowed(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            unreachable!("test uses dispatch_borrowed_into")
        }

        fn dispatch_borrowed_into(
            &self,
            _program: &Program,
            inputs: &[&[u8]],
            _config: &DispatchConfig,
            outputs: &mut OutputBuffers,
        ) -> Result<(), BackendError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 1 && self.final_outputs_addr != 0 {
                assert_eq!(outputs.as_ptr() as usize, self.final_outputs_addr);
                assert_eq!(outputs[0].as_ptr() as usize, self.final_slot_addr);
            }
            if outputs.is_empty() {
                outputs.push(Vec::new());
            }
            outputs[0].clear();
            outputs[0].extend_from_slice(inputs[0]);
            if call == 0 {
                outputs[0][0] = 7;
            } else {
                outputs[0][0] = outputs[0][0].saturating_add(1);
            }
            Ok(())
        }
    }

    #[test]
    fn split_into_preserves_caller_output_slot_after_named_output_collection() {
        let program = grid_sync_chain(&["a", "b"]);
        let mut outputs = vec![Vec::with_capacity(8)];
        let outputs_addr = outputs.as_ptr() as usize;
        let slot_addr = outputs[0].as_ptr() as usize;
        let backend = ReuseCheckingBackend {
            calls: AtomicUsize::new(0),
            final_outputs_addr: 0,
            final_slot_addr: 0,
        };
        let input = [0u8, 0, 0, 0];
        dispatch_with_grid_sync_split_into(
            &backend,
            &program,
            &[input.as_slice()],
            &DispatchConfig::default(),
            &mut outputs,
        )
        .expect("Fix: grid-sync split should write into caller-owned output storage");

        assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
        assert_eq!(outputs, vec![vec![8, 0, 0, 0]]);
        assert_eq!(outputs.as_ptr() as usize, outputs_addr);
        assert_eq!(outputs[0].as_ptr() as usize, slot_addr);
    }

    /// Each `dispatch_borrowed_into` reads `inputs[0][0]`, writes `+1`. With the
    /// ReadWrite buffer rotating between segments, a single pass over a
    /// two-segment program advances the accumulator by 2. The multi-hop
    /// `flows_to` closure relies on the WHOLE sequence being re-run
    /// `fixpoint_iterations` times (one dataflow hop per pass); a single pass
    /// is one hop, which silently dropped every flow through an intermediate
    /// variable to recall=0.
    struct IncrementingBackend {
        calls: AtomicUsize,
    }

    impl crate::backend::sealed::Sealed for IncrementingBackend {}

    impl VyreBackend for IncrementingBackend {
        fn id(&self) -> &'static str {
            "grid-sync-incrementing"
        }

        fn dispatch_borrowed(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            unreachable!("test uses dispatch_borrowed_into")
        }

        fn dispatch_borrowed_into(
            &self,
            _program: &Program,
            inputs: &[&[u8]],
            config: &DispatchConfig,
            outputs: &mut OutputBuffers,
        ) -> Result<(), BackendError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            // Each segment must run exactly once per outer pass: the whole
            // sequence carries the fixpoint, not any single segment.
            assert_eq!(
                config.fixpoint_iterations,
                Some(1),
                "segment dispatch must receive fixpoint_iterations=1; the outer split loop owns the iteration count"
            );
            if outputs.is_empty() {
                outputs.push(Vec::new());
            }
            outputs[0].clear();
            outputs[0].extend_from_slice(inputs[0]);
            outputs[0][0] = outputs[0][0].saturating_add(1);
            Ok(())
        }
    }

    #[test]
    fn split_into_loops_whole_sequence_fixpoint_iterations_times() {
        // Two segments separated by a GridSync barrier.
        let program = grid_sync_chain(&["a", "b"]);

        // Single pass (default): 2 segment launches, accumulator = 2.
        let backend = IncrementingBackend {
            calls: AtomicUsize::new(0),
        };
        let mut outputs = vec![Vec::new()];
        dispatch_with_grid_sync_split_into(
            &backend,
            &program,
            &[[0u8, 0, 0, 0].as_slice()],
            &DispatchConfig::default(),
            &mut outputs,
        )
        .expect("single-pass split dispatch");
        assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
        assert_eq!(outputs, vec![vec![2, 0, 0, 0]]);

        // Three fixpoint iterations: 3 passes × 2 segments = 6 launches, and
        // the accumulator advances one hop per pass to 6. This is the exact
        // property the multi-hop `flows_to` split depended on and the
        // single-pass implementation lacked.
        let backend = IncrementingBackend {
            calls: AtomicUsize::new(0),
        };
        let config = DispatchConfig {
            fixpoint_iterations: Some(3),
            ..DispatchConfig::default()
        };
        let mut outputs = vec![Vec::new()];
        dispatch_with_grid_sync_split_into(
            &backend,
            &program,
            &[[0u8, 0, 0, 0].as_slice()],
            &config,
            &mut outputs,
        )
        .expect("multi-pass split dispatch");
        assert_eq!(
            backend.calls.load(Ordering::SeqCst),
            6,
            "split must re-run the whole 2-segment sequence 3 times"
        );
        assert_eq!(
            outputs,
            vec![vec![6, 0, 0, 0]],
            "accumulator must advance one hop per fixpoint pass (2 segments × 3 passes)"
        );
    }

    /// A backend-allocated atomic output starts from zero even when split
    /// liveness rewrites its first writer as a read-write segment input.
    #[test]
    fn split_seeds_first_atomic_output_without_caller_bytes() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::let_bind("prior", Expr::atomic_add("out", Expr::u32(0), Expr::u32(1))),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                Node::Return,
            ],
        );
        let dispatch = |segment: &Program,
                        inputs: &[&[u8]],
                        _grid: Option<[u32; 3]>,
                        outputs: &mut Vec<Vec<u8>>|
         -> Result<(), String> {
            outputs.clear();
            if !segment_output_names(segment)
                .map_err(|error| error.to_string())?
                .is_empty()
            {
                assert_eq!(inputs, &[&[0, 0, 0, 0][..]]);
                outputs.push(1_u32.to_le_bytes().to_vec());
            }
            Ok(())
        };

        let outputs =
            dispatch_with_grid_sync_split_via(&program, &[], &DispatchConfig::default(), &dispatch)
                .expect("backend-allocated atomic output must receive its zero seed");

        assert_eq!(outputs, vec![1_u32.to_le_bytes().to_vec()]);
    }

    #[test]
    fn split_via_closure_entry_matches_backend_entry_on_the_same_grid_sync_program() {
        // The `&dyn VyreBackend` entry and the closure entry both delegate to
        // `dispatch_grid_sync_split_generic`, so on the same grid-sync program,
        // config, and inputs they must drive the same segment dispatches and
        // produce byte-identical output. This is the ONE-PLACE contract that
        // lets a host-loop dataflow solver route its fixpoint through the closure
        // entry with no separate split implementation.
        let program = grid_sync_chain(&["a", "b"]);
        let config = DispatchConfig {
            fixpoint_iterations: Some(3),
            ..DispatchConfig::default()
        };
        let inputs: [&[u8]; 1] = [[0u8, 0, 0, 0].as_slice()];

        let backend = IncrementingBackend {
            calls: AtomicUsize::new(0),
        };
        let mut backend_outputs = vec![Vec::new()];
        dispatch_with_grid_sync_split_into(
            &backend,
            &program,
            &inputs,
            &config,
            &mut backend_outputs,
        )
        .expect("backend split dispatch");

        // The closure delegates to an identical backend through the opaque
        // single-launch closure shape a host-loop solver supplies (grid override
        // only, per-segment fixpoint fixed at 1 by the shared core).
        let closure_backend = IncrementingBackend {
            calls: AtomicUsize::new(0),
        };
        let dispatch = |program: &Program,
                        inputs: &[&[u8]],
                        grid: Option<[u32; 3]>,
                        outputs: &mut Vec<Vec<u8>>|
         -> Result<(), String> {
            let segment_config = DispatchConfig {
                grid_override: grid,
                fixpoint_iterations: Some(1),
                ..DispatchConfig::default()
            };
            closure_backend
                .dispatch_borrowed_into(program, inputs, &segment_config, outputs)
                .map_err(|error| error.to_string())
        };
        let via_outputs = dispatch_with_grid_sync_split_via(&program, &inputs, &config, &dispatch)
            .expect("closure split dispatch");

        assert_eq!(
            via_outputs, backend_outputs,
            "closure and backend split entries must produce identical output"
        );
        assert_eq!(
            closure_backend.calls.load(Ordering::SeqCst),
            backend.calls.load(Ordering::SeqCst),
            "both entries must drive the same number of segment dispatches (3 passes x 2 segments)"
        );
        assert_eq!(via_outputs, vec![vec![6u8, 0, 0, 0]]);
    }

    struct OwnedFinalReserveBackend {
        calls: AtomicUsize,
    }

    impl crate::backend::sealed::Sealed for OwnedFinalReserveBackend {}

    impl VyreBackend for OwnedFinalReserveBackend {
        fn id(&self) -> &'static str {
            "grid-sync-owned-final-reserve"
        }

        fn dispatch_borrowed(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            unreachable!("test uses dispatch_borrowed_into")
        }

        fn dispatch_borrowed_into(
            &self,
            _program: &Program,
            inputs: &[&[u8]],
            _config: &DispatchConfig,
            outputs: &mut OutputBuffers,
        ) -> Result<(), BackendError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 1 {
                assert!(
                    outputs.capacity() >= 1,
                    "owned grid-sync split wrapper must pre-reserve final output slots before the final segment dispatch"
                );
            }
            if outputs.is_empty() {
                outputs.push(Vec::new());
            }
            outputs[0].clear();
            outputs[0].extend_from_slice(inputs[0]);
            outputs[0][0] = outputs[0][0].saturating_add(1);
            Ok(())
        }
    }

    #[test]
    fn split_owned_wrapper_reserves_final_output_vector_before_final_segment() {
        let program = grid_sync_chain(&["a", "b"]);
        let backend = OwnedFinalReserveBackend {
            calls: AtomicUsize::new(0),
        };
        let input = [4u8, 0, 0, 0];

        let outputs = dispatch_with_grid_sync_split(
            &backend,
            &program,
            &[input.as_slice()],
            &DispatchConfig::default(),
        )
        .expect("Fix: owned grid-sync split should reserve and return final outputs");

        assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
        assert_eq!(outputs, vec![vec![6, 0, 0, 0]]);
    }

    #[test]
    fn grid_sync_split_records_segment_telemetry() {
        let program = grid_sync_chain(&["a", "b", "c"]);
        let backend = ReuseCheckingBackend {
            calls: AtomicUsize::new(0),
            final_outputs_addr: 0,
            final_slot_addr: 0,
        };
        let before = crate::observability::snapshot_dispatch_telemetry();
        let input = [0u8, 0, 0, 0];
        let mut outputs = Vec::new();

        dispatch_with_grid_sync_split_into(
            &backend,
            &program,
            &[input.as_slice()],
            &DispatchConfig::default(),
            &mut outputs,
        )
        .expect("Fix: grid-sync split should dispatch every segment");

        let after = crate::observability::snapshot_dispatch_telemetry();
        assert_eq!(backend.calls.load(Ordering::SeqCst), 3);
        assert!(after.grid_sync_splits > before.grid_sync_splits);
        assert!(after.grid_sync_segments >= before.grid_sync_segments + 3);
        assert!(after.grid_sync_points >= before.grid_sync_points + 2);
    }

    struct IntermediateReuseBackend {
        calls: AtomicUsize,
        first_outputs_addr: AtomicUsize,
        first_slot_addr: AtomicUsize,
    }

    impl crate::backend::sealed::Sealed for IntermediateReuseBackend {}

    impl VyreBackend for IntermediateReuseBackend {
        fn id(&self) -> &'static str {
            "grid-sync-intermediate-reuse"
        }

        fn dispatch_borrowed(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            unreachable!("test uses dispatch_borrowed_into")
        }

        fn dispatch_borrowed_into(
            &self,
            _program: &Program,
            inputs: &[&[u8]],
            _config: &DispatchConfig,
            outputs: &mut OutputBuffers,
        ) -> Result<(), BackendError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if outputs.is_empty() {
                outputs.push(Vec::with_capacity(8));
            }
            if call == 0 {
                self.first_outputs_addr
                    .store(outputs.as_ptr() as usize, Ordering::SeqCst);
                self.first_slot_addr
                    .store(outputs[0].as_ptr() as usize, Ordering::SeqCst);
            } else if call == 1 {
                assert_eq!(
                    outputs.as_ptr() as usize,
                    self.first_outputs_addr.load(Ordering::SeqCst)
                );
                assert_eq!(
                    outputs[0].as_ptr() as usize,
                    self.first_slot_addr.load(Ordering::SeqCst)
                );
            }
            outputs[0].clear();
            outputs[0].extend_from_slice(inputs[0]);
            outputs[0][0] = outputs[0][0].saturating_add(1);
            Ok(())
        }
    }

    #[test]
    fn split_reuses_intermediate_output_slot_between_segments() {
        let program = grid_sync_chain(&["a", "b", "c"]);
        let backend = IntermediateReuseBackend {
            calls: AtomicUsize::new(0),
            first_outputs_addr: AtomicUsize::new(0),
            first_slot_addr: AtomicUsize::new(0),
        };
        let input = [1u8, 0, 0, 0];
        let mut outputs = vec![Vec::with_capacity(8)];

        dispatch_with_grid_sync_split_into(
            &backend,
            &program,
            &[input.as_slice()],
            &DispatchConfig::default(),
            &mut outputs,
        )
        .expect("Fix: grid-sync split should reuse intermediate output scratch");

        assert_eq!(backend.calls.load(Ordering::SeqCst), 3);
        assert_eq!(outputs, vec![vec![4, 0, 0, 0]]);
    }

    /// Emulates a backend that lacks native grid-sync: for the single output
    /// buffer `out`, it starts from the forwarded prior value (when the segment
    /// consumes it) or zeros, then applies that segment's literal `Store out[i]
    /// = v` writes, exactly the per-slot store shape a fused multi-rule program
    /// produces. Proves end-to-end that earlier segments' slots survive.
    struct SlotStoringBackend {
        calls: AtomicUsize,
    }

    impl crate::backend::sealed::Sealed for SlotStoringBackend {}

    impl VyreBackend for SlotStoringBackend {
        fn id(&self) -> &'static str {
            "grid-sync-slot-storing"
        }

        fn dispatch_borrowed(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            unreachable!("test uses dispatch_borrowed_into")
        }

        fn dispatch_borrowed_into(
            &self,
            program: &Program,
            inputs: &[&[u8]],
            _config: &DispatchConfig,
            outputs: &mut OutputBuffers,
        ) -> Result<(), BackendError> {
            // Locate `out`'s positional input/output slots using the SAME
            // role convention the host split planner uses.
            let mut in_pos = None;
            let mut cur_in = 0usize;
            let mut out_pos = None;
            let mut cur_out = 0usize;
            for buffer in program.buffers() {
                if matches!(buffer.access(), BufferAccess::Workgroup) {
                    continue;
                }
                let consumes = segment_buffer_consumes_input(buffer);
                let produces = segment_buffer_produces_output(buffer);
                if buffer.name() == "out" {
                    if consumes {
                        in_pos = Some(cur_in);
                    }
                    if produces {
                        out_pos = Some(cur_out);
                    }
                }
                if consumes {
                    cur_in += 1;
                }
                if produces {
                    cur_out += 1;
                }
            }
            let out_pos = out_pos.expect("every writing segment must produce `out`");
            let mut state = match in_pos {
                Some(i) => inputs[i].to_vec(),
                None => vec![0u8; 16],
            };
            apply_out_stores(entry_sequence(program), &mut state);

            self.calls.fetch_add(1, Ordering::SeqCst);
            while outputs.len() <= out_pos {
                outputs.push(Vec::new());
            }
            outputs[out_pos].clear();
            outputs[out_pos].extend_from_slice(&state);
            Ok(())
        }
    }

    #[test]
    fn split_preserves_earlier_segment_output_slots_end_to_end() {
        // Regression: a fused multi-arm program where arm A's result-store is in
        // segment 0 (slot at element 0) and arm B's in the final segment (slot
        // at element 2). Before the accumulator fix the final segment's
        // write-only `out` zeroed element 0, dropping arm A entirely (a co-fused
        // rule whose result-store does not land in the final grid-sync segment
        // returned recall=0). Both slots must now survive.
        let program = cross_segment_store_program();
        let backend = SlotStoringBackend {
            calls: AtomicUsize::new(0),
        };
        let mut outputs = vec![Vec::new()];
        dispatch_with_grid_sync_split_into(
            &backend,
            &program,
            &[],
            &DispatchConfig::default(),
            &mut outputs,
        )
        .expect("split dispatch");
        assert_eq!(
            backend.calls.load(Ordering::SeqCst),
            2,
            "two segments, single fixpoint pass"
        );
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 16, "output buffer is 4 × u32 = 16 bytes");
        assert_eq!(
            outputs[0][0], 0xAA,
            "segment 0's slot (element 0) must survive the final segment's write"
        );
        assert_eq!(
            outputs[0][8], 0xBB,
            "the final segment's slot (element 2) is also present"
        );
    }

    /// Copies its input to its output and bumps byte 0 toward a saturation cap.
    /// Once the cap is reached the output equals the input, so a full pass over
    /// the split leaves the carried accumulator unchanged (a fixpoint).
    struct SaturatingBackend {
        calls: AtomicUsize,
        cap: u8,
    }

    impl crate::backend::sealed::Sealed for SaturatingBackend {}

    impl VyreBackend for SaturatingBackend {
        fn id(&self) -> &'static str {
            "grid-sync-saturating"
        }

        fn dispatch_borrowed(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            unreachable!("test uses dispatch_borrowed_into")
        }

        fn dispatch_borrowed_into(
            &self,
            _program: &Program,
            inputs: &[&[u8]],
            _config: &DispatchConfig,
            outputs: &mut OutputBuffers,
        ) -> Result<(), BackendError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if outputs.is_empty() {
                outputs.push(Vec::new());
            }
            outputs[0].clear();
            outputs[0].extend_from_slice(inputs[0]);
            if outputs[0][0] < self.cap {
                outputs[0][0] += 1;
            }
            Ok(())
        }
    }

    #[test]
    fn split_outer_loop_early_exits_when_accumulator_reaches_fixpoint() {
        // Two segments (one GridSync barrier). With a generous iteration budget
        // of 10, byte 0 saturates at 3, after which a whole pass leaves the
        // accumulator unchanged. The outer loop must stop once two consecutive
        // passes match instead of burning all 10 iterations.
        let program = grid_sync_chain(&["a", "b"]);
        let backend = SaturatingBackend {
            calls: AtomicUsize::new(0),
            cap: 3,
        };
        let config = DispatchConfig {
            fixpoint_iterations: Some(10),
            ..DispatchConfig::default()
        };
        let mut outputs = vec![Vec::new()];
        dispatch_with_grid_sync_split_into(
            &backend,
            &program,
            &[[0u8, 0, 0, 0].as_slice()],
            &config,
            &mut outputs,
        )
        .expect("converging split dispatch");
        // pass0 -> 2, pass1 -> 3 (saturates mid-pass), pass2 -> 3 (unchanged) =>
        // break after pass2. 3 passes x 2 segments = 6 launches, NOT 10x2=20.
        assert_eq!(
            backend.calls.load(Ordering::SeqCst),
            6,
            "outer loop must early-exit one pass after the accumulator stops changing, not run all 10 iterations"
        );
        assert_eq!(
            outputs,
            vec![vec![3, 0, 0, 0]],
            "early-exit must return the converged fixpoint value, identical to running every iteration"
        );
    }

    #[test]
    fn split_non_converging_accumulator_runs_full_iteration_budget() {
        // The dual of the early-exit test: an accumulator that changes every
        // pass (never reaches a fixpoint within budget) must run all
        // iterations (early-exit must not fire on a still-advancing closure).
        let program = grid_sync_chain(&["a", "b"]);
        // cap=255 so it never saturates within 4 passes (8 increments).
        let backend = SaturatingBackend {
            calls: AtomicUsize::new(0),
            cap: 255,
        };
        let config = DispatchConfig {
            fixpoint_iterations: Some(4),
            ..DispatchConfig::default()
        };
        let mut outputs = vec![Vec::new()];
        dispatch_with_grid_sync_split_into(
            &backend,
            &program,
            &[[0u8, 0, 0, 0].as_slice()],
            &config,
            &mut outputs,
        )
        .expect("non-converging split dispatch");
        assert_eq!(
            backend.calls.load(Ordering::SeqCst),
            8,
            "a still-advancing accumulator must run the full 4 iterations x 2 segments"
        );
        assert_eq!(outputs, vec![vec![8, 0, 0, 0]]);
    }
}
