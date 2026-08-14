//! Device-resident dispatch: segments launched against resident resources that
//! stay bound across every segment and fixpoint pass.

use std::collections::HashMap;

use vyre_foundation::ir::{Ident, Program};

use super::barrier_split::{contains_grid_sync, try_split_on_grid_sync};
use super::segment_buffers::original_output_names;
use super::{
    elapsed_wall_ns, grid_sync_segment_error, reject_empty_grid_sync_split,
    reserve_grid_sync_hash_map, reserve_grid_sync_vec,
};
use crate::backend::{
    BackendError, DispatchConfig, OutputBuffers, ResidentDispatchStep, ResidentReadRange, Resource,
    TimedDispatchResult, VyreBackend,
};
use crate::binding::{Binding, BindingPlan, BindingRole};

/// Resident-resource variant of
/// [`crate::grid_sync::dispatch_with_grid_sync_split_timed`].
///
/// This keeps the same resource handles bound for every segment. Read-write
/// buffers therefore refresh in place on the backend's device-resident storage
/// between segment launches instead of downloading bytes to the host and
/// re-uploading them as the next segment's inputs.
///
/// # Errors
/// Propagates any [`BackendError`] raised by a segment resident dispatch.
pub fn dispatch_resident_with_grid_sync_split_timed(
    backend: &dyn VyreBackend,
    program: &Program,
    resources: &[Resource],
    config: &DispatchConfig,
) -> Result<TimedDispatchResult, BackendError> {
    // These are the explicit non-native grid-sync routes (host split /
    // resident fixpoint). They split unconditionally when the program carries a
    // grid-sync barrier: native cooperative launch has a residency ceiling, so
    // `supports_grid_sync()` no longer implies "this program runs natively".
    // The orchestrator (or the registry's `should_split_grid_sync`) decides
    // native-vs-split per program; once here, always split.
    if !contains_grid_sync(program) {
        return backend.dispatch_resident_timed(program, resources, config);
    }
    let segments = try_split_on_grid_sync(program)?;
    reject_empty_grid_sync_split(&segments)?;
    let started = std::time::Instant::now();
    let mut final_outputs = Vec::new();
    let mut device_ns = Some(0_u64);
    let mut enqueue_ns = Some(0_u64);
    let mut wait_ns = Some(0_u64);
    for (segment_idx, segment) in segments.iter().enumerate() {
        let timed = backend
            .dispatch_resident_timed(segment, resources, config)
            .map_err(|error| grid_sync_segment_error(error, segment_idx, segments.len()))?;
        if segment_idx + 1 == segments.len() {
            final_outputs = timed.outputs;
        }
        device_ns = crate::accounting::sum_optional_timing(
            device_ns,
            timed.device_ns,
            "device timing",
            "grid-sync segmented",
            "per-segment",
        )?;
        enqueue_ns = crate::accounting::sum_optional_timing(
            enqueue_ns,
            timed.enqueue_ns,
            "enqueue timing",
            "grid-sync segmented",
            "per-segment",
        )?;
        wait_ns = crate::accounting::sum_optional_timing(
            wait_ns,
            timed.wait_ns,
            "wait timing",
            "grid-sync segmented",
            "per-segment",
        )?;
    }
    Ok(TimedDispatchResult {
        outputs: final_outputs,
        wall_ns: elapsed_wall_ns(started)?,
        device_ns,
        enqueue_ns,
        wait_ns,
    })
}

/// Device-resident counterpart of
/// [`crate::grid_sync::dispatch_with_grid_sync_split_into`].
///
/// The host-split path round-trips every live buffer host↔device between each
/// split segment AND on every fixpoint pass. A fused multi-rule
/// `results_packed` accumulator is hundreds of MiB, so a program that splits
/// into hundreds of segments moves tens of GiB across PCIe per dispatch, that
/// transfer, not launch latency, is the host-split wall.
///
/// This variant uploads the program's inputs into backend-resident resources
/// ONCE, keeps them bound across every segment and every fixpoint pass, so a
/// multi-rule accumulator threads IN PLACE on device storage with no host copy
/// and no clobber (and reads back only the final output ranges a single time).
/// Net host↔device traffic drops from `O(segments × passes × live_bytes)` to
/// `O(inputs + outputs)`.
///
/// Every split segment from [`try_split_on_grid_sync`] carries the full program
/// buffer table (only the executable entry sequence differs), so one resident
/// resource slice binds to every segment. Resident dispatch never clears a
/// bound buffer between launches, so each rule's result-store accumulates into
/// the shared device `results_packed` exactly as the un-split program would.
///
/// `outputs` is shaped byte-identically to
/// [`crate::grid_sync::dispatch_with_grid_sync_split_into`]: one `Vec<u8>` per
/// original output buffer, in declaration order, so a caller can swap paths
/// without changing readback.
///
/// Requires a backend implementing the resident half of the [`VyreBackend`]
/// contract (`allocate_resident` / `upload_resident` /
/// `dispatch_resident_repeated_sequence_read_ranges_into` / `free_resident`).
/// A backend without residency fails loudly with `UnsupportedFeature` at the
/// first resident call; callers route those to
/// [`crate::grid_sync::dispatch_with_grid_sync_split_into`].
///
/// # Errors
/// Propagates any [`BackendError`] from splitting, resident allocation, upload,
/// segment dispatch, or readback. Resident resources allocated by this call are
/// always freed before returning, on success and on error.
pub fn dispatch_resident_grid_sync_fixpoint_into(
    backend: &dyn VyreBackend,
    program: &Program,
    inputs: &[&[u8]],
    config: &DispatchConfig,
    outputs: &mut OutputBuffers,
) -> Result<(), BackendError> {
    // These are the explicit non-native grid-sync routes (host split /
    // resident fixpoint). They split unconditionally when the program carries a
    // grid-sync barrier: native cooperative launch has a residency ceiling, so
    // `supports_grid_sync()` no longer implies "this program runs natively".
    // The orchestrator (or the registry's `should_split_grid_sync`) decides
    // native-vs-split per program; once here, always split.
    if !contains_grid_sync(program) {
        return backend.dispatch_borrowed_into(program, inputs, config, outputs);
    }
    let segments = try_split_on_grid_sync(program)?;
    reject_empty_grid_sync_split(&segments)?;
    crate::observability::record_grid_sync_split(segments.len());

    // Allocate one resident resource per non-shared binding (caller inputs
    // uploaded; output/scratch buffers zeroed so an accumulator's unfired
    // slots stay 0), then run the fixpoint and read back final outputs.
    let resident = allocate_resident_program_resources(backend, program, inputs)?;
    let result =
        run_resident_grid_sync_fixpoint(backend, program, &segments, &resident, config, outputs);
    // Free every resident resource before returning, success or error.
    let free_result = free_resident_program_resources(backend, resident);
    result.and(free_result)
}

/// Resident resources backing one [`crate::grid_sync::dispatch_resident_grid_sync_fixpoint_into`]
/// call: the binding-ordered slice every segment dispatches against, plus a
/// name → (handle, byte-len) map for output readback.
struct ResidentProgramResources {
    /// One resource per non-shared binding, in [`BindingPlan`] order, the
    /// slice the backend's resident dispatch binds positionally.
    ordered: Vec<Resource>,
    /// Buffer-name → (resident handle clone, byte length) for output readback
    /// by name. The handle is a cheap id clone; freeing `ordered` frees it.
    by_name: HashMap<Ident, (Resource, usize)>,
}

/// Allocate + initialize one resident resource per non-shared program binding.
///
/// Inputs are uploaded from the caller slice; output / write-only / scratch
/// buffers that consume no input are zeroed, mirroring the borrowed path's
/// memset of input-less buffers so a fused accumulator's unfired slots read 0.
fn allocate_resident_program_resources(
    backend: &dyn VyreBackend,
    program: &Program,
    inputs: &[&[u8]],
) -> Result<ResidentProgramResources, BackendError> {
    let plan = BindingPlan::from_borrowed_inputs(program, inputs)?;
    let mut ordered = Vec::new();
    reserve_grid_sync_vec(
        &mut ordered,
        plan.bindings.len(),
        "resident grid-sync resources",
    )?;
    let mut by_name = HashMap::new();
    reserve_grid_sync_hash_map(
        &mut by_name,
        plan.bindings.len(),
        "resident grid-sync resource name map",
    )?;
    for binding in &plan.bindings {
        if binding.role == BindingRole::Shared {
            continue;
        }
        // Logical length is the caller input slice length (input bindings) or
        // the buffer's static size (outputs/scratch). The host path binds the
        // unused standard scanner buffers (counts/offsets/lengths/metadata) as
        // zero-length `&[]`; resident allocation rejects 0 bytes, so allocate
        // one element (element-aligned, so the backend's element-size
        // validation holds) for those, the kernel never reads a 0/1-element
        // unused buffer, so the placeholder is bound but inert (proven equal to
        // the host path by the resident/host differential gate).
        let byte_len = resident_binding_byte_len(binding, inputs)?;
        let alloc_len = byte_len.max(binding.element_size.max(1));
        let resource = backend.allocate_resident(alloc_len)?;
        // Upload exactly `alloc_len` bytes so the backend's full-buffer upload
        // contract holds: the caller input when it is non-empty, else zeros
        // (output/scratch buffers, and the inert zero-length standard inputs).
        match binding.input_index {
            Some(index) if !inputs.get(index).copied().unwrap_or(&[]).is_empty() => {
                let bytes = inputs[index];
                backend.upload_resident(&resource, bytes)?;
            }
            _ => {
                let zeros = zeroed_upload_buffer(alloc_len)?;
                backend.upload_resident(&resource, &zeros)?;
            }
        }
        by_name.insert(
            Ident::from(binding.name.as_ref()),
            (resource.clone(), byte_len),
        );
        ordered.push(resource);
    }
    Ok(ResidentProgramResources { ordered, by_name })
}

/// Byte length to allocate for a binding's resident resource: the caller input
/// slice length for input-consuming bindings, else the buffer's static size.
fn resident_binding_byte_len(binding: &Binding, inputs: &[&[u8]]) -> Result<usize, BackendError> {
    if let Some(index) = binding.input_index {
        if let Some(bytes) = inputs.get(index) {
            return Ok(bytes.len());
        }
    }
    binding.static_byte_len.ok_or_else(|| BackendError::InvalidProgram {
        fix: format!(
            "Fix: resident grid-sync output buffer `{}` has no static byte length; dynamic-sized outputs are not supported on the resident grid-sync path. Declare a fixed `count` on the buffer or route this program through dispatch_with_grid_sync_split_into.",
            binding.name
        ),
    })
}

/// Allocate a zero-filled host staging buffer of `byte_len` for initializing a
/// resident output/scratch resource.
fn zeroed_upload_buffer(byte_len: usize) -> Result<Vec<u8>, BackendError> {
    let mut zeros = Vec::new();
    crate::allocation::try_reserve_vec_to_capacity(&mut zeros, byte_len).map_err(|error| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: failed to reserve a {byte_len}-byte zero-init staging buffer for a resident grid-sync output: {error}. Shard the program into smaller buffers."
            ),
        }
    })?;
    zeros.resize(byte_len, 0);
    Ok(zeros)
}

/// Run the fixpoint sequence resident: every segment dispatched against the
/// shared resident resource slice, the whole sequence repeated to the program's
/// fixpoint bound, then the final outputs read back by name into `outputs`.
fn run_resident_grid_sync_fixpoint(
    backend: &dyn VyreBackend,
    program: &Program,
    segments: &[Program],
    resident: &ResidentProgramResources,
    config: &DispatchConfig,
    outputs: &mut OutputBuffers,
) -> Result<(), BackendError> {
    let iterations = crate::fixpoint_iterations::resolve_fixpoint_iterations(
        config,
        "resident grid-sync split",
    )?;
    let repeat_count = iterations;

    // Every split segment shares the full program buffer layout, so the same
    // resident resource slice binds positionally to each one.
    let mut steps = Vec::new();
    reserve_grid_sync_vec(&mut steps, segments.len(), "resident grid-sync steps")?;
    for segment in segments {
        steps.push(ResidentDispatchStep {
            program: segment,
            resources: resident.ordered.as_slice(),
            grid_override: config.grid_override,
            // Carry the workgroup too: `grid_override` is sized for this
            // workgroup, so dropping it would launch a grid that under-covers
            // the work and silently drops findings.
            workgroup_override: config.workgroup_override,
        });
    }

    // Read back each original output buffer (declaration order) so the output
    // shape is byte-identical to the host-split path.
    let output_names = original_output_names(program)?;
    let mut read_ranges = Vec::new();
    reserve_grid_sync_vec(
        &mut read_ranges,
        output_names.len(),
        "resident grid-sync read ranges",
    )?;
    for name in &output_names {
        let (resource, byte_len) =
            resident.by_name.get(name).ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: resident grid-sync final output `{name}` has no resident resource; it was not declared as a non-shared program buffer."
                ),
            })?;
        read_ranges.push(ResidentReadRange {
            resource,
            byte_offset: 0,
            byte_len: *byte_len,
        });
    }

    // Size `outputs` to one slot per output buffer, reusing existing
    // allocations, then hand the readback mutable references in order.
    while outputs.len() < output_names.len() {
        outputs.push(Vec::new());
    }
    outputs.truncate(output_names.len());
    for slot in outputs.iter_mut() {
        slot.clear();
    }
    let mut output_refs: Vec<&mut Vec<u8>> = outputs.iter_mut().collect();

    backend.dispatch_resident_repeated_sequence_read_ranges_into(
        &[],
        &steps,
        repeat_count,
        &read_ranges,
        output_refs.as_mut_slice(),
    )
}

/// Free every resident resource allocated for a
/// [`dispatch_resident_grid_sync_fixpoint_into`] call. Attempts every free even
/// if one fails, returning the first error so a leak is surfaced loudly.
fn free_resident_program_resources(
    backend: &dyn VyreBackend,
    resident: ResidentProgramResources,
) -> Result<(), BackendError> {
    let ResidentProgramResources { ordered, by_name } = resident;
    // `by_name` holds handle clones of the same resources in `ordered`; drop
    // it first so each underlying handle is freed exactly once via `ordered`.
    drop(by_name);
    let mut first_error: Option<BackendError> = None;
    for resource in ordered {
        if let Err(error) = backend.free_resident(resource) {
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_sync::barrier_split::entry_sequence;
    use crate::grid_sync::test_programs::{
        apply_out_stores, cross_segment_store_program, grid_sync_chain,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};
    use vyre_foundation::memory_model::MemoryOrdering;

    struct ResidentReuseBackend {
        calls: AtomicUsize,
        owner: crate::ResidentOwner,
    }

    impl crate::backend::private::Sealed for ResidentReuseBackend {}

    impl VyreBackend for ResidentReuseBackend {
        fn id(&self) -> &'static str {
            "grid-sync-resident-reuse"
        }

        fn dispatch(
            &self,
            _program: &Program,
            _inputs: &[Vec<u8>],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            unreachable!("test uses dispatch_resident_timed")
        }

        fn dispatch_borrowed_into(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
            _outputs: &mut OutputBuffers,
        ) -> Result<(), BackendError> {
            unreachable!("resident grid-sync split must not refresh through host borrowed inputs")
        }

        fn dispatch_resident_timed(
            &self,
            _program: &Program,
            resources: &[Resource],
            _config: &DispatchConfig,
        ) -> Result<TimedDispatchResult, BackendError> {
            let bound: Vec<u64> = resources
                .iter()
                .map(|resource| match resource {
                    Resource::Resident(handle) => self
                        .owner
                        .resolve(*handle, "grid-sync resident reuse test backend")
                        .expect("Fix: test backend must only receive its own resident handles"),
                    Resource::Borrowed(_) => panic!(
                        "Fix: resident grid-sync split must bind device handles, not host bytes."
                    ),
                })
                .collect();
            assert_eq!(
                bound,
                vec![11, 22],
                "Fix: resident grid-sync split must keep the original device handles bound across every segment."
            );
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(TimedDispatchResult {
                outputs: vec![vec![call as u8]],
                wall_ns: 10,
                device_ns: Some(2),
                enqueue_ns: Some(3),
                wait_ns: Some(4),
            })
        }
    }

    #[test]
    fn resident_split_reuses_same_device_resources_across_segments() {
        let program = grid_sync_chain(&["a", "b", "c"]);
        let owner = crate::ResidentOwner::new().expect("Fix: owner ids must be available");
        let backend = ResidentReuseBackend {
            calls: AtomicUsize::new(0),
            owner,
        };

        let timed = dispatch_resident_with_grid_sync_split_timed(
            &backend,
            &program,
            &[
                Resource::Resident(owner.handle(11)),
                Resource::Resident(owner.handle(22)),
            ],
            &DispatchConfig::default(),
        )
        .expect("Fix: resident grid-sync split should run each segment on the same device handles");

        assert_eq!(backend.calls.load(Ordering::SeqCst), 3);
        assert_eq!(timed.outputs, vec![vec![2]]);
        assert_eq!(timed.device_ns, Some(6));
        assert_eq!(timed.enqueue_ns, Some(9));
        assert_eq!(timed.wait_ns, Some(12));
    }

    /// In-memory device for the resident fixpoint path: holds one byte vector
    /// per resident handle, applies a segment's `out` stores IN PLACE to the
    /// bound device buffer (no clear between launches), and reads ranges back.
    /// `allocate_resident` fills fresh buffers with 0xFF so a test can prove the
    /// zero-init upload actually ran.
    struct ResidentDeviceBackend {
        owner: crate::ResidentOwner,
        next_id: std::sync::atomic::AtomicU64,
        buffers: std::sync::Mutex<HashMap<u64, Vec<u8>>>,
        freed: std::sync::Mutex<Vec<u64>>,
        dispatches: AtomicUsize,
    }

    impl ResidentDeviceBackend {
        fn new() -> Self {
            Self {
                owner: crate::ResidentOwner::new().expect("Fix: owner ids must be available"),
                next_id: std::sync::atomic::AtomicU64::new(1),
                buffers: std::sync::Mutex::new(HashMap::new()),
                freed: std::sync::Mutex::new(Vec::new()),
                dispatches: AtomicUsize::new(0),
            }
        }

        fn resident_id(&self, resource: &Resource) -> u64 {
            match resource {
                Resource::Resident(handle) => self
                    .owner
                    .resolve(*handle, "grid-sync resident fixpoint test backend")
                    .expect("Fix: test backend must only receive its own resident handles"),
                Resource::Borrowed(_) => {
                    panic!(
                        "Fix: resident grid-sync fixpoint must bind Resident handles, not Borrowed"
                    )
                }
            }
        }
    }

    impl crate::backend::private::Sealed for ResidentDeviceBackend {}

    impl VyreBackend for ResidentDeviceBackend {
        fn id(&self) -> &'static str {
            "grid-sync-resident-device"
        }

        fn dispatch(
            &self,
            _program: &Program,
            _inputs: &[Vec<u8>],
            _config: &DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, BackendError> {
            unreachable!("resident fixpoint test uses resident dispatch")
        }

        fn dispatch_borrowed_into(
            &self,
            _program: &Program,
            _inputs: &[&[u8]],
            _config: &DispatchConfig,
            _outputs: &mut OutputBuffers,
        ) -> Result<(), BackendError> {
            unreachable!("resident fixpoint must thread device handles, never host borrowed inputs")
        }

        fn allocate_resident(&self, byte_len: usize) -> Result<Resource, BackendError> {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            // Fresh device memory is garbage (0xFF here) so the zero-init upload
            // path is actually exercised by the test assertions.
            self.buffers
                .lock()
                .unwrap()
                .insert(id, vec![0xFFu8; byte_len]);
            Ok(Resource::Resident(self.owner.handle(id)))
        }

        fn upload_resident(&self, resource: &Resource, bytes: &[u8]) -> Result<(), BackendError> {
            let id = self.resident_id(resource);
            let mut buffers = self.buffers.lock().unwrap();
            let buf = buffers.get_mut(&id).expect("resident handle exists");
            assert!(
                bytes.len() <= buf.len(),
                "upload {} bytes into a {}-byte resident buffer",
                bytes.len(),
                buf.len()
            );
            buf[..bytes.len()].copy_from_slice(bytes);
            Ok(())
        }

        fn download_resident_range_into(
            &self,
            resource: &Resource,
            byte_offset: usize,
            byte_len: usize,
            output: &mut Vec<u8>,
        ) -> Result<(), BackendError> {
            let id = self.resident_id(resource);
            let buffers = self.buffers.lock().unwrap();
            let buf = buffers.get(&id).expect("resident handle exists");
            output.clear();
            output.extend_from_slice(&buf[byte_offset..byte_offset + byte_len]);
            Ok(())
        }

        fn free_resident(&self, resource: Resource) -> Result<(), BackendError> {
            let id = self.resident_id(&resource);
            self.buffers.lock().unwrap().remove(&id);
            self.freed.lock().unwrap().push(id);
            Ok(())
        }

        fn dispatch_resident_timed(
            &self,
            program: &Program,
            resources: &[Resource],
            _config: &DispatchConfig,
        ) -> Result<TimedDispatchResult, BackendError> {
            self.dispatches.fetch_add(1, Ordering::SeqCst);
            // Find `out`'s index among the non-shared bindings  -  the same
            // order `allocate_resident_program_resources` builds `resources` in.
            let plan = BindingPlan::build(program)?;
            let mut out_slot = None;
            let mut pos = 0usize;
            for binding in &plan.bindings {
                if binding.role == BindingRole::Shared {
                    continue;
                }
                if binding.name.as_ref() == "out" {
                    out_slot = Some(pos);
                }
                pos += 1;
            }
            let out_slot = out_slot.expect("program declares `out`");
            let id = self.resident_id(&resources[out_slot]);
            let mut buffers = self.buffers.lock().unwrap();
            let buf = buffers.get_mut(&id).expect("resident `out` handle exists");

            // Apply the segment's `out` stores IN PLACE  -  never clearing the
            // buffer, so earlier segments' slots persist (the accumulator).
            apply_out_stores(entry_sequence(program), buf.as_mut_slice());

            Ok(TimedDispatchResult {
                outputs: Vec::new(),
                wall_ns: 1,
                device_ns: Some(1),
                enqueue_ns: Some(1),
                wait_ns: Some(1),
            })
        }
    }

    #[test]
    fn resident_fixpoint_accumulates_across_segments_zero_inits_and_frees() {
        // Same cross-anchor shape as the host-path regression: arm A stores slot
        // 0 in segment 0, arm B stores slot 2 in the final segment. The resident
        // path keeps ONE device `out` buffer bound across both segments, so both
        // slots must survive WITHOUT the host-path accumulator role-rewrite  -
        // the persistent device buffer is never cleared between launches.
        let program = cross_segment_store_program();
        let backend = ResidentDeviceBackend::new();
        let mut outputs = vec![Vec::new()];
        dispatch_resident_grid_sync_fixpoint_into(
            &backend,
            &program,
            &[],
            &DispatchConfig::default(),
            &mut outputs,
        )
        .expect("resident grid-sync fixpoint dispatch");

        assert_eq!(
            backend.dispatches.load(Ordering::SeqCst),
            2,
            "two segments, single fixpoint pass under the default config"
        );
        assert_eq!(outputs.len(), 1, "one output buffer (`out`)");
        assert_eq!(outputs[0].len(), 16, "4 × u32 = 16 bytes");
        assert_eq!(
            outputs[0][0], 0xAA,
            "segment 0's slot survives  -  resident accumulation, no clobber"
        );
        assert_eq!(outputs[0][8], 0xBB, "the final segment's slot is present");
        // Zero-init proof: every byte the kernel did not write is 0, not the
        // 0xFF garbage `allocate_resident` seeded  -  the output buffer was
        // zeroed before dispatch.
        assert_eq!(outputs[0][4], 0x00, "untouched slot 1 was zero-initialized");
        assert_eq!(
            outputs[0][12], 0x00,
            "untouched slot 3 was zero-initialized"
        );
        // Every resident resource is freed exactly once.
        assert_eq!(
            backend.freed.lock().unwrap().len(),
            1,
            "the single `out` resident buffer is freed"
        );
        assert!(
            backend.buffers.lock().unwrap().is_empty(),
            "no resident buffer leaks after dispatch"
        );
    }

    #[test]
    fn resident_fixpoint_repeats_to_fixpoint_bound() {
        // With a fixpoint bound > 1, the whole segment sequence repeats that many
        // times against the same resident buffers (idempotent stores here, so the
        // result is unchanged, but the launch count proves the repeat wiring).
        let program = cross_segment_store_program();
        let backend = ResidentDeviceBackend::new();
        let mut config = DispatchConfig::default();
        config.fixpoint_iterations = Some(3);
        let mut outputs = vec![Vec::new()];
        dispatch_resident_grid_sync_fixpoint_into(&backend, &program, &[], &config, &mut outputs)
            .expect("resident grid-sync fixpoint dispatch");
        assert_eq!(
            backend.dispatches.load(Ordering::SeqCst),
            6,
            "2 segments × 3 fixpoint passes"
        );
        assert_eq!(outputs[0][0], 0xAA);
        assert_eq!(outputs[0][8], 0xBB);
    }
}
