use std::ffi::c_void;

use smallvec::SmallVec;
use vyre_driver::binding::BindingRole;
use vyre_driver::{BackendError, BindingPlan, DispatchConfig};
use vyre_foundation::ir::Program;

use crate::backend::allocations::{DispatchAllocations, HostTransferAllocations};
use crate::backend::dispatch::CudaBackend;
use crate::backend::ordering::sort_unstable_by_key_if_needed;
use crate::backend::resident::{
    resident_bindings_from_handles, CudaDispatchBinding, CudaResidentBuffer, ResidentViewCache,
};
use crate::backend::resident_dispatch::dense_index_validation::validate_dense_resident_input_indices;
use crate::backend::resident_dispatch::descriptor_cursor::{
    next_dispatch_binding, next_resident_handle,
};
use crate::backend::resident_dispatch::PreparedStep;
use crate::backend::resident_dispatch_support::CudaResidentDispatchStep;
use crate::backend::resident_upload_fusion::{
    fuse_resident_upload_copies, push_resident_upload_copy, ResidentUploadCopy,
};
use crate::backend::staging_reserve::{reserve_smallvec, reserved_vec};

type ParamUpload = (u64, Option<(u64, *const c_void, usize)>);

pub(super) fn order_resident_fallback_inputs_by_logical_index(
    input_storage: &mut [(usize, Vec<u8>)],
    expected_len: usize,
) -> Result<(), BackendError> {
    sort_unstable_by_key_if_needed(input_storage, |(input_index, _)| *input_index);
    validate_dense_resident_input_indices(
        input_storage.iter().map(|(input_index, _)| *input_index),
        expected_len,
        "resident fallback input storage",
    )
}

impl CudaBackend {
    pub(super) fn resolve_resident_sequence_launch_ptrs(
        &self,
        step: &PreparedStep<'_>,
        resident_view_cache: &mut ResidentViewCache,
    ) -> Result<SmallVec<[u64; 8]>, BackendError> {
        let mut launch_ptrs = SmallVec::<[u64; 8]>::new();
        reserve_smallvec(
            &mut launch_ptrs,
            step.prepared.bindings.bindings.len(),
            "resident sequence launch pointers",
        )?;
        let mut next_handle = 0usize;
        for binding in &step.prepared.bindings.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let handle =
                next_resident_handle(&step.handles, &mut next_handle, "resident sequence launch")?;
            let resident = self.resident_store.view_cached(
                handle,
                resident_view_cache,
                "resident sequence view cache",
            )?;
            resident.validate_binding(
                "resident sequence",
                &binding.name,
                binding.static_byte_len,
                handle.handle,
            )?;
            launch_ptrs.push(resident.ptr);
        }
        Ok(launch_ptrs)
    }

    pub(crate) fn dispatch_resident_via_borrowed_into(
        &self,
        program: &Program,
        bindings: &[CudaDispatchBinding<'_>],
        config: &DispatchConfig,
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        // Bind this device's CUDA context to the CALLING thread before any launch.
        // The backend is `Send + Sync` and W3-5 cross-device parallel dispatch drives
        // one resident scan per device from its own thread, so the context must be
        // made current here (idempotent when already current), matching the batch,
        // async, and fused-sequence resident entry points (ONE PLACE pattern). Without
        // this, a resident scan issued from a thread other than the acquiring one
        // faults with CUDA_ERROR_INVALID_CONTEXT.
        self.warmup()?;
        self.telemetry.record_resident_borrowed_fallback_dispatch();
        let plan = BindingPlan::build(program)?;
        let required_bindings = plan
            .bindings
            .len()
            .checked_sub(plan.shared_indices.len())
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident fallback binding plan has {} binding(s) but {} shared binding index(es). Rebuild the dispatch plan before launching.",
                    plan.bindings.len(),
                    plan.shared_indices.len()
                ),
            })?;
        if bindings.len() != required_bindings {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident fallback expected {required_bindings} bound resource(s) but received {}.",
                    bindings.len()
                ),
            });
        }
        let mut input_storage =
            reserved_vec(plan.input_indices.len(), "resident fallback input storage")?;
        let mut output_handles =
            reserved_vec(plan.output_indices.len(), "resident fallback output handle")?;
        let mut next_binding = 0usize;
        for binding in &plan.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let source =
                next_dispatch_binding(bindings, &mut next_binding, "resident borrowed dispatch")?;
            if let Some(input_index) = binding.input_index {
                // This diagnostic fallback runs everything through the borrowed
                // path, so a resident input is read back to host bytes here. A
                // borrowed input is already host bytes and is copied because
                // `dispatch_borrowed` wants one uniform slice list.
                let bytes = match source {
                    CudaDispatchBinding::Resident(handle) => self.download_resident(handle)?,
                    CudaDispatchBinding::Borrowed(bytes) => bytes.to_vec(),
                };
                input_storage.push((input_index, bytes));
            }
            if let Some(output_index) = binding.output_index {
                // Only a resident output needs the result written back to the
                // device; a borrowed output is returned to the caller by value.
                if let CudaDispatchBinding::Resident(handle) = source {
                    output_handles.push((output_index, handle));
                }
            }
        }
        order_resident_fallback_inputs_by_logical_index(
            input_storage.as_mut_slice(),
            plan.input_indices.len(),
        )?;
        let mut input_refs = SmallVec::<[&[u8]; 8]>::new();
        reserve_smallvec(
            &mut input_refs,
            input_storage.len(),
            "resident fallback input reference",
        )?;
        input_refs.extend(input_storage.iter().map(|(_, bytes)| bytes.as_slice()));
        let dispatch_outputs = self.dispatch_borrowed(program, &input_refs, config)?;
        let mut output_uploads = SmallVec::<[(CudaResidentBuffer, &[u8]); 8]>::new();
        reserve_smallvec(
            &mut output_uploads,
            output_handles.len(),
            "resident fallback output upload",
        )?;
        for &(output_index, handle) in &output_handles {
            let output =
                dispatch_outputs
                    .get(output_index)
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident fallback missing output slot {output_index}; keep borrowed dispatch output ordering aligned with BindingPlan."
                        ),
                    })?;
            if !output.is_empty() {
                output_uploads.push((handle, output.as_slice()));
            }
        }
        self.upload_resident_many(&output_uploads)?;
        drop(output_uploads);
        vyre_driver::replace_output_buffers_preserving_slots(dispatch_outputs, outputs);
        Ok(())
    }

    pub(crate) fn dispatch_resident_via_borrowed(
        &self,
        program: &Program,
        bindings: &[CudaDispatchBinding<'_>],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let mut outputs = reserved_vec(0, "borrowed resident dispatch outputs")?;
        self.dispatch_resident_via_borrowed_into(program, bindings, config, &mut outputs)?;
        Ok(outputs)
    }

    /// Resolve the kernel parameter pointer for one resident dispatch,
    /// uploading the parameter words when the plan has no static pointer.
    ///
    /// Three answers, and every resident path needs all three: a plan that
    /// already owns a device-side parameter block hands its pointer straight
    /// back, a plan with no parameters at all launches with a null block rather
    /// than allocating an empty one, and anything else stages the words into a
    /// transient allocation whose upload the caller enqueues.
    ///
    /// `role` names the dispatch path and reaches only the four accounting
    /// labels. It was four separately spelled label constants per caller, which
    /// is four chances to record one path's parameter bytes under another's
    /// budget.
    pub(super) fn resolve_resident_params_ptr(
        &self,
        param_words: &[u32],
        param_bytes: usize,
        static_params_ptr: Option<u64>,
        role: &str,
        allocations: &mut DispatchAllocations,
        host_transfers: &mut HostTransferAllocations,
    ) -> Result<ParamUpload, BackendError> {
        match static_params_ptr {
            Some(ptr) => Ok((ptr, None)),
            None if param_bytes == 0 => Ok((0, None)),
            None => self.prepare_resident_param_upload(
                param_words,
                param_bytes,
                role,
                allocations,
                host_transfers,
            ),
        }
    }

    fn prepare_resident_param_upload(
        &self,
        param_words: &[u32],
        param_bytes: usize,
        role: &str,
        allocations: &mut DispatchAllocations,
        host_transfers: &mut HostTransferAllocations,
    ) -> Result<ParamUpload, BackendError> {
        self.validate_transient_allocation_memory_budget(
            param_bytes,
            &format!("CUDA {role} parameter bytes"),
            &format!("CUDA {role} parameter upload"),
        )?;
        let params_allocation = self.transient_pool.acquire(param_bytes)?;
        self.telemetry.record_transient_allocation_bytes(
            crate::numeric::CUDA_NUMERIC.usize_to_u64(
                params_allocation.byte_len,
                &format!("{role} parameter allocation byte count"),
            )?,
        );
        let params_ptr = params_allocation.ptr;
        let param_host_ptr = host_transfers.push_u32_words(param_words)?;
        let upload_metric = format!("{role} parameter upload byte count");
        let upload_bytes =
            crate::numeric::CUDA_NUMERIC.usize_to_u64(param_bytes, &upload_metric)?;
        self.telemetry.record_host_to_device_bytes(upload_bytes);
        self.telemetry.record_host_upload_operations(1);
        self.telemetry.record_param_upload_bytes(upload_bytes);
        allocations.set_params(params_allocation);
        Ok((params_ptr, Some((params_ptr, param_host_ptr, param_bytes))))
    }

    pub(super) fn prepare_resident_sequence_upload_copies<'a>(
        &self,
        uploads: &[(CudaResidentBuffer, &'a [u8])],
    ) -> Result<(SmallVec<[ResidentUploadCopy<'a>; 8]>, u64), BackendError> {
        let mut upload_copies = SmallVec::<[ResidentUploadCopy<'a>; 8]>::new();
        reserve_smallvec(
            &mut upload_copies,
            uploads.len(),
            "resident sequence upload copies",
        )?;
        let mut uploaded_bytes = 0_u64;
        let mut resident_view_cache = ResidentViewCache::new();
        reserve_smallvec(
            &mut resident_view_cache,
            uploads.len(),
            "resident sequence upload view cache",
        )?;
        for &(handle, bytes) in uploads {
            let buffer = self.resident_store.view_cached(
                handle,
                &mut resident_view_cache,
                "resident sequence upload view cache",
            )?;
            if bytes.len() != buffer.byte_len {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident sequence upload for handle {} expected {} bytes but received {}.",
                        handle.handle,
                        buffer.byte_len,
                        bytes.len()
                    ),
                });
            }
            push_resident_upload_copy(
                &mut upload_copies,
                &mut uploaded_bytes,
                handle.handle.id(),
                buffer.ptr,
                bytes,
                "sequence upload",
            )?;
        }
        fuse_resident_upload_copies(upload_copies)
    }

    pub(super) fn push_prepared_resident_sequence_step<'a>(
        &self,
        step: &'a CudaResidentDispatchStep<'a>,
        prepared_steps: &mut SmallVec<[PreparedStep<'a>; 8]>,
        target_indices: &mut SmallVec<[usize; 16]>,
        all_handles: &mut SmallVec<[CudaResidentBuffer; 32]>,
    ) -> Result<(), BackendError> {
        all_handles.extend(step.handles.iter().copied());
        if let Some(index) = prepared_steps.iter().position(|cached| {
            std::ptr::addr_eq(cached.program, step.program)
                && cached.handles.as_slice() == step.handles
                && cached.config == &step.config
        }) {
            target_indices.push(index);
            return Ok(());
        }
        let step_bindings = resident_bindings_from_handles(step.handles)?;
        let prepared =
            self.prepare_resident_dispatch(step.program, &step_bindings, &step.config)?;
        let (ptx_src, ptx_source_key) =
            self.ptx_for_program_cached_with_key(step.program, &step.config)?;
        let module_key = self.module_cache_key_for_ptx_source_key(ptx_source_key)?;
        let step_index = prepared_steps.len();
        prepared_steps.push(PreparedStep {
            program: step.program,
            handles: SmallVec::<[CudaResidentBuffer; 8]>::from_slice(step.handles),
            config: &step.config,
            ptx_src,
            module_key,
            prepared,
        });
        target_indices.push(step_index);
        Ok(())
    }
}
