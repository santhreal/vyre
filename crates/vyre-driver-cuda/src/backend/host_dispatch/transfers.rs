use std::ffi::c_void;
use std::fmt::Write as _;

use cudarc::driver::sys::CUstream;
use vyre_driver::accounting::checked_add_usize_lazy;
use vyre_driver::transfer_accounting::TransferAccountingPolicy;
use vyre_driver::{BackendError, PendingDispatch};
use vyre_foundation::ir::Program;

use super::super::plan::CudaDispatchPlan;

#[derive(Clone, Copy)]
pub(crate) struct HostUpload {
    pub(crate) dst: u64,
    pub(crate) src: *const c_void,
    pub(crate) byte_len: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct DeviceClear {
    pub(crate) dst: u64,
    pub(crate) byte_len: usize,
}

pub(crate) struct CudaReadyPending {
    pub(crate) outputs: Vec<Vec<u8>>,
}

pub(crate) const CUDA_HOST_TRANSFER_ACCOUNTING: TransferAccountingPolicy =
    TransferAccountingPolicy::new("CUDA", "split the dispatch into bounded chunks");

impl vyre_driver::sealed::Sealed for CudaReadyPending {}

impl PendingDispatch for CudaReadyPending {
    fn is_ready(&self) -> bool {
        true
    }

    fn await_result(self: Box<Self>) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(self.outputs)
    }
}

pub(crate) fn add_transfer_bytes(
    total: &mut u64,
    bytes: usize,
    label: &str,
) -> Result<(), BackendError> {
    CUDA_HOST_TRANSFER_ACCOUNTING.add_bytes(total, bytes, label)
}

pub(crate) fn add_transfer_operation(total: &mut u64, label: &str) -> Result<(), BackendError> {
    CUDA_HOST_TRANSFER_ACCOUNTING.add_operation(total, label)
}

/// Reject a program whose static workgroup scratch exceeds the device's
/// per-workgroup shared memory limit.
///
/// The limit is `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`, the
/// static allocation ceiling, which is smaller than the per-SM figure. A
/// program that crosses it fails at `cuModuleLoadData` with a diagnostic
/// that names PTX rather than shared memory, so the hunt starts in the
/// wrong place. Naming the measured bytes, the cap, and the contributing
/// buffers ends it immediately.
///
/// Buffers whose element type has no static width are skipped rather than
/// guessed: understating scratch here would reintroduce the silent case
/// this check exists to prevent, and such a buffer cannot be lowered to
/// fixed-size shared storage anyway.
///
/// Do not "complete" this by estimating a width. The gate is ADDITIVE: the
/// real module load still runs behind it, so a program whose scratch this
/// undercounts degrades to the pre-existing `CUDA_ERROR_INVALID_PTX`
/// message and never to silence. A miss here is bounded by the diagnostic
/// it replaces, while a false reject from a guessed width would break
/// working programs across the whole dispatch path.
pub(crate) fn check_workgroup_scratch_budget(
    program: &Program,
    limit_bytes: u32,
) -> Result<(), BackendError> {
    let mut total: u64 = 0;
    let mut breakdown = String::new();
    for buffer in program.buffers() {
        if buffer.access() != vyre_foundation::ir::BufferAccess::Workgroup {
            continue;
        }
        let Ok(Some(bytes)) = buffer.static_byte_len() else {
            continue;
        };
        // Checked, not saturating: saturating to u64::MAX would happen to exceed
        // the limit and error, but it reports a scratch total the program does not
        // have, and the file's accounting contract forbids saturating arithmetic
        // for exactly that reason.
        total = vyre_driver::accounting::checked_add_u64_usize_offset_lazy(
            total,
            bytes,
            || {
                BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA workgroup buffer `{}` reports {bytes} bytes, which does not fit u64. Reduce its element count or move the scratch to a storage buffer.",
                    buffer.name()
                ),
            }
            },
            || {
                BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA workgroup scratch total overflowed u64 while adding `{}` at {bytes} bytes to a running total of {total} bytes. Reduce the workgroup buffer element counts or move the scratch to a storage buffer.",
                    buffer.name()
                ),
            }
            },
        )?;
        if !breakdown.is_empty() {
            breakdown.push_str(", ");
        }
        let _ = write!(breakdown, "`{}` {bytes} bytes", buffer.name());
    }
    if total <= u64::from(limit_bytes) {
        return Ok(());
    }
    Err(BackendError::InvalidProgram {
        fix: format!(
            "CUDA workgroup scratch for this program is {total} bytes, over the device \
             per-workgroup static shared memory limit of {limit_bytes} bytes. \
             Contributing buffers: {breakdown}. \
             Fix: reduce the workgroup buffer element counts, narrow the workgroup width \
             they are sized against, or move the scratch to a storage buffer."
        ),
    })
}

#[inline]
pub(crate) fn host_transfer_capacities(
    prepared: &CudaDispatchPlan,
) -> Result<(usize, usize), BackendError> {
    let output_capacity = prepared.output_binding_indices.len();
    let upload_capacity = host_upload_batch_capacity(prepared)?;
    let transfer_capacity = checked_add_usize_lazy(upload_capacity, output_capacity, || {
        BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA host transfer capacity overflowed usize for {upload_capacity} upload slot(s) plus {output_capacity} output slot(s); split the dispatch."
                ),
            }
    })?;
    Ok((transfer_capacity, output_capacity))
}

#[inline]
pub(crate) fn host_upload_batch_capacity(
    prepared: &CudaDispatchPlan,
) -> Result<usize, BackendError> {
    let input_slots = prepared.bindings.input_indices.len();
    checked_add_usize_lazy(
        input_slots,
        usize::from(!prepared.launch.param_words.is_empty()),
        || {
            BackendError::InvalidProgram {
            fix: "Fix: CUDA host upload batch capacity overflowed usize while adding the params upload slot; split the dispatch."
                .to_string(),
        }
        },
    )
}

#[inline]
pub(crate) fn enqueue_host_uploads_async(
    uploads: &[HostUpload],
    stream: CUstream,
) -> Result<(), BackendError> {
    for upload in uploads {
        if upload.byte_len == 0 {
            continue;
        }
        // SAFETY: FFI to libcuda.so. Pointer args were validated by the
        // matching alloc / store API; lifetimes are documented in the
        // surrounding function. cuda_check (or matching CUresult guard)
        // propagates non-success codes as BackendError.
        unsafe {
            super::super::copy::h2d_async_checked(upload.dst, upload.src, upload.byte_len, stream)?;
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn enqueue_device_clears_async(
    clears: &[DeviceClear],
    stream: CUstream,
) -> Result<(), BackendError> {
    for clear in clears {
        // SAFETY: FFI to libcuda.so. Pointer args were validated by the
        // matching alloc / store API; lifetimes are documented in the
        // surrounding function. cuda_check (or matching CUresult guard)
        // propagates non-success codes as BackendError.
        unsafe {
            super::super::copy::memset_d8_async_checked(clear.dst, 0, clear.byte_len, stream)?;
        }
    }
    Ok(())
}

// Inline: covers `host_transfer_capacities`, `host_upload_batch_capacity`, which no integration
// test can name.
#[cfg(test)]
mod tests {
    use super::{host_transfer_capacities, host_upload_batch_capacity};
    use crate::backend::CudaDispatchPlan;
    use smallvec::smallvec;
    use std::sync::Arc;
    use vyre_driver::LaunchPlan;
    use vyre_driver::{Binding, BindingPlan, BindingRole};

    #[test]
    fn host_upload_batch_capacity_counts_inputs_once_plus_params() {
        let plan = CudaDispatchPlan {
            bindings: BindingPlan {
                bindings: vec![
                    Binding {
                        name: Arc::from("a"),
                        binding: 0,
                        buffer_index: 0,
                        role: BindingRole::Input,
                        element_size: 4,
                        preferred_alignment: 4,
                        element_count: 16,
                        static_byte_len: Some(64),
                        input_index: Some(0),
                        output_index: None,
                    },
                    Binding {
                        name: Arc::from("b"),
                        binding: 1,
                        buffer_index: 1,
                        role: BindingRole::InputOutput,
                        element_size: 4,
                        preferred_alignment: 4,
                        element_count: 16,
                        static_byte_len: Some(64),
                        input_index: Some(1),
                        output_index: Some(0),
                    },
                    Binding {
                        name: Arc::from("out"),
                        binding: 2,
                        buffer_index: 2,
                        role: BindingRole::Output,
                        element_size: 4,
                        preferred_alignment: 4,
                        element_count: 16,
                        static_byte_len: Some(64),
                        input_index: None,
                        output_index: Some(1),
                    },
                ],
                input_indices: vec![0, 1],
                output_indices: vec![1, 2],
                shared_indices: Vec::new(),
            },
            output_binding_indices: smallvec![1, 2],
            launch: LaunchPlan::new(),
            cooperative: false,
            fixpoint_iterations: 1,
        };

        assert_eq!(
            host_upload_batch_capacity(&plan).expect("Fix: capacity must fit"),
            2,
            "zero-byte launch params must not reserve a fake H2D upload slot"
        );
        assert_eq!(
            host_transfer_capacities(&plan).expect("Fix: capacity must fit"),
            (4, 2),
            "pinned-host transfer storage must reserve inputs + outputs only when params are empty"
        );

        let mut plan_with_params = plan;
        plan_with_params.launch.param_words.push(7);
        assert_eq!(
            host_upload_batch_capacity(&plan_with_params).expect("Fix: capacity must fit"),
            3,
            "non-empty launch params must reserve one H2D upload slot"
        );
        assert_eq!(
            host_transfer_capacities(&plan_with_params).expect("Fix: capacity must fit"),
            (5, 2),
            "pinned-host transfer storage must reserve inputs + params + outputs when params exist"
        );
    }
}
