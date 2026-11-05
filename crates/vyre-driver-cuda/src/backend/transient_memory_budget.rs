use vyre_driver::BackendError;
use vyre_driver::BindingRole;

use super::launch_params::launch_param_byte_len;
use super::plan::CudaDispatchPlan;
use super::resident::CudaDispatchBinding;
use super::resident_dispatch::next_dispatch_binding;

pub(crate) const CUDA_TRANSIENT_DISPATCH_BUDGET_NUMERATOR: u64 = 9;
pub(crate) const CUDA_TRANSIENT_DISPATCH_BUDGET_DENOMINATOR: u64 = 10;

pub(crate) fn cuda_transient_dispatch_budget_bytes(total_memory: u64) -> u64 {
    let numerator = u128::from(total_memory) * u128::from(CUDA_TRANSIENT_DISPATCH_BUDGET_NUMERATOR);
    (numerator / u128::from(CUDA_TRANSIENT_DISPATCH_BUDGET_DENOMINATOR)) as u64
}

pub(crate) fn cuda_live_free_memory_bytes() -> Result<u64, BackendError> {
    let (free, _total) = cudarc::driver::result::mem_get_info().map_err(|error| {
        BackendError::DispatchFailed {
            code: None,
            message: format!(
                "CUDA live-memory query failed: {error}. Fix: keep the CUDA context bound before memory preflight and treat query failure as a GPU release-path configuration error, not a CPU escape."
            ),
        }
    })?;
    cuda_usize_bytes_to_u64(free, "CUDA live free memory bytes")
}

pub(crate) fn cuda_transient_dispatch_available_budget_bytes(
    total_memory: u64,
    resident_bytes: u64,
    transient_pool_bytes: u64,
) -> u64 {
    let budget = u128::from(cuda_transient_dispatch_budget_bytes(total_memory));
    let used = u128::from(resident_bytes) + u128::from(transient_pool_bytes);
    if used >= budget {
        0
    } else {
        (budget - used) as u64
    }
}

pub(crate) fn cuda_transient_dispatch_live_available_budget_bytes(
    total_memory: u64,
    live_free_memory: u64,
    resident_bytes: u64,
    transient_pool_bytes: u64,
) -> u64 {
    let accounted = cuda_transient_dispatch_available_budget_bytes(
        total_memory,
        resident_bytes,
        transient_pool_bytes,
    );
    let live = cuda_transient_dispatch_budget_bytes(live_free_memory);
    accounted.min(live)
}

pub(crate) fn cuda_transient_dispatch_required_bytes(
    prepared: &CudaDispatchPlan,
    inputs: &[&[u8]],
) -> Result<u64, BackendError> {
    let mut required_bytes = 0u64;
    for binding in &prepared.bindings.bindings {
        if binding.role == BindingRole::Shared {
            continue;
        }
        let byte_len = match binding.input_index {
            Some(input_index) => inputs
                .get(input_index)
                .map(|input| input.len())
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA dispatch memory preflight expected input index {input_index} for `{}` but only {} input(s) were supplied.",
                        binding.name,
                        inputs.len()
                    ),
                })?,
            None => binding.static_byte_len.ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA dispatch memory preflight needs a static byte length for output `{}`; set BufferDecl::with_count or output_byte_range before launch.",
                    binding.name
                ),
            })?,
        };
        required_bytes = checked_dispatch_bytes_add(
            required_bytes,
            cuda_dispatch_allocation_bucket(byte_len, "CUDA dispatch buffer bytes")?,
            "CUDA dispatch buffer bytes",
        )?;
    }
    let param_bytes =
        launch_param_byte_len(&prepared.launch.param_words, "dispatch memory preflight")?;
    let param_allocation_bytes = if param_bytes == 0 {
        0
    } else {
        cuda_dispatch_allocation_bucket(param_bytes, "CUDA dispatch parameter bytes")?
    };
    checked_dispatch_bytes_add(
        required_bytes,
        param_allocation_bytes,
        "CUDA dispatch parameter bytes",
    )
}

/// Transient device bytes a mixed dispatch stages for its borrowed bindings.
///
/// Resident bindings already own their device memory and contribute nothing,
/// which is the whole point of leaving them resident.
pub(crate) fn cuda_mixed_dispatch_staging_bytes(
    prepared: &CudaDispatchPlan,
    bindings: &[CudaDispatchBinding<'_>],
) -> Result<u64, BackendError> {
    let mut required_bytes = 0u64;
    let mut next_binding = 0usize;
    for binding in &prepared.bindings.bindings {
        if binding.role == BindingRole::Shared {
            continue;
        }
        let source = next_dispatch_binding(
            bindings,
            &mut next_binding,
            "mixed dispatch memory preflight",
        )?;
        let CudaDispatchBinding::Borrowed(bytes) = source else {
            continue;
        };
        let byte_len = match binding.input_index {
            Some(_) => bytes.len(),
            None => binding.static_byte_len.ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA dispatch memory preflight needs a static byte length for borrowed output `{}`; set BufferDecl::with_count or output_byte_range before launch.",
                    binding.name
                ),
            })?,
        };
        required_bytes = checked_dispatch_bytes_add(
            required_bytes,
            cuda_dispatch_allocation_bucket(byte_len, "CUDA mixed dispatch staging bytes")?,
            "CUDA mixed dispatch staging bytes",
        )?;
    }
    Ok(required_bytes)
}

pub(crate) fn validate_cuda_transient_dispatch_budget(
    required_bytes: u64,
    budget_bytes: u64,
    context: &str,
) -> Result<(), BackendError> {
    if required_bytes > budget_bytes {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: {context} requires {required_bytes} transient CUDA device bytes but the live-device preflight budget is {budget_bytes} bytes. Reduce input/output size, shard the dispatch, use resident handles with explicit reuse, or raise the CUDA memory budget deliberately."
            ),
        });
    }
    Ok(())
}

pub(crate) fn checked_dispatch_bytes_add(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, BackendError> {
    left.checked_add(right)
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: {field} overflowed u64 during CUDA memory preflight. Shard the dispatch before CUDA allocation."
            ),
        })
}

pub(crate) fn cuda_dispatch_allocation_bucket(
    byte_len: usize,
    field: &str,
) -> Result<u64, BackendError> {
    let bucket = byte_len
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: {field} request of {byte_len} bytes cannot be rounded to the CUDA allocation bucket. Shard the dispatch before CUDA allocation."
            ),
        })?;
    cuda_usize_bytes_to_u64(bucket, field)
}

pub(crate) fn cuda_usize_bytes_to_u64(byte_len: usize, field: &str) -> Result<u64, BackendError> {
    u64::try_from(byte_len).map_err(|_| BackendError::InvalidProgram {
        fix: format!(
            "Fix: {field} value of {byte_len} bytes cannot fit u64 CUDA memory telemetry. Shard the dispatch or widen budget accounting."
        ),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smallvec::smallvec;
    use vyre_driver::{BackendError, Binding, BindingPlan, BindingRole, LaunchPlan};

    use super::*;
    use crate::backend::CudaDispatchPlan;

    fn plan(static_output_bytes: usize) -> CudaDispatchPlan {
        CudaDispatchPlan {
            bindings: BindingPlan {
                bindings: vec![
                    Binding {
                        name: Arc::from("input"),
                        binding: 0,
                        buffer_index: 0,
                        role: BindingRole::Input,
                        element_size: 1,
                        preferred_alignment: 1,
                        element_count: 8,
                        static_byte_len: Some(8),
                        input_index: Some(0),
                        output_index: None,
                    },
                    Binding {
                        name: Arc::from("output"),
                        binding: 1,
                        buffer_index: 1,
                        role: BindingRole::Output,
                        element_size: 1,
                        preferred_alignment: 1,
                        element_count: static_output_bytes as u32,
                        static_byte_len: Some(static_output_bytes),
                        input_index: None,
                        output_index: Some(0),
                    },
                ],
                input_indices: vec![0],
                output_indices: vec![1],
                shared_indices: Vec::new(),
            },
            output_binding_indices: smallvec![1],
            launch: LaunchPlan {
                element_count: 1,
                workgroup: [1, 1, 1],
                grid: [1, 1, 1],
                param_words: vec![0, 0],
                max_binding_alignment: 1,
            },
            cooperative: false,
            fixpoint_iterations: 1,
        }
    }

    #[test]
    fn transient_dispatch_memory_preflight_sums_buffers_and_params() {
        let input = [0u8; 8];
        let required = cuda_transient_dispatch_required_bytes(&plan(16), &[input.as_slice()])
            .expect("Fix: valid dispatch memory plan should sum");

        assert_eq!(required, 8 + 16 + 8);
    }

    #[test]
    fn transient_dispatch_memory_preflight_does_not_charge_empty_params() {
        let input = [0u8; 8];
        let mut plan = plan(16);
        plan.launch.param_words.clear();
        let required = cuda_transient_dispatch_required_bytes(&plan, &[input.as_slice()])
            .expect("Fix: valid zero-param dispatch memory plan should sum");

        assert_eq!(
            required,
            8 + 16,
            "Fix: CUDA memory preflight must not charge a rounded one-byte allocation for empty launch params."
        );
    }

    #[test]
    fn transient_dispatch_memory_preflight_counts_bucketed_allocation_pressure() {
        let input = [0u8; 9];
        let required = cuda_transient_dispatch_required_bytes(&plan(17), &[input.as_slice()])
            .expect("Fix: valid dispatch memory plan should sum bucketed allocation pressure");

        assert_eq!(required, 16 + 32 + 8);
    }

    #[test]
    fn transient_dispatch_memory_preflight_rejects_over_budget_before_allocation() {
        let error = validate_cuda_transient_dispatch_budget(1025, 1024, "CUDA test dispatch")
            .expect_err("over-budget dispatch must fail before CUDA allocation");

        match error {
            BackendError::InvalidProgram { fix } => {
                assert!(fix.contains("CUDA test dispatch requires 1025"));
                assert!(fix.contains("preflight budget is 1024"));
                assert!(fix.contains("Shard") || fix.contains("shard"));
            }
            other => panic!("expected InvalidProgram, got {other:?}"),
        }
    }

    #[test]
    fn transient_dispatch_budget_uses_conservative_live_vram_fraction() {
        assert_eq!(cuda_transient_dispatch_budget_bytes(1000), 900);
        assert_eq!(
            cuda_transient_dispatch_budget_bytes(u64::MAX),
            16_602_069_666_338_596_453,
            "Fix: CUDA transient budget must widen before multiplying so huge live-memory probes do not saturate before division."
        );
    }

    #[test]
    fn transient_dispatch_live_available_budget_caps_against_free_vram() {
        assert_eq!(
            cuda_transient_dispatch_live_available_budget_bytes(10_000, 1_000, 0, 0),
            900,
            "Fix: CUDA preflight must cap dispatch pressure against live free VRAM, not just total board memory."
        );
        assert_eq!(
            cuda_transient_dispatch_live_available_budget_bytes(10_000, 8_000, 2_000, 1_000),
            6_000,
            "Fix: CUDA preflight must still subtract resident and transient allocations from the total-device budget."
        );
        assert_eq!(
            cuda_transient_dispatch_live_available_budget_bytes(10_000, 0, 0, 0),
            0,
            "Fix: zero live free VRAM must produce zero preflight budget instead of allowing optimistic allocation."
        );
    }
}
