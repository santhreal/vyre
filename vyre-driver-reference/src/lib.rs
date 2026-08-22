//! Registry adapter that exposes `vyre-reference` as a `VyreBackend`, and the
//! `ProgramDispatcher` bridge parity suites dispatch through.

mod program_dispatch;

pub use program_dispatch::ReferenceEvalDispatcher;

use std::sync::Arc;

use vyre_driver::sealed;
use vyre_driver::{
    core_supported_ops, BackendCapability, BackendError, BackendPrecedence, BackendRegistration,
};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_foundation::ir::{BufferAccess, Program};
use vyre_reference::value::Value;

/// Stable backend id for the pure-Rust reference interpreter.
pub const CPU_REF_BACKEND_ID: &str = "cpu-ref";
/// Validated identity for the non-production reference target.
pub const CPU_REF_TARGET_ID: vyre_foundation::operation::TargetId =
    vyre_foundation::operation::TargetId::expect_valid(CPU_REF_BACKEND_ID);

/// Dispatch backend backed by `vyre_reference::reference_eval`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuRefBackend;

impl sealed::Sealed for CpuRefBackend {}

impl VyreBackend for CpuRefBackend {
    fn id(&self) -> &'static str {
        CPU_REF_BACKEND_ID
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn dispatch_borrowed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        if config.cooperative && !self.supports_grid_sync() {
            return Err(BackendError::UnsupportedFeature {
                name: "cpu-ref cooperative grid dispatch".to_string(),
                backend: CPU_REF_BACKEND_ID.to_string(),
            });
        }
        let values = reference_values(program, inputs)?;
        // The interpreter infers its grid from buffer SHAPES, which cannot express
        // the per-invocation count of a byte-scan program (the haystack is packed
        // 4 bytes/u32 and the scan length is a runtime value). When the caller
        // declares the true element-grid coverage via `dispatch_elements`, pass it
        // as the interpreter's dispatch floor so high positions are covered exactly
        // as the real GPU dispatch would, otherwise the tail is silently skipped
        // (the Law-10 under-coverage this backend used to exhibit). `None` (every
        // megakernel, whose `grid_override` is a work-queue length, not an element
        // count) keeps buffer-shape inference so its grid is never over-run.
        // An explicit dispatch grid fully specifies the workgroup coverage (its
        // N-D shape, e.g. one query per `grid.y` block for batched persistent-BFS),
        // so it wins over the 1-D `dispatch_elements` floor; the shape-inference
        // path only applies when neither is set. See `DispatchConfig::dispatch_grid`.
        let result = match (config.dispatch_grid, config.dispatch_elements) {
            (Some(grid), _) => vyre_reference::reference_eval_with_grid(program, &values, grid),
            (None, Some(elements)) => {
                vyre_reference::reference_eval_with_dispatch(program, &values, elements)
            }
            (None, None) => vyre_reference::reference_eval(program, &values),
        };
        result
            .map(|outputs| outputs.iter().map(Value::to_bytes).collect())
            .map_err(|error| {
                BackendError::new(format!(
                    "cpu-ref reference dispatch failed: {error}. Fix: validate the Program and input buffer ABI before dispatch."
                ))
            })
    }

    fn supported_ops(&self) -> &std::collections::HashSet<vyre_foundation::ir::OpId> {
        core_supported_ops()
    }

    fn max_workgroup_size(&self) -> [u32; 3] {
        [1024, 1, 1]
    }

    fn max_compute_workgroups_per_dimension(&self) -> u32 {
        u32::MAX
    }
}

fn reference_values(program: &Program, inputs: &[&[u8]]) -> Result<Vec<Value>, BackendError> {
    // `is_reference_input` is the interpreter's own ABI predicate, and its
    // complement `is_backend_allocated_output` is the SINGLE cross-backend
    // contract in vyre-foundation. Do NOT re-inline either (drift would make
    // this backend disagree with the interpreter on outputs). A backend-allocated
    // output is allocated by the callee, so it consumes neither a caller input
    // nor a `Value`: handing the interpreter a zeroed stand-in for one is the
    // legacy shape a device artifact rejects.
    let mut next_input = 0usize;
    let mut values = Vec::new();
    for buffer in program.buffers() {
        if buffer.access() == BufferAccess::Workgroup {
            continue;
        }
        if buffer.is_backend_allocated_output() {
            continue;
        }
        let input = inputs.get(next_input).ok_or_else(|| {
            BackendError::new(format!(
                "cpu-ref is missing an input buffer for `{}`. Fix: pass one buffer per reference input in Program::buffers order; a synthesized zero buffer would answer an ABI failure with fabricated data.",
                buffer.name()
            ))
        })?;
        next_input += 1;
        values.push(Value::Bytes(Arc::from(*input)));
    }
    if next_input != inputs.len() {
        return Err(BackendError::new(format!(
            "cpu-ref received {} extra input buffer(s). Fix: pass inputs in Program::buffers order without trailing buffers.",
            inputs.len() - next_input
        )));
    }
    Ok(values)
}

fn acquire_cpu_ref() -> Result<Box<dyn VyreBackend>, BackendError> {
    Ok(Box::new(CpuRefBackend))
}

/// Backend id this crate submits into the backend registry on this target.
///
/// WHY: the registration below lives in this crate's object file, and a linker
/// keeps that object only when a symbol inside it is referenced. Naming the
/// crate with `use vyre_driver_reference as _;` references nothing, and reading
/// [`CPU_REF_BACKEND_ID`] is a `const` that inlines at the use site, so neither
/// keeps the registration. Calling this function does, which is why the backend
/// registry owner calls it instead of importing the crate for effect.
#[must_use]
pub fn registered_backend_id() -> Option<&'static str> {
    Some(CPU_REF_BACKEND_ID)
}

inventory::submit! {
    BackendRegistration {
        id: CPU_REF_BACKEND_ID,
        target_id: CPU_REF_TARGET_ID,
        payload_format: None,
        reference_oracle: true,
        factory: acquire_cpu_ref,
        supported_ops: core_supported_ops,
        semantic_operations: vyre_driver::dialect_only_supported_ops,
        target_compiler: None,
        materializer: None,
    }
}

inventory::submit! {
    BackendCapability {
        id: CPU_REF_BACKEND_ID,
        dispatches: true,
    }
}

inventory::submit! {
    BackendPrecedence {
        id: CPU_REF_BACKEND_ID,
        rank: 900,
    }
}
