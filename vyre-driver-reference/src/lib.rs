//! Registry adapter for the reference parity backend and semantic executor.

mod materializer;
mod program_dispatch;

pub use program_dispatch::ReferenceSemanticExecutor;

use std::sync::Arc;

use vyre_driver::sealed;
use vyre_driver::{core_supported_ops, BackendError};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_foundation::ir::{BufferAccess, Program};
use vyre_reference::value::Value;

/// Stable backend id for the pure-Rust reference interpreter.
pub const CPU_REF_BACKEND_ID: &str = "cpu-ref";
/// Validated identity for the non-production reference target.
pub const CPU_REF_TARGET_ID: vyre_foundation::operation::TargetId =
    vyre_foundation::operation::TargetId::expect_valid(CPU_REF_BACKEND_ID);

/// Lane count the interpreter's subgroup simulator models.
///
/// [`vyre_reference::subgroup::SubgroupSimulator`] is built at this width, and
/// a ballot the oracle returns is only comparable to a device answer when both
/// report the same width.
pub const REFERENCE_SUBGROUP_WIDTH: u32 = 32;

/// Workgroup-scoped scratch the interpreter admits, in bytes.
///
/// The interpreter allocates scratch in host memory, so this figure exists to
/// keep the oracle from refusing a program a device accepts rather than to
/// describe a hardware bank. It sits above the largest per-workgroup scratch
/// any shipped target offers, which is 227 KiB of opt-in shared memory on the
/// widest CUDA part.
pub const REFERENCE_SHARED_SCRATCH_BYTES: u32 = 256 * 1024;

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

    /// Both modes are lowered here.
    ///
    /// The interpreter rounds every operation separately, so the contracted
    /// mode is already satisfied. The strict mode additionally replaces each
    /// approximable f32 operation with its exact expansion, which is the
    /// program a strict device kernel executes: the oracle has to run the same
    /// IR for a bit-identity comparison to mean anything.
    fn honors_float_lowering(&self, mode: vyre_foundation::fp_parity::FloatLoweringMode) -> bool {
        matches!(
            mode,
            vyre_foundation::fp_parity::FloatLoweringMode::Contracted
                | vyre_foundation::fp_parity::FloatLoweringMode::StrictIeee
        )
    }

    /// A cooperative request selects a launch, not a semantics. The interpreter
    /// satisfies a whole-grid fence either way, so both requests reach the same
    /// evaluation.
    fn dispatch_borrowed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        interpret(program, inputs, config)
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

    /// The interpreter evaluates subgroup expressions through
    /// [`vyre_reference::subgroup::SubgroupSimulator`], so a program that uses
    /// them reaches the oracle instead of being refused before compilation.
    fn supports_subgroup_ops(&self) -> bool {
        true
    }

    fn subgroup_size(&self) -> Option<u32> {
        Some(REFERENCE_SUBGROUP_WIDTH)
    }

    fn max_shared_memory_bytes(&self) -> u32 {
        REFERENCE_SHARED_SCRATCH_BYTES
    }

    /// The interpreter satisfies a whole-grid fence inside one dispatch.
    ///
    /// `vyre_reference` flattens every fence-carrying scope, partitions the body
    /// at each top-level fence, and runs the whole grid through one segment
    /// before the next over one shared memory. That is what a cooperative launch
    /// buys on a device, so the oracle has the capability and reports it.
    ///
    /// Reporting `false` is not conservative here. It sends a fenced program
    /// through the launch-boundary cut, which mints a retained carrier the
    /// segments hand to each other through device-resident storage. A one-shot
    /// host submission has no resident storage, so the later segment read a
    /// zeroed carrier and the oracle answered with the wrong bytes.
    fn supports_grid_sync(&self) -> bool {
        true
    }
}

/// Run one Program through the interpreter under a dispatch configuration.
///
/// The dispatch backend and the artifact materializer both reach the oracle
/// here, so a program submitted as an artifact and the same program dispatched
/// directly evaluate under one grid rule and one strict-mode expansion. Two
/// copies of this would let the artifact route and the direct route disagree
/// about the answer the oracle gives.
fn interpret(
    program: &Program,
    inputs: &[&[u8]],
    config: &DispatchConfig,
) -> Result<Vec<Vec<u8>>, BackendError> {
    let expanded = strict_expanded(program, config)?;
    let program = expanded.as_ref().unwrap_or(program);
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
    let result = match (config.coverage_grid(), config.dispatch_elements) {
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

/// The program with every approximable f32 operation expanded, or `None` when
/// the dispatch did not ask for strict IEEE lowering.
fn strict_expanded(
    program: &Program,
    config: &DispatchConfig,
) -> Result<Option<Program>, BackendError> {
    if !config.float_lowering.blocks_contraction() {
        return Ok(None);
    }
    vyre_foundation::fp_expansion::expand_strict_transcendentals(program).map_err(|error| {
        BackendError::new(format!(
            "cpu-ref cannot lower the strict IEEE float mode: {error}. Fix: give the operation an \
             exact f32 expansion in vyre_foundation::fp_expansion, so the oracle evaluates the \
             program a strict device kernel executes."
        ))
    })
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

vyre_driver::register_backend! {
    id: CPU_REF_BACKEND_ID,
    target_id: CPU_REF_TARGET_ID,
    payload_format: Some(program_dispatch::REFERENCE_TARGET_FORMAT),
    reference_oracle: true,
    factory: acquire_cpu_ref,
    target_compiler: Some(program_dispatch::target_compiler_factory),
    materializer: Some(materializer::materializer_factory),
    rank: 900,
}
