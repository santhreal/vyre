//! Reference-only semantic executor and reference target compilation dialect.

mod program_dispatch;

pub use program_dispatch::{target_profile, ReferenceSemanticExecutor};

pub use vyre_driver::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;
use vyre_reference::value::Value;

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
/// Pure evaluation helper backed by `vyre_reference::reference_eval`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuRefEvaluator;

impl CpuRefEvaluator {
    /// Execute a program on input byte buffers using reference semantics.
    pub fn evaluate(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        interpret(program, inputs, config)
    }

    /// Execute a program on input byte buffers using default reference dispatch configuration.
    pub fn evaluate_default(
        &self,
        program: &Program,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        self.evaluate(program, inputs, &DispatchConfig::default())
    }
}

fn interpret(
    program: &Program,
    inputs: &[&[u8]],
    config: &DispatchConfig,
) -> Result<Vec<Vec<u8>>, BackendError> {
    let expanded = strict_expanded(program, config)?;
    let program = expanded.as_ref().unwrap_or(program);
    let values = reference_values(program, inputs)?;
    let result = vyre_reference::reference_eval(program, &values);
    result
        .map(|outputs| outputs.iter().map(Value::to_bytes).collect())
        .map_err(|error| {
            BackendError::new(format!(
                "reference evaluation failed: {error}. Fix: validate the Program and input buffer ABI before evaluation."
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
            "reference evaluator cannot lower float mode `{}`: {error}. Fix: give the operation an exact f32 \
             expansion in vyre_foundation::fp_expansion, so the oracle evaluates the program a \
             strict device kernel executes.",
            config.float_lowering.cache_label()
        ))
    })
}

fn reference_values(program: &Program, inputs: &[&[u8]]) -> Result<Vec<Value>, BackendError> {
    // `vyre_reference::reference_input_values` is the interpreter's own input
    // ABI. This backend walked the buffers itself and selected them as
    // `access() != Workgroup && !is_backend_allocated_output()`, which admits a
    // `Shared` buffer, a `Persistent` buffer, and a non-read-write
    // `pipeline_live_out` that no backend stages from the host, so the oracle
    // asked for one value more than a device dispatch and every later buffer
    // read the one before it.
    vyre_reference::reference_input_values(program, inputs).map_err(|mismatch| {
        BackendError::new(format!(
            "reference input buffers do not match the program: {mismatch}. Fix: pass one buffer per \
             reference input in Program::buffers order; a synthesized zero buffer would answer an \
             ABI failure with fabricated data."
        ))
    })
}
