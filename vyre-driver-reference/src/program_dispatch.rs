//! The reference interpreter behind the `ProgramDispatcher` boundary.
//!
//! `CpuRefBackend` exposes `vyre-reference` to the driver registry. Parity
//! suites need the same interpreter behind the narrower
//! [`ProgramDispatcher`] surface that every `_via` consumer takes, and this
//! crate is where a host execution route is allowed to live.

use vyre_foundation::ir::{BufferAccess, Program};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};
use vyre_reference::value::Value;

/// The one `ProgramDispatcher` that actually executes a vyre `Program`.
///
/// Backing the dispatch boundary with `vyre_reference::reference_eval` lets
/// every `*_via` production entry point be tested end to end against its `_cpu`
/// oracle without a GPU backend. Hand-written per-op CPU oracles recognize only
/// the programs they were written for; this one runs any program.
///
/// The bridge models the real backend's dispatch-input contract exactly, so a
/// `_via` consumer that passes here also runs correctly on hardware. The
/// canonical mapping is `vyre-driver`'s `role_for_buffer` /
/// [`vyre_foundation::ir::BufferDecl::is_backend_allocated_output`]: a buffer is
/// backend-allocated (the backend creates it, no dispatch input) only when it is
/// `is_output` / `WriteOnly` / `pipeline_live_out && ReadWrite`. Every other
/// non-workgroup buffer - `ReadOnly` (role `Input`), plain `ReadWrite` (role
/// `InputOutput`, whose zero or initial contents the caller supplies), and
/// `Uniform` - consumes one dispatch input, in buffer order. The real backend
/// validates this strictly (`inputs.len() == input_indices.len()`), so a
/// consumer must pass one `Vec<u8>` per input-consuming buffer in buffer order,
/// zero-filled for plain-`ReadWrite` outputs. `reference_eval` has the same
/// requirement, so this bridge forwards each input-consuming buffer's dispatch
/// bytes straight through. No zero-synthesis, which would silently diverge from
/// the backend when a plain-`ReadWrite` buffer precedes a `ReadOnly` input. The
/// returned values are the writable buffers in binding order, matching the
/// dispatch contract.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceEvalDispatcher;

impl ProgramDispatcher for ReferenceEvalDispatcher {
    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut values: Vec<Value> = Vec::new();
        let mut next_input = inputs.iter();
        for buffer in program.buffers() {
            if buffer.access() == BufferAccess::Workgroup {
                continue;
            }
            // Backend-allocated outputs (is_output / WriteOnly / pipeline_live_out&&RW) are created
            // by the backend and consume NO dispatch input, mirroring `role_for_buffer`.
            if buffer.is_backend_allocated_output() {
                continue;
            }
            // Every remaining buffer is input-consuming per `role_for_buffer`: ReadOnly (Input),
            // plain ReadWrite (InputOutput, whose zero/initial contents come from the caller),
            // Uniform. Take the next dispatch input in buffer order, as the real backend does.
            let bytes = next_input.next().ok_or_else(|| {
                DispatchError::BadInputs(format!(
                    "ReferenceEvalDispatcher: program declares more input-consuming buffers than the \
                     {} dispatch inputs provided (at buffer `{}`). The backend requires one input per \
                     {{ReadOnly, plain-ReadWrite, Uniform}} buffer in buffer order; pass a zero-filled \
                     slot for each plain-ReadWrite output.",
                    inputs.len(),
                    buffer.name()
                ))
            })?;
            values.push(Value::from(bytes.clone()));
        }
        // Faithful to the backend's strict count validation: reject leftover inputs so an
        // over-feeding consumer is caught here, not silently on hardware.
        if next_input.next().is_some() {
            return Err(DispatchError::BadInputs(format!(
                "ReferenceEvalDispatcher: {} dispatch inputs provided but the program has fewer \
                 input-consuming buffers. The backend requires exactly one input per {{ReadOnly, \
                 plain-ReadWrite, Uniform}} buffer; do not pass slots for backend-allocated outputs.",
                inputs.len()
            )));
        }
        let outputs = vyre_reference::reference_eval(program, &values).map_err(|err| {
            DispatchError::BackendError(format!(
                "ReferenceEvalDispatcher: reference_eval failed. {err}"
            ))
        })?;
        Ok(outputs.iter().map(Value::to_bytes).collect())
    }
}
