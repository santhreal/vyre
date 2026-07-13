#![allow(unused_imports)]
#![allow(dead_code)]

pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

/// SIGNED 16.16 fixed-point multiply, bits [16..48] of the SIGNED 64-bit product, IDENTICAL to the IR's
/// [`vyre_primitives::fixed_mul_16_16_expr`]. Operands are two's-complement i32 in a u32; a weighted-Jacobi
/// residual / sheaf coupling / gradient is routinely negative, so the multiply MUST be signed (the old
/// unsigned `((a as u64 * b as u64) >> 16)` silently corrupted negative operands, see BACKLOG
/// `FIXED-amg-fixed-path-unsigned-mul-negatives`). For non-negative operands it is bit-identical to the
/// unsigned form.
///
/// ONE PLACE: every `_via_reference_parity` oracle that needs a 16.16 multiply uses THIS, rather than
/// re-defining a per-file copy (7 copies previously drifted: 6 of them silently kept the unsigned form
/// after the kernel was corrected).
pub(crate) fn fixed_mul(a: u32, b: u32) -> u32 {
    ((i64::from(a as i32) * i64::from(b as i32)) >> 16) as i32 as u32
}

/// SIGNED integer division by a KNOWN-POSITIVE divisor (truncating toward zero), IDENTICAL to the IR's
/// [`vyre_primitives::fixed_sdiv_by_positive_expr`]. Mirrors the fixed weighted-Jacobi `delta` divide,
/// whose numerator is negative whenever the residual is negative.
pub(crate) fn fixed_sdiv_by_positive(numerator: u32, denominator: u32) -> u32 {
    ((numerator as i32) / (denominator as i32)) as u32
}

use vyre_foundation::ir::{BufferAccess, Program};
use vyre_reference::value::Value;
use vyre_self_substrate::optimizer::dispatcher::{DispatchError, OptimizerDispatcher};

/// The one canonical `OptimizerDispatcher` that actually EXECUTES a vyre Program (rather than
/// hand-writing a per-op CPU oracle like `optimizer::dispatcher::oracle::CpuOracleDispatcher`,
/// which only recognizes `persistent_bfs` / `exploded`). Backing the dispatch boundary with
/// `vyre_reference::reference_eval` lets every `*_via` production entry point be tested end to end
/// against its `_cpu` oracle without a GPU backend, the "reference dispatcher" the
/// `OptimizerDispatcher` trait doc anticipates.
///
/// ONE PLACE: every `_via` end-to-end parity test uses THIS dispatcher rather than re-deriving the
/// (subtle) buffer-bridging rule below.
///
/// This bridge models the REAL wgpu/cuda backend's dispatch-input contract EXACTLY, so a `_via`
/// consumer that passes here also runs correctly on hardware. The canonical mapping lives in
/// `vyre-driver`'s `role_for_buffer` / [`vyre_foundation::ir::BufferDecl::is_backend_allocated_output`]:
/// a buffer is BACKEND-ALLOCATED (the backend creates it, no dispatch input) ONLY when it is
/// `is_output` / `WriteOnly` / `pipeline_live_out&&ReadWrite`; EVERY other non-workgroup buffer 
/// `ReadOnly` (role `Input`), plain `ReadWrite` (role `InputOutput`, whose zero/initial contents the
/// caller supplies), and `Uniform`: CONSUMES one dispatch input, in buffer order. The real backend
/// validates this strictly (`inputs.len() == input_indices.len()`), so a consumer must pass one
/// `Vec<u8>` per input-consuming buffer in buffer order (zero-filled slots for plain-RW outputs,
/// exactly as `mori_zwanzig`/`kfac` do). `reference_eval` has the SAME requirement (an initial
/// `Value` for every non-backend-allocated non-workgroup buffer in binding order), so this bridge
/// forwards each input-consuming buffer's dispatch bytes straight through. NO zero-synthesis, which
/// would silently diverge from the backend when a plain-RW buffer precedes a `ReadOnly` input (e.g.
/// kfac's `blocks_out` RW at binding 0 before `blocks_in` RO at binding 1). The returned values are
/// the writable buffers in binding order, matching the dispatch contract ("declared outputs in
/// canonical order").
///
/// ONE PLACE: every `_via` end-to-end parity test uses THIS dispatcher rather than re-deriving the
/// (subtle) buffer-bridging rule (and it is the SAME rule the production backend uses).
pub(crate) struct ReferenceEvalDispatcher;

impl OptimizerDispatcher for ReferenceEvalDispatcher {
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
            // Backend-allocated outputs (is_output / WriteOnly / pipeline_live_out&&RW) are created by
            // the backend and consume NO dispatch input (skip them, mirroring `role_for_buffer`).
            if buffer.is_backend_allocated_output() {
                continue;
            }
            // Every remaining buffer is input-consuming per `role_for_buffer`: ReadOnly (Input),
            // plain ReadWrite (InputOutput (its zero/initial contents come from the caller), Uniform).
            // Take the next dispatch input in buffer order, exactly as the real backend indexes them.
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
        // Faithful to the backend's strict count validation: reject leftover inputs so an over-feeding
        // consumer (one input per buffer INCLUDING backend-allocated outputs) is caught here, not
        // silently on hardware.
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
