//! Subject: `fuzz(vyre): T10 - GPU dispatch`
//!
//! Arbitrary bytes → `Program::from_wire` → validate → wgpu dispatch with
//! zero-filled inputs (when sizes are known and bounded).
//!
//! Invariants:
//! 1. No panic on arbitrary wire bytes.
//! 2. `from_wire` errors carry a `Fix:` hint.
//! 3. GPU dispatch is attempted only when inputs fit under the byte cap.
//!
//! Run with: `./cargo_full fuzz build dispatch`

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use vyre::ir::Program;
use vyre::validate;
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::optimizer::optimize;

/// `vyre_foundation::serial::wire::MAX_PROGRAM_BYTES` - reject larger blobs early.
const MAX_WIRE_BYTES: usize = 64 * 1024 * 1024;

/// Cap synthesized dispatch inputs to avoid OOM on huge buffer declarations.
const MAX_DISPATCH_INPUT_BYTES: usize = 8 * 1024 * 1024;

static BACKEND: OnceLock<Option<WgpuBackend>> = OnceLock::new();

fn backend() -> Option<&'static WgpuBackend> {
    BACKEND.get_or_init(|| WgpuBackend::acquire().ok()).as_ref()
}

/// Zero-filled bytes for exactly the buffers a dispatch stages from the host.
///
/// `BufferDecl::consumes_host_input` is the single definition of that list, and
/// this returns it in binding order, so the vector is the dispatch input list
/// rather than something that has to be filtered again. Spelling the rule out
/// as `ReadOnly | ReadWrite | Uniform` was a copy that disagreed with it on a
/// `Shared`-kind buffer, a `Persistent`-kind buffer, an output and a pipeline
/// live-out, and every dispatch built that way is refused for arity before the
/// fuzzer reaches the backend.
fn zeroed_dispatch_inputs(program: &Program, max_total: usize) -> Option<Vec<Vec<u8>>> {
    let mut total = 0usize;
    let mut inputs = Vec::with_capacity(program.buffers().len());
    for buffer in program.buffers() {
        if !buffer.consumes_host_input() {
            continue;
        }
        let byte_len = usize::try_from(buffer.count())
            .ok()
            .and_then(|count| count.checked_mul(buffer.element().min_bytes()))?;
        if byte_len == 0 {
            return None;
        }
        total = total.checked_add(byte_len)?;
        if total > max_total {
            return None;
        }
        inputs.push(vec![0u8; byte_len]);
    }
    Some(inputs)
}

fuzz_target!(|data: &[u8]| {
    let data = if data.len() > MAX_WIRE_BYTES {
        &data[..MAX_WIRE_BYTES]
    } else {
        data
    };

    let program = match Program::from_wire(data) {
        Ok(program) => program,
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("Fix:"),
                "from_wire error missing Fix: hint: {msg}"
            );
            return;
        }
    };

    let validation_errors = validate(&program);
    if !validation_errors.is_empty() {
        return;
    }

    let required = vyre_foundation::program_caps::scan(&program);
    if let Some(backend) = backend() {
        if vyre_foundation::program_caps::check_backend_capabilities(
            backend.id(),
            &vyre_driver::validation::ProgramValidationCaps::from_backend(backend).support(),
            &required,
        )
        .is_err()
        {
            return;
        }
    }

    let Some(gpu_inputs) = zeroed_dispatch_inputs(&program, MAX_DISPATCH_INPUT_BYTES) else {
        return;
    };
    if gpu_inputs.is_empty() {
        return;
    }

    let Some(backend) = backend() else {
        return;
    };

    let Ok(lowered) = optimize(program) else {
        return;
    };
    let _ = backend.dispatch(&lowered, &gpu_inputs, &DispatchConfig::default());
});
