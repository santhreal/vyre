//! Subject: `fuzz(vyre): T10 — GPU dispatch`
//!
//! Arbitrary bytes → `Program::from_wire` → validate → wgpu dispatch with
//! zero-filled inputs (when sizes are known and bounded).
//!
//! Invariants:
//! 1. No panic on arbitrary wire bytes.
//! 2. `from_wire` errors carry a `Fix:` hint.
//! 3. GPU dispatch is attempted only when inputs fit under the byte cap.
//!
//! Run with: `cargo fuzz build dispatch`

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use vyre::ir::{BufferAccess, Program};
use vyre::{validate, DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::optimizer::pre_lowering::optimize;

/// `vyre_foundation::serial::wire::MAX_PROGRAM_BYTES` — reject larger blobs early.
const MAX_WIRE_BYTES: usize = 64 * 1024 * 1024;

/// Cap synthesized dispatch inputs to avoid OOM on huge buffer declarations.
const MAX_DISPATCH_INPUT_BYTES: usize = 8 * 1024 * 1024;

static BACKEND: OnceLock<Option<WgpuBackend>> = OnceLock::new();

fn backend() -> Option<&'static WgpuBackend> {
    BACKEND
        .get_or_init(|| WgpuBackend::acquire().ok())
        .as_ref()
}

fn zeroed_dispatch_inputs(program: &Program, max_total: usize) -> Option<Vec<Vec<u8>>> {
    let mut total = 0usize;
    let mut inputs = vec![Vec::new(); program.buffers().len()];
    for (idx, buffer) in program.buffers().iter().enumerate() {
        match buffer.access() {
            BufferAccess::ReadOnly | BufferAccess::ReadWrite | BufferAccess::Uniform => {
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
                inputs[idx] = vec![0u8; byte_len];
            }
            BufferAccess::Workgroup => {}
            _ => {}
        }
    }
    Some(inputs)
}

fn gpu_dispatch_inputs(program: &Program, all_inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    program
        .buffers()
        .iter()
        .enumerate()
        .filter_map(|(buffer_idx, buffer)| {
            matches!(
                buffer.access(),
                BufferAccess::ReadOnly | BufferAccess::ReadWrite | BufferAccess::Uniform
            )
            .then(|| all_inputs.get(buffer_idx).cloned())
        })
        .flatten()
        .collect()
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
        if let Err(missing) = vyre_foundation::program_caps::check_backend_capabilities(
            backend.id(),
            backend.supports_subgroup_ops(),
            backend.supports_f16(),
            backend.supports_bf16(),
            backend.supports_indirect_dispatch(),
            true,
            backend.supports_distributed_collectives(),
            backend.max_workgroup_size(),
            &required,
        ) {
            let _ = missing;
            return;
        }
    }

    let Some(all_inputs) = zeroed_dispatch_inputs(&program, MAX_DISPATCH_INPUT_BYTES) else {
        return;
    };
    let gpu_inputs = gpu_dispatch_inputs(&program, &all_inputs);
    if gpu_inputs.is_empty() {
        return;
    }

    let Some(backend) = backend() else {
        return;
    };

    let lowered = optimize(program);
    let _ = backend.dispatch(&lowered, &gpu_inputs, &DispatchConfig::default());
});
