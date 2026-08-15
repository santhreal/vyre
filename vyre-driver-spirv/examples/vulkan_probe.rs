//! Minimal Vulkan probe: acquire the SPIR-V backend, lower one program, dispatch
//! it, and print what the device returned.
//!
//! The program is the one `tests/dispatch.rs` asserts against the CPU reference,
//! so a probe that succeeds and a test that fails are talking about the same
//! dispatch.

#![allow(unsafe_code)]

use vyre_driver::{DispatchConfig, VyreBackend};

#[path = "../tests/support/elementwise.rs"]
mod elementwise;
use elementwise::{bytes_to_u32_values, elementwise_add_program, u32_values_to_bytes};

fn main() {
    println!("Probing Vulkan dispatch...");
    let backend = vyre_driver_spirv::SpirvBackendRegistration::acquire()
        .expect("Fix: Failed to acquire backend");

    println!("Building program...");
    let program = elementwise_add_program(4);
    let a = u32_values_to_bytes(&[1, 2, 3, 4]);
    let b = u32_values_to_bytes(&[10, 20, 30, 40]);

    println!("Lowering to SPIR-V...");
    let spv = vyre_driver_spirv::SpirvBackend::program_to_spv(&program)
        .expect("Fix: SPIR-V lowering failed");
    println!("SPIR-V: {} words", spv.len());

    println!("Dispatching...");
    match backend.dispatch(&program, &[a, b], &DispatchConfig::default()) {
        Ok(outputs) => {
            println!("Dispatch succeeded! {} output buffers", outputs.len());
            for (index, output) in outputs.iter().enumerate() {
                println!("  output[{index}]: {:?}", bytes_to_u32_values(output));
            }
        }
        Err(error) => println!("Dispatch failed: {error}"),
    }
}
