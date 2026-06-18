//! Integration test for the CUDA backend.


mod common;
#[path = "resident_dispatch_contracts/basic_resident_contracts.rs"]
mod basic_resident_contracts;
#[path = "resident_dispatch_contracts/sequence_readback_contracts.rs"]
mod sequence_readback_contracts;
#[path = "resident_dispatch_contracts/repeated_sequence_contracts.rs"]
mod repeated_sequence_contracts;
#[path = "resident_dispatch_contracts/optimizer_combined_contracts.rs"]
mod optimizer_combined_contracts;
#[path = "resident_dispatch_contracts/source_accounting_contracts.rs"]
mod source_accounting_contracts;

use common::{bytes_u32, resident_dispatch_source, u32_bytes};
use std::sync::Arc;

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_cuda::{CudaBackend, CudaBackendRegistration, CudaOptimizerDispatcher};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_self_substrate::optimizer::dispatcher::{
    OptimizerDispatcher, ResidentDispatchStep, ResidentReadRange,
};

fn cuda_resident_borrowed_fallback_active() -> bool {
    if std::env::var_os("VYRE_CUDA_RESIDENT_BORROWED_FALLBACK").is_none() {
        return false;
    }
    #[cfg(debug_assertions)]
    {
        return true;
    }
    #[cfg(not(debug_assertions))]
    {
        std::env::var("VYRE_CUDA_ALLOW_BORROWED_FALLBACK")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
            .unwrap_or(false)
    }
}

fn expected_readback_bytes(native_resident: u64, fallback_resident: u64) -> u64 {
    if cuda_resident_borrowed_fallback_active() {
        fallback_resident
    } else {
        native_resident
    }
}
