//! Live CUDA execution contracts for lane coverage and readback semantics.

mod common;
use common::{
    bytes_f32 as bytes_to_f32, bytes_u32, compiled_cuda_outputs_with_config, f32_bytes, i32_bytes,
    ordered_f32_bits, u16_bytes, u32_bytes,
};

use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

mod execution_contracts_cuda_dispatch_writes_every_output_lane_for_identity {

    include!("contract_cases/execution_contracts__cuda_dispatch_writes_every_output_lane_for_identity.rs");
}
mod execution_contracts_cuda_large_storage_atomic_sum_crosses_workgroup_boundary {
    include!("contract_cases/execution_contracts__cuda_large_storage_atomic_sum_crosses_workgroup_boundary.rs");
}
