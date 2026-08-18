//! Live CUDA execution contracts for lane coverage and readback semantics.

mod harness;
use harness::{
    bytes_f32 as bytes_to_f32, bytes_u32, compiled_cuda_outputs_with_config, f32_bytes, i32_bytes,
    ordered_f32_bits, u16_bytes, u32_bytes,
};

use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};
fn acquire_cuda_backend() -> CudaBackend {
    CudaBackend::acquire().expect("Fix: CUDA backend must acquire on the GPU-required host.")
}

fn make_atomic_sum_program(count: u32, with_output_range: bool) -> Program {
    let mut sum_decl =
        BufferDecl::storage("sum", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1);
    if with_output_range {
        sum_decl = sum_decl.with_output_byte_range(0..4);
    }
    Program::wrapped(
        vec![
            sum_decl,
            BufferDecl::read("values", 1, DataType::U32).with_count(count),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(count)),
                vec![Node::let_bind(
                    "old_sum",
                    Expr::atomic_add("sum", Expr::u32(0), Expr::load("values", Expr::var("idx"))),
                )],
            ),
        ],
    )
}

#[path = "contract_cases/execution_contracts__cuda_dispatch_writes_every_output_lane_for_identity.rs"]
mod execution_contracts_cuda_dispatch_writes_every_output_lane_for_identity;
#[path = "contract_cases/execution_contracts__cuda_large_storage_atomic_sum_crosses_workgroup_boundary.rs"]
mod execution_contracts_cuda_large_storage_atomic_sum_crosses_workgroup_boundary;
