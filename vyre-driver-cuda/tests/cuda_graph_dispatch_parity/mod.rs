//! cudaGraph dispatch parity and latency tests.
//!
//! Records a real Program kernel launch into a CUDA graph via
//! `CudaBackend::record_cuda_graph`, then replays it many times via
//! `dispatch_via_cuda_graph` and asserts:
//!
//! 1. **Byte-identity parity**  -  every replay's outputs match the same
//!    Program dispatched via the regular `CudaBackend::dispatch` path.
//! 2. **Latency ceiling**  -  per-replay wall-clock remains under the hot
//!    dispatch budget for latency-bound kernels.
//! 3. **Shape validation**  -  passing inputs of the wrong byte length
//!    returns `BackendError::InvalidProgram` with a structured fix string.

#[path = "../harness/mod.rs"]
mod harness;
mod latency_cache_contracts;
mod replay_parity_contracts;
mod telemetry_shape_contracts;

use harness::{bool_bytes, bytes_u32, u32_bytes};
use std::sync::Arc;
use std::time::Instant;

use vyre_driver::{BackendError, DispatchConfig};
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Simple program: out[i] = in[i] + 1, 8 threads. Small enough that the
/// dispatch overhead dominates kernel time, so the cudaGraph speedup is
/// the headline number.
fn add_one_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(8),
            BufferDecl::output("out", 1, DataType::U32).with_count(8),
        ],
        [128, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::add(Expr::load("input", Expr::gid_x()), Expr::u32(1)),
        )],
    )
}

fn bool_not_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::Bool).with_count(8),
            BufferDecl::output("out", 1, DataType::Bool).with_count(8),
        ],
        [128, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::not(Expr::load("input", Expr::gid_x())),
        )],
    )
}
