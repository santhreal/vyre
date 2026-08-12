//! PTX codegen smoke tests  -  validate emitted PTX structure without GPU hardware.

use vyre_driver::DispatchConfig;
use vyre_driver_cuda::codegen::{
    program_to_ptx, program_to_ptx_for_sm, program_to_ptx_for_sm_and_subgroup,
};
use vyre_foundation::ir::{BinOp, BufferDecl, DataType, Expr, Node, Program, UnOp};

fn default_config() -> DispatchConfig {
    DispatchConfig::default()
}

fn identity_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(4),
            BufferDecl::output("output", 1, DataType::U32).with_count(4),
        ],
        [64, 1, 1],
        vec![Node::store(
            "output",
            Expr::gid_x(),
            Expr::load("input", Expr::gid_x()),
        )],
    )
}

mod ptx_codegen_smoke_base64_decode_ptx_compiles_with_ptxas {

    include!("contract_cases/ptx_codegen_smoke__base64_decode_ptx_compiles_with_ptxas.rs");
}
mod ptx_codegen_smoke_ptx_emits_bitwise_ops {
    include!("contract_cases/ptx_codegen_smoke__ptx_emits_bitwise_ops.rs");
}
mod ptx_codegen_smoke_ptx_emits_select {
    include!("contract_cases/ptx_codegen_smoke__ptx_emits_select.rs");
}
