//! PTX codegen smoke tests  -  validate emitted PTX structure without GPU hardware.

#![cfg(feature = "device-tests")]

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
fn assert_ptx_emits_expr_insns(ops: &[(&str, Expr, &str)]) {
    for (name, expr, expected_insn) in ops {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), expr.clone())],
        );
        let secondary_text = program_to_ptx(&program, &default_config())
            .unwrap_or_else(|e| panic!("Fix: {name} must lower to PTX: {e}"));
        assert!(
            secondary_text.contains(expected_insn),
            "Fix: {name} must emit {expected_insn}, got:\n{secondary_text}"
        );
    }
}

fn assert_ptx_emits_active_mask_subgroup_insns(ops: &[(&str, Expr, &str)]) {
    for (name, expr, expected_insn) in ops {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), expr.clone())],
        );
        let secondary_text = program_to_ptx(&program, &default_config())
            .unwrap_or_else(|e| panic!("Fix: {name} must lower to PTX: {e}"));
        assert!(
            secondary_text.contains("activemask.b32") && secondary_text.contains(expected_insn),
            "Fix: {name} must emit active-mask guarded {expected_insn}."
        );
    }
}

fn shared_memory_smoke_program(workgroup_size: [u32; 3]) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::workgroup("scratch", 16, DataType::U32),
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
        ],
        workgroup_size,
        vec![
            Node::store("scratch", Expr::u32(0), Expr::u32(7)),
            Node::Barrier {
                ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
            },
            Node::store("out", Expr::u32(0), Expr::load("scratch", Expr::u32(0))),
        ],
    )
}

#[path = "contract_cases/ptx_codegen_smoke__base64_decode_ptx_compiles_with_ptxas.rs"]
mod ptx_codegen_smoke_base64_decode_ptx_compiles_with_ptxas;
#[path = "contract_cases/ptx_codegen_smoke__ptx_emits_bitwise_ops.rs"]
mod ptx_codegen_smoke_ptx_emits_bitwise_ops;
#[path = "contract_cases/ptx_codegen_smoke__ptx_emits_select.rs"]
mod ptx_codegen_smoke_ptx_emits_select;

/// A column walk over a 32-row tile staged in workgroup memory.
///
/// Lane `t` writes and reads element `t * 32`, so on 32 four-byte banks every
/// lane addresses the same bank: the classifier's 32-way case. The tile is
/// reached only by scalar loads and stores under one barrier, so the whole
/// binding is permutable and the mitigation the strategy selects is applied at
/// the shared address site.
fn column_walk_tile_program() -> Program {
    let column = || Expr::mul(Expr::gid_x(), Expr::u32(32));
    Program::wrapped(
        vec![
            BufferDecl::workgroup("tile", 1024, DataType::U32),
            BufferDecl::output("out", 0, DataType::U32).with_count(32),
        ],
        [32, 1, 1],
        vec![
            Node::store("tile", column(), Expr::gid_x()),
            Node::Barrier {
                ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
            },
            Node::store("out", Expr::gid_x(), Expr::load("tile", column())),
        ],
    )
}

/// The mitigation reaches the emitted kernel through the production route, not
/// only through the selector that ranks it.
///
/// One element of padding per 32-element row is the cheapest accepted candidate
/// for this geometry, so the tile is declared at 32 rows of 33 four-byte
/// elements and every access scales its row by the padded length. A rewrite
/// applied at one of the two sites would read back a different element than it
/// wrote, so both are counted.
#[test]
fn a_column_walk_over_workgroup_memory_is_padded_in_emitted_ptx() {
    let secondary_text = program_to_ptx(&column_walk_tile_program(), &default_config())
        .expect("Fix: a workgroup column walk must lower to PTX.");
    assert!(
        secondary_text.contains(".shared .align 4 .b8 shared_buf_")
            && secondary_text.contains("[4224];"),
        "Fix: a padded tile is declared at its grown extent, 32 rows of 33 \
         four-byte elements; got:\n{secondary_text}"
    );
    assert_eq!(
        secondary_text.matches(", 33;").count(),
        2,
        "Fix: both the store and the load scale their row by the padded row \
         length; got:\n{secondary_text}"
    );
}
