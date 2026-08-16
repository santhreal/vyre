//! Contract tests for IR first-class Tile nodes and Tile wire serialization / validation.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Ident, Layout, Node, Program, Residency,
    SubgroupReduceOp, Tile, UnOp,
};
use vyre_foundation::serial::wire::decode::from_wire;
use vyre_foundation::serial::wire::encode::to_wire;
use vyre_foundation::validate::{
    validate_with_options, BackendCapabilities, ValidationOptions,
};

fn sample_tile_program() -> Program {
    let tile_a = Tile::new(DataType::F32, vec![16, 16], Layout::RowMajor, Residency::Register);
    let tile_b = Tile::new(DataType::F32, vec![16, 16], Layout::ColumnMajor, Residency::Subgroup);
    let tile_acc = Tile::new(DataType::F32, vec![16, 16], Layout::RowMajor, Residency::Register);

    Program::wrapped(
        vec![
            BufferDecl::storage("in_a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(256),
            BufferDecl::storage("in_b", 1, BufferAccess::ReadOnly, DataType::F32).with_count(256),
            BufferDecl::output("out", 2, DataType::F32).with_count(256),
        ],
        [32, 1, 1],
        vec![
            Node::tile_decl("acc", tile_acc),
            Node::tile_load("t_a", tile_a, "in_a", vec![Expr::u32(0), Expr::u32(0)], Layout::RowMajor),
            Node::tile_load("t_b", tile_b, "in_b", vec![Expr::u32(0), Expr::u32(0)], Layout::ColumnMajor),
            Node::tile_matmul("acc", "t_a", "t_b"),
            Node::tile_reduce("max_val", "acc", SubgroupReduceOp::Max, 1),
            Node::tile_elementwise(
                "acc_norm",
                vec![Ident::from("acc"), Ident::from("max_val")],
                vec![Node::let_bind(
                    "acc_norm",
                    Expr::UnOp {
                        op: UnOp::Exp,
                        operand: Box::new(Expr::sub(Expr::var("acc"), Expr::var("max_val"))),
                    },
                )],
            ),
            Node::tile_store("out", vec![Expr::u32(0), Expr::u32(0)], "acc_norm"),
        ],
    )
}

#[test]
fn tile_program_wire_roundtrip() {
    let prog = sample_tile_program();
    let wire_bytes = to_wire(&prog).expect("to_wire failed");
    let decoded = from_wire(&wire_bytes).expect("from_wire failed");

    assert_eq!(prog.buffers().len(), decoded.buffers().len());
    assert_eq!(prog.entry().len(), decoded.entry().len());

    let re_encoded = to_wire(&decoded).expect("re-encode failed");
    assert_eq!(wire_bytes, re_encoded);
}

#[test]
fn tile_program_validates_with_tensor_cores() {
    let prog = sample_tile_program();
    let caps = BackendCapabilities {
        supports_tensor_cores: true,
        max_shared_memory_bytes: 65536,
        regs_per_thread_max: 512,
        subgroup_size: 32,
        ..BackendCapabilities::default()
    };
    let opts = ValidationOptions::default().with_backend_capabilities(caps);
    let res = validate_with_options(&prog, opts);
    assert!(res.is_ok(), "Validation errors: {:#?}", res.errors);
}

#[test]
fn tile_program_rejects_missing_tensor_cores() {
    let prog = sample_tile_program();
    let caps = BackendCapabilities {
        supports_tensor_cores: false,
        ..BackendCapabilities::default()
    };
    let opts = ValidationOptions::default().with_backend_capabilities(caps);
    let res = validate_with_options(&prog, opts);
    assert!(!res.is_ok());
}

#[test]
fn tile_program_shared_memory_overflow_rejected() {
    let tile_huge = Tile::new(
        DataType::F32,
        vec![128, 128],
        Layout::RowMajor,
        Residency::Workgroup,
    );
    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in_a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(16384),
            BufferDecl::output("out", 1, DataType::F32).with_count(16384),
        ],
        [32, 1, 1],
        vec![Node::tile_decl("huge_tile", tile_huge)],
    );

    let caps = BackendCapabilities {
        supports_tensor_cores: true,
        max_shared_memory_bytes: 4096, // 128*128*4 = 65536 bytes > 4096 bytes
        regs_per_thread_max: 255,
        subgroup_size: 32,
        ..BackendCapabilities::default()
    };
    let opts = ValidationOptions::default().with_backend_capabilities(caps);
    let res = validate_with_options(&prog, opts);
    assert!(!res.is_ok());
}

#[test]
fn tile_program_rejects_subgroup_incompatibility() {
    let tile_unaligned = Tile::new(
        DataType::F32,
        vec![15], // 15 elements is not a multiple of subgroup size 32
        Layout::RowMajor,
        Residency::Subgroup,
    );
    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in_a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(100),
            BufferDecl::output("out", 1, DataType::F32).with_count(100),
        ],
        [32, 1, 1],
        vec![Node::tile_decl("unaligned_tile", tile_unaligned)],
    );

    let caps = BackendCapabilities {
        supports_tensor_cores: true,
        max_shared_memory_bytes: 65536,
        regs_per_thread_max: 255,
        subgroup_size: 32,
        ..BackendCapabilities::default()
    };
    let opts = ValidationOptions::default().with_backend_capabilities(caps);
    let res = validate_with_options(&prog, opts);
    assert!(!res.is_ok());
}

#[test]
fn tile_program_rejects_register_overflow() {
    let tile_huge_reg = Tile::new(
        DataType::F32,
        vec![64, 64], // 4096 floats = 4096 words > 255 regs
        Layout::RowMajor,
        Residency::Register,
    );
    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in_a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4096),
            BufferDecl::output("out", 1, DataType::F32).with_count(4096),
        ],
        [32, 1, 1],
        vec![Node::tile_decl("huge_reg_tile", tile_huge_reg)],
    );

    let caps = BackendCapabilities {
        supports_tensor_cores: true,
        max_shared_memory_bytes: 65536,
        regs_per_thread_max: 255,
        subgroup_size: 32,
        ..BackendCapabilities::default()
    };
    let opts = ValidationOptions::default().with_backend_capabilities(caps);
    let res = validate_with_options(&prog, opts);
    assert!(!res.is_ok());
}

#[test]
fn tile_program_rejects_write_only_load_and_read_only_store() {
    let tile = Tile::new(DataType::F32, vec![4, 4], Layout::RowMajor, Residency::Register);
    let prog_bad_load = Program::wrapped(
        vec![
            BufferDecl::storage("wo_buf", 0, BufferAccess::WriteOnly, DataType::F32).with_count(16),
            BufferDecl::output("out", 1, DataType::F32).with_count(16),
        ],
        [32, 1, 1],
        vec![Node::tile_load("t", tile.clone(), "wo_buf", vec![Expr::u32(0), Expr::u32(0)], Layout::RowMajor)],
    );
    let res_load = validate_with_options(&prog_bad_load, ValidationOptions::default());
    assert!(!res_load.is_ok());

    let prog_bad_store = Program::wrapped(
        vec![
            BufferDecl::storage("ro_buf", 0, BufferAccess::ReadOnly, DataType::F32).with_count(16),
            BufferDecl::output("out", 1, DataType::F32).with_count(16),
        ],
        [32, 1, 1],
        vec![
            Node::tile_decl("t", tile),
            Node::tile_store("ro_buf", vec![Expr::u32(0), Expr::u32(0)], "t"),
        ],
    );
    let res_store = validate_with_options(&prog_bad_store, ValidationOptions::default());
    assert!(!res_store.is_ok());
}

#[test]
fn tile_encoder_rejects_over_limit_extents() {
    use vyre_foundation::serial::wire::MAX_TENSOR_RANK;
    let huge_extents = vec![1u32; MAX_TENSOR_RANK + 1];
    let tile = Tile::new(DataType::F32, huge_extents, Layout::RowMajor, Residency::Register);
    let prog = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::F32).with_count(1)],
        [1, 1, 1],
        vec![Node::tile_decl("huge", tile)],
    );
    let err = to_wire(&prog).expect_err("should reject over-limit extents");
    assert!(err.contains("Fix:"));
}
