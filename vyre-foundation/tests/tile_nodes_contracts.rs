//! Contract tests for IR first-class Tile nodes and Tile wire serialization / validation.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Ident, Layout, Node, Program, Residency,
    SubgroupReduceOp, Tile, UnOp,
};
use vyre_foundation::serial::wire::decode::from_wire;
use vyre_foundation::serial::wire::encode::to_wire;
use vyre_foundation::validate::{validate_with_options, BackendCapabilities, ValidationOptions};

fn sample_tile_program() -> Program {
    let tile_a = Tile::new(
        DataType::F32,
        vec![16, 16],
        Layout::RowMajor,
        Residency::Register,
    );
    let tile_b = Tile::new(
        DataType::F32,
        vec![16, 16],
        Layout::ColumnMajor,
        Residency::Subgroup,
    );
    let tile_acc = Tile::new(
        DataType::F32,
        vec![16, 16],
        Layout::RowMajor,
        Residency::Register,
    );

    Program::wrapped(
        vec![
            BufferDecl::storage("in_a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(256),
            BufferDecl::storage("in_b", 1, BufferAccess::ReadOnly, DataType::F32).with_count(256),
            BufferDecl::output("out", 2, DataType::F32).with_count(256),
        ],
        [32, 1, 1],
        vec![
            Node::tile_decl("acc", tile_acc),
            Node::tile_load(
                "t_a",
                tile_a,
                "in_a",
                vec![Expr::u32(0), Expr::u32(0)],
                Layout::RowMajor,
            ),
            Node::tile_load(
                "t_b",
                tile_b,
                "in_b",
                vec![Expr::u32(0), Expr::u32(0)],
                Layout::ColumnMajor,
            ),
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
    let tile = Tile::new(
        DataType::F32,
        vec![4, 4],
        Layout::RowMajor,
        Residency::Register,
    );
    let prog_bad_load = Program::wrapped(
        vec![
            BufferDecl::storage("wo_buf", 0, BufferAccess::WriteOnly, DataType::F32).with_count(16),
            BufferDecl::output("out", 1, DataType::F32).with_count(16),
        ],
        [32, 1, 1],
        vec![Node::tile_load(
            "t",
            tile.clone(),
            "wo_buf",
            vec![Expr::u32(0), Expr::u32(0)],
            Layout::RowMajor,
        )],
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
    let tile = Tile::new(
        DataType::F32,
        huge_extents,
        Layout::RowMajor,
        Residency::Register,
    );
    let prog = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::F32).with_count(1)],
        [1, 1, 1],
        vec![Node::tile_decl("huge", tile)],
    );
    let err = to_wire(&prog).expect_err("should reject over-limit extents");
    assert!(err.contains("Fix:"));
}
#[test]
fn tile_node_variants_report_tile_operands_as_uses_and_survive_dce() {
    use vyre_foundation::ir::{node_variant_name, NODE_VARIANT_NAMES};
    use vyre_foundation::optimizer::fact_cache::FactCache;
    use vyre_foundation::optimizer::passes::fusion_cse::dce::dce;
    let tile_2x2 = Tile::new(
        DataType::F32,
        vec![2, 2],
        Layout::RowMajor,
        Residency::Register,
    );

    let tile_nodes: Vec<(Node, Vec<Ident>)> = vec![
        (Node::tile_decl("decl_t", tile_2x2.clone()), vec![]),
        (
            Node::tile_load(
                "load_t",
                tile_2x2.clone(),
                "buf",
                vec![Expr::u32(0)],
                Layout::RowMajor,
            ),
            vec![],
        ),
        (
            Node::tile_store("buf", vec![Expr::u32(0)], "store_t"),
            vec![Ident::from("store_t")],
        ),
        (
            Node::tile_matmul("mat_acc", "mat_a", "mat_b"),
            vec![
                Ident::from("mat_acc"),
                Ident::from("mat_a"),
                Ident::from("mat_b"),
            ],
        ),
        (
            Node::tile_reduce("red_out", "red_in", SubgroupReduceOp::Max, 1),
            vec![Ident::from("red_in")],
        ),
        (
            Node::tile_elementwise(
                "elem_out",
                vec![Ident::from("elem_in_a"), Ident::from("elem_in_b")],
                vec![Node::let_bind(
                    "elem_out",
                    Expr::add(Expr::var("elem_in_a"), Expr::var("elem_in_b")),
                )],
            ),
            vec![Ident::from("elem_in_a"), Ident::from("elem_in_b")],
        ),
    ];

    let tile_variant_names: Vec<&str> = NODE_VARIANT_NAMES
        .iter()
        .copied()
        .filter(|name| name.starts_with("Tile"))
        .collect();
    assert_eq!(
        tile_variant_names.len(),
        tile_nodes.len(),
        "Fix: every tile node variant in NODE_VARIANT_NAMES must be represented in tile_nodes test suite"
    );

    for (node, expected_uses) in &tile_nodes {
        let variant_name = node_variant_name(node);
        assert!(
            tile_variant_names.contains(&variant_name),
            "node variant `{variant_name}` must be in tile_variant_names"
        );
        let prog = Program::wrapped(
            vec![
                BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::F32)
                    .with_count(16),
                BufferDecl::output("out", 1, DataType::F32).with_count(16),
            ],
            [1, 1, 1],
            vec![node.clone()],
        );

        let cache = FactCache::derive_use_only(&prog);
        for expected in expected_uses {
            let count = cache.use_count_of(expected);
            assert!(
                count >= 1,
                "tile node {:?} must record use of operand `{expected}`, got use count {count}",
                node
            );
        }
    }

    let fused_prog = Program::wrapped(
        vec![
            BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::F32).with_count(16),
            BufferDecl::output("out", 1, DataType::F32).with_count(16),
        ],
        [1, 1, 1],
        vec![
            Node::tile_decl("scores", tile_2x2),
            Node::tile_elementwise(
                "exp_scores",
                vec![Ident::from("scores")],
                vec![Node::let_bind("exp_scores", Expr::f32(1.0))],
            ),
            Node::tile_store("buf", vec![Expr::u32(0)], "exp_scores"),
        ],
    );
    let opt_result = dce(fused_prog);
    let opt_entry = opt_result.entry();
    let elementwise_node = opt_entry
        .iter()
        .find_map(|node| match node {
            Node::TileElementwise { body, .. } => Some(body),
            Node::Region { body, .. } => body.iter().find_map(|n| match n {
                Node::TileElementwise { body, .. } => Some(body),
                _ => None,
            }),
            _ => None,
        })
        .expect("TileElementwise must be preserved");
    assert_eq!(
        elementwise_node.len(),
        1,
        "TileElementwise inner Let binding must not be eliminated by DCE"
    );
}
