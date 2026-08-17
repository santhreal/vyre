//! Test: rewrite layer contract.
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
use vyre_lower::rewrites::{
    all_registered_contracts, apply_lowering_rewrites, canonicalize_for_emit, classify_rule,
    lowering_owned_rules, rewrite_const_buffer_promote, rewrite_dead_ops, rewrite_vector_memory,
    rewrite_vector_memory_with_alias_facts, LoweringRewriteRule, RewriteOwnership,
    ALL_REWRITE_RULES,
};
use vyre_lower::{
    verify, BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelDescriptor,
    KernelOp, KernelOpKind, LiteralValue, MemoryClass,
};

fn workspace_root() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root()
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("failed to read {}: {error}", path.display());
    })
}

#[derive(Debug, Deserialize)]
struct KernelFamilySurfaceManifest {
    schema_version: u32,
    contract: String,
    family: Vec<KernelFamilySurface>,
}

#[derive(Debug, Deserialize)]
struct KernelFamilySurface {
    family_id: String,
    owner_lane: String,
    root: String,
    public_reexport: String,
    schedule_config: String,
    evidence_writer: String,
    forbid_section_dividers: bool,
    forbidden_private_import_prefixes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct KernelFamilySchedule {
    schema_version: u32,
    contract: String,
    family_id: String,
    stage_order: Vec<String>,
    evidence_policy: String,
}

#[test]
fn descriptor_lowering_owns_backend_neutral_rewrites_module() {
    assert!(
        workspace_root().join("vyre-lower/src/rewrites").exists(),
        "verified lowering must own the canonical backend-neutral rewrites module"
    );
}

#[test]
fn exhaustive_rewrite_registry_covers_all_variants_without_omissions() {
    let contracts = all_registered_contracts();
    assert_eq!(
        contracts.len(),
        ALL_REWRITE_RULES.len(),
        "every declared rewrite rule must have an applicability contract"
    );

    for rule in ALL_REWRITE_RULES {
        let contract = classify_rule(*rule);
        assert_eq!(contract.rule, *rule);
        assert!(
            !contract.id.trim().is_empty(),
            "rule {:?} id must not be empty",
            rule
        );
        assert!(
            !contract.rationale.trim().is_empty(),
            "rule {:?} rationale must not be empty",
            rule
        );
        assert!(
            !contract.target_structure.trim().is_empty(),
            "rule {:?} target_structure must not be empty",
            rule
        );
        assert!(
            !contract.preconditions.is_empty(),
            "rule {:?} preconditions must declare at least one requirement",
            rule
        );
    }
}

#[test]
fn lowering_owned_rewrites_preserve_program_semantics() {
    let owned = lowering_owned_rules();
    assert_eq!(
        owned,
        vec![
            LoweringRewriteRule::RepresentationCanonicalize,
            LoweringRewriteRule::ConstBufferPromote,
            LoweringRewriteRule::DeadOpElimination,
            LoweringRewriteRule::VectorLoadFusion,
        ],
        "only verified profitable backend-neutral transforms belong in lowering"
    );

    for rule in owned {
        let contract = rule.contract();
        assert_eq!(contract.ownership, RewriteOwnership::LoweringOwned);
        assert!(
            contract.preserves_program_semantics,
            "lowering rewrite {:?} must preserve Program semantics",
            rule
        );
        assert_eq!(
            contract.target_structure, "KernelDescriptor",
            "lowering rewrite {:?} must target KernelDescriptor",
            rule
        );
    }
}

#[test]
fn non_lowering_rules_are_explicitly_classified_and_refused_by_lowering() {
    // Texture promotion is an emitter-owned binding decoration strategy.
    let texture = LoweringRewriteRule::TexturePromote.contract();
    assert!(matches!(
        texture.ownership,
        RewriteOwnership::EmitterOwned { .. }
    ));
    assert!(!texture.rule.is_lowering_owned());

    // Shared memory promotion is a foundation semantic / tiling strategy.
    let shared = LoweringRewriteRule::SharedMemPromote.contract();
    assert!(matches!(
        shared.ownership,
        RewriteOwnership::FoundationOptimizerOwned { .. }
    ));
    assert!(!shared.rule.is_lowering_owned());

    // Loop LICM is a foundation semantic optimizer pass.
    let licm = LoweringRewriteRule::LoopLicmLoadHoist.contract();
    assert!(matches!(
        licm.ownership,
        RewriteOwnership::FoundationOptimizerOwned { .. }
    ));
    assert!(!licm.rule.is_lowering_owned());

    // CSE is a foundation semantic optimizer pass.
    let cse = LoweringRewriteRule::CommonSubexpressionElim.contract();
    assert!(matches!(
        cse.ownership,
        RewriteOwnership::FoundationOptimizerOwned { .. }
    ));
    assert!(!cse.rule.is_lowering_owned());

    // AoS to SoA layout transformation is a host dispatch / binding signature transformation.
    let layout = LoweringRewriteRule::LayoutAosToSoa.contract();
    assert!(matches!(
        layout.ownership,
        RewriteOwnership::HostDispatchOwned { .. }
    ));
    assert!(!layout.rule.is_lowering_owned());
}
#[test]
fn const_buffer_promotion_rewrite_transforms_eligible_bindings_and_loads() {
    let desc = KernelDescriptor {
        id: "const_promote_contract".into(),
        bindings: BindingLayout {
            slots: vec![
                BindingSlot {
                    slot: 0,
                    name: "weights".into(),
                    element_type: DataType::F32,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    element_count: Some(512),
                },
                BindingSlot {
                    slot: 1,
                    name: "activations".into(),
                    element_type: DataType::F32,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    element_count: Some(512),
                },
            ],
        },
        dispatch: Dispatch {
            workgroup_size: [128, 1, 1],
        },
        body: KernelBody {
            ops: vec![
                lit(0, 0), // index 0 (result 0)
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 0],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![1, 2],
                    result: Some(3),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![1, 0, 3],
                    result: None,
                },
            ],
            literals: vec![LiteralValue::U32(0)],
            child_bodies: vec![],
        },
    };

    let rewritten = rewrite_const_buffer_promote(&desc);

    // Slot 0 was eligible and promoted.
    assert_eq!(
        rewritten.bindings.slots[0].memory_class,
        MemoryClass::Constant
    );
    assert_eq!(rewritten.body.ops[1].kind, KernelOpKind::LoadConstant);
    assert_eq!(rewritten.body.ops[2].kind, KernelOpKind::LoadConstant);

    // Slot 1 was ReadWrite and remains Global.
    assert_eq!(
        rewritten.bindings.slots[1].memory_class,
        MemoryClass::Global
    );
    assert_eq!(rewritten.body.ops[4].kind, KernelOpKind::StoreGlobal);

    // Verifier succeeds.
    assert!(verify(&rewritten).is_ok());
}
#[test]
fn vector_memory_rewrite_transforms_eligible_load_and_store_chains_at_vec2_and_vec4() {
    // Positive vec4 load chain
    let mut load_ops = Vec::new();
    let mut literals = Vec::new();
    for i in 0..4u32 {
        literals.push(LiteralValue::U32(i));
        load_ops.push(lit(i, i));
        load_ops.push(op(KernelOpKind::LoadGlobal, [0, i], 10 + i));
    }
    load_ops.push(effect(KernelOpKind::StoreGlobal, [1, 0, 10]));

    let desc = descriptor("vec4_load_kernel")
        .slot(BindingSlot {
            slot: 0,
            name: "in_table".into(),
            element_type: DataType::F32,
            memory_class: MemoryClass::Global,
            visibility: BindingVisibility::ReadOnly,
            element_count: Some(512),
        })
        .slot(global_rw(1, DataType::F32, "out_table"))
        .dispatch(64, 1, 1)
        .body(body().literals(literals).ops(load_ops))
        .build();

    let rewritten = rewrite_vector_memory(&desc);
    assert_eq!(
        rewritten.body.ops[4].kind,
        KernelOpKind::VectorLoadGlobal { width: 4 }
    );
    assert_eq!(
        rewritten.body.ops[5].kind,
        KernelOpKind::ExtractLane { lane: 0 }
    );
    assert_eq!(
        rewritten.body.ops[6].kind,
        KernelOpKind::ExtractLane { lane: 1 }
    );
    assert_eq!(
        rewritten.body.ops[7].kind,
        KernelOpKind::ExtractLane { lane: 2 }
    );
    assert_eq!(
        rewritten.body.ops[8].kind,
        KernelOpKind::ExtractLane { lane: 3 }
    );
    assert_eq!(rewritten.body.ops[5].result, Some(10));
    assert_eq!(rewritten.body.ops[6].result, Some(11));
    assert_eq!(rewritten.body.ops[7].result, Some(12));
    assert_eq!(rewritten.body.ops[8].result, Some(13));
    assert!(verify(&rewritten).is_ok());
    // Positive vec2 store chain
    let store_desc = descriptor("vec2_store_kernel")
        .slot(global_rw(0, DataType::U32, "out_buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(1),
                    LiteralValue::U32(100),
                    LiteralValue::U32(200),
                ])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 2]),
                    effect(KernelOpKind::StoreGlobal, [0, 1, 3]),
                ]),
        )
        .build();

    let rewritten_store = rewrite_vector_memory(&store_desc);
    assert_eq!(rewritten_store.body.ops.len(), 5);
    assert_eq!(
        rewritten_store.body.ops[4].kind,
        KernelOpKind::VectorStoreGlobal { width: 2 }
    );
    assert_eq!(rewritten_store.body.ops[4].operands, vec![0, 0, 2, 3]);
    assert!(verify(&rewritten_store).is_ok());
    // Positive vec4 store chain
    let vec4_store_desc = descriptor("vec4_store_kernel")
        .slot(global_rw(0, DataType::U32, "out_buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(1),
                    LiteralValue::U32(2),
                    LiteralValue::U32(3),
                    LiteralValue::U32(10),
                    LiteralValue::U32(20),
                    LiteralValue::U32(30),
                    LiteralValue::U32(40),
                ])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    lit(4, 4),
                    lit(5, 5),
                    lit(6, 6),
                    lit(7, 7),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 4]),
                    effect(KernelOpKind::StoreGlobal, [0, 1, 5]),
                    effect(KernelOpKind::StoreGlobal, [0, 2, 6]),
                    effect(KernelOpKind::StoreGlobal, [0, 3, 7]),
                ]),
        )
        .build();

    let rewritten_vec4_store = rewrite_vector_memory(&vec4_store_desc);
    assert_eq!(rewritten_vec4_store.body.ops.len(), 9);
    assert_eq!(
        rewritten_vec4_store.body.ops[8].kind,
        KernelOpKind::VectorStoreGlobal { width: 4 }
    );
    assert_eq!(
        rewritten_vec4_store.body.ops[8].operands,
        vec![0, 0, 4, 5, 6, 7]
    );
    assert!(verify(&rewritten_vec4_store).is_ok());
}

#[test]
fn vector_memory_rewrite_rejects_misaligned_accesses() {
    // Offset 1 is misaligned for vec2 (requires % 2 == 0)
    let desc = descriptor("misaligned_vec2")
        .slot(global_rw(0, DataType::U32, "buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(1),
                    LiteralValue::U32(2),
                    LiteralValue::U32(10),
                    LiteralValue::U32(20),
                ])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 2]),
                    effect(KernelOpKind::StoreGlobal, [0, 1, 3]),
                ]),
        )
        .build();

    let rewritten = rewrite_vector_memory(&desc);
    assert_eq!(rewritten, desc, "misaligned base offset must remain scalar");
}

#[test]
fn vector_memory_rewrite_rejects_aliasing_hazards() {
    use vyre_lower::analyses::alias_facts::AliasFactSet;

    let desc = descriptor("alias_hazard")
        .slot(global_rw(0, DataType::U32, "buf_a"))
        .slot(global_rw(1, DataType::U32, "buf_b"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(1),
                    LiteralValue::U32(10),
                    LiteralValue::U32(20),
                ])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    op(KernelOpKind::LoadGlobal, [0, 0], 10),
                    // Intervening write to same buffer between loads is a RAW hazard
                    effect(KernelOpKind::StoreGlobal, [0, 0, 2]),
                    op(KernelOpKind::LoadGlobal, [0, 1], 11),
                ]),
        )
        .build();

    let rewritten = rewrite_vector_memory(&desc);
    assert_eq!(
        rewritten, desc,
        "intervening write to same slot must reject load vectorization"
    );

    // Intervening write to different slot with unproven alias
    let mut alias_facts = AliasFactSet::default();
    // Do not insert no-alias fact
    let rewritten_alias = rewrite_vector_memory_with_alias_facts(&desc, &alias_facts);
    assert_eq!(rewritten_alias, desc);
}

#[test]
fn vector_memory_rewrite_rejects_intervening_side_effects() {
    let desc = descriptor("side_effect_hazard")
        .slot(global_rw(0, DataType::U32, "buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(1),
                    LiteralValue::U32(10),
                    LiteralValue::U32(20),
                ])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 2]),
                    // Intervening Atomic side effect
                    op(
                        KernelOpKind::Atomic {
                            op: vyre_foundation::ir::AtomicOp::Add,
                            ordering: vyre_foundation::ir::MemoryOrdering::Relaxed,
                        },
                        [0, 0, 2],
                        4,
                    ),
                    effect(KernelOpKind::StoreGlobal, [0, 1, 3]),
                ]),
        )
        .build();

    let rewritten = rewrite_vector_memory(&desc);
    assert_eq!(
        rewritten, desc,
        "intervening atomic effect must reject vectorization"
    );
}

#[test]
fn vector_memory_rewrite_rejects_scheduling_fences() {
    let desc = descriptor("fence_hazard")
        .slot(global_rw(0, DataType::U32, "buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(1),
                    LiteralValue::U32(10),
                    LiteralValue::U32(20),
                ])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 2]),
                    effect(
                        KernelOpKind::Barrier {
                            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
                        },
                        [],
                    ),
                    effect(KernelOpKind::StoreGlobal, [0, 1, 3]),
                ]),
        )
        .build();

    let rewritten = rewrite_vector_memory(&desc);
    assert_eq!(
        rewritten, desc,
        "intervening barrier fence must reject vectorization"
    );
}

#[test]
fn vector_memory_rewrite_rejects_structured_control_boundaries() {
    let desc = descriptor("structured_control_hazard")
        .slot(global_rw(0, DataType::U32, "buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(1),
                    LiteralValue::U32(10),
                    LiteralValue::U32(20),
                    LiteralValue::Bool(true),
                ])
                .child(body().ops([effect(KernelOpKind::StoreGlobal, [0, 1, 3])]))
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    lit(4, 4),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 2]),
                    effect(KernelOpKind::StructuredIfThen, [4, 0]),
                ]),
        )
        .build();

    let rewritten = rewrite_vector_memory(&desc);
    assert_eq!(
        rewritten, desc,
        "accesses across structured control boundary must not be fused across bodies"
    );
}

#[test]
fn vector_memory_rewrite_rejects_unsupported_widths() {
    // Width 3 starting at 0: unsupported width (must be vec2 or vec4)
    // Note: a 3-element chain starting at 0 has an aligned vec2 prefix [0, 1],
    // so it canonicalizes the vec2 prefix and leaves element 2 scalar.
    // Width 1 is not a chain at all.
    let desc = descriptor("singleton")
        .slot(global_rw(0, DataType::U32, "buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(10)])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 1]),
                ]),
        )
        .build();

    let rewritten = rewrite_vector_memory(&desc);
    assert_eq!(rewritten, desc, "single store is not a vector chain");
}

#[test]
fn vector_memory_rewrite_rejects_value_dependency_hazards() {
    // Value of second store is computed between the stores from an impure or dependent op
    let desc = descriptor("value_dependency_hazard")
        .slot(global_rw(0, DataType::U32, "buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(1),
                    LiteralValue::U32(10),
                ])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 2]),
                    // Op 4 computes the value for the second store
                    op(KernelOpKind::BinOpKind(BinOp::Add), [2, 2], 4),
                    effect(KernelOpKind::StoreGlobal, [0, 1, 4]),
                ]),
        )
        .build();

    let rewritten = rewrite_vector_memory(&desc);
    assert_eq!(
        rewritten, desc,
        "intervening value computation between stores must reject store fusion"
    );
}

#[test]
fn vector_load_fusion_collapses_memory_ops_and_preserves_ssa_projections() {
    // 4 contiguous LoadGlobal ops feeding downstream BinOp Add consumers
    let desc = descriptor("contiguous_load_ssa_projections")
        .slot(global_rw(0, DataType::F32, "in_buf"))
        .slot(global_rw(1, DataType::F32, "out_buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(1),
                    LiteralValue::U32(2),
                    LiteralValue::U32(3),
                ])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    op(KernelOpKind::LoadGlobal, [0, 0], 10),
                    op(KernelOpKind::LoadGlobal, [0, 1], 11),
                    op(KernelOpKind::LoadGlobal, [0, 2], 12),
                    op(KernelOpKind::LoadGlobal, [0, 3], 13),
                    // Downstream SSA consumers reading load results
                    op(KernelOpKind::BinOpKind(BinOp::Add), [10, 11], 20),
                    op(KernelOpKind::BinOpKind(BinOp::Add), [12, 13], 21),
                    effect(KernelOpKind::StoreGlobal, [1, 0, 20]),
                    effect(KernelOpKind::StoreGlobal, [1, 1, 21]),
                ]),
        )
        .build();

    let rewritten = rewrite_vector_memory(&desc);

    // Verify exact op sequence: 4 literals, 1 VectorLoadGlobal, 4 ExtractLane, 2 BinOp Add, 1 VectorStoreGlobal
    assert_eq!(rewritten.body.ops.len(), 12);
    assert_eq!(
        rewritten.body.ops[4].kind,
        KernelOpKind::VectorLoadGlobal { width: 4 }
    );
    assert_eq!(rewritten.body.ops[4].operands, vec![0, 0]);
    let vec_res = rewritten.body.ops[4]
        .result
        .expect("VectorLoadGlobal must define result");

    assert_eq!(
        rewritten.body.ops[5].kind,
        KernelOpKind::ExtractLane { lane: 0 }
    );
    assert_eq!(rewritten.body.ops[5].operands, vec![vec_res]);
    assert_eq!(rewritten.body.ops[5].result, Some(10));

    assert_eq!(
        rewritten.body.ops[6].kind,
        KernelOpKind::ExtractLane { lane: 1 }
    );
    assert_eq!(rewritten.body.ops[6].operands, vec![vec_res]);
    assert_eq!(rewritten.body.ops[6].result, Some(11));

    assert_eq!(
        rewritten.body.ops[7].kind,
        KernelOpKind::ExtractLane { lane: 2 }
    );
    assert_eq!(rewritten.body.ops[7].operands, vec![vec_res]);
    assert_eq!(rewritten.body.ops[7].result, Some(12));

    assert_eq!(
        rewritten.body.ops[8].kind,
        KernelOpKind::ExtractLane { lane: 3 }
    );
    assert_eq!(rewritten.body.ops[8].operands, vec![vec_res]);
    assert_eq!(rewritten.body.ops[8].result, Some(13));

    // Downstream consumers preserve operand result identities
    assert_eq!(
        rewritten.body.ops[9].kind,
        KernelOpKind::BinOpKind(BinOp::Add)
    );
    assert_eq!(rewritten.body.ops[9].operands, vec![10, 11]);
    assert_eq!(rewritten.body.ops[9].result, Some(20));

    assert_eq!(
        rewritten.body.ops[10].kind,
        KernelOpKind::BinOpKind(BinOp::Add)
    );
    assert_eq!(rewritten.body.ops[10].operands, vec![12, 13]);
    assert_eq!(rewritten.body.ops[10].result, Some(21));

    assert_eq!(
        rewritten.body.ops[11].kind,
        KernelOpKind::VectorStoreGlobal { width: 2 }
    );
    assert_eq!(rewritten.body.ops[11].operands, vec![1, 0, 20, 21]);

    assert!(verify(&rewritten).is_ok());
}

#[test]
fn vector_store_fusion_collapses_op_counts() {
    let desc = descriptor("contiguous_store_op_collapse")
        .slot(global_rw(0, DataType::U32, "out_buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(1),
                    LiteralValue::U32(2),
                    LiteralValue::U32(3),
                    LiteralValue::U32(100),
                    LiteralValue::U32(200),
                    LiteralValue::U32(300),
                    LiteralValue::U32(400),
                ])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    lit(4, 4),
                    lit(5, 5),
                    lit(6, 6),
                    lit(7, 7),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 4]),
                    effect(KernelOpKind::StoreGlobal, [0, 1, 5]),
                    effect(KernelOpKind::StoreGlobal, [0, 2, 6]),
                    effect(KernelOpKind::StoreGlobal, [0, 3, 7]),
                ]),
        )
        .build();

    let before_op_count = desc.body.ops.len();
    let rewritten = rewrite_vector_memory(&desc);
    let after_op_count = rewritten.body.ops.len();

    // Exactly 3 store ops eliminated (4 scalar stores collapsed into 1 vector store)
    assert_eq!(before_op_count - after_op_count, 3);
    assert_eq!(
        rewritten.body.ops[8].kind,
        KernelOpKind::VectorStoreGlobal { width: 4 }
    );
    assert_eq!(rewritten.body.ops[8].operands, vec![0, 0, 4, 5, 6, 7]);
    assert!(verify(&rewritten).is_ok());
}

#[test]
fn vector_memory_rewrite_preserves_scalar_for_unsupported_byte_types() {
    // Byte-element buffer (U8) is not in supported 4-byte scalar set for vector memory rewrite
    let desc = descriptor("u8_buffer_load")
        .slot(global_rw(0, DataType::U8, "byte_buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(1),
                    LiteralValue::U32(2),
                    LiteralValue::U32(3),
                ])
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    op(KernelOpKind::LoadGlobal, [0, 0], 10),
                    op(KernelOpKind::LoadGlobal, [0, 1], 11),
                    op(KernelOpKind::LoadGlobal, [0, 2], 12),
                    op(KernelOpKind::LoadGlobal, [0, 3], 13),
                ]),
        )
        .build();

    let rewritten = rewrite_vector_memory(&desc);
    assert_eq!(rewritten, desc, "byte-element buffer must remain scalar");
}

#[test]
fn dead_op_elimination_rewrite_strips_unreferenced_pure_ops() {
    let desc = KernelDescriptor {
        id: "dead_op_contract".into(),
        bindings: BindingLayout {
            slots: vec![BindingSlot {
                slot: 0,
                name: "out".into(),
                element_type: DataType::U32,
                memory_class: MemoryClass::Global,
                visibility: BindingVisibility::ReadWrite,
                element_count: Some(64),
            }],
        },
        dispatch: Dispatch {
            workgroup_size: [64, 1, 1],
        },
        body: KernelBody {
            ops: vec![
                lit(0, 0), // result 0, used
                lit(0, 1), // result 1, DEAD
                lit(0, 2), // result 2, DEAD
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 0, 0],
                    result: None,
                },
            ],
            literals: vec![LiteralValue::U32(99)],
            child_bodies: vec![],
        },
    };

    let rewritten = rewrite_dead_ops(&desc);

    assert_eq!(rewritten.body.ops.len(), 2);
    assert_eq!(rewritten.body.ops[0].result, Some(0));
    assert_eq!(rewritten.body.ops[1].kind, KernelOpKind::StoreGlobal);
    assert!(verify(&rewritten).is_ok());
}

#[test]
fn representation_canonicalize_orders_pure_producers_before_consumers() {
    let desc = KernelDescriptor {
        id: "canonicalize_contract".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch {
            workgroup_size: [1, 1, 1],
        },
        body: KernelBody {
            ops: vec![
                lit(0, 0),
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![1, 2],
                    result: Some(3),
                },
                lit(0, 1),
                lit(0, 2),
            ],
            literals: vec![LiteralValue::U32(10)],
            child_bodies: vec![],
        },
    };

    let canonical = canonicalize_for_emit(&desc);

    assert_eq!(canonical.body.ops[1].result, Some(1));
    assert_eq!(canonical.body.ops[2].result, Some(2));
    assert_eq!(canonical.body.ops[3].result, Some(3));
    assert_eq!(canonicalize_for_emit(&canonical), canonical);
    if let Err(errors) = verify(&canonical) {
        panic!("verify(&canonical) failed: {errors:?}");
    }
}

#[test]
fn apply_lowering_rewrites_pipeline_is_idempotent_and_verifies() {
    let desc = KernelDescriptor {
        id: "pipeline_contract".into(),
        bindings: BindingLayout {
            slots: vec![
                BindingSlot {
                    slot: 0,
                    name: "lut".into(),
                    element_type: DataType::U32,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    element_count: Some(64),
                },
                BindingSlot {
                    slot: 1,
                    name: "dest".into(),
                    element_type: DataType::U32,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    element_count: Some(64),
                },
            ],
        },
        dispatch: Dispatch {
            workgroup_size: [64, 1, 1],
        },
        body: KernelBody {
            ops: vec![
                lit(0, 0), // index 0 (result 0)
                lit(0, 1), // dead literal (result 1)
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 0],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 0],
                    result: Some(3),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![2, 3],
                    result: Some(4),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![1, 0, 4],
                    result: None,
                },
            ],
            literals: vec![LiteralValue::U32(0)],
            child_bodies: vec![],
        },
    };

    let pass1 = apply_lowering_rewrites(&desc);
    let pass2 = apply_lowering_rewrites(&pass1);

    assert_eq!(pass1, pass2);
    assert_eq!(pass1.bindings.slots[0].memory_class, MemoryClass::Constant);
    assert_eq!(pass1.body.ops[1].kind, KernelOpKind::LoadConstant);
    assert_eq!(pass1.body.ops[2].kind, KernelOpKind::LoadConstant);
    assert!(verify(&pass1).is_ok());
}

#[test]
fn kernel_family_surfaces_have_single_reexport_schedule_and_evidence_contracts() {
    let root = workspace_root();
    let manifest_path = root.join("vyre-lower/rules/kernel_family_surfaces.toml");
    let manifest: KernelFamilySurfaceManifest = toml::from_str(&read(&manifest_path))
        .expect("Fix: kernel family surface manifest must be valid TOML.");

    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.contract, "vyre-kernel-family-surfaces:v1");
    assert!(
        !manifest.family.is_empty(),
        "Fix: kernel family surface manifest must declare at least one family."
    );

    let mut public_reexports = std::collections::BTreeSet::new();
    let mut failures = Vec::new();
    for family in &manifest.family {
        if family.family_id.trim().is_empty() {
            failures.push("family_id is blank".to_string());
        }
        if family.owner_lane.trim().is_empty() {
            failures.push(format!("{} owner_lane is blank", family.family_id));
        }
        for (label, rel_path) in [
            ("root", &family.root),
            ("public_reexport", &family.public_reexport),
            ("schedule_config", &family.schedule_config),
            ("evidence_writer", &family.evidence_writer),
        ] {
            if !root.join(rel_path).exists() {
                failures.push(format!(
                    "{} {label} `{rel_path}` does not exist",
                    family.family_id
                ));
            }
        }
        if !public_reexports.insert(family.public_reexport.as_str()) {
            failures.push(format!(
                "duplicate public re-export point `{}`",
                family.public_reexport
            ));
        }
        if let Ok(schedule_text) = fs::read_to_string(root.join(&family.schedule_config)) {
            let schedule: KernelFamilySchedule =
                toml::from_str(&schedule_text).unwrap_or_else(|error| {
                    panic!(
                        "Fix: schedule config `{}` must be valid TOML: {error}",
                        family.schedule_config
                    )
                });
            if schedule.schema_version != 1 {
                failures.push(format!(
                    "{} schedule schema_version must be 1",
                    family.family_id
                ));
            }
            if schedule.contract != "vyre-kernel-family-schedule:v1" {
                failures.push(format!(
                    "{} schedule contract must be vyre-kernel-family-schedule:v1",
                    family.family_id
                ));
            }
            if schedule.family_id != family.family_id {
                failures.push(format!(
                    "{} schedule family_id `{}` does not match manifest",
                    family.family_id, schedule.family_id
                ));
            }
            if schedule.stage_order.len() < 3 {
                failures.push(format!(
                    "{} schedule must declare at least three stages",
                    family.family_id
                ));
            }
            if schedule.evidence_policy.trim().is_empty() {
                failures.push(format!(
                    "{} schedule evidence_policy is blank",
                    family.family_id
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Kernel family organization contract failed:\n{}",
        failures.join("\n")
    );
}
