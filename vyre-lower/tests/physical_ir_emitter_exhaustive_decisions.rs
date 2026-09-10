//! Proves that every physical-IR variant has an explicit per-target emitter decision and no catch-all downgrade exists.
//!
//! WHY: In a GPU compiler, target emitters must make an explicit, deterministic decision
//! for every physical-IR `KernelOpKind` variant. Silent catch-all arms (`_ => ...`) or
//! implicit CPU fallback downgrades violate the compiler execution doctrine:
//! 1. An unhandled op must fail at compile time (via exhaustive match) or return an explicit structured error.
//! 2. No catch-all downgrade may convert an un-emittable kernel into silent host execution or a stub.
//! 3. All target dialects (naga/wgsl, ptx, spirv, metal) must have explicit coverage.

use std::collections::BTreeSet;
use std::path::Path;
use vyre_foundation::ir::{AtomicOp, BinOp, DataType, MemoryOrdering, SubgroupReduceOp, UnOp};
use vyre_lower::variant_space::kernel_op_kind_variants;
use vyre_lower::{
    AsyncTransaction, AsyncWaitSpec, FragmentValue, KernelOpKind, MatrixMmaElement,
    MatrixMmaLayout, MatrixMmaSpec, MatrixTileShape, MemoryProxyFence, Name, OpaqueExprData,
    OpaqueNodeData, TransactionScope,
};
/// Target emitter decision category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitterDecision {
    /// Emitter supports this op natively in the target dialect.
    NativeEmit,
    /// Emitter transforms/decomposes this op into target instructions.
    Decomposed,
    /// Emitter explicitly rejects this op with a structured unsupported diagnostic.
    ExplicitReject,
}

/// Helper that checks per-target emitter decisions using an EXHAUSTIVE match with NO catch-all `_ =>` arm.
///
/// If a new variant is added to `KernelOpKind`, this function will fail to compile
/// until an explicit decision is recorded for all targets.
fn classify_physical_ir_variant_decision(kind: &KernelOpKind, target: &str) -> EmitterDecision {
    match target {
        "naga" | "wgpu" => match kind {
            KernelOpKind::Literal
            | KernelOpKind::Copy
            | KernelOpKind::LocalInvocationId
            | KernelOpKind::GlobalInvocationId
            | KernelOpKind::WorkgroupId
            | KernelOpKind::SubgroupLocalId
            | KernelOpKind::SubgroupSize
            | KernelOpKind::LoopIndex { .. }
            | KernelOpKind::LoopCarrierInit { .. }
            | KernelOpKind::LoopCarrier { .. }
            | KernelOpKind::LoopCarrierEnd { .. }
            | KernelOpKind::LoadGlobal
            | KernelOpKind::LoadShared
            | KernelOpKind::LoadConstant
            | KernelOpKind::BufferLength
            | KernelOpKind::StoreGlobal
            | KernelOpKind::StoreShared
            | KernelOpKind::VectorLoadGlobal { .. }
            | KernelOpKind::VectorStoreGlobal { .. }
            | KernelOpKind::ExtractLane { .. }
            | KernelOpKind::BinOpKind(_)
            | KernelOpKind::UnOpKind(_)
            | KernelOpKind::Cast { .. }
            | KernelOpKind::Select
            | KernelOpKind::Fma
            | KernelOpKind::StructuredIfThen
            | KernelOpKind::StructuredIfThenElse
            | KernelOpKind::StructuredBlock
            | KernelOpKind::Region { .. }
            | KernelOpKind::StructuredForLoop { .. }
            | KernelOpKind::AsyncLoad(_)
            | KernelOpKind::AsyncStore(_)
            | KernelOpKind::AsyncWait(_)
            | KernelOpKind::Trap { .. }
            | KernelOpKind::Resume { .. }
            | KernelOpKind::Barrier { .. }
            | KernelOpKind::Return
            | KernelOpKind::SubgroupBallot
            | KernelOpKind::SubgroupReduce { .. }
            | KernelOpKind::SubgroupShuffle
            | KernelOpKind::SubgroupBroadcast
            | KernelOpKind::Atomic { .. } => EmitterDecision::NativeEmit,

            KernelOpKind::IndirectDispatch { .. }
            | KernelOpKind::MatrixMma(_)
            | KernelOpKind::Call { .. }
            | KernelOpKind::OpaqueExpr(_)
            | KernelOpKind::OpaqueNode(_) => EmitterDecision::ExplicitReject,
        },

        "ptx" | "cuda" => match kind {
            KernelOpKind::Literal
            | KernelOpKind::Copy
            | KernelOpKind::LocalInvocationId
            | KernelOpKind::GlobalInvocationId
            | KernelOpKind::WorkgroupId
            | KernelOpKind::SubgroupLocalId
            | KernelOpKind::SubgroupSize
            | KernelOpKind::LoopIndex { .. }
            | KernelOpKind::LoopCarrierInit { .. }
            | KernelOpKind::LoopCarrier { .. }
            | KernelOpKind::LoopCarrierEnd { .. }
            | KernelOpKind::LoadGlobal
            | KernelOpKind::LoadShared
            | KernelOpKind::LoadConstant
            | KernelOpKind::BufferLength
            | KernelOpKind::StoreGlobal
            | KernelOpKind::StoreShared
            | KernelOpKind::VectorLoadGlobal { .. }
            | KernelOpKind::VectorStoreGlobal { .. }
            | KernelOpKind::ExtractLane { .. }
            | KernelOpKind::BinOpKind(_)
            | KernelOpKind::UnOpKind(_)
            | KernelOpKind::Cast { .. }
            | KernelOpKind::Select
            | KernelOpKind::Fma
            | KernelOpKind::StructuredIfThen
            | KernelOpKind::StructuredIfThenElse
            | KernelOpKind::StructuredBlock
            | KernelOpKind::Region { .. }
            | KernelOpKind::StructuredForLoop { .. }
            | KernelOpKind::AsyncLoad(_)
            | KernelOpKind::AsyncStore(_)
            | KernelOpKind::AsyncWait(_)
            | KernelOpKind::Trap { .. }
            | KernelOpKind::Resume { .. }
            | KernelOpKind::Barrier { .. }
            | KernelOpKind::Return
            | KernelOpKind::SubgroupBallot
            | KernelOpKind::SubgroupReduce { .. }
            | KernelOpKind::SubgroupShuffle
            | KernelOpKind::SubgroupBroadcast
            | KernelOpKind::Atomic { .. }
            | KernelOpKind::MatrixMma(_) => EmitterDecision::NativeEmit,

            KernelOpKind::IndirectDispatch { .. }
            | KernelOpKind::Call { .. }
            | KernelOpKind::OpaqueExpr(_)
            | KernelOpKind::OpaqueNode(_) => EmitterDecision::ExplicitReject,
        },

        "spirv" => match kind {
            KernelOpKind::Literal
            | KernelOpKind::Copy
            | KernelOpKind::LocalInvocationId
            | KernelOpKind::GlobalInvocationId
            | KernelOpKind::WorkgroupId
            | KernelOpKind::SubgroupLocalId
            | KernelOpKind::SubgroupSize
            | KernelOpKind::LoopIndex { .. }
            | KernelOpKind::LoopCarrierInit { .. }
            | KernelOpKind::LoopCarrier { .. }
            | KernelOpKind::LoopCarrierEnd { .. }
            | KernelOpKind::LoadGlobal
            | KernelOpKind::LoadShared
            | KernelOpKind::LoadConstant
            | KernelOpKind::BufferLength
            | KernelOpKind::StoreGlobal
            | KernelOpKind::StoreShared
            | KernelOpKind::VectorLoadGlobal { .. }
            | KernelOpKind::VectorStoreGlobal { .. }
            | KernelOpKind::ExtractLane { .. }
            | KernelOpKind::BinOpKind(_)
            | KernelOpKind::UnOpKind(_)
            | KernelOpKind::Cast { .. }
            | KernelOpKind::Select
            | KernelOpKind::Fma
            | KernelOpKind::StructuredIfThen
            | KernelOpKind::StructuredIfThenElse
            | KernelOpKind::StructuredBlock
            | KernelOpKind::Region { .. }
            | KernelOpKind::StructuredForLoop { .. }
            | KernelOpKind::AsyncLoad(_)
            | KernelOpKind::AsyncStore(_)
            | KernelOpKind::AsyncWait(_)
            | KernelOpKind::Trap { .. }
            | KernelOpKind::Resume { .. }
            | KernelOpKind::Barrier { .. }
            | KernelOpKind::Return
            | KernelOpKind::SubgroupBallot
            | KernelOpKind::SubgroupReduce { .. }
            | KernelOpKind::SubgroupShuffle
            | KernelOpKind::SubgroupBroadcast
            | KernelOpKind::Atomic { .. } => EmitterDecision::NativeEmit,

            KernelOpKind::IndirectDispatch { .. }
            | KernelOpKind::MatrixMma(_)
            | KernelOpKind::Call { .. }
            | KernelOpKind::OpaqueExpr(_)
            | KernelOpKind::OpaqueNode(_) => EmitterDecision::ExplicitReject,
        },

        "metal" => match kind {
            KernelOpKind::Literal
            | KernelOpKind::Copy
            | KernelOpKind::LocalInvocationId
            | KernelOpKind::GlobalInvocationId
            | KernelOpKind::WorkgroupId
            | KernelOpKind::SubgroupLocalId
            | KernelOpKind::SubgroupSize
            | KernelOpKind::LoopIndex { .. }
            | KernelOpKind::LoopCarrierInit { .. }
            | KernelOpKind::LoopCarrier { .. }
            | KernelOpKind::LoopCarrierEnd { .. }
            | KernelOpKind::LoadGlobal
            | KernelOpKind::LoadShared
            | KernelOpKind::LoadConstant
            | KernelOpKind::BufferLength
            | KernelOpKind::StoreGlobal
            | KernelOpKind::StoreShared
            | KernelOpKind::VectorLoadGlobal { .. }
            | KernelOpKind::VectorStoreGlobal { .. }
            | KernelOpKind::ExtractLane { .. }
            | KernelOpKind::BinOpKind(_)
            | KernelOpKind::UnOpKind(_)
            | KernelOpKind::Cast { .. }
            | KernelOpKind::Select
            | KernelOpKind::Fma
            | KernelOpKind::StructuredIfThen
            | KernelOpKind::StructuredIfThenElse
            | KernelOpKind::StructuredBlock
            | KernelOpKind::Region { .. }
            | KernelOpKind::StructuredForLoop { .. }
            | KernelOpKind::AsyncLoad(_)
            | KernelOpKind::AsyncStore(_)
            | KernelOpKind::AsyncWait(_)
            | KernelOpKind::Trap { .. }
            | KernelOpKind::Resume { .. }
            | KernelOpKind::Barrier { .. }
            | KernelOpKind::Return
            | KernelOpKind::SubgroupBallot
            | KernelOpKind::SubgroupReduce { .. }
            | KernelOpKind::SubgroupShuffle
            | KernelOpKind::SubgroupBroadcast
            | KernelOpKind::Atomic { .. } => EmitterDecision::NativeEmit,

            KernelOpKind::IndirectDispatch { .. }
            | KernelOpKind::MatrixMma(_)
            | KernelOpKind::Call { .. }
            | KernelOpKind::OpaqueExpr(_)
            | KernelOpKind::OpaqueNode(_) => EmitterDecision::ExplicitReject,
        },

        other => panic!("Fix: unknown target `{other}` under test"),
    }
}

/// Build a representative instance of every physical IR `KernelOpKind` variant.
fn all_physical_ir_variants() -> Vec<KernelOpKind> {
    vec![
        KernelOpKind::Literal,
        KernelOpKind::Copy,
        KernelOpKind::LocalInvocationId,
        KernelOpKind::GlobalInvocationId,
        KernelOpKind::WorkgroupId,
        KernelOpKind::SubgroupLocalId,
        KernelOpKind::SubgroupSize,
        KernelOpKind::LoopIndex {
            loop_var: Name::from("i"),
        },
        KernelOpKind::LoopCarrierInit {
            name: Name::from("acc"),
        },
        KernelOpKind::LoopCarrier {
            name: Name::from("acc"),
        },
        KernelOpKind::LoopCarrierEnd {
            name: Name::from("acc"),
        },
        KernelOpKind::LoadGlobal,
        KernelOpKind::LoadShared,
        KernelOpKind::LoadConstant,
        KernelOpKind::BufferLength,
        KernelOpKind::StoreGlobal,
        KernelOpKind::StoreShared,
        KernelOpKind::VectorLoadGlobal { width: 4 },
        KernelOpKind::VectorStoreGlobal { width: 4 },
        KernelOpKind::ExtractLane { lane: 0 },
        KernelOpKind::BinOpKind(BinOp::Add),
        KernelOpKind::UnOpKind(UnOp::BitNot),
        KernelOpKind::Cast {
            target: DataType::U32,
        },
        KernelOpKind::Select,
        KernelOpKind::Fma,
        KernelOpKind::StructuredIfThen,
        KernelOpKind::StructuredIfThenElse,
        KernelOpKind::StructuredBlock,
        KernelOpKind::Region {
            generator: Name::from("region_1"),
        },
        KernelOpKind::StructuredForLoop {
            loop_var: Name::from("i"),
        },
        KernelOpKind::AsyncLoad(Box::new(AsyncTransaction {
            tag: Name::from("tx_load"),
            visibility: TransactionScope::Workgroup,
            stage: None,
        })),
        KernelOpKind::AsyncStore(Box::new(AsyncTransaction {
            tag: Name::from("tx_store"),
            visibility: TransactionScope::Workgroup,
            stage: None,
        })),
        KernelOpKind::AsyncWait(Box::new(AsyncWaitSpec {
            transaction: AsyncTransaction {
                tag: Name::from("tx_wait"),
                visibility: TransactionScope::Workgroup,
                stage: None,
            },
            fence: MemoryProxyFence::Workgroup,
        })),
        KernelOpKind::Trap {
            tag: Name::from("trap_1"),
        },
        KernelOpKind::Resume {
            tag: Name::from("resume_1"),
        },
        KernelOpKind::Barrier {
            ordering: MemoryOrdering::SeqCst,
        },
        KernelOpKind::Return,
        KernelOpKind::SubgroupBallot,
        KernelOpKind::SubgroupReduce {
            op: SubgroupReduceOp::Add,
        },
        KernelOpKind::SubgroupShuffle,
        KernelOpKind::SubgroupBroadcast,
        KernelOpKind::Atomic {
            op: AtomicOp::Add,
            ordering: MemoryOrdering::SeqCst,
        },
        KernelOpKind::IndirectDispatch { count_offset: 0 },
        KernelOpKind::MatrixMma(Box::new(MatrixMmaSpec {
            tile: MatrixTileShape {
                m: 16,
                n: 16,
                k: 16,
            },
            left: FragmentValue {
                element: MatrixMmaElement::F16,
                layout: MatrixMmaLayout::RowMajor,
                lanes: 32,
                access: None,
            },
            right: FragmentValue {
                element: MatrixMmaElement::F16,
                layout: MatrixMmaLayout::ColMajor,
                lanes: 32,
                access: None,
            },
            accumulator: FragmentValue {
                element: MatrixMmaElement::F32,
                layout: MatrixMmaLayout::RowMajor,
                lanes: 32,
                access: None,
            },
        })),
        KernelOpKind::Call {
            op_id: Name::from("call_1"),
        },
        KernelOpKind::OpaqueExpr(Box::new(OpaqueExprData {
            extension_id: 1,
            extension_kind: "test_ext".to_string(),
            payload: vec![1, 2, 3],
        })),
        KernelOpKind::OpaqueNode(Box::new(OpaqueNodeData {
            extension_kind: "test_node".to_string(),
            payload: vec![4, 5, 6],
        })),
    ]
}

/// The variant name `kind` reports, taken from its derived `Debug` form.
fn variant_name(kind: &KernelOpKind) -> String {
    let debug = format!("{kind:?}");
    debug
        .split(['(', '{', ' '])
        .next()
        .expect("a Debug form opens with the variant identifier")
        .to_string()
}

/// The target dialects the decision table records an answer for.
const TARGETS: [&str; 4] = ["naga", "ptx", "spirv", "metal"];

#[test]
fn every_kernel_op_kind_the_source_declares_has_a_representative_and_a_decision() {
    let declared = kernel_op_kind_variants();
    let variants = all_physical_ir_variants();

    for target in TARGETS {
        let decided: BTreeSet<String> = variants
            .iter()
            .filter(|kind| {
                matches!(
                    classify_physical_ir_variant_decision(kind, target),
                    EmitterDecision::NativeEmit
                        | EmitterDecision::Decomposed
                        | EmitterDecision::ExplicitReject
                )
            })
            .map(variant_name)
            .collect();

        assert_eq!(
            decided, declared,
            "Fix: target `{target}` must record a decision for every declared \
             `KernelOpKind` variant. `all_physical_ir_variants` holds one \
             representative per variant and is written out, so a variant added to the \
             enum enters `declared` on its own and has to be added there too"
        );
    }
}

#[test]
fn no_target_rejects_an_op_without_which_no_program_can_run() {
    // A target that refuses to load, store, name a constant or return cannot
    // emit any program, so a reject recorded for one of these is a table entry
    // no emitter can honour.
    let indispensable = [
        KernelOpKind::Literal,
        KernelOpKind::LoadGlobal,
        KernelOpKind::StoreGlobal,
        KernelOpKind::Return,
    ];

    for target in TARGETS {
        for kind in &indispensable {
            assert_ne!(
                classify_physical_ir_variant_decision(kind, target),
                EmitterDecision::ExplicitReject,
                "Fix: target `{target}` records a reject for `{}`, which leaves it \
                 unable to emit any program at all",
                variant_name(kind)
            );
        }
    }
}

#[test]
fn emitter_dispatch_names_every_variant_rather_than_falling_through_a_catch_all() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("this crate is a workspace member, so its manifest has a parent");
    let declared = kernel_op_kind_variants();

    for relative_path in [
        "vyre-emit-naga/src/emitter/op_dispatch/mod.rs",
        "vyre-emit-ptx/src/emitter/dispatch.rs",
    ] {
        let path = workspace_root.join(relative_path);
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "Fix: {} is the op dispatch this contract audits and must be readable: {e}",
                path.display()
            )
        });

        let unnamed: Vec<&String> = declared
            .iter()
            .filter(|variant| !names_variant(&content, variant))
            .collect();
        assert!(
            unnamed.is_empty(),
            "Fix: {} must state an arm for every physical-IR variant. A variant the \
             dispatch never names reaches a catch-all instead of a decision: {unnamed:?}",
            path.display()
        );
    }
}

/// Whether `content` names `variant` as a whole identifier.
///
/// A substring test alone would accept `LoopCarrier` for `LoopCarrierEnd`, which
/// is the exact case a catch-all would hide.
fn names_variant(content: &str, variant: &str) -> bool {
    content.match_indices(variant).any(|(at, _)| {
        let before = content[..at].chars().next_back();
        let after = content[at + variant.len()..].chars().next();
        let boundary = |c: Option<char>| !c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        boundary(before) && boundary(after)
    })
}
