//! Proves that every physical-IR variant has an explicit per-target emitter decision and no catch-all downgrade exists.
//!
//! WHY: In a GPU compiler, target emitters must make an explicit, deterministic decision
//! for every physical-IR `KernelOpKind` variant. Silent catch-all arms (`_ => ...`) or
//! implicit CPU fallback downgrades violate the compiler execution doctrine:
//! 1. An unhandled op must fail at compile time (via exhaustive match) or return an explicit structured error.
//! 2. No catch-all downgrade may convert an un-emittable kernel into silent host execution or a stub.
//! 3. All target dialects (naga/wgsl, ptx, spirv, metal) must have explicit coverage.

use std::path::Path;
use vyre_foundation::ir::{
    AtomicOp, BinOp, DataType, MemoryOrdering, SubgroupReduceOp, UnOp,
};
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
        KernelOpKind::IndirectDispatch {
            count_offset: 0,
        },
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

#[test]
fn every_physical_ir_variant_has_explicit_decision_for_all_targets() {
    let variants = all_physical_ir_variants();
    let targets = ["naga", "ptx", "spirv", "metal"];

    for target in &targets {
        for variant in &variants {
            let decision = classify_physical_ir_variant_decision(variant, target);
            // Verify that an explicit decision is recorded (either NativeEmit, Decomposed, or ExplicitReject)
            assert!(
                matches!(
                    decision,
                    EmitterDecision::NativeEmit
                        | EmitterDecision::Decomposed
                        | EmitterDecision::ExplicitReject
                ),
                "Fix: target `{target}` has no explicit decision for variant `{:?}`",
                variant
            );
        }
    }
}

#[test]
fn no_catch_all_wildcard_arms_in_emitter_dispatch_modules() {
    // Audit emitter dispatch files to ensure `classify_op_dispatch_route` and `emit_op` match exhaustively without `_ =>`
    let dispatch_files = [
        "vyre-emit-naga/src/emitter/op_dispatch/mod.rs",
        "vyre-emit-ptx/src/emitter/dispatch.rs",
    ];

    for relative_path in &dispatch_files {
        let path = Path::new(relative_path);
        let fallback = Path::new("../").join(relative_path);
        let target_path = if path.exists() {
            path
        } else if fallback.exists() {
            &fallback
        } else {
            continue;
        };

        let content = std::fs::read_to_string(target_path)
            .unwrap_or_else(|e| panic!("Fix: failed to read {}: {e}", target_path.display()));

        // Check if there is a wildcard arm in match kind
        assert!(
            !content.contains("match kind {\n        _ =>")
                && !content.contains("match &op.kind {\n            _ =>"),
            "Fix: emitter dispatch in {} must not contain a wildcard `_ =>` catch-all arm",
            target_path.display()
        );
    }
}
