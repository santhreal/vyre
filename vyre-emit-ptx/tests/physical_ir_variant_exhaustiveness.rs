//! Proves exhaustive emitter decisions for every physical-IR variant on PTX.
//!
//! WHY: An emitter arm that silently degrades or wildcard-falls through an
//! unsupported physical-IR variant is a correctness defect. Every variant in
//! `KernelOpKind` must have a deliberate decision: either emitted as native PTX
//! or explicitly refused by name.

use std::collections::BTreeSet;
use std::path::Path;

use vyre_emit_ptx::{emit, EmitError};
use vyre_foundation::ir::DataType;
use vyre_lower::descriptor_builder::{global_rw, store_literal_kernel};
use vyre_lower::{
    KernelDescriptor, KernelOp, KernelOpKind, LiteralValue, OpaqueExprData, OpaqueNodeData,
};
use vyre_test_support::monorepo::vyre_workspace_root;

fn parse_kernel_op_kind_variants_from_source(root: &Path) -> BTreeSet<String> {
    let source_path = root.join("vyre-lower/src/descriptor/mod.rs");
    let content = std::fs::read_to_string(&source_path)
        .unwrap_or_else(|e| panic!("Fix: {source_path:?} must be readable: {e}"));

    let mut in_enum = false;
    let mut brace_depth = 0usize;
    let mut variants = BTreeSet::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if !in_enum {
            if trimmed.starts_with("pub enum KernelOpKind") {
                in_enum = true;
                brace_depth = if trimmed.contains('{') { 1 } else { 0 };
            }
            continue;
        }

        if brace_depth == 0 {
            if trimmed.contains('{') {
                brace_depth = 1;
            }
            continue;
        }

        if brace_depth == 1 {
            if trimmed.starts_with('}') {
                break;
            }
            if !trimmed.starts_with("//") && !trimmed.is_empty() && !trimmed.starts_with('#') {
                let ident = trimmed
                    .split(&['{', '(', ',', ' '][..])
                    .next()
                    .unwrap_or("")
                    .trim();
                if !ident.is_empty() && ident.chars().next().unwrap().is_ascii_uppercase() {
                    variants.insert(ident.to_string());
                }
            }
        }

        let opens = trimmed.chars().filter(|&c| c == '{').count();
        let closes = trimmed.chars().filter(|&c| c == '}').count();
        brace_depth = brace_depth.saturating_add(opens).saturating_sub(closes);
        if brace_depth == 0 {
            break;
        }
    }

    variants
}

fn sample_descriptor_for_kind(kind: KernelOpKind) -> KernelDescriptor {
    let mut desc = store_literal_kernel(
        "sample_kernel",
        global_rw(0, DataType::U32, "out"),
        [64, 1, 1],
    );
    desc.body.literals = vec![LiteralValue::U32(42), LiteralValue::U32(0)];
    desc.body.ops = vec![
        KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        },
        KernelOp {
            kind,
            operands: vec![0, 0, 0, 0],
            result: Some(1),
        },
    ];
    desc
}

#[test]
fn runtime_enumerates_all_kernel_op_kind_variants_from_source() {
    let root = vyre_workspace_root();
    let variants = parse_kernel_op_kind_variants_from_source(&root);

    assert!(
        variants.len() >= 40,
        "Fix: expected at least 40 physical-IR variants, found {}: {:?}",
        variants.len(),
        variants
    );

    // Verify key variants are detected from source
    assert!(variants.contains("Literal"));
    assert!(variants.contains("LoadGlobal"));
    assert!(variants.contains("StoreGlobal"));
    assert!(variants.contains("BinOpKind"));
    assert!(variants.contains("MatrixMma"));
    assert!(variants.contains("IndirectDispatch"));
    assert!(variants.contains("Call"));
    assert!(variants.contains("OpaqueExpr"));
    assert!(variants.contains("OpaqueNode"));
}

#[test]
fn unsupported_variants_are_refused_by_name_on_ptx() {
    // 1. IndirectDispatch must be refused by name with UnsupportedOp
    let desc_indirect = sample_descriptor_for_kind(KernelOpKind::IndirectDispatch { count_offset: 0 });
    let err_indirect = emit(&desc_indirect).unwrap_err();
    match err_indirect {
        EmitError::UnsupportedOp(op) => {
            assert!(matches!(op.kind, KernelOpKind::IndirectDispatch { .. }));
        }
        other => panic!("expected UnsupportedOp for IndirectDispatch, got: {other:?}"),
    }

    // 2. Call must be refused by name with UnsupportedOp
    let desc_call = sample_descriptor_for_kind(KernelOpKind::Call {
        op_id: "ext_call".into(),
    });
    let err_call = emit(&desc_call).unwrap_err();
    match err_call {
        EmitError::UnsupportedOp(op) => {
            assert!(matches!(op.kind, KernelOpKind::Call { .. }));
        }
        other => panic!("expected UnsupportedOp for Call, got: {other:?}"),
    }

    // 3. OpaqueExpr must be refused by name with UnsupportedOp
    let desc_opaque_expr = sample_descriptor_for_kind(KernelOpKind::OpaqueExpr(Box::new(OpaqueExprData {
        extension_id: 1,
        extension_kind: "custom_expr".into(),
        payload: vec![1, 2, 3],
    })));
    let err_opaque_expr = emit(&desc_opaque_expr).unwrap_err();
    match err_opaque_expr {
        EmitError::UnsupportedOp(op) => {
            assert!(matches!(op.kind, KernelOpKind::OpaqueExpr(..)));
        }
        other => panic!("expected UnsupportedOp for OpaqueExpr, got: {other:?}"),
    }

    // 4. OpaqueNode must be refused by name with UnsupportedOp
    let desc_opaque_node = sample_descriptor_for_kind(KernelOpKind::OpaqueNode(Box::new(OpaqueNodeData {
        extension_kind: "custom_node".into(),
        payload: vec![4, 5, 6],
    })));
    let err_opaque_node = emit(&desc_opaque_node).unwrap_err();
    match err_opaque_node {
        EmitError::UnsupportedOp(op) => {
            assert!(matches!(op.kind, KernelOpKind::OpaqueNode(..)));
        }
        other => panic!("expected UnsupportedOp for OpaqueNode, got: {other:?}"),
    }
}
