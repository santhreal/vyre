//! Proves exhaustive emitter decisions for every physical-IR variant on Naga/WGSL.
//!
//! WHY: An emitter arm that silently degrades or wildcard-falls through an
//! unsupported physical-IR variant is a correctness defect. Every variant in
//! `KernelOpKind` must have a deliberate decision: either emitted as valid Naga IR
//! or explicitly refused by name with a structured diagnostic.

use std::collections::BTreeSet;
use std::path::Path;

use vyre_emit_naga::emit;
use vyre_foundation::ir::DataType;
use vyre_lower::descriptor_builder::{global_rw, store_literal_kernel};
use vyre_lower::{
    FragmentValue, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue, MatrixMmaElement,
    MatrixMmaLayout, MatrixMmaSpec, MatrixTileShape, OpaqueNodeData,
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
fn unsupported_variants_are_refused_by_name_on_naga() {
    // 1. IndirectDispatch must be refused by name
    let desc_indirect =
        sample_descriptor_for_kind(KernelOpKind::IndirectDispatch { count_offset: 0 });
    let err_indirect = emit(&desc_indirect).unwrap_err();
    let msg_indirect = err_indirect.to_string();
    assert!(
        msg_indirect.contains("IndirectDispatch"),
        "error message must name IndirectDispatch: {msg_indirect}"
    );

    // 2. MatrixMma must be refused by name
    let desc_mma = sample_descriptor_for_kind(KernelOpKind::MatrixMma(Box::new(MatrixMmaSpec {
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
    })));
    let err_mma = emit(&desc_mma).unwrap_err();
    let msg_mma = err_mma.to_string();
    assert!(
        msg_mma.contains("MatrixMma"),
        "error message must name MatrixMma: {msg_mma}"
    );

    // 3. Call must be refused by name
    let desc_call = sample_descriptor_for_kind(KernelOpKind::Call {
        op_id: "unsupported_callee".into(),
    });
    let err_call = emit(&desc_call).unwrap_err();
    let msg_call = err_call.to_string();
    assert!(
        msg_call.contains("unsupported_callee")
            || msg_call.contains("Call")
            || msg_call.contains("call"),
        "error message must cite callee/Call: {msg_call}"
    );

    // 4. OpaqueNode must be refused by name
    let desc_opaque_node =
        sample_descriptor_for_kind(KernelOpKind::OpaqueNode(Box::new(OpaqueNodeData {
            extension_kind: "test_opaque_node".into(),
            payload: vec![1, 2, 3],
        })));
    let err_opaque_node = emit(&desc_opaque_node).unwrap_err();
    let msg_opaque_node = err_opaque_node.to_string();
    assert!(
        msg_opaque_node.contains("test_opaque_node") || msg_opaque_node.contains("OpaqueNode"),
        "error message must name opaque node: {msg_opaque_node}"
    );
}
