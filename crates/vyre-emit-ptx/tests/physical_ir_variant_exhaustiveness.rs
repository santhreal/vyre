//! Proves exhaustive emitter decisions for every physical-IR variant on PTX.
//!
//! WHY: An emitter arm that silently degrades or wildcard-falls through an
//! unsupported physical-IR variant is a correctness defect. Every variant in
//! `KernelOpKind` must have a deliberate decision: either emitted as native PTX
//! or explicitly refused by name.
//!
//! The variant space and the probe descriptor belong to `vyre-lower`, which
//! declares the enum. What stays here is the PTX answer for each refused
//! variant, which is this crate's alone.

use vyre_emit_ptx::{emit, EmitError};
use vyre_lower::descriptor_builder::probe_kernel;
use vyre_lower::{KernelOpKind, OpaqueExprData, OpaqueNodeData};

#[test]
fn unsupported_variants_are_refused_by_name_on_ptx() {
    // 1. IndirectDispatch must be refused by name with UnsupportedOp
    let desc_indirect = probe_kernel(KernelOpKind::IndirectDispatch { count_offset: 0 });
    let err_indirect = emit(&desc_indirect).unwrap_err();
    match err_indirect {
        EmitError::UnsupportedOp(op) => {
            assert!(matches!(op.kind, KernelOpKind::IndirectDispatch { .. }));
        }
        other => panic!("expected UnsupportedOp for IndirectDispatch, got: {other:?}"),
    }

    // 2. Call must be refused by name with UnsupportedOp
    let desc_call = probe_kernel(KernelOpKind::Call {
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
    let desc_opaque_expr = probe_kernel(KernelOpKind::OpaqueExpr(Box::new(OpaqueExprData {
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
    let desc_opaque_node = probe_kernel(KernelOpKind::OpaqueNode(Box::new(OpaqueNodeData {
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
