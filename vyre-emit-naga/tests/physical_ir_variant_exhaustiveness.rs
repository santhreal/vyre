//! Proves exhaustive emitter decisions for every physical-IR variant on Naga/WGSL.
//!
//! WHY: An emitter arm that silently degrades or wildcard-falls through an
//! unsupported physical-IR variant is a correctness defect. Every variant in
//! `KernelOpKind` must have a deliberate decision: either emitted as valid Naga IR
//! or explicitly refused by name with a structured diagnostic.
//!
//! The variant space and the probe descriptor belong to `vyre-lower`, which
//! declares the enum. What stays here is the Naga answer for each refused
//! variant, which is this crate's alone.

use vyre_emit_naga::emit;
use vyre_lower::descriptor_builder::{mma_f16_m16n8k16, probe_kernel};
use vyre_lower::{KernelOpKind, OpaqueNodeData};

#[test]
fn unsupported_variants_are_refused_by_name_on_naga() {
    // 1. IndirectDispatch must be refused by name
    let desc_indirect = probe_kernel(KernelOpKind::IndirectDispatch { count_offset: 0 });
    let err_indirect = emit(&desc_indirect).unwrap_err();
    let msg_indirect = err_indirect.to_string();
    assert!(
        msg_indirect.contains("IndirectDispatch"),
        "error message must name IndirectDispatch: {msg_indirect}"
    );

    // 2. MatrixMma must be refused by name
    let desc_mma = probe_kernel(mma_f16_m16n8k16());
    let err_mma = emit(&desc_mma).unwrap_err();
    let msg_mma = err_mma.to_string();
    assert!(
        msg_mma.contains("MatrixMma"),
        "error message must name MatrixMma: {msg_mma}"
    );

    // 3. Call must be refused by name
    let desc_call = probe_kernel(KernelOpKind::Call {
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
    let desc_opaque_node = probe_kernel(KernelOpKind::OpaqueNode(Box::new(OpaqueNodeData {
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
