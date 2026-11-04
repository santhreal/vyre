//! Dispatch diagnostics. Every message an op route can refuse with is written
//! once here so the same defect never reads two ways.

use std::fmt::Write as _;

use vyre_foundation::ir::DataType;
use vyre_lower::KernelOpKind;

pub(super) fn missing_literal_pool_index_message(literal_index: u32) -> String {
    let mut message: String = Default::default();
    message.push_str("literal op references missing literal-pool index ");
    let _ = write!(&mut message, "{literal_index}");
    message
}

pub(super) fn missing_binding_slot_message(kind: &KernelOpKind) -> String {
    let mut message: String = Default::default();
    let _ = write!(&mut message, "{kind:?} missing binding slot");
    message
}

pub(super) fn non_byte_load_route_message(data_type: DataType) -> String {
    let mut message: String = Default::default();
    message.push_str("emit_byte_element_load called with non-byte DataType ");
    let _ = write!(&mut message, "{data_type:?}");
    message.push_str("; this is an emitter routing bug");
    message
}

pub(super) fn call_reached_message(op_id: &str) -> String {
    let mut message: String = Default::default();
    message.push_str("Call op `");
    message.push_str(op_id);
    message.push_str(
        "` reached descriptor Naga emission. Fix: expand calls into KernelDescriptor ops before emission.",
    );
    message
}

pub(super) fn opaque_node_message(extension_kind: &str, payload_len: usize) -> String {
    payload_message(
        "opaque node `",
        extension_kind,
        payload_len,
        "` with ",
        " payload bytes has no descriptor Naga lowering. Fix: lower this extension into concrete KernelDescriptor ops before descriptor emission.",
    )
}

pub(super) fn wide_literal_payload_message(extension_kind: &str, payload_len: usize) -> String {
    payload_message(
        "wide-literal opaque `",
        extension_kind,
        payload_len,
        "` carries ",
        " payload bytes, expected 8. Fix: encode literals through Expr::u64/i64/f64 builders.",
    )
}

fn payload_message(
    prefix: &str,
    extension_kind: &str,
    payload_len: usize,
    count_prefix: &str,
    suffix: &str,
) -> String {
    let mut message = String::new();
    message.push_str(prefix);
    message.push_str(extension_kind);
    message.push_str(count_prefix);
    let _ = write!(&mut message, "{payload_len}");
    message.push_str(suffix);
    message
}

pub(super) fn wide_literal_kind_gate_message(kind: &str) -> String {
    let mut message: String = Default::default();
    message.push_str("wide-literal kind `");
    message.push_str(kind);
    message.push_str(
        "` reached descriptor opaque emission after the kind gate. Fix: update the kind gate and decoder together.",
    );
    message
}

pub(super) fn opaque_expression_message(extension_kind: &str, extension_id: u32) -> String {
    let mut message: String = Default::default();
    message.push_str("opaque expression `");
    message.push_str(extension_kind);
    message.push_str("` (id=");
    let _ = write!(&mut message, "{extension_id:#010x}");
    message.push_str(
        ") has no descriptor Naga lowering. Fix: lower this extension into concrete KernelDescriptor ops or add a descriptor extension emitter before Naga emission.",
    );
    message
}
