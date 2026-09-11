//! Quantized datatype contracts for the reference oracle.
//!
//! The spec exposes INT4/FP4/NF4/FP8 datatypes for GPU inference paths. The
//! CPU oracle must preserve their fixed-width storage bytes exactly. A load
//! past the buffer is refused under the strict default; diagnostic permissive
//! mode counts the absorbed load instead of refusing it, and the payload it
//! absorbs keeps the element's storage width rather than degrading to empty
//! `Bytes`.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::{value::Value, ReferenceErrorClass, ReferenceRequest};

fn load_store_program(ty: DataType, index: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, ty.clone()).with_count(1),
            BufferDecl::output("out", 1, ty).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("input", Expr::u32(index)),
        )],
    )
}

fn run_single_load_store(ty: DataType, input: Vec<u8>, index: u32) -> Vec<u8> {
    let outputs = vyre_reference::ReferenceRequest::standard(
        &load_store_program(ty, index),
        &[Value::Bytes(input.into())],
    )
    .outputs()
    .expect("quantized load/store oracle program must execute");
    outputs[0].to_bytes()
}

/// Diagnostic permissive mode counts an out-of-bounds load instead of
/// refusing it.
///
/// The absorbed payload is not reachable from here, and that is the point:
/// permissive mode issues no output value a device could be graded against.
/// The width of that payload is a property of [`Value::try_zero_for`] and is
/// asserted directly against it below.
fn absorbed_oob_load_count(ty: DataType, input: Vec<u8>) -> u64 {
    let program = load_store_program(ty, 99);
    let inputs = [Value::Bytes(input.into())];
    let report = ReferenceRequest::standard(&program, &inputs)
        .execute_permissive()
        .expect("Fix: diagnostic mode must absorb an out-of-bounds load rather than refuse it.");
    report.oob_report.oob_loads
}

/// Under the strict default a load past the buffer is refused for every
/// quantized width. These cases used to assert the absorbed typed zero, which
/// is a value the device never produces.
#[test]
fn quantized_out_of_bounds_load_refuses_under_the_strict_default() {
    for ty in [
        DataType::I4,
        DataType::FP4,
        DataType::NF4,
        DataType::F8E4M3,
        DataType::F8E5M2,
        DataType::F16,
        DataType::BF16,
        DataType::I16,
        DataType::U16,
    ] {
        let error = vyre_reference::ReferenceRequest::standard(
            &load_store_program(ty.clone(), 99),
            &[Value::Bytes(vec![0xFF, 0xFF].into())],
        )
        .outputs()
        .expect_err("Fix: a quantized load past the buffer must be refused.");
        assert_eq!(
            error.error_class(),
            ReferenceErrorClass::OutOfBoundsAccess,
            "{ty} out-of-bounds load must refuse as an out-of-bounds access, got {error:?}"
        );
    }
}

#[test]
fn quantized_scalar_load_store_preserves_raw_storage_bits() {
    for (ty, encoded) in [
        (DataType::I4, vec![0x0F]),
        (DataType::FP4, vec![0x06]),
        (DataType::NF4, vec![0x08]),
        (DataType::F8E4M3, vec![0x7F]),
        (DataType::F8E5M2, vec![0x7B]),
    ] {
        let out = run_single_load_store(ty.clone(), encoded.clone(), 0);
        assert_eq!(
            out.len(),
            encoded.len(),
            "{ty} output length must match input"
        );
        assert_eq!(out, encoded, "{ty} in-bounds load/store must be byte-exact");
    }
}

/// The absorbed payload of a one-byte quantized element keeps one byte.
#[test]
fn absorbed_quantized_scalar_load_keeps_its_one_byte_width() {
    for ty in [
        DataType::I4,
        DataType::FP4,
        DataType::NF4,
        DataType::F8E4M3,
        DataType::F8E5M2,
    ] {
        assert!(
            absorbed_oob_load_count(ty.clone(), vec![0xFF]) > 0,
            "{ty} absorbed load must be counted, or this measures an in-bounds load"
        );
        assert_eq!(
            Value::try_zero_for(ty.clone())
                .expect("Fix: the declared element type must have a defined zero")
                .to_bytes(),
            vec![0],
            "{ty} absorbed load must keep a one-byte typed zero, not empty Bytes"
        );
    }
}

/// The absorbed payload of a two-byte element keeps two bytes.
#[test]
fn absorbed_half_and_bfloat_loads_keep_their_two_byte_width() {
    for ty in [DataType::F16, DataType::BF16, DataType::I16, DataType::U16] {
        assert!(
            absorbed_oob_load_count(ty.clone(), vec![0xFF, 0xFF]) > 0,
            "{ty} absorbed load must be counted, or this measures an in-bounds load"
        );
        assert_eq!(
            Value::try_zero_for(ty.clone())
                .expect("Fix: the declared element type must have a defined zero")
                .to_bytes(),
            vec![0, 0],
            "{ty} absorbed load must preserve its two-byte storage shape"
        );
    }
}

#[test]
fn packed_i4_reference_buffer_len_reports_logical_elements() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::I4).with_count(8),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::buf_len("input"))],
    );

    let outputs =
        vyre_reference::ReferenceRequest::standard(&program, &[Value::Bytes(vec![0u8; 4].into())])
            .outputs()
            .expect("Fix: packed I4 buffer length oracle must execute.");

    assert_eq!(
        outputs[0].to_bytes(),
        8u32.to_le_bytes(),
        "Fix: four bytes of I4 storage must report eight logical elements to Expr::buf_len."
    );
}
