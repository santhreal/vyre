use super::*;

#[test]
fn subnormal_sqrt_sin_cos_produce_canonical_results() {
    let pos_sub = f32::from_bits(0x0000_0001);
    let neg_sub = f32::from_bits(0x8000_0001);

    assert_eq!(
        float_bits(eval_expr_value(&Expr::UnOp {
            op: UnOp::Sqrt,
            operand: Box::new(Expr::f32(pos_sub)),
        })),
        0x0000_0000
    );
    assert_eq!(
        float_bits(eval_expr_value(&Expr::UnOp {
            op: UnOp::Sqrt,
            operand: Box::new(Expr::f32(neg_sub)),
        })),
        0x8000_0000
    );
    assert_eq!(
        float_bits(eval_expr_value(&Expr::UnOp {
            op: UnOp::Sin,
            operand: Box::new(Expr::f32(pos_sub)),
        })),
        0x0000_0000
    );
    assert_eq!(
        float_bits(eval_expr_value(&Expr::UnOp {
            op: UnOp::Sin,
            operand: Box::new(Expr::f32(neg_sub)),
        })),
        0x8000_0000
    );
    assert_eq!(
        float_bits(eval_expr_value(&Expr::UnOp {
            op: UnOp::Cos,
            operand: Box::new(Expr::f32(pos_sub)),
        })),
        1.0f32.to_bits()
    );
    assert_eq!(
        float_bits(eval_expr_value(&Expr::UnOp {
            op: UnOp::Cos,
            operand: Box::new(Expr::f32(neg_sub)),
        })),
        1.0f32.to_bits()
    );
}

// ---------------------------------------------------------------------------
// 3. Atomic ops
// ---------------------------------------------------------------------------

/// An atomic past the buffer refuses instead of returning an old value of
/// zero. A device performs no bounds check, so the zero this used to hand
/// back was an answer no backend produces.
#[test]
fn atomic_at_an_out_of_bounds_index_refuses() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("buf", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        Vec::new(),
    );
    let mut memory =
        ReferenceMemory::empty().with_storage("buf", Buffer::new(vec![0xAB; 4], DataType::U32));
    let error = reference_eval_expr(
        &program,
        &mut memory,
        InvocationIds::ZERO,
        &Expr::atomic_add("buf", Expr::u32(999), Expr::u32(1)),
    )
    .expect_err("Fix: an atomic past the buffer must be refused, not absorbed.");
    assert_eq!(
        error.error_class(),
        vyre_reference::ReferenceErrorClass::OutOfBoundsAccess,
        "Fix: an out-of-bounds atomic must refuse as an out-of-bounds access, got {error:?}."
    );
}

#[test]
fn atomic_on_u64_buffer_touches_lower_half_only() {
    // The interpreter treats atomics as 4-byte ops regardless of declared element type.
    // This test documents that gap: an atomic add on a U64 buffer only modifies the
    // low 32 bits of each 64-bit slot.
    let program = Program::wrapped(
        vec![BufferDecl::read_write("buf", 0, DataType::U64).with_count(1)],
        [1, 1, 1],
        Vec::new(),
    );
    let mut memory = ReferenceMemory::empty().with_storage(
        "buf",
        Buffer::new(
            0x0000_0001_0000_0000u64.to_le_bytes().to_vec(),
            DataType::U64,
        ),
    );
    let old = reference_eval_expr(&program, &mut memory, InvocationIds::ZERO, &Expr::atomic_add("buf", Expr::u32(0), Expr::u32(1)))
    .expect("Fix: atomic on U64 buffer must evaluate");
    // old value read as low 32 bits
    assert_eq!(old, Value::U32(0));

    let loaded = reference_eval_expr(&program, &mut memory, InvocationIds::ZERO, &Expr::load("buf", Expr::u32(0)))
    .expect("Fix: load after atomic must succeed");
    // U64 value should now be 0x0000_0001_0000_0001
    assert_eq!(
        loaded,
        Value::U64(0x0000_0001_0000_0001),
        "atomic add on U64 must only touch lower 32 bits"
    );
}

#[test]
fn multiple_atomics_on_same_location_are_deterministic() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("buf", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        Vec::new(),
    );
    let mut memory = ReferenceMemory::empty().with_storage("buf", Buffer::new(vec![0; 4], DataType::U32));

    let first = reference_eval_expr(&program, &mut memory, InvocationIds::ZERO, &Expr::atomic_add("buf", Expr::u32(0), Expr::u32(1)))
    .unwrap();
    let second = reference_eval_expr(&program, &mut memory, InvocationIds::ZERO, &Expr::atomic_add("buf", Expr::u32(0), Expr::u32(1)))
    .unwrap();

    assert_eq!(first, Value::U32(0), "first atomic must see old=0");
    assert_eq!(second, Value::U32(1), "second atomic must see old=1");

    let final_val = reference_eval_expr(&program, &mut memory, InvocationIds::ZERO, &Expr::load("buf", Expr::u32(0)))
    .unwrap();
    assert_eq!(final_val, Value::U32(2));
}

// ---------------------------------------------------------------------------
// 4. Buffer access
// ---------------------------------------------------------------------------

fn read_and_output_prog(in_count: u32, node: Node) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::U32).with_count(in_count),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![node],
    )
}

/// Every out-of-bounds access refuses at the access site.
///
/// These cases used to assert the absorbed answer: a load returned a typed
/// zero, a store vanished, an atomic returned an old value of zero. The
/// interpreter is the parity oracle, and a device does no bounds checking, so
/// an absorbed access certified a result no backend can reproduce. Strict
/// refusal is now the default on every entry point and each case names the
/// buffer and the index instead of inventing a value.
#[test]
fn out_of_bounds_load_refuses_instead_of_returning_a_typed_zero() {
    let program = read_and_output_prog(
        1,
        Node::store("out", Expr::u32(0), Expr::load("in", Expr::u32(999))),
    );
    assert_out_of_bounds(
        reference_eval(&program, &[Value::from(vec![0xAB; 4])]),
        "a load past the buffer",
    );
}

/// A store past the buffer refuses rather than vanishing.
///
/// Validation rejects a constant out-of-bounds index, so the index is loaded
/// from a buffer to force runtime evaluation.
#[test]
fn out_of_bounds_store_refuses_instead_of_vanishing() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("idx", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::load("idx", Expr::u32(0)),
            Expr::u32(0xDEAD_BEEF),
        )],
    );
    assert_out_of_bounds(
        reference_eval(&program, &[Value::from(999u32.to_le_bytes().to_vec())]),
        "a store past the buffer",
    );
}

/// A buffer with no elements has no element 0 to load.
#[test]
fn load_from_a_zero_sized_buffer_refuses() {
    let program = read_and_output_prog(
        0,
        Node::store("out", Expr::u32(0), Expr::load("in", Expr::u32(0))),
    );
    assert_out_of_bounds(
        reference_eval(&program, &[Value::from(vec![])]),
        "a load from a zero-sized buffer",
    );
}

/// An explicitly empty readback range allocates no bytes, so it holds no
/// element to store into and yields empty output bytes.
#[test]
fn an_empty_output_range_holds_no_element_and_yields_no_bytes() {
    let decls =
        || vec![BufferDecl::output("out", 0, DataType::U32).with_output_byte_range(0usize..0usize)];

    assert_out_of_bounds(
        reference_eval(
            &Program::wrapped(
                decls(),
                [1, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(0xDEAD_BEEF))],
            ),
            &[],
        ),
        "a store into an empty output range",
    );

    let outputs = reference_eval(&Program::wrapped(decls(), [1, 1, 1], Vec::new()), &[])
        .expect("Fix: an empty output range must still be collected as an output.");
    assert_eq!(
        outputs.len(),
        1,
        "Fix: a zero-sized output buffer is still a declared output."
    );
    assert_eq!(
        outputs[0].to_bytes(),
        Vec::<u8>::new(),
        "Fix: a zero-sized output must yield empty bytes."
    );
}

/// An index whose byte offset overflows is out of bounds, not a zero.
#[test]
fn u32_max_index_load_refuses() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("in", Expr::u32(u32::MAX)),
        )],
    );
    assert_out_of_bounds(
        reference_eval(&program, &[Value::from(vec![0xAB; 4])]),
        "a load at an index whose byte offset overflows",
    );
}

fn assert_out_of_bounds(result: Result<Vec<Value>, vyre_reference::ReferenceError>, what: &str) {
    let error = result
        .err()
        .unwrap_or_else(|| panic!("Fix: {what} must be refused, not absorbed into an output."));
    assert_eq!(
        error.error_class(),
        vyre_reference::ReferenceErrorClass::OutOfBoundsAccess,
        "Fix: {what} must refuse as an out-of-bounds access, got {error:?}."
    );
}
