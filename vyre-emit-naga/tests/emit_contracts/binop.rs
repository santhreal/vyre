//! Test: BinOp literal folding semantics; also coerce_value_to_type with
//! extended-width type handles (VYRE-NAGA-001 regression).
use super::*;
use naga::{Expression, Literal};
use vyre_foundation::fp_parity::FloatLoweringMode;
use vyre_lower::descriptor_builder::{body, descriptor, lit, op};

/// BinOp::WrappingSub of two u32 literals must fold to wrapping_sub, not
/// saturating_sub. 0u32 WrappingSub 1u32 must produce 0xFFFF_FFFFu32, the
/// two's-complement wrap-around that the GPU would compute at runtime. The
/// previous saturating_sub produced Literal::U32(0), which is a silently
/// wrong constant (any downstream op using it would compute with the wrong
/// value without any error).
#[test]
fn fold_literal_wrapping_sub_u32_underflow_wraps() {
    let desc = descriptor("wrapping_sub_fold")
        .body(
            body()
                .ops([
                    // result 0: literal 0u32 (left operand)
                    lit(0, 0),
                    // result 1: literal 1u32 (right operand)
                    lit(1, 1),
                    // result 2: 0u32 WrappingSub 1u32, must fold to 0xFFFF_FFFF,
                    // NOT 0 (saturating). Operands are [left_result_id,
                    // right_result_id].
                    op(KernelOpKind::BinOpKind(BinOp::WrappingSub), [0, 1], 2),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(1)]),
        )
        .build();
    let module = emit(&desc).expect("WrappingSub of u32 literals must emit without error");

    // The fold must have produced Literal::U32(0xFFFF_FFFF) in the
    // expression arena. A saturating_sub would leave Literal::U32(0) here
    // instead, which is a silently wrong constant.
    let entry = &module.entry_points[0];
    let has_wrapping_result = entry
        .function
        .expressions
        .iter()
        .any(|(_, expr)| matches!(expr, Expression::Literal(Literal::U32(0xFFFF_FFFF))));
    assert!(
        has_wrapping_result,
        "fold_literal_binop WrappingSub(0u32, 1u32) must produce Literal::U32(0xFFFF_FFFF); \
         got saturating result (Literal::U32(0)) instead. GPU u32 subtraction wraps, not saturates"
    );
}

/// BinOp::Sub of two u32 literals with underflow must also wrap, because
/// WGSL u32 arithmetic is two's-complement at runtime; fold semantics must
/// match runtime semantics.
#[test]
fn fold_literal_plain_sub_u32_underflow_wraps() {
    let desc = descriptor("plain_sub_fold")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(KernelOpKind::BinOpKind(BinOp::Sub), [0, 1], 2),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(1)]),
        )
        .build();
    let module = emit(&desc).expect("Sub of u32 literals must emit without error");
    let entry = &module.entry_points[0];
    let has_wrapping_result = entry
        .function
        .expressions
        .iter()
        .any(|(_, expr)| matches!(expr, Expression::Literal(Literal::U32(0xFFFF_FFFF))));
    assert!(
        has_wrapping_result,
        "fold_literal_binop Sub(0u32, 1u32) must produce Literal::U32(0xFFFF_FFFF); \
         WGSL u32 subtraction wraps at runtime"
    );
}

/// VYRE-NAGA-001: `coerce_value_to_type` must correctly handle
/// extended-width type handles (u64_ty/i64_ty/f64_ty).
///
/// Before the fix the function's dispatch chain covered only four types
/// (bool/u32/i32/f32) and fell through with `return value` for any other
/// handle, including `u64_ty`. When a Select op used `u64_ty` as its
/// result type and one Select arm produced a Bool expression (e.g. from
/// a prior comparison), `coerce_value_to_type(bool_expr, u64_ty)` returned
/// the bool unchanged, leaving a type mismatch in the naga IR that either
/// caused a downstream naga validation failure or silently produced a
/// wrong shader.
///
/// The fix adds `u64_ty` (and `i64_ty`/`f64_ty`) to the dispatch chain so
/// they resolve to `ScalarKind::Uint`/`Sint`/`Float` and the subsequent
/// coercion branches fire correctly.
///
/// This test constructs a Select whose condition is a bool (thread-id == 0)
/// and whose accept/reject arms are both `vyre.literal.u64` values. The
/// Select result type is therefore `u64_ty`. With the old code the coerce
/// calls on lines 477-478 would fall to `return value` and leave the arms
/// typed as u64 (which is fine here, the test proves the emit succeeds);
/// the real regression is when one arm is a non-u64 expression. We test the
/// emit succeeds end-to-end AND that `Literal::U64` values appear, proving
/// the wide-literal path ran and the coerce didn't panic or bail out.
#[test]
fn select_with_u64_arms_emits_without_coerce_passthrough_panic() {
    // Descriptor:
    //   result 0: vyre.literal.u64(100)
    //   result 1: vyre.literal.u64(200)
    //   result 2: Literal U32(0) for LocalInvocationId comparison
    //   result 3: LocalInvocationId (thread id)
    //   result 4: BinOp::Eq on result 3 and result 2 → Bool
    //   result 5: Select(condition=result4, accept=result0, reject=result1)
    //: forces coerce_value_to_type(u64_expr, u64_ty) on both arms
    //             (after the fix, u64_ty maps to ScalarKind::Uint; before the
    //             fix it fell to `return value` which is harmless for matching
    //             kinds but would panic for mismatched kinds, testing this
    //             shape proves the dispatch path is reachable without panicking)
    let desc = descriptor("select_u64_arms")
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    // result 0: vyre.literal.u64(100)
                    op(
                        KernelOpKind::OpaqueExpr(Box::new(vyre_lower::OpaqueExprData {
                            extension_id: 10,
                            extension_kind: "vyre.literal.u64".to_owned(),
                            payload: 100u64.to_le_bytes().to_vec(),
                        })),
                        [],
                        0,
                    ),
                    // result 1: vyre.literal.u64(200)
                    op(
                        KernelOpKind::OpaqueExpr(Box::new(vyre_lower::OpaqueExprData {
                            extension_id: 11,
                            extension_kind: "vyre.literal.u64".to_owned(),
                            payload: 200u64.to_le_bytes().to_vec(),
                        })),
                        [],
                        1,
                    ),
                    // result 2: Literal U32(0) for comparison
                    lit(0, 2),
                    // result 3: LocalInvocationId (U32)
                    op(KernelOpKind::LocalInvocationId, [0], 3),
                    // result 4: thread_id == 0  →  Bool
                    op(KernelOpKind::BinOpKind(BinOp::Eq), [3, 2], 4),
                    // result 5: Select(cond=Bool, accept=u64(100), reject=u64(200))
                    // coerce_value_to_type(u64_expr, u64_ty) must not panic or fall through.
                    // Before the fix: else branch returned value unchanged.
                    // After the fix: u64_ty → ScalarKind::Uint; actual==target → identity.
                    op(KernelOpKind::Select, [4, 0, 1], 5),
                ])
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let module = emit(&desc).expect(
        "Select with u64-typed arms must emit without error; \
                 coerce_value_to_type must recognise u64_ty as ScalarKind::Uint",
    );

    // Verify both wide literals appear in the expression arena, proves the
    // u64 emit path ran end-to-end and was not short-circuited.
    let entry = &module.entry_points[0];
    let has_u64_100 = entry
        .function
        .expressions
        .iter()
        .any(|(_, expr)| matches!(expr, Expression::Literal(Literal::U64(100))));
    let has_u64_200 = entry
        .function
        .expressions
        .iter()
        .any(|(_, expr)| matches!(expr, Expression::Literal(Literal::U64(200))));
    assert!(
        has_u64_100,
        "Literal::U64(100) must appear in the emitted expression arena"
    );
    assert!(
        has_u64_200,
        "Literal::U64(200) must appear in the emitted expression arena"
    );
    // The Select must be present.
    let has_select = entry
        .function
        .expressions
        .iter()
        .any(|(_, expr)| matches!(expr, Expression::Select { .. }));
    assert!(
        has_select,
        "Expression::Select must appear in the emitted expression arena"
    );
}

/// A descriptor computing `buf[0] * buf[0] + -1.0` into `buf[0]`.
///
/// The operands are loads rather than literals so constant folding cannot
/// evaluate the multiply away: a folded product carries no rounding for the
/// strict mode to publish, and the emitted module would be identical under both
/// modes for a reason unrelated to the barrier.
fn multiply_add_f32_descriptor() -> KernelDescriptor {
    use vyre_lower::descriptor_builder::{binop, load_global, store_global};

    descriptor("f32_multiply_add")
        .slot(global_rw(0, DataType::F32, "buf").with_count(8))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::F32(-1.0)])
                .ops([
                    lit(0, 0),
                    load_global(0, 0, 1),
                    binop(BinOp::Mul, 1, 1, 2),
                    lit(1, 3),
                    binop(BinOp::Add, 2, 3, 4),
                    store_global(0, 0, 4),
                ]),
        )
        .build()
}

/// Count the `f32 -> u32 -> f32` reinterpretation pairs in an entry point.
///
/// A single `As` proves nothing: the emitter uses integer reinterpretation for
/// bitcast ops and for the 64-bit lowering too. The barrier is the PAIR, and
/// only a pair whose inner expression is the outer one's operand republishes a
/// value rather than converting it.
fn round_barrier_pairs(module: &naga::Module) -> usize {
    let expressions = &module.entry_points[0].function.expressions;
    expressions
        .iter()
        .filter(|(_, expr)| {
            let Expression::As {
                expr: inner,
                kind: naga::ScalarKind::Float,
                convert: None,
            } = expr
            else {
                return false;
            };
            matches!(
                expressions.try_get(*inner),
                Ok(Expression::As {
                    kind: naga::ScalarKind::Uint,
                    convert: None,
                    ..
                })
            )
        })
        .count()
}

/// The strict mode emits the rounding barrier and the default mode does not.
///
/// `FloatLoweringMode::StrictIeee` is answered by publishing every f32 product
/// through a `u32` reinterpretation, which leaves the following add no unrounded
/// product to absorb. Nothing asserted that the barrier reached the module:
/// `honors_float_lowering` returns `true` for both modes on every backend that
/// routes through this emitter, so a mode that silently emitted the contracted
/// module would be reported as honored and would answer a strict dispatch with
/// contracted arithmetic. The device suites that would have caught it compare
/// against an oracle, so they fail for a rounding reason and cannot say whether
/// the barrier was absent or merely ineffective. This separates the two.
#[test]
fn the_strict_mode_publishes_the_product_and_the_default_mode_does_not() {
    let desc = multiply_add_f32_descriptor();

    let contracted = emit_with_float_mode(&desc, FloatLoweringMode::Contracted)
        .expect("the contracted mode must emit a multiply-add module");
    let strict = emit_with_float_mode(&desc, FloatLoweringMode::StrictIeee)
        .expect("the strict mode must emit a multiply-add module");

    assert_valid_wgsl(&contracted, "contracted f32 multiply-add");
    assert_valid_wgsl(&strict, "strict f32 multiply-add");

    assert_eq!(
        round_barrier_pairs(&contracted),
        0,
        "the default mode admits contraction, so it must not pay for a reinterpretation pair"
    );
    assert_eq!(
        round_barrier_pairs(&strict),
        1,
        "the strict mode must publish the one f32 product through a u32 reinterpretation; \
         without it the target is free to fuse the multiply and the add into one rounding"
    );

    // The pair alone is a value identity, and a target compiler that folds it
    // recovers the single expression every contraction rule permits fusing.
    // Naming the result is what makes the product a statement the following add
    // reads, so the name is part of the contract rather than a readability aid.
    let named: Vec<&str> = strict.entry_points[0]
        .function
        .named_expressions
        .values()
        .map(String::as_str)
        .collect();
    assert!(
        named.iter().any(|name| name.starts_with("vyre_rounded_")),
        "the strict mode must name the published product so the writer emits it as its own \
         `let`; named expressions were {named:?}"
    );
    assert!(
        contracted.entry_points[0]
            .function
            .named_expressions
            .values()
            .all(|name| !name.starts_with("vyre_rounded_")),
        "the default mode must not name a published product, because it publishes none"
    );
}

/// The two modes are different modules, so a cache key over one is not the other.
///
/// Bit-identical modules under the two modes would make every cache-key test in
/// the driver vacuous: the key would separate two entries holding the same
/// artifact and a strict dispatch would be served contracted arithmetic no
/// matter how carefully the key was built.
#[test]
fn the_two_float_modes_do_not_emit_the_same_module() {
    let desc = multiply_add_f32_descriptor();
    let contracted = emit_with_float_mode(&desc, FloatLoweringMode::Contracted)
        .expect("the contracted mode must emit a multiply-add module");
    let strict = emit_with_float_mode(&desc, FloatLoweringMode::StrictIeee)
        .expect("the strict mode must emit a multiply-add module");
    assert_ne!(
        contracted.entry_points[0].function.expressions.len(),
        strict.entry_points[0].function.expressions.len(),
        "the strict module must carry expressions the contracted one does not"
    );
}

/// `emit` is the contracted mode, and every mode is covered by one of the two.
///
/// Derived from `FloatLoweringMode::EVERY` rather than from the pair that exists
/// today: a third mode added to the enum turns this red until someone states
/// whether it emits the barrier.
#[test]
fn every_float_mode_states_whether_it_emits_the_barrier() {
    let desc = multiply_add_f32_descriptor();
    for &mode in FloatLoweringMode::EVERY {
        let module = emit_with_float_mode(&desc, mode).unwrap_or_else(|error| {
            panic!("mode {mode:?} must emit the multiply-add module: {error:?}")
        });
        let pairs = round_barrier_pairs(&module);
        let expected = usize::from(mode.blocks_contraction());
        assert_eq!(
            pairs, expected,
            "mode {mode:?} reports blocks_contraction() == {} and emitted {pairs} barrier pair(s). \
             A mode that blocks contraction must publish the product; one that does not must not \
             pay for the pair.",
            mode.blocks_contraction()
        );
    }
}
