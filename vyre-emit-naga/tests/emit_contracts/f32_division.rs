//! Test: an f32 quotient is emitted as the correctly-rounded one.
//!
//! WHY: `BinaryOperator::Divide` states IEEE-754 division in the IR and in the
//! reference oracle, and the PTX emitter selects `div.rn.f32`, but Vulkan
//! requires `OpFDiv` only to land within 2.5 ULP. A target answers it with a
//! reciprocal estimate and a refinement that stops one step short of the final
//! rounding: measured through the wgpu backend on an NVIDIA adapter, five of
//! eight f32 quotients came back one ULP from the correctly-rounded value,
//! `36.0 / 200.0` among them. One ULP passes the elementary window on its own
//! and does not survive composition, so `tensor_train_decompose` read a Gram
//! matrix 164 ULP from the oracle and failed conformance.
//!
//! The class this closes is every operator whose all-f32 lowering reaches a
//! `Divide`. The variant space is the frozen builtin operator tables read at
//! run time rather than a list written here, so an operator added to
//! `vyre-spec` that lowers to an uncorrected f32 quotient turns this red.
//!
//! What it does not catch: whether the target's own shader compiler keeps the
//! two `Fma` expressions fused. The refinement is exact only under a single
//! rounding per multiply-add, and a module is compiled a second time by the
//! platform. That half is measured on the device by the conformance pair, not
//! here.

use super::*;
use naga::{BinaryOperator, Expression, Literal, MathFunction};
use vyre_lower::descriptor_builder::binop_over_loads;
use vyre_test_support::spec_variant_tables::{builtin_bin_ops, builtin_un_ops};

/// One loaded f32, `unop` applied, stored back as f32.
fn unop_over_load(unop: UnOp) -> KernelDescriptor {
    vyre_lower::descriptor_builder::unop_over_load("f32_unop", DataType::F32, unop)
}

/// How many expressions of each shape the refinement is counted by.
struct Shapes {
    divides: usize,
    fmas: usize,
    selects: usize,
}

fn shapes(module: &naga::Module) -> Shapes {
    let mut counts = Shapes {
        divides: 0,
        fmas: 0,
        selects: 0,
    };
    for function in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        for (_, expr) in function.expressions.iter() {
            match expr {
                Expression::Binary {
                    op: BinaryOperator::Divide,
                    ..
                } => counts.divides += 1,
                Expression::Math {
                    fun: MathFunction::Fma,
                    ..
                } => counts.fmas += 1,
                Expression::Select { .. } => counts.selects += 1,
                _ => {}
            }
        }
    }
    counts
}

/// The refinement in full, read back out of the arena.
///
/// Counting alone would admit two `Fma` expressions wired to the wrong
/// operands, so the shape is walked: the selected value is a fused
/// multiply-add of a residual and a reciprocal onto the plain quotient, and
/// the arm taken when that value is not finite is the plain quotient itself.
fn assert_refined_quotient(module: &naga::Module, what: &str) {
    let function = &module.entry_points[0].function;
    let arena = &function.expressions;
    let get = |handle| arena.try_get(handle).expect("handle is in this arena");

    let (_, select) = arena
        .iter()
        .find(|(_, expr)| matches!(expr, Expression::Select { .. }))
        .unwrap_or_else(|| panic!("{what}: no Select, so the quotient is a bare Divide"));
    let Expression::Select {
        condition,
        accept,
        reject,
    } = select
    else {
        unreachable!("filtered above")
    };

    assert!(
        matches!(
            get(*reject),
            Expression::Binary {
                op: BinaryOperator::Divide,
                ..
            }
        ),
        "{what}: the arm taken for a non-finite correction must be the plain quotient, \
         which is what carries every special value the operator already had"
    );

    let Expression::Math {
        fun: MathFunction::Fma,
        arg: residual,
        arg1: Some(reciprocal),
        arg2: Some(quotient),
        ..
    } = get(*accept)
    else {
        panic!("{what}: the selected value must be a fused multiply-add of the correction")
    };
    assert!(
        matches!(
            get(*quotient),
            Expression::Binary {
                op: BinaryOperator::Divide,
                ..
            }
        ),
        "{what}: the correction must be folded onto the plain quotient"
    );
    assert!(
        matches!(
            get(*reciprocal),
            Expression::Binary {
                op: BinaryOperator::Divide,
                left,
                ..
            } if matches!(get(*left), Expression::Literal(Literal::F32(one)) if *one == 1.0)
        ),
        "{what}: the correction must be scaled by a reciprocal of the divisor"
    );
    assert!(
        matches!(
            get(*residual),
            Expression::Math {
                fun: MathFunction::Fma,
                ..
            }
        ),
        "{what}: the residual must be a fused multiply-add, which is what makes it exact"
    );

    let Expression::Binary {
        op: BinaryOperator::LessEqual,
        left: magnitude,
        right: bound,
    } = get(*condition)
    else {
        panic!("{what}: the correction must be admitted only where it is finite")
    };
    assert!(
        matches!(
            get(*magnitude),
            Expression::Math {
                fun: MathFunction::Abs,
                ..
            }
        ) && matches!(get(*bound), Expression::Literal(Literal::F32(max)) if *max == f32::MAX),
        "{what}: finiteness is stated as a magnitude against the largest finite f32"
    );
}

#[test]
fn f32_divide_is_corrected_to_the_nearest_quotient() {
    let module = emit(&binop_over_loads("f32_div", DataType::F32, BinOp::Div))
        .expect("f32 Div must emit");
    assert_valid_wgsl(&module, "f32 Div");
    assert_refined_quotient(&module, "BinOp::Div on f32");
}

#[test]
fn f32_reciprocal_takes_the_same_correction_as_divide() {
    let module = emit(&unop_over_load(UnOp::Reciprocal)).expect("f32 Reciprocal must emit");
    assert_valid_wgsl(&module, "f32 Reciprocal");
    assert_refined_quotient(&module, "UnOp::Reciprocal");
}

#[test]
fn integer_divide_is_left_as_one_instruction() {
    // The correction is f32 arithmetic. An integer quotient is exact already,
    // and wrapping it in a fused multiply-add would be both wrong and slower,
    // so the u32 and i32 arms must keep the single `Divide` naga emits. The
    // unsigned arm additionally carries the oracle's divide-by-zero sentinel,
    // which is a Select over one Divide and no Fma.
    for elem in [DataType::U32, DataType::I32] {
        let module = emit(&binop_over_loads("int_div", elem.clone(), BinOp::Div))
            .unwrap_or_else(|error| panic!("{elem:?} Div must emit: {error:?}"));
        let counts = shapes(&module);
        assert_eq!(
            counts.fmas, 0,
            "{elem:?} Div must not reach the f32 correction: {} fused multiply-add(s) emitted",
            counts.fmas
        );
        assert_eq!(
            counts.divides, 1,
            "{elem:?} Div must stay one Divide, got {}",
            counts.divides
        );
    }
}

#[test]
fn no_builtin_operator_reaches_an_uncorrected_f32_quotient() {
    // The variant space is the frozen builtin tables, so an operator added to
    // `vyre-spec` whose f32 lowering divides without the correction fails here
    // rather than reaching a device and diverging from the oracle by one ULP.
    //
    // Every all-f32 program below has no integer value in it, so a `Divide` in
    // the emitted module is necessarily an f32 quotient. The correction emits
    // two `Divide` and two `Fma` per quotient, which is the invariant checked:
    // a bare quotient leaves a `Divide` with no `Fma` to pay for it.
    let mut checked = 0;
    for binop in builtin_bin_ops() {
        let Ok(module) = emit(&binop_over_loads("scan", DataType::F32, binop.clone())) else {
            continue;
        };
        let counts = shapes(&module);
        if counts.divides == 0 {
            continue;
        }
        checked += 1;
        assert!(
            counts.fmas >= counts.divides,
            "BinOp::{binop:?} emits {} f32 Divide expression(s) against {} fused multiply-add(s). \
             An f32 quotient must go through `emit_f32_divide`, whose residual correction is what \
             puts it in the same ulp as the reference oracle.",
            counts.divides,
            counts.fmas
        );
        assert!(
            counts.selects >= 1,
            "BinOp::{binop:?} emits an f32 Divide with no Select, so a non-finite correction \
             would replace the special value the plain quotient states"
        );
    }
    for unop in builtin_un_ops() {
        let Ok(module) = emit(&unop_over_load(unop.clone())) else {
            continue;
        };
        let counts = shapes(&module);
        if counts.divides == 0 {
            continue;
        }
        checked += 1;
        assert!(
            counts.fmas >= counts.divides,
            "UnOp::{unop:?} emits {} f32 Divide expression(s) against {} fused multiply-add(s). \
             An f32 quotient must go through `emit_f32_divide`.",
            counts.divides,
            counts.fmas
        );
        assert!(
            counts.selects >= 1,
            "UnOp::{unop:?} emits an f32 Divide with no Select, so a non-finite correction \
             would replace the special value the plain quotient states"
        );
    }
    assert!(
        checked >= 2,
        "the scan reached {checked} dividing operator(s), so it stopped covering \
         `BinOp::Div` and `UnOp::Reciprocal` and proves nothing"
    );
}
