//! Contracts for `vyre_driver::strategy`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::strategy::{
    select_precision_lowering, select_strategy, LoweredExpr, LoweringStrategy,
    PrecisionLoweringPlan,
};
use vyre_foundation::ir::{BinOp, Expr};
use vyre_foundation::optimizer::passes::algebraic::precision_hint::{
    PrecisionHint, TranscendentalOp,
};
use vyre_foundation::validate::BackendCapabilities;

#[derive(Debug)]
struct MockNativeStrategy;

impl LoweringStrategy for MockNativeStrategy {
    fn name(&self) -> &str {
        "mock-native"
    }
    fn can_apply(&self, caps: &BackendCapabilities, op: &BinOp) -> bool {
        caps.has_mul_high && matches!(op, BinOp::MulHigh)
    }
    fn priority(&self) -> u32 {
        100
    }
    fn lower(&self, _op: &BinOp, left: &Expr, right: &Expr) -> LoweredExpr {
        // In real impl: emit OpUMulExtended
        LoweredExpr::Expr(Expr::mulhi(left.clone(), right.clone()))
    }
}

#[derive(Debug)]
struct MockFallbackStrategy;

impl LoweringStrategy for MockFallbackStrategy {
    fn name(&self) -> &str {
        "mock-fallback"
    }
    fn can_apply(&self, _caps: &BackendCapabilities, op: &BinOp) -> bool {
        matches!(op, BinOp::MulHigh)
    }
    fn priority(&self) -> u32 {
        10
    }
    fn lower(&self, _op: &BinOp, left: &Expr, right: &Expr) -> LoweredExpr {
        // In real impl: 16-bit decomposition
        LoweredExpr::Expr(Expr::mul(left.clone(), right.clone()))
    }
}

#[test]
fn selects_highest_priority() {
    let strategies: Vec<Box<dyn LoweringStrategy>> =
        vec![Box::new(MockFallbackStrategy), Box::new(MockNativeStrategy)];
    let caps = BackendCapabilities {
        has_mul_high: true,
        ..Default::default()
    };
    let selected = select_strategy(&strategies, &caps, &BinOp::MulHigh);
    assert_eq!(selected.unwrap().name(), "mock-native");
}

#[test]
fn falls_back_when_native_unavailable() {
    let strategies: Vec<Box<dyn LoweringStrategy>> =
        vec![Box::new(MockFallbackStrategy), Box::new(MockNativeStrategy)];
    let caps = BackendCapabilities {
        has_mul_high: false,
        ..Default::default()
    };
    let selected = select_strategy(&strategies, &caps, &BinOp::MulHigh);
    assert_eq!(selected.unwrap().name(), "mock-fallback");
}

#[test]
fn returns_none_for_unsupported_op() {
    let strategies: Vec<Box<dyn LoweringStrategy>> = vec![Box::new(MockNativeStrategy)];
    let caps = BackendCapabilities {
        has_mul_high: true,
        ..Default::default()
    };
    let selected = select_strategy(&strategies, &caps, &BinOp::Add);
    assert!(selected.is_none());
}

#[test]
fn precision_hint_selects_native_f16_when_supported() {
    let caps = BackendCapabilities {
        has_native_f16: true,
        ..Default::default()
    };
    let plan = select_precision_lowering(
        &caps,
        &PrecisionHint::F16Eligible {
            max_abs_operand: 4.0,
        },
    );
    assert_eq!(
        plan,
        PrecisionLoweringPlan::NativeF16 {
            max_abs_operand: 4.0
        }
    );
}

#[test]
fn precision_hint_keeps_f32_without_native_f16() {
    let plan = select_precision_lowering(
        &BackendCapabilities::default(),
        &PrecisionHint::F16Eligible {
            max_abs_operand: 4.0,
        },
    );
    assert_eq!(plan, PrecisionLoweringPlan::DefaultF32);
}

#[test]
fn transcendental_hint_selects_polynomial_when_supported() {
    let caps = BackendCapabilities {
        has_transcendental_polynomial_emit: true,
        ..Default::default()
    };
    let plan = select_precision_lowering(
        &caps,
        &PrecisionHint::TranscendentalPolynomial {
            op: TranscendentalOp::Sin,
            argument_bound: 0.2,
        },
    );
    assert_eq!(
        plan,
        PrecisionLoweringPlan::PolynomialTranscendental {
            op: TranscendentalOp::Sin,
            argument_bound: 0.2,
            degree: 3,
        }
    );
}

#[test]
fn transcendental_hint_uses_higher_degree_for_wider_sin_range() {
    let caps = BackendCapabilities {
        has_transcendental_polynomial_emit: true,
        ..Default::default()
    };
    let plan = select_precision_lowering(
        &caps,
        &PrecisionHint::TranscendentalPolynomial {
            op: TranscendentalOp::Sin,
            argument_bound: 0.75,
        },
    );
    assert_eq!(
        plan,
        PrecisionLoweringPlan::PolynomialTranscendental {
            op: TranscendentalOp::Sin,
            argument_bound: 0.75,
            degree: 5,
        }
    );
}
