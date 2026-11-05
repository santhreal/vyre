#![allow(missing_docs)]

use vyre_macros::vyre_ast_registry;

vyre_ast_registry! {
    ContractExpr {
        Literal(u32),
        Binary { left: u32, right: u32 },
    }

    ContractNode {
        Return,
        Store(u32, u32),
    }

    ContractFloat {
        Scalar(f32),
        Wide(f64),
        Scaled { value: f32, scale: f64, count: u32 },
    }
}

#[test]
fn ast_registry_supports_multiple_enums_without_name_cross_talk() {
    assert_eq!(
        contractexpr_op_id(&ContractExpr::Literal(7)),
        "vyre.contractexpr.literal"
    );
    assert_eq!(
        contractnode_op_id(&ContractNode::Store(1, 2)),
        "vyre.contractnode.store"
    );

    assert_eq!(ContractExpr::Literal(7), ContractExpr::Literal(7));
    assert_ne!(ContractExpr::Literal(7), ContractExpr::Literal(8));
    assert_eq!(ContractNode::Return, ContractNode::Return);
    assert_ne!(ContractNode::Store(1, 2), ContractNode::Store(2, 1));
}

/// The emitted name list is the run-time enumeration of the declared variants,
/// and `*_variant_name` agrees with it for a value of each variant.
///
/// This is the mechanism that lets a downstream crate notice a variant it was
/// never told about: these enums are `#[non_exhaustive]`, so nothing outside
/// the defining crate can match exhaustively, and a hand-written list of
/// variants elsewhere would go stale in silence.
#[test]
fn ast_registry_enumerates_every_declared_variant_by_name() {
    assert_eq!(CONTRACTEXPR_VARIANT_NAMES, ["Literal", "Binary"]);
    assert_eq!(CONTRACTNODE_VARIANT_NAMES, ["Return", "Store"]);

    assert_eq!(
        contractexpr_variant_name(&ContractExpr::Literal(7)),
        "Literal"
    );
    assert_eq!(
        contractexpr_variant_name(&ContractExpr::Binary { left: 1, right: 2 }),
        "Binary"
    );
    assert_eq!(contractnode_variant_name(&ContractNode::Return), "Return");
    assert_eq!(
        contractnode_variant_name(&ContractNode::Store(1, 2)),
        "Store"
    );

    for name in CONTRACTNODE_VARIANT_NAMES {
        assert!(
            [
                contractnode_variant_name(&ContractNode::Return),
                contractnode_variant_name(&ContractNode::Store(1, 2)),
            ]
            .contains(name),
            "every declared name must be reachable from some value: {name} was not"
        );
    }
}

/// A float field of an AST node compares by bit pattern, not by IEEE equality.
///
/// IEEE equality is the wrong relation for a literal held in compiler IR. It
/// reports that a NaN literal differs from itself, so a program carrying one is
/// never equal to itself: every pass that rebuilds the tree then reports a
/// rewrite it did not make, and the optimizer fixpoint runs to its iteration
/// cap and fails the compile instead of converging. That is how strict-IEEE
/// `sin` and `cos`, whose expansion answers an out-of-domain argument with a
/// NaN literal, failed every dispatch before emission.
///
/// The dual is just as wrong: IEEE reports `0.0 == -0.0`, and the sign of a
/// zero survives division and copysign, so two programs that differ in it are
/// two programs.
///
/// Both directions are asserted for a tuple field, a named field, and `f64` as
/// well as `f32`, so a manifest that grows a float field of either shape is
/// covered by construction.
#[test]
fn a_float_field_compares_by_bits_and_not_by_ieee_equality() {
    assert_eq!(
        ContractFloat::Scalar(f32::NAN),
        ContractFloat::Scalar(f32::NAN),
        "a NaN literal must equal itself or no program containing one is equal to itself"
    );
    assert_eq!(ContractFloat::Wide(f64::NAN), ContractFloat::Wide(f64::NAN));
    assert_eq!(
        ContractFloat::Scaled {
            value: f32::NAN,
            scale: f64::NAN,
            count: 1,
        },
        ContractFloat::Scaled {
            value: f32::NAN,
            scale: f64::NAN,
            count: 1,
        }
    );

    assert_ne!(
        ContractFloat::Scalar(0.0),
        ContractFloat::Scalar(-0.0),
        "the sign of a zero is observable through division and copysign"
    );
    assert_ne!(ContractFloat::Wide(0.0), ContractFloat::Wide(-0.0));
    assert_ne!(
        ContractFloat::Scaled {
            value: 0.0,
            scale: 0.0,
            count: 1,
        },
        ContractFloat::Scaled {
            value: -0.0,
            scale: 0.0,
            count: 1,
        }
    );

    assert_ne!(
        ContractFloat::Scalar(f32::NAN),
        ContractFloat::Scalar(f32::from_bits(f32::NAN.to_bits() | 1)),
        "two NaN payloads are two literals"
    );
    assert_ne!(ContractFloat::Scalar(1.0), ContractFloat::Scalar(2.0));
}
