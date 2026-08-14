#![allow(missing_docs)]

mod support;

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
