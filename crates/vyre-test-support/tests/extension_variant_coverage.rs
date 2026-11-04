//! Integration test verifying extension registrations and resolvers.

use vyre_test_support::extension_variants::{
    assert_extension_registry_complete, registered_expr_extension_kinds,
    registered_node_extension_kinds,
};

#[test]
fn extension_registrations_are_complete() {
    assert_extension_registry_complete();
    let expr_exts = registered_expr_extension_kinds();
    let node_exts = registered_node_extension_kinds();
    // At least WideLiteralExpr extensions are registered in vyre-foundation
    assert!(
        !expr_exts.is_empty(),
        "expr extensions should be registered"
    );
    let _ = node_exts;
}
