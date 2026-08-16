//! Contract test verifying derived host oracle migration.

use std::path::Path;
use vyre_reference::host_oracle_migration::{
    assert_host_oracle_migration_complete, derive_host_function_inventory,
    HOST_FUNCTION_DISCOVERY_FLOOR,
};

#[test]
fn host_oracle_inventory_derives_and_exceeds_floor() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root exists");

    let inventory = derive_host_function_inventory(root)
        .expect("host function inventory derivation must succeed");

    assert!(
        inventory.len() >= HOST_FUNCTION_DISCOVERY_FLOOR,
        "discovered {} functions, must meet or exceed floor {}",
        inventory.len(),
        HOST_FUNCTION_DISCOVERY_FLOOR
    );

    // Verify all discovered functions have valid paths and non-empty names
    for func in &inventory {
        assert!(!func.function_name.is_empty());
        assert!(!func.crate_name.is_empty());
        assert!(!func.classification_reason.is_empty());
    }
}

#[test]
fn host_oracle_migration_assert_passes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root exists");

    assert_host_oracle_migration_complete(root);
}
