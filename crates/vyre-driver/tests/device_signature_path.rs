//! Tests for the built-in device signature table.

use vyre_driver::DeviceSignatureTable;

#[test]
fn blackwell_120_signature_can_be_loaded() {
    let table = DeviceSignatureTable::builtins().expect("Fix: builtin device signatures load");

    let signature = table
        .get("blackwell_120")
        .expect("Fix: the builtin table contains blackwell_120");

    assert_eq!(signature.id, "blackwell_120");
}
