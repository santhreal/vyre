//! Tests for the built-in device signature table.

use vyre_runtime::DeviceSignatureTable;

#[test]
fn blackwell_120_signature_can_be_loaded() {
    let table = DeviceSignatureTable::builtins_ref().expect("Fix: builtin device signatures load");

    let signature = table
        .get("blackwell_120")
        .expect("Fix: the builtin table contains blackwell_120");

    assert_eq!(signature.id, "blackwell_120");
}
