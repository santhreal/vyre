//! Canonical value pins for the registered hash operations.
//!
//! Each op declares its own `test_inputs`/`expected_output`; the harness only
//! checks that a program reproduces whatever the registration declares. These
//! pins are the independent half: they hold the registration's declared output
//! against the published checksum vector, so an edited fixture fails here
//! instead of silently redefining the op.
#![cfg(feature = "hash")]

use vyre_foundation::operation::OperationRegistry;
use vyre_reference::value::Value;

/// Evaluate a registered op over its declared inputs and pin every output byte.
fn assert_registered_witness(id: &str, expected: Vec<Vec<Vec<u8>>>) {
    let entry = OperationRegistry::global()
        .get(id)
        .unwrap_or_else(|| panic!("missing canonical operation registration for {id}"));
    let inputs = (entry.test_inputs.expect("declared test inputs"))();
    let declared = (entry.expected_output.expect("declared expected output"))();
    assert_eq!(declared, expected, "declared witness drift for {id}");

    let build = entry.build.expect("neutral builder");
    for (case, (input_set, expected_outputs)) in inputs.iter().zip(expected.iter()).enumerate() {
        let outputs = vyre_reference::reference_eval(
            &build(),
            &input_set
                .iter()
                .cloned()
                .map(Value::from)
                .collect::<Vec<_>>(),
        )
        .unwrap_or_else(|error| panic!("reference run failed for {id}: {error}"))
        .into_iter()
        .map(|value| value.to_bytes())
        .collect::<Vec<_>>();
        assert_eq!(
            outputs, *expected_outputs,
            "CPU witness drift for {id} case {case}"
        );
    }
}

#[test]
fn adler32_witness_is_pinned() {
    // Adler-32("abc") = 0x024D0127.
    assert_registered_witness(
        "vyre-libs::hash::adler32",
        vec![vec![vec![0x27, 0x01, 0x4d, 0x02]]],
    );
}

#[test]
fn crc32_witness_is_pinned() {
    // CRC-32("abc") = 0x352441C2 (reflected polynomial 0xEDB88320).
    assert_registered_witness(
        "vyre-libs::hash::crc32",
        vec![vec![vec![0xc2, 0x41, 0x24, 0x35]]],
    );
}

#[test]
fn fnv1a32_witness_is_pinned() {
    // FNV-1a32("a") = 0xE40C292C.
    assert_registered_witness(
        "vyre-libs::hash::fnv1a32",
        vec![vec![vec![0x2c, 0x29, 0x0c, 0xe4]]],
    );
}

#[test]
fn fnv1a64_witness_is_pinned() {
    // FNV-1a64("a") = 0xAF63DC4C8601EC8C, stored as low then high word.
    assert_registered_witness(
        "vyre-libs::hash::fnv1a64",
        vec![vec![vec![0x8c, 0xec, 0x01, 0x86, 0x4c, 0xdc, 0x63, 0xaf]]],
    );
}

#[test]
fn multi_hash_witness_is_pinned() {
    // One walk over "abc" emitting CRC-32, FNV-1a32, Adler-32 in that order.
    assert_registered_witness(
        "vyre-libs::hash::multi_hash",
        vec![vec![vec![
            0xc2, 0x41, 0x24, 0x35, 0x0b, 0xe9, 0x47, 0x1a, 0x27, 0x01, 0x4d, 0x02,
        ]]],
    );
}
