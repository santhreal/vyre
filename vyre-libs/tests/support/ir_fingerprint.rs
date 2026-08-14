//! Pinned-fingerprint guards for collapsed clone families.
//!
//! Collapsing a clone family onto one owner is only safe if the survivor emits
//! exactly what every former copy emitted, so a family guard pins the canonical
//! wire fingerprint of every entry point involved.

use vyre_foundation::ir::Program;

/// Canonical wire fingerprint of `program` as lowercase hex.
pub(crate) fn fingerprint_hex(program: &Program) -> String {
    program
        .fingerprint()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Assert every entry point still fingerprints to its pinned hex, in the pinned
/// order.
///
/// A drift report carries the freshly measured table so re-pinning a deliberate
/// IR change is mechanical, and every entry is reported at once rather than
/// stopping at the first one.
pub(crate) fn assert_pinned_ir_fingerprints(
    entry_points: &[(&'static str, Program)],
    expected: &[(&str, &str)],
) {
    let actual: Vec<(&'static str, String)> = entry_points
        .iter()
        .map(|(name, program)| (*name, fingerprint_hex(program)))
        .collect();
    assert_eq!(
        actual.len(),
        expected.len(),
        "fixture count drifted from the pinned table"
    );
    let mut report = String::new();
    let mut drifted = false;
    for ((name, got), (pinned_name, pinned)) in actual.iter().zip(expected.iter()) {
        assert_eq!(name, pinned_name, "fixture order drifted from the table");
        if got != pinned {
            drifted = true;
        }
        report.push_str(&format!(
            "    (\n        \"{name}\",\n        \"{got}\",\n    ),\n"
        ));
    }
    assert!(
        !drifted,
        "generated IR changed for at least one clone-family entry point. \
         Recorded fingerprints:\n{report}"
    );
}
