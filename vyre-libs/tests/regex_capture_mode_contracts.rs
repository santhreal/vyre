//! Regex capture mode contracts test suite.

const CONTRACTS: &str = include_str!("../../docs/optimization/REGEX_CAPTURE_MODE_CONTRACTS.toml");

const REQUIRED_MODES: &[&str] = &[
    "noncapture",
    "count",
    "span",
    "named_capture",
    "repeated_capture",
    "group_extraction",
];

const REQUIRED_OUTPUT_FIELDS: &[&str] = &[
    "match_id",
    "pattern_id",
    "start",
    "end",
    "group_id",
    "group_name",
    "nullable",
];

#[test]
fn regex_capture_mode_contracts_cover_required_modes_and_fields() {
    for mode in REQUIRED_MODES {
        assert!(
            CONTRACTS.contains(&format!("mode_id = \"{mode}\"")),
            "Fix: regex capture mode contracts must include `{mode}`"
        );
    }
    for field in REQUIRED_OUTPUT_FIELDS {
        assert!(
            CONTRACTS.contains(&format!("\"{field}\"")),
            "Fix: regex capture mode contracts must declare output field `{field}`"
        );
    }
}

#[test]
fn regex_capture_mode_contracts_gate_extraction_on_verifier() {
    for required in [
        "mode_id = \"named_capture\"",
        "mode_id = \"repeated_capture\"",
        "mode_id = \"group_extraction\"",
        "verifier_required = true",
        "accelerator_eligible = false",
        "unmatched-group-null",
    ] {
        assert!(
            CONTRACTS.contains(required),
            "Fix: capture extraction contracts must include `{required}`"
        );
    }
}

#[test]
fn regex_capture_mode_contracts_keep_whole_match_modes_accelerator_eligible() {
    for required in [
        "mode_id = \"noncapture\"",
        "mode_id = \"count\"",
        "mode_id = \"span\"",
        "whole_match_only",
        "match_count_per_pattern",
        "whole_match_span",
        "accelerator_eligible = true",
    ] {
        assert!(
            CONTRACTS.contains(required),
            "Fix: whole-match capture modes must include `{required}`"
        );
    }
    let mode_rows = CONTRACTS.matches("[[mode]]").count();
    assert_eq!(
        CONTRACTS
            .matches("evidence_path = \"vyre-libs/tests/regex_capture_mode_contracts.rs\"")
            .count(),
        mode_rows,
        "Fix: every capture mode row must point at this proof gate"
    );
}

/// The code-side [`CaptureMode`] routing table MUST agree with this TOML field
/// for field, the whole point of the enum is that a consumer routes on the type
/// instead of parsing the contract, so a drift between the two is a latent
/// mis-route. This gate locks every mode's `mode_id`, `output_shape`, routing
/// bits, and `null_policy` to the exact TOML text, and checks the enum covers
/// every declared mode and no extras.
#[cfg(feature = "matching-regex")]
#[test]
fn capture_mode_enum_matches_the_toml_contract_row_for_row() {
    use vyre_libs::scan::CaptureMode;

    // Every mode the enum knows appears in the TOML with byte-identical fields.
    for mode in CaptureMode::ALL {
        let row = mode.contract_row();
        for required in [
            format!("mode_id = \"{}\"", row.mode_id),
            format!("output_shape = \"{}\"", row.output_shape),
            format!("null_policy = \"{}\"", row.null_policy),
        ] {
            assert!(
                CONTRACTS.contains(&required),
                "code CaptureMode::{mode:?} declares `{required}` but the TOML contract does not"
            );
        }
        // The routing bits must sit inside this mode's own [[mode]] block, not
        // merely somewhere in the file. Slice from this mode_id to the next.
        let mode_marker = format!("mode_id = \"{}\"", row.mode_id);
        let start = CONTRACTS
            .find(&mode_marker)
            .unwrap_or_else(|| panic!("mode_id `{}` missing from TOML", row.mode_id));
        let block = &CONTRACTS[start..];
        let block_end = block[mode_marker.len()..]
            .find("[[mode]]")
            .map_or(block.len(), |offset| offset + mode_marker.len());
        let block = &block[..block_end];
        assert!(
            block.contains(&format!(
                "accelerator_eligible = {}",
                row.accelerator_eligible
            )),
            "TOML block for `{}` must set accelerator_eligible = {}",
            row.mode_id,
            row.accelerator_eligible
        );
        assert!(
            block.contains(&format!("verifier_required = {}", row.verifier_required)),
            "TOML block for `{}` must set verifier_required = {}",
            row.mode_id,
            row.verifier_required
        );
    }

    // The enum must cover EVERY mode the TOML declares (no code-side omission).
    let toml_mode_count = CONTRACTS.matches("[[mode]]").count();
    assert_eq!(
        CaptureMode::ALL.len(),
        toml_mode_count,
        "CaptureMode has {} variants but the TOML declares {toml_mode_count} modes, they must match",
        CaptureMode::ALL.len()
    );
    for required in REQUIRED_MODES {
        assert!(
            CaptureMode::from_mode_id(required).is_some(),
            "CaptureMode::from_mode_id must resolve the required mode `{required}`"
        );
    }
}
