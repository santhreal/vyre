//! Regex unsupported diagnostic registry test suite.

const REGISTRY: &str = include_str!("../../docs/optimization/REGEX_UNSUPPORTED_DIAGNOSTICS.toml");

const REQUIRED_DIAGNOSTICS: &[(&str, &str, bool)] = &[
    (
        "backreference",
        "VYRE_SCAN_UNSUPPORTED_BACKREFERENCE",
        false,
    ),
    (
        "lookaround",
        "VYRE_SCAN_APPROXIMATED_LOOKAROUND_REQUIRES_VERIFIER",
        true,
    ),
    (
        "unicode_classes_gpu",
        "VYRE_SCAN_UNSUPPORTED_UNICODE_MODE_GPU",
        true,
    ),
    (
        "capture_extraction",
        "VYRE_SCAN_CAPTURE_EXTRACTION_REQUIRES_VERIFIER",
        true,
    ),
    (
        "huge_alternation",
        "VYRE_SCAN_UNSUPPORTED_HUGE_ALTERNATION_BUDGET",
        false,
    ),
    (
        "nested_repeats",
        "VYRE_SCAN_UNSUPPORTED_NESTED_REPEAT_BUDGET",
        false,
    ),
];

const REQUIRED_FIELDS: &[&str] = &[
    "construct_id",
    "diagnostic_code",
    "syntax_span_policy",
    "dialect_expectation",
    "safe_fallback",
    "verifier_required",
    "remediation",
    "evidence_path",
];

#[test]
fn regex_unsupported_diagnostic_registry_covers_required_constructs() {
    for (construct, code, verifier_required) in REQUIRED_DIAGNOSTICS {
        assert!(
            REGISTRY.contains(&format!("construct_id = \"{construct}\"")),
            "Fix: unsupported regex diagnostic registry must include `{construct}`"
        );
        assert!(
            REGISTRY.contains(&format!("diagnostic_code = \"{code}\"")),
            "Fix: unsupported regex diagnostic `{construct}` must use `{code}`"
        );
        assert!(
            REGISTRY.contains(&format!("verifier_required = {verifier_required}")),
            "Fix: unsupported regex diagnostic `{construct}` must record verifier_required={verifier_required}"
        );
    }
}

#[test]
fn regex_unsupported_diagnostic_registry_records_operator_fields() {
    for field in REQUIRED_FIELDS {
        assert!(
            REGISTRY.contains(&format!("\"{field}\"")),
            "Fix: unsupported regex diagnostic registry must declare required field `{field}`"
        );
        assert!(
            REGISTRY.contains(&format!("{field} =")),
            "Fix: unsupported regex diagnostic registry must populate field `{field}`"
        );
    }

    let row_count = REGISTRY.matches("[[diagnostic]]").count();
    assert_eq!(
        row_count,
        REQUIRED_DIAGNOSTICS.len(),
        "Fix: unsupported regex diagnostic registry must have one row per required diagnostic"
    );
    assert_eq!(
        REGISTRY
            .matches("evidence_path = \"vyre-libs/tests/regex_unsupported_diagnostic_registry.rs\"")
            .count(),
        row_count,
        "Fix: every unsupported regex diagnostic row must point at this proof gate"
    );
}

/// Behavioral evidence (the registry rows point their `evidence_path` here): the
/// real public compile path must emit each registry construct's diagnostic code
/// from a representative pattern, the four W2-3 constructs the frontend now
/// distinctly detects (backreference, huge alternation, nested repeats as compile
/// errors; capture extraction as a non-error verifier signal), plus the two
/// pre-existing ones (lookaround, unicode class).
#[cfg(feature = "matching-regex")]
#[test]
fn frontend_emits_registry_diagnostic_codes_from_real_patterns() {
    use vyre_libs::scan::{compile_regex_set, RegexConstruct};

    // Compile-error constructs -> RegexCompileError::diagnostic_code().
    let cases: &[(&str, &str)] = &[
        (r"(a)\1", "VYRE_SCAN_UNSUPPORTED_BACKREFERENCE"),
        (
            r"a\bc",
            "VYRE_SCAN_APPROXIMATED_LOOKAROUND_REQUIRES_VERIFIER",
        ),
        (
            "[\u{0100}-\u{0200}]",
            "VYRE_SCAN_UNSUPPORTED_UNICODE_MODE_GPU",
        ),
        (
            r"(?:a{40}){40}",
            "VYRE_SCAN_UNSUPPORTED_NESTED_REPEAT_BUDGET",
        ),
    ];
    for (pattern, code) in cases {
        let err = compile_regex_set(&[pattern]).expect_err("construct must be rejected");
        assert_eq!(
            err.diagnostic_code(),
            Some(*code),
            "pattern {pattern:?} must emit {code}; got error {err}"
        );
    }

    // A huge alternation built from the exported code's own construct (so the
    // arm count is derived, not a magic literal).
    let huge: String = (0..2100)
        .map(|i| format!("v{i}"))
        .collect::<Vec<_>>()
        .join("|");
    let alt_err = compile_regex_set(&[huge.as_str()]).expect_err("huge alternation rejected");
    assert_eq!(
        alt_err.diagnostic_code(),
        Some("VYRE_SCAN_UNSUPPORTED_HUGE_ALTERNATION_BUDGET"),
    );

    // Capture extraction is a NON-error verifier signal on a successful compile.
    let captured = compile_regex_set(&[r"(abc)d"]).expect("captures compile for whole-match");
    assert_eq!(
        captured.capture_extraction_diagnostic_code(),
        Some("VYRE_SCAN_CAPTURE_EXTRACTION_REQUIRES_VERIFIER"),
        "a captured pattern must surface the capture-verifier code without erroring"
    );

    // The exported ONE-PLACE map agrees with the registry codes.
    assert_eq!(
        vyre_libs::scan::regex_construct_diagnostic_code(RegexConstruct::HugeAlternation),
        "VYRE_SCAN_UNSUPPORTED_HUGE_ALTERNATION_BUDGET"
    );
}

#[test]
fn regex_unsupported_diagnostic_registry_names_span_fallback_and_remediation() {
    for required in [
        "exact_construct_span",
        "class_span",
        "capture_group_span",
        "alternation_span",
        "repeat_subtree_span",
        "prefilter-plus-verifier",
        "host-reference-or-verifier",
        "whole-match-plus-capture-verifier",
        "reject-or-split-pattern-set",
    ] {
        assert!(
            REGISTRY.contains(required),
            "Fix: unsupported regex diagnostic registry must include `{required}`"
        );
    }
    assert!(
        REGISTRY.matches("remediation = ").count() == REQUIRED_DIAGNOSTICS.len(),
        "Fix: every unsupported regex diagnostic must include remediation text"
    );
}
