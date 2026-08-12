//! External relation import certificate compatibility contracts.

const CERTIFICATES: &str =
    include_str!("../../docs/optimization/EXTERNAL_RELATION_IMPORT_CERTIFICATES.toml");

/// Relation-import evidence must retain every witness boundary and stable edge identifier.
/// Removing one makes the generated analyzer evidence impossible to replay across repositories.
#[test]
fn relation_import_certificates_preserve_witness_boundaries() {
    for required in [
        "endpoint_domains",
        "call_string_ids",
        "sanitizer_ids",
        "tuple_digest",
        "source_span_mapping",
        "malformed_bytes_policy",
        "witness_path_policy",
    ] {
        assert!(
            CERTIFICATES.contains(required),
            "relation import certificate must expose {required}"
        );
    }

    assert!(CERTIFICATES.contains("fail_closed"));
    assert!(CERTIFICATES.contains("preserve_source_span_chain"));
    assert!(CERTIFICATES.contains("external-taint-edge"));
}
