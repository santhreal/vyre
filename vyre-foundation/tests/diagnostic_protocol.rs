//! Shared diagnostic protocol regression contracts.

use vyre_foundation::diagnostics::{Diagnostic, DiagnosticStage, OpLocation, RetryClass, Severity};

/// WHY: public workflow failures cross compiler, packaging, runtime, and driver
/// boundaries. Serialization must preserve the code, stage, typed location,
/// cause, correction, and retry decision instead of collapsing them into prose.
#[test]
fn diagnostic_round_trip_preserves_workflow_identity() {
    let diagnostic = Diagnostic::error("MKC016_DIGEST_MISMATCH", "artifact digest mismatch")
        .with_stage(DiagnosticStage::Admit)
        .with_location(
            OpLocation::op("artifact.authenticate")
                .with_graph_node(7)
                .with_graph_value(11)
                .with_path("artifact.envelope"),
        )
        .with_cause(
            "digest_mismatch",
            "declared digest differs from canonical body",
        )
        .with_fix("discard the envelope and recompile the source graph")
        .with_retry(RetryClass::RecompileSource);

    let encoded = serde_json::to_vec(&diagnostic).expect("diagnostic must serialize");
    let decoded: Diagnostic =
        serde_json::from_slice(&encoded).expect("diagnostic must deserialize");

    assert_eq!(decoded, diagnostic);
    assert_eq!(decoded.severity, Severity::Error);
    assert_eq!(decoded.code.as_str(), "MKC016_DIGEST_MISMATCH");
    assert_eq!(decoded.stage, DiagnosticStage::Admit);
    assert_eq!(decoded.retry, RetryClass::RecompileSource);
    let location = decoded.location.expect("typed location must survive");
    assert_eq!(location.graph_node, Some(7));
    assert_eq!(location.graph_value, Some(11));
    assert_eq!(location.path.as_deref(), Some("artifact.envelope"));
    let cause = decoded.cause.expect("structured cause must survive");
    assert_eq!(cause.kind, "digest_mismatch");
}
