//! Production scan compiler, target payload, materialization, and submission contract.

use vyre_driver::backend::backend_registration;
use vyre_driver_wgpu as _;
use vyre_foundation::match_result::Match;
use vyre_scan::{build_scan_session, GpuLiteralSet, ScanArtifactError};

fn wgpu_registration() -> &'static vyre_driver::BackendRegistration {
    backend_registration("wgpu")
        .expect("WGPU compiler and materializer registration must be linked")
}

/// WHY: production scan execution must pass through the canonical compiler,
/// authenticated target payload, materializer, and typed submission path.
#[test]
fn scan_executes_through_authenticated_artifact_submission() -> Result<(), ScanArtifactError> {
    let haystack = b"zabc";
    let neutral = build_scan_session(&["ab", "bc"], "input", "hits", haystack.len() as u32);
    let expected = neutral.reference_scan(haystack);

    let session = neutral.materialize(wgpu_registration())?;
    let first_artifact = session.artifact_digest();
    let first_payload = session.payload_digest();
    let actual = session.scan(haystack, 16)?;

    assert_eq!(actual, vec![Match::new(0, 1, 3), Match::new(1, 2, 4)]);
    assert_eq!(actual, expected);
    assert_ne!(first_artifact.0, [0; 32]);
    assert_ne!(first_payload.0, [0; 32]);
    let cap_error = session
        .scan(haystack, 10_001)
        .expect_err("caller cap above the compiler-owned hit ABI must fail closed");
    assert!(
        cap_error
            .to_string()
            .contains("exceeds compiled hit capacity 10000"),
        "unexpected cap error: {cap_error}"
    );
    Ok(())
}

/// WHY: resident scan resources must be owned, uploaded, submitted, and freed by the
/// same authenticated artifact materializer generation rather than a raw backend pipeline.
#[test]
fn resident_literal_scan_uses_authenticated_artifact_resources(
) -> Result<(), Box<dyn std::error::Error>> {
    let haystack = b"zabcab";
    let matcher = GpuLiteralSet::compile(&[b"ab".as_slice(), b"bc".as_slice()]);
    let expected = matcher.reference_scan(haystack);
    let session = matcher.prepare_resident_scan("wgpu", haystack.len() + 16, 16)?;
    let mut actual = vec![Match::new(99, 0, 0)];
    let mut scratch = Vec::new();

    session.scan_into(haystack, &mut actual, &mut scratch)?;
    assert_eq!(actual, expected);
    session.scan_into(haystack, &mut actual, &mut scratch)?;
    assert_eq!(actual, expected);
    session.free()?;
    Ok(())
}
