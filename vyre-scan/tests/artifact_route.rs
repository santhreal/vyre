//! Production scan compiler, target payload, materialization, and submission contract.

use vyre_driver::backend::backend_registration;
use vyre_driver_wgpu as _;
use vyre_foundation::match_result::ByteRange;
use vyre_scan::{
    build_scan_session, DirectGpuScanner, GpuLiteralSet, MatchScan, Pipeline, ScanArtifactError,
};

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

    assert_eq!(
        actual,
        vec![ByteRange::new(0, 1, 3), ByteRange::new(1, 2, 4)]
    );
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
    let mut actual = vec![ByteRange::new(99, 0, 0)];
    let mut scratch = Vec::new();

    session.scan_into(haystack, &mut actual, &mut scratch)?;
    assert_eq!(actual, expected);
    session.scan_into(haystack, &mut actual, &mut scratch)?;
    assert_eq!(actual, expected);
    session.free()?;
    Ok(())
}

/// WHY: the general NFA resident route must reuse authenticated artifact state
/// across submissions and preserve reference scan semantics.
#[test]
fn resident_nfa_scan_uses_authenticated_artifact_resources(
) -> Result<(), Box<dyn std::error::Error>> {
    let haystack = b"zabcab";
    let neutral = build_scan_session(&["ab", "bc"], "input", "hits", haystack.len() as u32);
    let expected = neutral.reference_scan(haystack);
    let session = neutral.prepare_resident("wgpu", haystack.len() + 16, 10_000)?;
    let mut actual = Vec::new();
    let mut scratch = Vec::new();

    session.scan_into(haystack, &mut actual, &mut scratch)?;
    assert_eq!(actual, expected);
    session.scan_into(haystack, &mut actual, &mut scratch)?;
    assert_eq!(actual, expected);
    session.free()?;
    Ok(())
}

/// WHY: region-presence resources and executable state must share one authenticated
/// materializer generation across repeated submissions.
#[test]
fn resident_presence_uses_authenticated_artifact_resources(
) -> Result<(), Box<dyn std::error::Error>> {
    let haystack = b"abxxbc";
    let matcher = GpuLiteralSet::compile(&[b"ab".as_slice(), b"bc".as_slice()]);
    let session = matcher.prepare_resident_presence("wgpu", haystack.len() + 16, 2)?;
    let mut actual = Vec::new();
    let mut scratch = Vec::new();

    session.scan_into(haystack, &[0, 4], 0, &mut actual, &mut scratch)?;
    assert_eq!(actual, vec![1, 2]);
    session.scan_into(haystack, &[0, 4], 0, &mut actual, &mut scratch)?;
    assert_eq!(actual, vec![1, 2]);
    session.free()?;
    Ok(())
}

/// WHY: fused presence and position scans must share one authenticated artifact
/// generation while preserving output identity across repeated resident submissions.
#[test]
fn resident_fused_scan_uses_authenticated_artifact_resources(
) -> Result<(), Box<dyn std::error::Error>> {
    let haystack = b"abxxbc";
    let matcher = GpuLiteralSet::compile(&[b"ab".as_slice(), b"bc".as_slice()]);
    let session = matcher.prepare_resident_fused_scan("wgpu", haystack.len() + 16, 2, 8)?;
    let mut presence = Vec::new();
    let mut matches = Vec::new();
    let mut scratch = Vec::new();

    for _ in 0..2 {
        session.scan_into(
            haystack,
            &[0, 4],
            0,
            &mut presence,
            &mut matches,
            &mut scratch,
        )?;
        assert_eq!(presence, vec![1, 2]);
        assert_eq!(
            matches,
            vec![ByteRange::new(0, 0, 2), ByteRange::new(1, 4, 6)]
        );
    }
    session.free()?;
    Ok(())
}

/// WHY: public engine and post-processing facades must preserve the artifact-only
/// execution boundary instead of accepting a raw backend dispatcher.
#[test]
fn public_scan_facades_submit_registered_artifacts() -> Result<(), Box<dyn std::error::Error>> {
    let haystack = b"zabc";
    let expected = vec![ByteRange::new(0, 1, 4), ByteRange::new(1, 2, 4)];

    let direct = DirectGpuScanner::compile(&[b"abc".as_slice(), b"bc".as_slice()]);
    assert_eq!(direct.scan("wgpu", haystack, 8)?, expected);

    let matcher = GpuLiteralSet::compile(&[b"abc".as_slice(), b"bc".as_slice()]);
    let dynamic: &dyn MatchScan = &matcher;
    assert_eq!(dynamic.scan("wgpu", haystack, 8)?, expected);

    let pipeline =
        Pipeline::with_post_process(matcher, vyre_scan::post_process::try_reference_post_process);
    let processed = pipeline.scan_processed("wgpu", haystack, 8)?;
    assert_eq!(processed.len(), expected.len());
    assert_eq!(processed[0].pattern_id, 0);
    assert_eq!(processed[1].pattern_id, 1);

    Ok(())
}
