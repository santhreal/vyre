//! W1-3 gate: `GpuLiteralSet` cached-artifact wire format is versioned and
//! fails CLOSED on a version/magic mismatch, a silent format drift can never
//! corrupt a scan by loading a stale blob into a newer runtime (Law 10).
//!
//! The wire envelope is `magic[4] || le_u32(version) || sections...`. These
//! tests exercise the envelope contract through the public
//! `GpuLiteralSet::to_bytes` / `from_bytes` without depending on the private
//! magic/version constants: a valid blob's own header bytes are read and mutated
//! in place, so the test tracks whatever the current version is.

use vyre_libs::scan::GpuLiteralSet;

fn sample_matcher() -> GpuLiteralSet {
    GpuLiteralSet::compile(&[b"AKIA".as_slice(), b"ghp_", b"token", b"a"])
}

/// The header version is a little-endian u32 at bytes[4..8].
fn header_version(blob: &[u8]) -> u32 {
    u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]])
}

fn set_header_version(blob: &mut [u8], version: u32) {
    blob[4..8].copy_from_slice(&version.to_le_bytes());
}

#[test]
fn round_trip_reconstructs_an_identical_matcher() {
    let matcher = sample_matcher();
    let blob = matcher.to_bytes().expect("serialize");
    let restored = GpuLiteralSet::from_bytes(&blob).expect("deserialize a self-produced blob");

    // Prove the blob is REAL, not merely structurally valid: the restored
    // matcher must scan identically to the original on a corpus with hits.
    let haystack = b"__AKIA__ghp___token__aaa__";
    let a = matcher.reference_scan(haystack);
    let b = restored.reference_scan(haystack);
    assert_eq!(
        a, b,
        "restored matcher must scan identically to the original"
    );
    assert!(!a.is_empty(), "expected the sample corpus to have matches");
}

#[test]
fn future_version_fails_closed_with_named_versions() {
    let matcher = sample_matcher();
    let mut blob = matcher.to_bytes().expect("serialize");
    let current = header_version(&blob);
    // Bump to an unknown FUTURE version no runtime (current or legacy) accepts.
    let future = current + 1;
    set_header_version(&mut blob, future);

    // `GpuLiteralSet` is not `Debug`, so match rather than `expect_err`.
    let err = match GpuLiteralSet::from_bytes(&blob) {
        Ok(_) => panic!("a future wire version must be rejected, never silently loaded"),
        Err(err) => err,
    };
    let text = err.to_string();
    assert!(
        text.contains(&future.to_string()),
        "the fail-closed error must name the offending version {future}: got `{text}`"
    );
}

#[test]
fn zero_version_fails_closed() {
    let matcher = sample_matcher();
    let mut blob = matcher.to_bytes().expect("serialize");
    set_header_version(&mut blob, 0);
    assert!(
        GpuLiteralSet::from_bytes(&blob).is_err(),
        "version 0 (uninitialized/corrupt header) must fail closed"
    );
}

#[test]
fn corrupt_magic_fails_closed() {
    let matcher = sample_matcher();
    let mut blob = matcher.to_bytes().expect("serialize");
    // Flip the first magic byte so it no longer matches the consumer's expected
    // magic (a different payload type must be rejected, not misparsed).
    blob[0] ^= 0xFF;
    assert!(
        GpuLiteralSet::from_bytes(&blob).is_err(),
        "a magic mismatch must fail closed, never misparse another payload as a literal set"
    );
}

#[test]
fn truncated_blob_fails_closed() {
    let matcher = sample_matcher();
    let blob = matcher.to_bytes().expect("serialize");
    // A header-only prefix (magic + version, no sections) must be rejected.
    let truncated = &blob[..blob.len().min(8)];
    assert!(
        GpuLiteralSet::from_bytes(truncated).is_err(),
        "a truncated blob must fail closed, never decode partial sections"
    );
    // An empty blob likewise.
    assert!(
        GpuLiteralSet::from_bytes(&[]).is_err(),
        "an empty blob must fail closed"
    );
}
