//! WGPU parity coverage for the shared literal-set matcher.

#![allow(deprecated)]
use vyre::scan::literal_set::{ByteRange, GpuLiteralSet};
use vyre_driver_wgpu as _;

#[test]
fn literal_set_parity_abc() {
    let patterns: &[&[u8]] = &[b"abc", b"bc"];
    let engine = GpuLiteralSet::compile(patterns);
    let haystack = b"zabc";

    let reference_matches = engine.reference_scan(haystack);
    assert_eq!(reference_matches.len(), 2);
    assert_eq!(reference_matches[0], ByteRange::new(0, 1, 4));
    assert_eq!(reference_matches[1], ByteRange::new(1, 2, 4));

    let gpu_matches = engine.scan("wgpu", haystack, 10_000).unwrap();
    assert_eq!(gpu_matches, reference_matches);
}
