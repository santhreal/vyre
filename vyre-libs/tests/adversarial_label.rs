//! Failure-oriented adversarial tests for label primitives.
//!
//! Focus: hostile boundaries, overflow, invalid offsets, property invariants.
#![cfg(feature = "label")]

use vyre_reference::composition_witness::resolve_family_witness as reference_resolve_family;

#[test]
fn resolve_family_empty() {
    let got = reference_resolve_family(&[], 0xFF);
    assert!(got.is_empty());
}

#[test]
fn resolve_family_all_zeros() {
    let got = reference_resolve_family(&[0, 0, 0, 0], 0xFF);
    assert_eq!(got, vec![0]);
}

#[test]
fn resolve_family_all_hits() {
    let got = reference_resolve_family(&[0xFF; 64], 0xFF);
    assert_eq!(got, vec![0xFFFFFFFF; 2]);
}

#[test]
fn resolve_family_mask_zero() {
    let got = reference_resolve_family(&[0xFF; 64], 0);
    assert_eq!(got, vec![0; 2]);
}

#[test]
fn resolve_family_u32_max_overflow_boundary() {
    // 33 nodes requires 2 words
    let tags: Vec<u32> = (0..33).map(|i| if i == 32 { 0x1 } else { 0 }).collect();
    let got = reference_resolve_family(&tags, 0x1);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0], 0);
    assert_eq!(got[1], 0x1);
}

#[test]
fn resolve_family_single_node_boundary() {
    let got = reference_resolve_family(&[0x01], 0x01);
    assert_eq!(got, vec![0x1]);
}

#[test]
fn resolve_family_31_nodes_fits_one_word() {
    let tags = vec![0x01; 31];
    let got = reference_resolve_family(&tags, 0x01);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0], 0x7FFFFFFF);
}

#[test]
fn resolve_family_32_nodes_exact_word() {
    let tags = vec![0x01; 32];
    let got = reference_resolve_family(&tags, 0x01);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0], 0xFFFFFFFF);
}

#[test]
fn resolve_family_partial_hits() {
    let tags = vec![0x01, 0x02, 0x03, 0x04];
    let got = reference_resolve_family(&tags, 0x02);
    assert_eq!(got, vec![0b0110]);
}
