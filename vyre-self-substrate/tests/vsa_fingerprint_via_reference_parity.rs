//! End-to-end parity for `data::vsa_fingerprint::fingerprint_via`.
//!
//! `hypervector_xor_bind` (the VSA cache-fingerprint binding primitive) had NO IR-execution
//! coverage anywhere: `rg -l hypervector_xor_bind vyre-primitives/tests/` = zero files, and its
//! only self-substrate consumer test uses an `XorDispatcher` MOCK that ignores the `_program`
//! argument and hand-computes the XOR, so `fingerprint_via` validated the two-stage dispatch
//! plumbing but NEVER executed the kernel (the mock-dispatcher-coherence gap).
//!
//! This runs the real `hypervector_xor_bind` Program through the shared `ReferenceEvalDispatcher`
//!, twice, chained (`fingerprint = kind ⊕ signature ⊕ region`), and asserts it reproduces the
//! host `reference_fingerprint` oracle over generated component hypervector triples across a range
//! of dimensionalities. XOR binding is exact integer arithmetic, so the dispatched result must
//! equal the host bit-for-bit.
#![forbid(unsafe_code)]
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::data::vsa_fingerprint::{fingerprint_via, reference_fingerprint};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn fingerprint_via_matches_host_over_generated_hypervectors() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x0B1D_5A7Cu32;
    let mut nonzero_cases = 0u32;
    for case in 0..400u32 {
        let dim_words = 1 + xorshift(&mut state) % 12; // 1..=12 lanes
        let kind_hv: Vec<u32> = (0..dim_words).map(|_| xorshift(&mut state)).collect();
        let signature_hv: Vec<u32> = (0..dim_words).map(|_| xorshift(&mut state)).collect();
        let region_hv: Vec<u32> = (0..dim_words).map(|_| xorshift(&mut state)).collect();

        let via = fingerprint_via(&dispatcher, &kind_hv, &signature_hv, &region_hv).expect(
            "fingerprint_via must dispatch the XOR-bind Program through the reference backend",
        );
        let host = reference_fingerprint(&kind_hv, &signature_hv, &region_hv);
        if host.iter().any(|&w| w != 0) {
            nonzero_cases += 1;
        }
        assert_eq!(
            via, host,
            "case {case} (dim_words={dim_words}): fingerprint _via {via:?} != host oracle {host:?} \
             (kind={kind_hv:?}, signature={signature_hv:?}, region={region_hv:?})"
        );
    }
    assert!(
        nonzero_cases > 380,
        "only {nonzero_cases}/400 fingerprints were non-zero, the binding is not being exercised"
    );
}

#[test]
fn fingerprint_via_is_the_triple_xor_bind() {
    // fingerprint = kind ⊕ signature ⊕ region, word-wise. Hand-check a small case through the
    // full two-stage dispatch so the chained-dispatch composition is pinned, not just per-stage.
    let dispatcher = ReferenceEvalDispatcher;
    let kind_hv = vec![0b0011u32, 0xFFFF_0000];
    let signature_hv = vec![0b0101u32, 0x0F0F_0F0F];
    let region_hv = vec![0b1001u32, 0x0000_FFFF];
    let expected: Vec<u32> = (0..kind_hv.len())
        .map(|i| kind_hv[i] ^ signature_hv[i] ^ region_hv[i])
        .collect();
    let via = fingerprint_via(&dispatcher, &kind_hv, &signature_hv, &region_hv)
        .expect("fingerprint_via must dispatch");
    let host = reference_fingerprint(&kind_hv, &signature_hv, &region_hv);
    assert_eq!(
        host, expected,
        "sanity: host oracle is the word-wise triple XOR"
    );
    assert_eq!(
        via, expected,
        "the chained XOR-bind dispatch must equal the direct triple XOR"
    );
}
