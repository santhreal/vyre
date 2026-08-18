//! Aho-Corasick scans over a 4 MiB haystack of unaligned, varied-length literals.
//!
//! Two cases share one fixture and one sampling path:
//! [`literals`] emits every match as a byte range, [`count`] returns cardinality
//! only. The fixture lives in [`haystack`], the match wire format in
//! [`match_triples`], the dispatch and reporting path in [`sample`], the CPU
//! oracle in [`baseline`], and the metric points in [`metrics`].

use crate::api::suite::SuiteKind;

mod baseline;
mod count;
mod haystack;
mod literals;
mod match_triples;
mod metrics;
mod sample;

#[cfg(test)]
#[path = "../../../tests/internal/cases/scan_ac_irregular/mod.rs"]
mod tests;

pub use literals::ScanAcIrregularLiterals;

const HAYSTACK_BYTES: usize = 4 * 1024 * 1024;
const MAX_MATCHES: u32 = 65_536;
/// Input slots the literal scan and the count preflight bind identically.
const CANDIDATE_END_MASK_INPUT_INDEX: usize = 7;
const CANDIDATE_SUFFIX2_MASK_INPUT_INDEX: usize = 8;
const CANDIDATE_SUFFIX3_BLOOM_INPUT_INDEX: usize = 9;
const MATCH_TRIPLE_WORDS: usize = 3;
const SUITES: &[SuiteKind] = &[
    SuiteKind::Smoke,
    SuiteKind::Release,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

pub(crate) const PATTERNS: &[&[u8]] = &[
    b"AKIA",
    b"ghp_",
    b"Authorization: Bearer ",
    b"password=",
    b"api_key",
    b"secret=",
    b"BEGIN RSA PRIVATE KEY",
    b"BEGIN OPENSSH PRIVATE KEY",
    b"eval(",
    b"strcpy(",
    b"memcpy(",
    b"TODO:",
    b"unsafe {",
    b"__attribute__((",
    b"container_of(",
    b"ioread32(",
];

pub(super) fn scan_ac_requirements() -> crate::api::case::BenchRequirements {
    crate::api::case::BenchRequirements {
        needs_gpu: true,
        needs_network: false,
        min_vram_bytes: Some(32 * 1024 * 1024),
        min_input_bytes: Some(HAYSTACK_BYTES as u64),
        feature_set: vec![
            "pattern-dfa".to_string(),
            "packed-byte".to_string(),
            "aho-corasick".to_string(),
        ],
    }
}

pub(super) fn scan_ac_candidate_masks(
    ac: &vyre_libs::pattern::classic_ac::ClassicAcAutomaton,
) -> ([u32; 8], [u32; 2048], Vec<u32>) {
    let candidate_end_mask =
        vyre_reference::composition_witness::classic_ac_candidate_end_byte_mask_words_witness(
            &ac.dfa.transitions,
            &ac.dfa.output_offsets,
            ac.dfa.state_count,
        );
    let candidate_suffix2_mask =
        vyre_reference::composition_witness::classic_ac_candidate_suffix2_mask_words_witness(
            &ac.dfa.transitions,
            &ac.dfa.output_offsets,
            ac.dfa.state_count,
        );
    let candidate_suffix3_bloom =
        vyre_reference::composition_witness::classic_ac_candidate_suffix3_bloom_words_witness(
            PATTERNS,
        );
    (
        candidate_end_mask,
        candidate_suffix2_mask,
        candidate_suffix3_bloom,
    )
}

pub(super) fn scan_ac_metadata(
    id: crate::api::case::BenchId,
    name: &'static str,
    description: &'static str,
    is_count_only: bool,
) -> crate::api::case::BenchMetadata {
    let mut tags = vec![
        "scan".to_string(),
        "pattern".to_string(),
        "dfa".to_string(),
        "aho-corasick".to_string(),
        "packed-byte".to_string(),
    ];
    if is_count_only {
        tags.push("count-only".to_string());
    }
    tags.extend(["irregular".to_string(), "release".to_string()]);
    crate::api::case::BenchMetadata {
        id,
        name: name.to_string(),
        description: description.to_string(),
        tags,
        layer: crate::api::case::BenchLayer::Libs,
        workload: crate::api::case::WorkloadClass::Macro,
        determinism: crate::api::case::DeterminismClass::Deterministic,
        owner_crate: "vyre-libs".to_string(),
    }
}
