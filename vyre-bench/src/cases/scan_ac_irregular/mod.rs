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
