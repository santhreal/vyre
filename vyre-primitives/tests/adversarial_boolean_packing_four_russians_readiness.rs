//! Adversarial contract tests for boolean packing substrate and
//! Four-Russians readiness contracts. Coverage: bitset_words sizing,
//! word-aligned packing invariants, set-relation predicates (equal,
//! subset_of, contains, test_bit), elementwise ops (and, or, xor,
//! not), popcount, and boundary element counts that stress cross-word
//! behaviour. GPU acquisition: none  -  all assertions use CPU
//! reference oracles. Implementation lives in two `include!`-d
//! chunks under `__split/`.
#![cfg(feature = "bitset")]

mod common;

include!("__split/adversarial_boolean_packing_four_russians_readiness_chunk1.rs");
include!("__split/adversarial_boolean_packing_four_russians_readiness_chunk2.rs");
