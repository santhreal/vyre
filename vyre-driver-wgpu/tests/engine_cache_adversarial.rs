//! Adversarial tests for `vyre-libs::matching::engine`.
//!
//! Targets:
//!   - `MatchScan` trait object safety and dispatch
//!   - `MatchEngineCache` round-trip resilience
//!   - `cached_load_or_compile` corruption recovery, concurrency, and
//!     filesystem edge cases.
//!
//! Run:
//! `cargo test -p vyre-libs --features matching-regex --test engine_cache_adversarial`

#![cfg(feature = "matching-nfa")]
#![allow(deprecated)]
use std::sync::{Arc, Barrier};
use std::thread;

use vyre::scan::{cached_load_or_compile, engine_cache_path, GpuLiteralSet, MatchScan};
use vyre_foundation::match_result::ByteRange;

// ---------------------------------------------------------------------------
// 1. Cache file corruption recovery (7 tests)
// ---------------------------------------------------------------------------

mod engine_cache_adversarial_cache_recovers_from_truncated_file {

    include!("contract_cases/engine_cache_adversarial__cache_recovers_from_truncated_file.rs");
}
mod engine_cache_adversarial_write_failure_still_returns_engine {
    include!("contract_cases/engine_cache_adversarial__write_failure_still_returns_engine.rs");
}
