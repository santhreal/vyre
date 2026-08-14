//! Benchmark execution and the semantics of benchmark evidence.
//!
//! `release_benchmarks` runs the long benchmark suites and writes their
//! artifacts, `bench_release` and `bench_crossback` are the coordinator and
//! the cross-backend view, and `benchmark_evidence_semantics` is what every
//! reader of those artifacts checks them against.

pub(crate) mod bench_crossback;
pub(crate) mod bench_release;
pub(crate) mod benchmark_evidence_semantics;
pub(crate) mod release_benchmarks;
