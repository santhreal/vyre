#![allow(
    clippy::doc_lazy_continuation,
    clippy::double_must_use,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::match_like_matches_macro,
    clippy::redundant_closure
)]
//! Vyre pass engine: the optimizer's own passes, executed as vyre Programs.
//!
//! A pass here is not a second implementation of a `vyre-foundation` pass. The
//! semantics stay foundation's. This crate encodes a
//! `vyre_foundation::ir::Program` into the canonical 5-buffer `ProgramGraph`
//! ABI, runs the pass as graph primitives over that encoding, and applies the
//! result back to the IR. Dead-code elimination is `persistent_bfs`
//! reachability, CSE is `union_find` over a structural-hash key, constant
//! folding is `level_wave` bottom-up evaluation. The compiler runs on the
//! primitives it ships.
//!
//! # The name
//!
//! It states the job, not a hardware tier. Execution goes through
//! `vyre_foundation::program_dispatch::ProgramDispatcher`, so which device runs
//! a pass is the dispatcher's answer and not the crate's: the parity tests here
//! measure against `vyre_libs::graph::dispatch::cpu_oracle`, a CPU dispatcher,
//! while a backend dispatcher runs the identical Program on device. A name with
//! `gpu` in it would be false.
//!
//! # Layering
//!
//! ```text
//!   vyre-foundation        IR, registry, CPU optimizer semantics
//!         ↑
//!   vyre-primitives        hardware intrinsics
//!         ↑
//!   vyre-libs              compositions, dispatch marshalling, CPU oracles
//!         ↑
//!   vyre-pass-engine       ← THIS CRATE (no driver deps)
//!         ↑
//!   vyre-driver / vyre-runtime / vyre-driver-{cuda,wgpu}
//! ```
//!
//! `vyre-foundation` cannot depend on this crate: the encoding needs
//! `vyre-primitives`, which needs `vyre-foundation`. Foundation therefore keeps
//! the CPU pass math it needs at `vyre_foundation::pass_substrate`, and this
//! crate imports those functions and adds dispatch around them rather than
//! reimplementing them.
//!
//! # Contents
//!
//! `optimizer/` is the whole crate.
//!
//! - `encode`, `expr_arena`  -  IR DAG to the canonical 5-buffer encoding.
//! - `dce_program`  -  the persistent-BFS DCE analysis Program with early exit
//!   on convergence, and its stable `OP_ID`.
//! - `dce_via_encoded`, `cse_via_encoded`, `const_fold_via_encoded`,
//!   `canonicalize_via_encoded`, `pattern_match_via_encoded`,
//!   `validate_via_encoded`  -  one pass each, dispatched through a
//!   `ProgramDispatcher`.
//! - `const_prop`, `cross_scope_cse`, `dead_branch`, `licm`  -  host-side
//!   rewrites over the Program the dispatched passes hand back.
//! - `pipeline`, `pipeline_resident`, `pipeline_resident_decode`  -  pass
//!   sequencing, including the resident form that keeps arena buffers on the
//!   device across passes and applies one combined delta at the end.
//!
//! Nine module trees that used to live here are `vyre-libs` modules now:
//! scheduling solvers, analysis, logic, data, math, graph, hardware-boundary
//! contracts, telemetry counters, and the parity-test program-sequence helper.
//! `docs/ARCHITECTURE.md` records where each one went.

#[cfg(feature = "optimizer")]
/// The encoder plus the passes that run the compiler against its own
/// primitives. Exposed at the lib root so external consumers (driver-cuda
/// parity tests, conform runners) can reach the per-pass `*_via_encoded`
/// entry points and optimizer contract metadata without descending into
/// private module paths.
pub mod optimizer;
