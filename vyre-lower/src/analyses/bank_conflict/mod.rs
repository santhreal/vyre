//! Shared-memory bank-conflict analysis for vyre kernels.
//!
//!
//! Shared memory on modern GPUs is divided into N banks. Each bank can
//! serve one read or write per cycle. When K threads in the same
//! subgroup access K different addresses that map to the **same
//! bank**, those accesses serialize  -  costing up to 32x throughput
//! for the worst case (32-way conflict).
//!
//! A common cause: a stride pattern where `addr % BANK_COUNT` is the
//! same for every thread. Classic example: a 32x32 tile in shared
//! memory accessed column-major with stride 32  -  all 32 threads in a
//! subgroup hit bank 0, full 32-way serialization.
//!
//! This crate detects bank-conflict candidates among shared-memory
//! load/store ops in a `KernelDescriptor`. Operates substrate-neutrally
//! on the post-lowering descriptor; emit-time concerns (per-substrate
//! bank count, swizzle-padding strategies) live in emitter crates.
//!
//! Phase 1 (this crate today): detection only. Walk every
//! `LoadShared`/`StoreShared` op, look at the index expression's
//! stride, classify as `NoConflict` / `Conflict` / `Unknown`, return
//! a `BankConflictReport`. Phase 2 (follow-up): rewrites that pad
//! shared-mem allocations or swizzle indices to break conflict
//! patterns.
//!
//! The bank count is a device fact the caller states; `analyze` takes it and
//! this crate holds no default for it.

pub(crate) mod analysis;
pub(crate) mod report;
pub(crate) mod strategy;

pub use analysis::analyze;
pub use report::{BankAccessSite, BankConflictKind, BankConflictReport, ConflictSeverity};
pub use strategy::{
    derive_shared_access_profiles, evaluate_mitigation_candidate, select_bank_conflict_strategy,
    AccessPhase, AccessPhaseProfile, BankConflictMitigation, MitigationEvaluation,
    PhaseConflictReport, SharedBindingAccessProfile, SharedPermutationBlock, TargetBankGeometry,
};
