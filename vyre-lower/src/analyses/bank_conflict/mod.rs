//! Shared-memory bank-conflict analysis for vyre kernels.
//!
//! Shared memory is divided into banks, and each bank serves one read or write
//! per cycle. When K threads in one subgroup address K different locations that
//! map to the same bank, those accesses serialize, costing up to 32x throughput
//! at a 32-way conflict.
//!
//! A stride whose `addr % BANK_COUNT` is equal for every thread produces one. A
//! 32x32 shared tile walked column-major at stride 32 puts all 32 threads on
//! bank 0.
//!
//! This module detects bank-conflict candidates among shared-memory
//! load/store ops in a `KernelDescriptor`, and derives the per-binding access
//! phase profiles a target needs before it may rewrite an index. Both operate
//! substrate-neutrally on the post-lowering descriptor. Which rewrite to apply,
//! and applying it, belong to the emitter that states the bank geometry: a
//! primary-binary emitter selects one strategy per permutable shared binding and
//! rewrites the element index at its single address site.
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
    CANDIDATE_MITIGATIONS,
};
