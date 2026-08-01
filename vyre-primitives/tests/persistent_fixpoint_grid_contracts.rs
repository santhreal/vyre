//! Contract tests for `fixpoint::persistent_fixpoint::persistent_fixpoint_grid`,
//! the grid-correct sibling of `persistent_fixpoint`.
//!
//! `persistent_fixpoint` drives convergence from an in-kernel
//! `Node::Loop` whose per-iteration barriers are `MemoryOrdering::SeqCst`,
//! which is WORKGROUP scope, and whose single shared `changed` word is
//! cleared by a plain lane-0 store and set by every group's `atomic_or`.
//! With more than one workgroup that is a race with two faces: a lost set
//! (a clear erases another group's flag, that group reads 0 and returns
//! early with unconverged state) and a false verdict (the flag read back
//! after the dispatch does not describe the convergence actually reached).
//!
//! `persistent_fixpoint_grid` replaces the in-kernel loop with top-level
//! waves separated by `MemoryOrdering::GridSync` barriers and gives
//! `changed` one never-cleared word per iteration, which is what makes its
//! early exit collective instead of a stranding hazard. Every test below
//! locks one of those properties and names the defect it excludes.
#![cfg(all(feature = "fixpoint", feature = "cpu-parity"))]

#[path = "persistent_fixpoint_grid_contracts/mod.rs"]
mod suite;
