//! Shared-memory store race legality analysis outside bank conflict classification.
//!
//! Rejects multi-invocation constant-index `StoreShared` operations unless
//! single-invocation execution, atomicity, or disjoint-index mapping guarantees
//! race freedom under the target memory model.

pub(crate) mod analysis;
pub(crate) mod report;

pub use analysis::analyze;
pub use report::{SharedStoreLegality, SharedStoreRaceReport, SharedStoreRaceSite};
