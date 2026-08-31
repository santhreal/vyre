//! Release evidence whose provenance is measured, or whose derivation needs a
//! vyre crate linked.
//!
//! `backend_matrix` probes the host devices a release claims support for,
//! `release_workload_matrix` derives the workload families from the benchmark
//! case registry, `release_evidence` gathers the recorded artifacts, and
//! `vyre_release_gate` decides whether the measurements still describe this
//! tree. The parts of the release surface that only read manifests live in
//! `xtask::release`.

pub mod backend_matrix;
pub mod release_evidence;
pub mod release_workload_matrix;
pub mod vyre_release_gate;
