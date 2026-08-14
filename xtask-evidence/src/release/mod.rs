//! Release evidence whose provenance is measured, not read.
//!
//! `backend_matrix` probes the host devices a release claims support for,
//! `release_evidence` gathers the recorded artifacts, and `vyre_release_gate`
//! decides whether the measurements still describe this tree. The parts of the
//! release surface that only read manifests live in `xtask::release`.

pub mod backend_matrix;
pub mod release_evidence;
pub mod vyre_release_gate;
