// What one disagreement between two backends records, what one operation that
// could not be measured records, and what the run reports once every operation
// has been attempted.

use blake3::Hash;

#[derive(Debug)]
pub(crate) struct Divergence {
    pub(crate) op_id: &'static str,
    pub(crate) backend_a: &'static str,
    pub(crate) backend_b: &'static str,
    pub(crate) input_hash: Hash,
    pub(crate) output_a_hash: Hash,
    pub(crate) output_b_hash: Hash,
    pub(crate) detail: String,
}

/// Backend field of a failure raised before any backend ran.
pub(crate) const HARNESS_BACKEND: &str = "matrix";

/// One operation the matrix could not measure.
///
/// A missing fixture, a program the validator rejects, or a backend that refuses
/// the dispatch says nothing about the other operations in the sweep. Recording
/// the failure and continuing is what makes one run report every operation
/// instead of the first broken one, so the counters and the divergence list in
/// [`Summary`] describe the whole registry.
#[derive(Debug)]
pub(crate) struct OpFailure {
    pub(crate) op_id: &'static str,
    /// Backend that refused, or [`HARNESS_BACKEND`] when the operation never
    /// reached one.
    pub(crate) backend: &'static str,
    /// Which part of the measurement could not be completed.
    pub(crate) stage: &'static str,
    pub(crate) detail: String,
}

impl OpFailure {
    /// Record a failure raised before any backend ran.
    pub(crate) fn harness(op_id: &'static str, stage: &'static str, detail: String) -> Self {
        Self {
            op_id,
            backend: HARNESS_BACKEND,
            stage,
            detail,
        }
    }

    /// Record a failure one named backend raised.
    pub(crate) fn backend(
        op_id: &'static str,
        backend: &'static str,
        stage: &'static str,
        detail: String,
    ) -> Self {
        Self {
            op_id,
            backend,
            stage,
            detail,
        }
    }
}

#[derive(Default, Debug)]
pub(crate) struct Summary {
    pub(crate) ops_total: usize,
    pub(crate) ops_covered: usize,
    pub(crate) backends_linked: usize,
    pub(crate) backends_runnable: usize,
    pub(crate) divergences: Vec<Divergence>,
    pub(crate) failures: Vec<OpFailure>,
}
