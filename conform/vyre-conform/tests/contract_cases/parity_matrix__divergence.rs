// What one disagreement between two backends records, and what the run
// reports once every operation has been compared.

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

#[derive(Default, Debug)]
pub(crate) struct Summary {
    pub(crate) ops_total: usize,
    pub(crate) ops_covered: usize,
    pub(crate) backends_linked: usize,
    pub(crate) backends_runnable: usize,
    pub(crate) divergences: Vec<Divergence>,
}
