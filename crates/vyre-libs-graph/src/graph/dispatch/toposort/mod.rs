//! Dispatch wrapper for the CSR topological sort in `crate::graph::toposort`.
//!
//! The wrapper validates the CSR, keys the encoded graph inputs so a reused
//! [`ToposortGpuScratch`] encodes one graph once, and validates the returned
//! order before returning it.

mod dispatch;

use crate::graph::dispatch::dispatch_bridge::{CachedProgram, ProgramCache};
use crate::graph::toposort::ToposortCsrStaticInputKey;

pub use dispatch::{
    topo_order_csr_via, topo_order_csr_via_with_scratch, topo_order_csr_via_with_scratch_into,
};
/// Caller-owned GPU dispatch scratch for topological-sort CSR queries.
#[derive(Debug, Default)]
pub struct ToposortGpuScratch {
    inputs: Vec<Vec<u8>>,
    program_cache: ProgramCache<u32, CachedToposortProgram>,
    static_input_key: Option<ToposortCsrStaticInputKey>,
}

type CachedToposortProgram = CachedProgram;
