//! Exploded-supergraph (IFDS encoding) substrate consumer.
//!
//! Wires the primitive-owned exploded-supergraph reference builder (zero
//! prior consumers) into the substrate so the optimizer can build
//! interprocedural-dataflow graphs directly. The IFDS encoding packs
//! `(proc_id, block_id, fact_id)` into a u32 node id, then composes
//! intra-/inter-procedural edges + GEN/KILL flow into a CSR ready for
//! reachability/closure analysis.

mod dispatch;

pub use dispatch::{
    build_ifds_csr_via, build_ifds_csr_via_into, build_ifds_csr_via_with_scratch_into,
};

use crate::graph::exploded::{
    IfdsCsrProgramCacheKey, IfdsCsrRuleColumns, IfdsCsrRuleInputFingerprint, IfdsCsrStaticInputKey,
};

use crate::graph::dispatch::dispatch_bridge::{CachedProgram, ProgramCache};

/// Caller-owned GPU dispatch scratch for exploded IFDS CSR construction.
#[derive(Debug, Default)]
pub struct IfdsCsrGpuScratch {
    rule_columns: IfdsCsrRuleColumns,
    rule_fingerprint: Option<IfdsCsrRuleInputFingerprint>,
    inputs: Vec<Vec<u8>>,
    static_input_key: Option<IfdsCsrStaticInputKey>,
    row_cursor: Vec<u32>,
    col_len_words: Vec<u32>,
    program_cache: ProgramCache<IfdsCsrProgramCacheKey, CachedIfdsCsrProgram>,
}

type CachedIfdsCsrProgram = CachedProgram;

impl IfdsCsrGpuScratch {
    #[cfg(test)]
    fn program_builds(&self) -> usize {
        self.program_cache.builds()
    }
}

#[cfg(test)]
pub use crate::graph::exploded::{ifds_node_count, round_trip_dense};

#[cfg(test)]
#[path = "../../../../tests/internal/graph/dispatch/exploded/mod.rs"]
mod tests;
