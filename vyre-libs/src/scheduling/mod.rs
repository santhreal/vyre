//! Scheduling, fusion, batching, and dispatch-strategy compositions.

#[cfg(test)]
pub mod branch_compaction;
#[cfg(test)]
pub mod frontier_partitioning;
pub mod frontier_typed_ir;
pub mod megakernel_schedule;
#[cfg(test)]
pub mod multi_corpus_batching;
pub mod planar_rewrite_pass_scheduler;
#[cfg(test)]
pub mod polyhedral_fusion;
pub mod spectral_schedule;
pub mod submodular_cache_eviction;
