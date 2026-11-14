//! Scheduling, fusion, batching, and dispatch-strategy compositions.

pub mod planar_rewrite_pass_scheduler;
pub mod spectral_schedule;
pub mod submodular_cache_eviction;

pub(crate) use crate::plumbing::host::dispatch_buffers::{
    checked_product_count, checked_square_cells,
};
