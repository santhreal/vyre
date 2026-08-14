//! Device-boundary contracts and dispatch scratch shared by the compositions.
//!
//! These are host-side contracts and allocation policy, not Category C
//! intrinsics: nothing here needs an emitter arm or an interpreter arm. The
//! hardware intrinsics live in `vyre_primitives::hardware`.

pub mod device_resident_token_fact_graph;
pub mod dispatch_program_cache;
pub mod gpu_probe_contract;
pub mod memory_ownership_contract;
pub mod scratch;
