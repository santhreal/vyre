//! One binary for every integration test in this crate.

#[path = "smoke.rs"]
pub mod smoke;

#[path = "contraction_output_tiling.rs"]
pub mod contraction_output_tiling;

#[path = "cooperative_tile_scaffold.rs"]
pub mod cooperative_tile_scaffold;
