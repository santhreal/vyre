//! One binary for every integration test in this crate.

#[path = "adversarial_decode.rs"]
pub mod adversarial_decode;

#[path = "fuse_decode_scan_error.rs"]
pub mod fuse_decode_scan_error;

#[path = "inflate_program.rs"]
pub mod inflate_program;

#[path = "inflate_stored_ir_parity_proptest.rs"]
pub mod inflate_stored_ir_parity_proptest;

#[path = "ir_aliasing.rs"]
pub mod ir_aliasing;

#[path = "loop_trip_count_bounds.rs"]
pub mod loop_trip_count_bounds;

#[path = "proptest_base64_decode.rs"]
pub mod proptest_base64_decode;

#[path = "proptest_hex_decode.rs"]
pub mod proptest_hex_decode;

#[path = "rle_segment_lengths_contracts.rs"]
pub mod rle_segment_lengths_contracts;

#[path = "rle_segment_lengths_ir_parity_proptest.rs"]
pub mod rle_segment_lengths_ir_parity_proptest;

#[path = "sweep_decode_base64_volume_oracle_matrix.rs"]
pub mod sweep_decode_base64_volume_oracle_matrix;

#[path = "sweep_decode_hex_oracle_matrix.rs"]
pub mod sweep_decode_hex_oracle_matrix;

#[path = "sweep_decode_hex_primitives_volume_oracle_matrix.rs"]
pub mod sweep_decode_hex_primitives_volume_oracle_matrix;
