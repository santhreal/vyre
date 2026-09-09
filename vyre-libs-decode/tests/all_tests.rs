//! One binary for every integration test in this crate.

#[path = "inflate_program.rs"]
pub mod inflate_program;

#[path = "ir_aliasing.rs"]
pub mod ir_aliasing;

#[path = "sweep_decode_base64_volume_oracle_matrix.rs"]
pub mod sweep_decode_base64_volume_oracle_matrix;

#[path = "sweep_decode_hex_oracle_matrix.rs"]
pub mod sweep_decode_hex_oracle_matrix;

#[path = "sweep_decode_hex_primitives_volume_oracle_matrix.rs"]
pub mod sweep_decode_hex_primitives_volume_oracle_matrix;

