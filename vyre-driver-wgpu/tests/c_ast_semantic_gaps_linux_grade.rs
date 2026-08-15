//! Backend parity arm for Linux-grade C semantic gaps: nested typedef shadowing,
//! GNU attributes on enums and parameters, asm aliases, and mixed and incomplete
//! initializers.
//!
//! The construct list, the four-stage scaffold and the property-graph row
//! contract are owned by `tests/support/c_frontend/parity_matrix` and
//! `tests/support/c_frontend/fixtures/semantic_gap_constructs`; the CPU arm in
//! `vyre-libs/tests/c_ast_semantic_gaps_linux_grade` iterates the same `CASES`
//! and owns every classification assertion. What this root adds is the GPU arm
//! those cases run on.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_semantic_gaps_linux_grade/gpu_parity.rs"]
mod gpu_parity;
#[path = "../../tests/support/c_frontend/fixtures/semantic_gap_constructs.rs"]
mod semantic_gap_constructs;

use c_ast_gpu_parity_support::{assert_family_parity, GpuArm};
