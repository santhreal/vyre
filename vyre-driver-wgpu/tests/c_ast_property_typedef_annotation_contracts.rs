//! Test: c ast property typedef annotation contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c_ast_property_typedef_annotation_contracts/arb_atom.rs"]
mod arb_atom;
mod c_ast_gpu_parity_support;
#[path = "c_ast_property_typedef_annotation_contracts/run_gpu_typedef_annotation.rs"]
mod run_gpu_typedef_annotation;
