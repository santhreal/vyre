//! Contract tests for c ast real corpus harness.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c_ast_real_corpus_harness/bytes.rs"]
mod bytes;
mod c_ast_gpu_parity_support;
#[path = "c_ast_real_corpus_harness/test_kernel_list_head_parity.rs"]
mod test_kernel_list_head_parity;
