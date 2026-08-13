//! CPU/GPU parity on real C sources: kernel list heads, libc errno, and complex declarators.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c_ast_real_corpus_harness/bytes.rs"]
mod bytes;
mod c_ast_gpu_parity_support;
#[path = "c_ast_real_corpus_harness/test_kernel_list_head_parity.rs"]
mod test_kernel_list_head_parity;
