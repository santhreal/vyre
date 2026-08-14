//! Live CUDA coverage for GPU-resident C macro expansion.

#![cfg(test)]

#[path = "common/c_preprocess_oracles.rs"]
mod c_preprocess_oracles;
mod common;

use std::path::{Path, PathBuf};

use c_preprocess_oracles::{CudaOracle, ReferenceOracle};
use common::with_live_backend;
use vyre_libs::parsing::c::preprocess::gpu_pipeline::{
    gpu_preprocess_translation_unit, IncludeLoader,
};

struct EmptyLoader;

impl IncludeLoader for EmptyLoader {
    fn load(
        &self,
        _path: &[u8],
        _is_system: bool,
        _is_next: bool,
        _from: &Path,
    ) -> Result<Option<(PathBuf, std::sync::Arc<[u8]>)>, String> {
        Ok(None)
    }
}

#[test]
fn cuda_c_preprocess_macro_expansion_matches_reference() {
    with_live_backend("c preprocess macro expansion", |backend| {
        let source = b"#define OBJ 123\n#define FN(x) x\nint a = OBJ + FN(alpha);\n";
        let loader = EmptyLoader;
        let expected = gpu_preprocess_translation_unit(
            &ReferenceOracle,
            &loader,
            Path::new("<macro-ref>"),
            source,
            &[],
        )
        .expect("reference macro expansion must succeed");
        let actual = gpu_preprocess_translation_unit(
            &CudaOracle(backend),
            &loader,
            Path::new("<macro-cuda>"),
            source,
            &[],
        )
        .expect("CUDA macro expansion must succeed");

        assert_eq!(
            actual.bytes, expected.bytes,
            "Fix: CUDA materialized macro expansion must match reference raw-U8 source/name/replacement byte arenas."
        );
        let out = String::from_utf8_lossy(&actual.bytes);
        assert!(
            out.contains("123") && out.contains("alpha") && !out.contains("OBJ"),
            "Fix: CUDA macro expansion must replace object and function-like macros; got {out:?}"
        );
    });
}
