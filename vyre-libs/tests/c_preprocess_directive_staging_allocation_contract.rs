//! Source contracts for C GPU-preprocess directive staging allocation.

use std::fs;
use std::path::PathBuf;

fn src_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn read_src(relative: &str) -> String {
    fs::read_to_string(src_path(relative)).unwrap_or_else(|err| {
        panic!("failed to read {relative}: {err}");
    })
}

#[test]
fn directive_staging_uses_checked_fallible_allocation_paths() {
    let directives = read_src("src/parsing/c/preprocess/gpu_pipeline/directives.rs");
    assert!(
        directives.contains("fn directive_word_bytes(")
            && directives.contains("fn directive_padded_u32_bytes(")
            && directives.contains("fn reserve_directive_vec<T>("),
        "directive extraction must centralize checked byte sizing and fallible reserve paths"
    );
    assert!(
        directives
            .contains("fn prepare_zero_init(&mut self, byte_len: usize) -> Result<(), String>")
            && directives.contains("try_reserve_exact(byte_len)"),
        "directive zero-init staging must reserve fallibly before resize"
    );
    assert!(
        directives.contains("u32::try_from(scratch.macro_names.len())"),
        "directive macro-name offsets must reject values outside the GPU u32 address space"
    );
    assert!(
        !directives.contains("prepare_zero_init(n_pad * 4)")
            && !directives.contains(".reserve((count + builtin_hashes.len()) * 4)")
            && !directives.contains("Vec::with_capacity(defined_macros.len() + 1)")
            && !directives.contains("scratch.macro_names.len() as u32"),
        "directive staging must not use unchecked reserve or offset arithmetic"
    );
}
