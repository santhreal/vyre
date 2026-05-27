//! Source contracts for shared C GPU-preprocess buffer staging helpers.

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
fn shared_gpu_buffer_pack_and_pad_helpers_are_checked_and_fallible() {
    let buffers = read_src("src/parsing/c/preprocess/gpu_pipeline/buffers.rs");
    assert!(
        buffers.contains("pub(super) fn u32_word_byte_len(")
            && buffers.contains("pub(super) fn padded_u32_byte_len(")
            && buffers.contains("pub(super) fn reserve_gpu_staging_bytes("),
        "shared GPU staging helpers must centralize checked sizing and fallible reserve"
    );
    assert!(
        buffers.contains("pub(super) fn unpack_u32_words_prefix(bytes: &[u8], count: usize) -> Result<Vec<u32>, String>")
            && buffers.contains("could not reserve {take} prefix u32 decode words")
            && !buffers.contains("Vec::with_capacity(take)")
            && !buffers.contains("clamped prefix u32 unpack must not fail"),
        "prefix u32 unpacking must reserve fallibly and propagate decode errors"
    );
    assert!(
        buffers.contains("pub(super) fn pack_u32_words(words: &[u32], pad_len: usize) -> Result<Vec<u8>, String>")
            && buffers.contains("pub(super) fn pack_u32_words_into(")
            && buffers.contains(") -> Result<(), String>"),
        "u32 table packing must return Result instead of silently saturating or panicking"
    );
    assert!(
        buffers.contains("pub(super) fn pad_to_u32_words(bytes: &[u8]) -> Result<Vec<u8>, String>")
            && buffers.contains("pub(super) fn pad_to_u32_words_into(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), String>"),
        "byte padding helpers must return Result instead of using infallible reserve"
    );
    assert!(
        buffers.contains("try_reserve_exact(additional)")
            && !buffers.contains("saturating_mul(4)")
            && !buffers.contains(".reserve(target)")
            && !buffers.contains(".reserve(min_words"),
        "shared GPU staging helpers must not use saturating byte counts or infallible reserve"
    );
}
