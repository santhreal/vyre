//! Source contracts for C GPU-preprocess prefix-scan staging allocation.

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
fn prefix_scan_staging_uses_checked_fallible_byte_sizing() {
    let scan = read_src("src/parsing/c/preprocess/gpu_pipeline/scan.rs");
    assert!(
        scan.contains("fn prefix_scan_word_bytes(")
            && scan.contains("fn prefix_scan_product_word_bytes("),
        "prefix scan staging must centralize checked word-to-byte sizing"
    );
    assert!(
        scan.contains("fn prepare_zero(") && scan.contains(") -> Result<(), String> {"),
        "prefix scan zero staging must return typed errors"
    );
    assert!(
        scan.contains("try_reserve_exact(byte_len)"),
        "prefix scan zero staging must reserve fallibly before resize"
    );
    assert!(
        !scan.contains("n as usize * 4")
            && !scan.contains("num_blocks as usize * 4")
            && !scan.contains("total_partials as usize * 4"),
        "prefix scan staging must not use unchecked u32-to-byte multiplication"
    );
    assert!(
        !scan.contains("saturating_mul(BLOCK_LANES)"),
        "prefix scan partial sizing must not silently saturate"
    );
}
