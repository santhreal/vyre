use std::fs;
use std::path::Path;

pub(crate) fn crate_file(path: &str) -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest.join(path)).unwrap_or_else(|error| {
        panic!("failed to read {path}: {error}");
    })
}

pub(crate) fn assert_byte_lru_core_tracks_resident_bytes() {
    let byte_lru_cache = crate_file("src/parsing/c/preprocess/gpu_pipeline/byte_lru_cache.rs");
    assert!(
        byte_lru_cache.contains("bytes: usize") && byte_lru_cache.contains("max_bytes: usize"),
        "Fix: shared GPU preprocessor cache core must track resident bytes and byte limit."
    );
}

pub(crate) fn assert_byte_lru_core_rejects_and_accounts() {
    let byte_lru_cache = crate_file("src/parsing/c/preprocess/gpu_pipeline/byte_lru_cache.rs");
    assert!(
        byte_lru_cache.contains("entry_bytes > self.max_bytes")
            && byte_lru_cache.contains(".checked_add(entry_bytes)")
            && byte_lru_cache.contains("checked_sub(entry.bytes)"),
        "Fix: shared GPU preprocessor cache core must reject oversized entries, evict to byte budget, and update byte accounting."
    );
}

pub(crate) fn assert_contains_all(source: &str, needles: &[&str], message: &str) {
    let missing = needles
        .iter()
        .copied()
        .filter(|needle| !source.contains(needle))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "{message} Missing required source fragment(s): {}",
        missing.join(" | ")
    );
}

pub(crate) fn assert_contains_none(source: &str, needles: &[&str], message: &str) {
    let present = needles
        .iter()
        .copied()
        .filter(|needle| source.contains(needle))
        .collect::<Vec<_>>();
    assert!(
        present.is_empty(),
        "{message} Forbidden source fragment(s): {}",
        present.join(" | ")
    );
}
