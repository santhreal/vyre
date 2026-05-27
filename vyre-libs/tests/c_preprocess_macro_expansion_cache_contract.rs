//! Structural contracts for the C macro expansion cache.

use std::fs;
use std::path::Path;

fn crate_file(path: &str) -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest.join(path)).unwrap_or_else(|error| {
        panic!("failed to read {path}: {error}");
    })
}

#[test]
fn macro_expansion_caches_are_byte_bounded() {
    let model = crate_file("src/parsing/c/preprocess/gpu_pipeline/macro_expansion/model.rs");
    assert!(
        model.contains("const MACRO_EXPANDED_SEGMENT_CACHE_MAX_BYTES: usize"),
        "Fix: expanded macro segment cache must have an explicit byte budget."
    );
    assert!(
        model.contains("const PACKED_MACRO_TABLE_CACHE_MAX_BYTES: usize"),
        "Fix: packed macro table cache must have an explicit byte budget."
    );
    assert!(
        model.contains("cached_expanded_segment_bytes(&value)")
            && model.contains("classified_tokens_bytes(&segment.classified)"),
        "Fix: expanded segment cache must size expanded bytes plus classified token residency."
    );
    assert!(
        model.contains("ByteBoundLruCache<MacroSegmentCacheKey, CachedExpandedSegment>")
            && model.contains("ByteBoundLruCache<[u8; 16], macro_table::PackedMacroTable>"),
        "Fix: macro expansion caches must use the shared byte-bounded LRU core instead of bespoke entry-only caching."
    );
    let byte_lru_cache = crate_file("src/parsing/c/preprocess/gpu_pipeline/byte_lru_cache.rs");
    assert!(
        byte_lru_cache.contains("entry_bytes > self.max_bytes")
            && byte_lru_cache.contains(".checked_add(entry_bytes)")
            && byte_lru_cache.contains("checked_sub(entry.bytes)"),
        "Fix: shared GPU preprocessor cache core must reject oversized entries, evict to byte budget, and update byte accounting."
    );
    assert!(
        model.contains("value.byte_len()"),
        "Fix: packed macro table cache must size packed table buffers before storing."
    );
    assert!(
        model.contains("packed macro table cache entry is {entry_bytes} bytes"),
        "Fix: oversized packed macro table entries must fail with an actionable byte-budget error."
    );

    let macro_table = crate_file("src/parsing/c/preprocess/gpu_pipeline/macro_table.rs");
    assert!(
        macro_table.contains("pub(crate) fn byte_len(&self) -> usize"),
        "Fix: packed macro tables must expose byte sizing to cache admission."
    );
}
