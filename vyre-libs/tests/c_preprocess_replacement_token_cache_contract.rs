//! Structural contracts for C preprocessing replacement-token caching.

use std::fs;
use std::path::Path;

fn crate_file(path: &str) -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest.join(path)).unwrap_or_else(|error| {
        panic!("failed to read {path}: {error}");
    })
}

#[test]
fn replacement_token_cache_is_entry_and_byte_bounded() {
    let model = crate_file("src/parsing/c/preprocess/gpu_pipeline/token_provenance/model.rs");
    assert!(
        model.contains("REPLACEMENT_TOKEN_CACHE_MAX_BYTES"),
        "Fix: replacement-token cache must expose an explicit byte budget."
    );

    let cache =
        crate_file("src/parsing/c/preprocess/gpu_pipeline/token_provenance/replacement_cache.rs");
    assert!(
        cache.contains("use crate::parsing::c::preprocess::gpu_pipeline::classified_size::classified_tokens_bytes;"),
        "Fix: replacement-token cache must share classified-token byte sizing with other preprocessor caches."
    );
    assert!(
        cache.contains(
            "ByteBoundLruCache<ReplacementTokenCacheKey, std::sync::Arc<ClassifiedTokens>>"
        ),
        "Fix: replacement-token cache must use the shared byte-bounded LRU core instead of bespoke entry-only caching."
    );
    assert!(
        cache.contains("classified_tokens_bytes(&value)"),
        "Fix: replacement-token cache admission must size retained classified tokens before storing."
    );
    let byte_lru_cache = crate_file("src/parsing/c/preprocess/gpu_pipeline/byte_lru_cache.rs");
    assert!(
        byte_lru_cache.contains("bytes: usize") && byte_lru_cache.contains("max_bytes: usize"),
        "Fix: shared GPU preprocessor cache core must track resident bytes and byte limit."
    );
    assert!(
        byte_lru_cache.contains("entry_bytes > self.max_bytes")
            && byte_lru_cache.contains(".checked_add(entry_bytes)")
            && byte_lru_cache.contains("checked_sub(entry.bytes)"),
        "Fix: shared GPU preprocessor cache core must reject oversized entries, evict to byte budget, and update byte accounting."
    );
}
