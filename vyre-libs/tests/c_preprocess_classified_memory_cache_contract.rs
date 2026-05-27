//! Structural contracts for classified-token preprocessing memory caching.

use std::fs;
use std::path::Path;

fn crate_file(path: &str) -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest.join(path)).unwrap_or_else(|error| {
        panic!("failed to read {path}: {error}");
    })
}

#[test]
fn classified_token_memory_cache_is_byte_bounded() {
    let classified_memory =
        crate_file("src/parsing/c/preprocess/gpu_pipeline/cache/classified_memory.rs");
    assert!(
        classified_memory.contains("const CLASSIFIED_CACHE_MAX_BYTES: usize"),
        "Fix: classified-token memory cache must be byte-bounded, not only entry-bounded."
    );
    assert!(
        classified_memory.contains("ByteBoundLruCache<ClassifiedCacheKey, Arc<ClassifiedTokens>>"),
        "Fix: classified-token cache must use the shared byte-bounded LRU core instead of bespoke entry-only caching."
    );
    assert!(
        classified_memory.contains("classified_tokens_bytes(&value)"),
        "Fix: classified-token cache admission must size token columns and source bytes before storing."
    );
    let byte_lru_cache = crate_file("src/parsing/c/preprocess/gpu_pipeline/byte_lru_cache.rs");
    assert!(
        byte_lru_cache.contains("bytes: usize") && byte_lru_cache.contains("max_bytes: usize"),
        "Fix: shared GPU preprocessor cache core must track resident bytes and a byte limit."
    );
    assert!(
        byte_lru_cache.contains("entry_bytes > self.max_bytes")
            && byte_lru_cache.contains(".checked_add(entry_bytes)"),
        "Fix: shared GPU preprocessor cache core must reject oversized entries and evict to byte budget."
    );

    let classified_size = crate_file("src/parsing/c/preprocess/gpu_pipeline/classified_size.rs");
    assert!(
        classified_size.contains("pub(super) fn classified_tokens_bytes"),
        "Fix: classified-token byte sizing must live in one shared helper."
    );
    assert!(
        classified_size.contains("tok_types")
            && classified_size.contains("tok_starts")
            && classified_size.contains("tok_lens")
            && classified_size.contains("directive_kinds")
            && classified_size.contains("classified.source.len()"),
        "Fix: classified-token byte sizing must include every token column plus retained source bytes."
    );

    let header_reuse = crate_file("src/parsing/c/preprocess/gpu_pipeline/header_reuse.rs");
    assert!(
        header_reuse.contains("use super::classified_size::classified_tokens_bytes;"),
        "Fix: header reuse and classified-token cache must share the same classified-token sizing helper."
    );
}
