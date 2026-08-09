use super::*;

/// A cache directory that cannot be written to does not break the scan.
///
/// The engine still compiles, still scans correctly, and no cache file is left
/// behind. A failed cache write is a performance event, never a correctness
/// one.
///
/// Inducing the failure took some care. The test used to pre-create the temp
/// file's exact path as a directory, spelled `<key>.tmp.<pid>`. Temp paths now
/// carry a per-process sequence as well, `<key>.tmp.<pid>.<n>`, added so two
/// concurrent writers in one process cannot collide. The planted directory
/// therefore sat unused, the write succeeded, and the test failed while
/// reporting that the cache file should not exist. Guessing the temp name is
/// the fragile part, so this uses a cache "directory" that is really a regular
/// file: every path under it fails with ENOTDIR, whatever the temp file ends
/// up being called, and it does not depend on file permissions or on the uid
/// the tests run as.
#[test]
fn write_failure_still_returns_engine() {
    let dir = tempfile::tempdir().expect("tempdir");
    let not_a_dir = dir.path().join("cache-dir-is-a-file");
    std::fs::write(&not_a_dir, b"this is a file, not a directory").expect("plant file");
    let key = "write-fails";
    let path = engine_cache_path(&not_a_dir, key).expect("cache_path");

    let mut compiles = 0;
    let engine: GpuLiteralSet = cached_load_or_compile(&not_a_dir, key, || {
        compiles += 1;
        GpuLiteralSet::compile(&[b"test".as_slice()])
    });
    assert_eq!(compiles, 1, "must compile when the cache write fails");
    assert_eq!(
        engine.reference_scan(b"test"),
        vec![Match::new(0, 0, 4)],
        "the engine must scan correctly even though it could not be cached"
    );
    assert!(
        !path.exists(),
        "no cache file may be published when the write failed"
    );
    assert!(
        not_a_dir.is_file(),
        "the failed write must not have replaced the planted file"
    );
}

/// A writable cache directory does publish the cache.
///
/// The positive twin. Without it, a helper that never wrote anything at all
/// would satisfy the failure case above.
#[test]
fn a_writable_cache_directory_publishes_the_engine() {
    let dir = tempfile::tempdir().expect("tempdir");
    let key = "write-succeeds";
    let path = engine_cache_path(dir.path(), key).expect("cache_path");

    let mut compiles = 0;
    let _: GpuLiteralSet = cached_load_or_compile(dir.path(), key, || {
        compiles += 1;
        GpuLiteralSet::compile(&[b"test".as_slice()])
    });
    assert_eq!(compiles, 1);
    assert!(path.is_file(), "the cache file must be published at {path:?}");

    // And a second call reads it back instead of recompiling.
    let mut recompiles = 0;
    let cached: GpuLiteralSet = cached_load_or_compile(dir.path(), key, || {
        recompiles += 1;
        GpuLiteralSet::compile(&[b"test".as_slice()])
    });
    assert_eq!(recompiles, 0, "the second call must hit the cache");
    assert_eq!(cached.reference_scan(b"test"), vec![Match::new(0, 0, 4)]);
}

/// No temp file is left behind, whichever way the write goes.
///
/// The temp path is an implementation detail, so this checks the observable
/// consequence instead: after a successful publish the only thing in the cache
/// directory is the cache file itself.
#[test]
fn publishing_the_cache_leaves_no_temp_files_behind() {
    let dir = tempfile::tempdir().expect("tempdir");
    let key = "no-litter";
    let _: GpuLiteralSet = cached_load_or_compile(dir.path(), key, || {
        GpuLiteralSet::compile(&[b"test".as_slice()])
    });
    let leftovers: Vec<String> = std::fs::read_dir(dir.path())
        .expect("read cache dir")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp."))
        .collect();
    assert!(
        leftovers.is_empty(),
        "the publish path must rename its temp file, not leave {leftovers:?}"
    );
}

#[test]
fn tempfile_rename_in_tmp() {
    let dir = tempfile::tempdir_in("/tmp").expect("tempdir in /tmp");
    let key = "tmp-rename";

    let engine: GpuLiteralSet = cached_load_or_compile(dir.path(), key, || {
        GpuLiteralSet::compile(&[b"test".as_slice()])
    });
    assert_eq!(engine.reference_scan(b"test"), vec![Match::new(0, 0, 4)]);

    let mut compiles = 0;
    let _: GpuLiteralSet = cached_load_or_compile(dir.path(), key, || {
        compiles += 1;
        GpuLiteralSet::compile(&[b"test".as_slice()])
    });
    assert_eq!(compiles, 0, "must hit cache when stored in /tmp");
}

#[test]
fn unicode_cache_key_and_dir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache_dir = dir.path().join("缓存目录🚀");
    let key = "キー_מפתח_مفتاح";

    let engine: GpuLiteralSet = cached_load_or_compile(&cache_dir, key, || {
        GpuLiteralSet::compile(&[b"unicode".as_slice()])
    });
    assert_eq!(engine.reference_scan(b"unicode"), vec![Match::new(0, 0, 7)]);
    assert!(
        engine_cache_path(&cache_dir, key).unwrap().is_file(),
        "unicode cache file must exist"
    );
}

#[test]
fn cache_dir_is_file_not_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache_dir = dir.path().join("is-a-file");
    std::fs::write(&cache_dir, b"not a dir").expect("write file");

    let key = "file-dir";
    let mut compiles = 0;
    let engine: GpuLiteralSet = cached_load_or_compile(&cache_dir, key, || {
        compiles += 1;
        GpuLiteralSet::compile(&[b"test".as_slice()])
    });
    assert_eq!(
        compiles, 1,
        "must compile when cache_dir is a file (create_dir_all fails)"
    );
    assert_eq!(engine.reference_scan(b"test"), vec![Match::new(0, 0, 4)]);
}

#[test]
fn cache_key_with_path_separator_does_not_panic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let key = "a/b/c";

    // The helper must not panic even if the cache key contains path
    // separators.  It may or may not successfully write the cache.
    let engine: GpuLiteralSet = cached_load_or_compile(dir.path(), key, || {
        GpuLiteralSet::compile(&[b"test".as_slice()])
    });
    assert_eq!(engine.reference_scan(b"test"), vec![Match::new(0, 0, 4)]);
}

// ---------------------------------------------------------------------------
// 4. Cache-key contract (7 tests)
// ---------------------------------------------------------------------------

#[test]
fn same_patterns_same_cache_key() {
    let a = GpuLiteralSet::compile(&[b"AKIA".as_slice(), b"ghp_".as_slice()]);
    let b = GpuLiteralSet::compile(&[b"AKIA".as_slice(), b"ghp_".as_slice()]);
    assert_eq!(
        MatchScan::cache_key(&a),
        MatchScan::cache_key(&b),
        "identical patterns must yield identical cache keys"
    );
}

#[test]
fn reordering_changes_cache_key() {
    let a = GpuLiteralSet::compile(&[b"first".as_slice(), b"second".as_slice()]);
    let b = GpuLiteralSet::compile(&[b"second".as_slice(), b"first".as_slice()]);
    assert_ne!(
        MatchScan::cache_key(&a),
        MatchScan::cache_key(&b),
        "reordering patterns must change cache key"
    );
}

#[test]
fn removing_pattern_changes_cache_key() {
    let full = GpuLiteralSet::compile(&[b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]);
    let partial = GpuLiteralSet::compile(&[b"a".as_slice(), b"b".as_slice()]);
    assert_ne!(
        MatchScan::cache_key(&full),
        MatchScan::cache_key(&partial),
        "removing a pattern must change cache key"
    );
}

#[test]
fn single_byte_mutation_changes_cache_key() {
    let a = GpuLiteralSet::compile(&[b"AKIA".as_slice()]);
    let b = GpuLiteralSet::compile(&[b"AKIB".as_slice()]);
    assert_ne!(
        MatchScan::cache_key(&a),
        MatchScan::cache_key(&b),
        "single-byte mutation must change cache key"
    );
}

#[test]
fn cross_process_determinism_literal_set_known_constant() {
    // FNV-1a64 over the wire buffer for [b"VYRE"] is deterministic.
    //
    // Repinned from 6fbbc5c22cb738b9 to the value the literal-set program
    // actually encodes to. The old constant was already stale at the 0.7.0
    // commit: a clean checkout of that commit produces this same key, so the
    // encoding drifted under some earlier change and nothing caught it, because
    // `cargo test --workspace --all-features` could not run to completion.
    // The two other places that check this key (vyre-core/src/scan/literal_set.rs
    // and vyre-libs/tests/cross_layer_parity.rs) recompute it from the wire
    // buffer instead of pinning a literal, which is why only this one drifted.
    let engine = GpuLiteralSet::compile(&[b"VYRE".as_slice()]);
    let key = MatchScan::cache_key(&engine);
    assert_eq!(
        key, "lit-264e7c96c5bbfcc9",
        "cache key must match known cross-process constant"
    );
}

#[cfg(feature = "matching-nfa")]
#[test]
fn cross_process_determinism_rule_pipeline_stable() {
    let pipe = build_rule_pipeline(&["abc", "de"], "input", "hits", 8);
    let key = MatchScan::cache_key(&pipe);
    let pipe2 = build_rule_pipeline(&["abc", "de"], "input", "hits", 8);
    assert_eq!(
        key,
        MatchScan::cache_key(&pipe2),
        "RulePipeline cache key must be stable across recomputations"
    );
}

#[test]
fn different_engines_different_keys() {
    let literal = GpuLiteralSet::compile(&[b"abc".as_slice()]);
    let literal_key = MatchScan::cache_key(&literal);
    assert!(
        literal_key.starts_with("lit-"),
        "literal key must use lit- prefix"
    );

    #[cfg(feature = "matching-nfa")]
    {
        let pipe = build_rule_pipeline(&["abc"], "input", "hits", 8);
        let pipe_key = MatchScan::cache_key(&pipe);
        assert!(
            pipe_key.starts_with("pipe-"),
            "pipeline key must use pipe- prefix"
        );
        assert_ne!(
            literal_key, pipe_key,
            "different engines must not share cache keys"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Trait object dispatch (7 tests)
// ---------------------------------------------------------------------------

#[test]
fn box_dyn_match_scan_gpu_literal_set_reference_scan() {
    let engine: Box<dyn MatchScan> = Box::new(GpuLiteralSet::compile(&[b"abc".as_slice()]));
    assert_eq!(engine.reference_scan(b"zabc"), vec![Match::new(0, 1, 4)]);
}

#[cfg(feature = "matching-nfa")]
#[test]
fn box_dyn_match_scan_rule_pipeline_reference_scan() {
    let engine: Box<dyn MatchScan> =
        Box::new(build_rule_pipeline(&["abc", "bc"], "input", "hits", 4));
    let matches = engine.reference_scan(b"zabc");
    assert!(matches.contains(&Match::new(0, 1, 4)));
    assert!(matches.contains(&Match::new(1, 2, 4)));
}

#[test]
fn vec_mixed_engines_reference_scan() {
    let mut engines: Vec<Box<dyn MatchScan>> =
        vec![Box::new(GpuLiteralSet::compile(&[b"abc".as_slice()]))];

    #[cfg(feature = "matching-nfa")]
    {
        engines.push(Box::new(build_rule_pipeline(&["abc"], "input", "hits", 8)));
    }

    for engine in &engines {
        let matches = engine.reference_scan(b"zabc");
        assert!(
            matches
                .iter()
                .any(|m| m.pattern_id == 0 && m.start == 1 && m.end == 4),
            "each engine in Vec<Box<dyn MatchScan>> must find 'abc' in 'zabc': got {matches:?}"
        );
    }
}

#[test]
fn scan_through_dyn_ref() {
    let engine = GpuLiteralSet::compile(&[b"abc".as_slice()]);
    let dyn_ref: &dyn MatchScan = &engine;

    let backend = vyre_driver_wgpu::WgpuBackend::new()
        .expect("Fix: scan_through_dyn_ref requires a live GPU");
    let matches = dyn_ref
        .scan(&backend, b"zabc", 10_000)
        .expect("scan through &dyn MatchScan must succeed");
    assert_eq!(matches, vec![Match::new(0, 1, 4)]);
}

#[test]
fn reference_scan_through_dyn_ref() {
    let engine = GpuLiteralSet::compile(&[b"abc".as_slice()]);
    let dyn_ref: &dyn MatchScan = &engine;
    assert_eq!(
        dyn_ref.reference_scan(b"zabc"),
        vec![Match::new(0, 1, 4)],
        "reference_scan through &dyn MatchScan must work"
    );
}

#[test]
fn cache_key_through_dyn_ref() {
    let engine = GpuLiteralSet::compile(&[b"abc".as_slice()]);
    let dyn_ref: &dyn MatchScan = &engine;
    let key = dyn_ref.cache_key();
    assert!(
        key.starts_with("lit-"),
        "cache_key through &dyn MatchScan must return expected prefix"
    );
}

#[cfg(feature = "matching-nfa")]
#[test]
fn rule_pipeline_scan_through_dyn_ref() {
    let engine = build_rule_pipeline(&["abc", "bc"], "input", "hits", 4);
    let dyn_ref: &dyn MatchScan = &engine;

    let backend = vyre_driver_wgpu::WgpuBackend::new()
        .expect("Fix: rule_pipeline_scan_through_dyn_ref requires a live GPU");
    let matches = dyn_ref
        .scan(&backend, b"zabc", 10_000)
        .expect("scan through &dyn MatchScan must succeed");
    assert!(
        matches.contains(&Match::new(0, 1, 4)),
        "expected abc match at (1,4), got {:?}",
        matches
    );
    assert!(
        matches.contains(&Match::new(1, 2, 4)),
        "expected bc match at (2,4), got {:?}",
        matches
    );
}
