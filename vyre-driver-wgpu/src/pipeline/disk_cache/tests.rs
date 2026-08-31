#![allow(missing_docs)]

use std::fs::File;

use super::io::{
    read_bounded_bytes, read_metadata, write_atomic, MAX_COMPILED_PIPELINE_CACHE_BLOB_BYTES,
    MAX_PIPELINE_CACHE_METADATA_BYTES,
};
use super::keys::{
    adapter_fingerprint, blake3_hex, hex_hash, metadata_fingerprint, CompiledPipelineCacheKey,
    NAGA_VERSION, WGSL_LOWERING_CONTRACT,
};
use super::*;

/// Serializes the tests that swap the process-wide disk-cache root.
static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Holds the serialising lock and puts the cache root back on the way out.
///
/// The root is process-wide. A case that panicked between swapping it and
/// restoring it left every later case reading a deleted temporary directory,
/// which is why poisoning had to propagate: the lock was the only thing
/// stopping the next case from reading a corrupted global. Restoring on unwind
/// removes that reason, so the lock is recovered instead, one real failure
/// stays one report, and the two neighbours that reported the poisoned lock
/// report their own verdicts again.
struct CacheRootGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    previous: Option<std::path::PathBuf>,
}

impl Drop for CacheRootGuard {
    fn drop(&mut self) {
        set_test_disk_pipeline_cache_root(self.previous.take());
    }
}

/// Serialize against the other root-swapping cases and install `root`.
///
/// `None` selects the default root, which is the state a case that swaps
/// nothing expects to find.
fn env_lock(root: Option<std::path::PathBuf>) -> CacheRootGuard {
    let lock = ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let previous = set_test_disk_pipeline_cache_root(root);
    CacheRootGuard {
        _lock: lock,
        previous,
    }
}

#[test]
fn fixed_digest_hex_hash_is_lowercase_and_stack_encoded() {
    let mut digest = [0_u8; 32];
    digest[0] = 0xab;
    digest[31] = 0x7f;

    let hex = hex_hash(&digest);

    assert_eq!(hex.len(), 64);
    assert!(hex.starts_with("ab00"));
    assert!(hex.ends_with("007f"));
    assert!(hex.bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(hex, hex.to_ascii_lowercase());
}

mod cache_key_contracts {
    use super::*;

    #[test]
    fn cache_key_isolates_wire_from_adapter() {
        // Two different (wire, fingerprint) pairs whose concatenation would
        // collide under a naïve concat hash must still produce different
        // cache keys because the domain separators intervene.
        let cfg = DispatchConfig::default();
        let k1 = wgsl_cache_key(b"ab", "cd", &cfg);
        let k2 = wgsl_cache_key(b"a", "bcd", &cfg);
        assert_ne!(
            k1, k2,
            "wire/adapter boundaries must not collapse into a single blob"
        );

        // Same (wire, fingerprint) pair must be deterministic across calls.
        let k3 = wgsl_cache_key(b"ab", "cd", &cfg);
        assert_eq!(k1, k3);
    }

    #[test]
    fn adapter_change_invalidates_cache_match() {
        // Given the same wire, a different adapter fingerprint must miss.
        let wire = b"some-wire-bytes".as_slice();
        let cfg = DispatchConfig::default();
        let k_a = wgsl_cache_key(wire, "adapter-alpha", &cfg);
        let k_b = wgsl_cache_key(wire, "adapter-beta", &cfg);
        assert_ne!(k_a, k_b);
    }

    #[test]
    fn manual_cache_key_strings_preserve_stable_format() {
        let adapter_info = wgpu::AdapterInfo {
            name: "test-adapter".to_string(),
            vendor: 0x1234,
            device: 0x5678,
            device_type: wgpu::DeviceType::Other,
            driver: "driver".to_string(),
            driver_info: "info".to_string(),
            backend: wgpu::Backend::Vulkan,
        };
        assert_eq!(
            adapter_fingerprint(&adapter_info),
            "Vulkan:00001234:00005678:driver:info"
        );

        let mut config = DispatchConfig::default();
        config.ulp_budget = Some(7);
        config.workgroup_override = Some([8, 16, 32]);
        assert_eq!(
            vyre_driver::dispatch_policy_cache_string(&config),
            "ulp=Some(7):wg=Some([8, 16, 32])"
        );
    }

    #[test]
    fn content_digest_rejects_corrupted_payload() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let meta_path = dir.path().join("meta.toml");

        let wgsl = "genuine shader content";
        let cache_key = wgsl_cache_key(b"key123", "fingerprint", &DispatchConfig::default());

        let metadata = DiskPipelineMetadata {
            version: DISK_PIPELINE_CACHE_VERSION,
            cache_key,
            wgsl_bytes: wgsl.len(),
            adapter_fingerprint: metadata_fingerprint("fingerprint"),
            program_abi_version: u32::from(WIRE_FORMAT_VERSION),
            naga_version: std::borrow::Cow::Borrowed(NAGA_VERSION),
            wgsl_lowering_contract: std::borrow::Cow::Borrowed(WGSL_LOWERING_CONTRACT),
            policy: vyre_driver::dispatch_policy_cache_string(&DispatchConfig::default()),
            wgsl_blake3: blake3_hex(wgsl.as_bytes()),
        };
        let mut file = std::fs::File::create(&meta_path).unwrap();
        file.write_all(toml::to_string(&metadata).unwrap().as_bytes())
            .unwrap();

        // Exact match -> true
        assert!(wgsl_metadata_matches(
            &meta_path,
            &cache_key,
            wgsl,
            "fingerprint",
            &DispatchConfig::default()
        ));

        // Match length, but corrupted bytes -> false
        let corrupted_wgsl = "genuine shader corpent";
        assert_eq!(corrupted_wgsl.len(), wgsl.len());
        assert!(!wgsl_metadata_matches(
            &meta_path,
            &cache_key,
            corrupted_wgsl,
            "fingerprint",
            &DispatchConfig::default()
        ));
    }

    #[test]
    fn wgsl_cache_key_includes_lowering_contract() {
        let cfg = DispatchConfig::default();
        let digest = b"normalized-program-digest";
        let fingerprint = "Vulkan:00000000:00000000:test:driver";
        let real = wgsl_cache_key(digest, fingerprint, &cfg);

        let mut legacy_hasher = blake3::Hasher::new();
        legacy_hasher.update(b"vyre-pipeline-cache-v7\0norm\0");
        legacy_hasher.update(digest);
        legacy_hasher.update(b"\0adapter\0");
        legacy_hasher.update(fingerprint.as_bytes());
        legacy_hasher.update(b"\0abi\0");
        legacy_hasher.update(&WIRE_FORMAT_VERSION.to_le_bytes());
        legacy_hasher.update(b"\0naga\0");
        legacy_hasher.update(NAGA_VERSION.as_bytes());
        legacy_hasher.update(b"\0policy\0");
        vyre_driver::update_dispatch_policy_cache_hash(&mut legacy_hasher, &cfg);

        assert_ne!(
            real,
            *legacy_hasher.finalize().as_bytes(),
            "WGSL cache keys must include the lowering contract so stale lowered shaders cannot survive emitter semantic changes"
        );
    }

    #[test]
    fn wgsl_metadata_rejects_stale_lowering_contract() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let meta_path = dir.path().join("meta.toml");
        let wgsl = "shader";
        let cache_key = wgsl_cache_key(b"key123", "fingerprint", &DispatchConfig::default());
        let metadata = DiskPipelineMetadata {
            version: DISK_PIPELINE_CACHE_VERSION,
            cache_key,
            wgsl_bytes: wgsl.len(),
            adapter_fingerprint: metadata_fingerprint("fingerprint"),
            program_abi_version: u32::from(WIRE_FORMAT_VERSION),
            naga_version: std::borrow::Cow::Borrowed(NAGA_VERSION),
            wgsl_lowering_contract: std::borrow::Cow::Borrowed("old-contract"),
            policy: vyre_driver::dispatch_policy_cache_string(&DispatchConfig::default()),
            wgsl_blake3: blake3_hex(wgsl.as_bytes()),
        };
        let mut file = std::fs::File::create(&meta_path).unwrap();
        file.write_all(toml::to_string(&metadata).unwrap().as_bytes())
            .unwrap();

        assert!(
            !wgsl_metadata_matches(
                &meta_path,
                &cache_key,
                wgsl,
                "fingerprint",
                &DispatchConfig::default()
            ),
            "WGSL metadata must reject entries produced under an old lowering contract"
        );
    }

    #[test]
    fn cache_writes_are_durable_on_explicit_flush_not_insert() {
        let _lock = env_lock(None);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entry.wgsl");
        write_atomic(&path, b"shader", "test cache data")
            .expect("Fix: cache write must install the entry before explicit flush.");

        flush_disk_pipeline_cache()
            .expect("Fix: explicit pipeline cache flush must fsync pending writes.");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"shader",
            "Fix: explicit flush must preserve the installed cache payload."
        );
    }

    #[test]
    fn oversized_pipeline_metadata_is_rejected_before_parse() {
        let dir = tempfile::tempdir().unwrap();
        let meta_path = dir.path().join("oversized.pipeline.toml");
        std::fs::write(
            &meta_path,
            vec![b'a'; MAX_PIPELINE_CACHE_METADATA_BYTES as usize + 1],
        )
        .unwrap();

        assert!(
            read_metadata::<CompiledPipelineMetadata>(&meta_path).is_err(),
            "Fix: oversized compiled-pipeline metadata must be rejected before TOML parsing"
        );
    }

    #[test]
    fn oversized_compiled_pipeline_blob_is_rejected_before_read() {
        let dir = tempfile::tempdir().unwrap();
        let blob_path = dir.path().join("oversized.pipeline.bin");
        let file = File::create(&blob_path).unwrap();
        file.set_len(MAX_COMPILED_PIPELINE_CACHE_BLOB_BYTES + 1)
            .unwrap();
        drop(file);

        let error = read_bounded_bytes(&blob_path, MAX_COMPILED_PIPELINE_CACHE_BLOB_BYTES)
            .expect_err("oversized compiled-pipeline blob must fail before allocation");
        assert_eq!(
            error.kind(),
            std::io::ErrorKind::InvalidData,
            "Fix: oversized compiled-pipeline blobs must return InvalidData, got {error:?}"
        );
    }

    #[test]
    fn stale_compiled_pipeline_adapter_metadata_misses() {
        let temp = tempfile::tempdir().unwrap();
        let _lock = env_lock(Some(temp.path().to_path_buf()));

        let key = CompiledPipelineCacheKey {
            hash: [7u8; 32],
            adapter_fingerprint: "current-adapter".to_string(),
            cache_key: "stale-adapter-key".to_string(),
            wgsl_blake3: blake3_hex(b"wgsl"),
        };
        let dir = disk_pipeline_cache_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let blob = b"driver-cache-bytes";
        std::fs::write(
            cache_entry_path(&dir, &key.cache_key, ".pipeline.bin"),
            blob,
        )
        .unwrap();
        let metadata = CompiledPipelineMetadata {
            version: DISK_PIPELINE_CACHE_VERSION,
            cache_key: key.hash,
            adapter_fingerprint: metadata_fingerprint("old-adapter"),
            wgsl_blake3: key.wgsl_blake3.clone(),
            program_abi_version: u32::from(WIRE_FORMAT_VERSION),
            naga_version: std::borrow::Cow::Borrowed(NAGA_VERSION),
            blob_bytes: blob.len(),
            blob_blake3: blake3_hex(blob),
        };
        std::fs::write(
            cache_entry_path(&dir, &key.cache_key, ".pipeline.toml"),
            toml::to_string(&metadata).unwrap(),
        )
        .unwrap();

        let result = load_compiled_pipeline_blob(&key).expect(
            "Fix: stale metadata must be a miss; restore this invariant before continuing.",
        );
        assert!(
            result.is_none(),
            "Fix: compiled-pipeline cache must miss when adapter fingerprint metadata is stale"
        );
    }

    #[test]
    fn normalized_cache_digest_erases_runtime_storage_lengths() {
        let entry = vec![vyre_foundation::ir::Node::return_()];
        let a = Program::wrapped(
            vec![
                vyre_foundation::ir::BufferDecl::read(
                    "haystack",
                    0,
                    vyre_foundation::ir::DataType::U32,
                )
                .with_count(8),
                vyre_foundation::ir::BufferDecl::output(
                    "matches",
                    1,
                    vyre_foundation::ir::DataType::U32,
                )
                .with_count(8)
                .with_output_byte_range(0..32),
            ],
            [64, 1, 1],
            entry.clone(),
        );
        let b = Program::wrapped(
            vec![
                vyre_foundation::ir::BufferDecl::read(
                    "haystack",
                    0,
                    vyre_foundation::ir::DataType::U32,
                )
                .with_count(1024),
                vyre_foundation::ir::BufferDecl::output(
                    "matches",
                    1,
                    vyre_foundation::ir::DataType::U32,
                )
                .with_count(1024)
                .with_output_byte_range(0..4096),
            ],
            [64, 1, 1],
            entry,
        );

        assert_eq!(
            vyre_driver::normalized_program_cache_digest(&a),
            vyre_driver::normalized_program_cache_digest(&b),
            "storage buffer lengths must not perturb the compile fingerprint"
        );
    }

    #[test]
    fn early_pipeline_cache_key_preserves_runtime_storage_lengths() {
        let adapter = wgpu::AdapterInfo {
            name: "cache-test".to_string(),
            vendor: 0x10de,
            device: 0x5090,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: "driver".to_string(),
            driver_info: "info".to_string(),
            backend: wgpu::Backend::Vulkan,
        };
        let entry = vec![vyre_foundation::ir::Node::return_()];
        let small = Program::wrapped(
            vec![
                vyre_foundation::ir::BufferDecl::read(
                    "haystack",
                    0,
                    vyre_foundation::ir::DataType::U32,
                )
                .with_count(8),
                vyre_foundation::ir::BufferDecl::output(
                    "matches",
                    1,
                    vyre_foundation::ir::DataType::U32,
                )
                .with_count(8)
                .with_output_byte_range(0..32),
            ],
            [64, 1, 1],
            entry.clone(),
        );
        let large = Program::wrapped(
            vec![
                vyre_foundation::ir::BufferDecl::read(
                    "haystack",
                    0,
                    vyre_foundation::ir::DataType::U32,
                )
                .with_count(4096),
                vyre_foundation::ir::BufferDecl::output(
                    "matches",
                    1,
                    vyre_foundation::ir::DataType::U32,
                )
                .with_count(4096)
                .with_output_byte_range(0..16_384),
            ],
            [64, 1, 1],
            entry,
        );

        assert_ne!(
            small.fingerprint(),
            large.fingerprint(),
            "test programs must differ at the raw Program fingerprint layer"
        );
        assert_ne!(
            early_pipeline_cache_key(&small, &adapter, &DispatchConfig::default()),
            early_pipeline_cache_key(&large, &adapter, &DispatchConfig::default()),
            "Fix: in-memory compiled-pipeline artifacts carry binding and output metadata, so shape-distinct Programs must not share an early cache key."
        );
    }

    /// Wire a Program to a disk cache key exactly as `load_or_compile_disk_wgsl`
    /// does: normalized digest, then `wgsl_cache_key`.
    fn disk_cache_key_for(program: &vyre_foundation::ir::Program) -> [u8; 32] {
        let norm_digest = vyre_driver::try_normalized_program_cache_digest(program)
            .expect("Fix: fixture Program must produce a normalized cache digest");
        wgsl_cache_key(
            &norm_digest,
            "Vulkan:00000000:00000000:test:driver",
            &DispatchConfig::default(),
        )
    }

    /// Two Programs whose buffers differ only by swapped binding slots must not
    /// share a WGSL disk cache key.
    ///
    /// This is the defect this test exists to lock out, and the wgpu key is where it
    /// was FATAL rather than merely wasteful. `wgsl_cache_key` mixes exactly one
    /// program-derived input, the normalized digest, so unlike the CUDA PTX key it
    /// has no second lane that could discriminate these two programs incidentally.
    /// Before the digest keyed `binding`, these two programs produced the same key,
    /// so the cache returned the shader compiled with the other bind-group layout
    /// and every dispatch wrote its results into the wrong buffer, silently and
    /// with no error anywhere.
    #[test]
    fn wgsl_disk_cache_key_separates_swapped_buffer_bindings() {
        let entry = vec![
            vyre_foundation::ir::Node::store(
                "a",
                vyre_foundation::ir::Expr::u32(0),
                vyre_foundation::ir::Expr::u32(1),
            ),
            vyre_foundation::ir::Node::store(
                "b",
                vyre_foundation::ir::Expr::u32(0),
                vyre_foundation::ir::Expr::u32(2),
            ),
        ];
        let straight = Program::wrapped(
            vec![
                vyre_foundation::ir::BufferDecl::output("a", 0, vyre_foundation::ir::DataType::U32)
                    .with_count(64),
                vyre_foundation::ir::BufferDecl::output("b", 1, vyre_foundation::ir::DataType::U32)
                    .with_count(64),
            ],
            [64, 1, 1],
            entry.clone(),
        );
        let swapped = Program::wrapped(
            vec![
                vyre_foundation::ir::BufferDecl::output("a", 1, vyre_foundation::ir::DataType::U32)
                    .with_count(64),
                vyre_foundation::ir::BufferDecl::output("b", 0, vyre_foundation::ir::DataType::U32)
                    .with_count(64),
            ],
            [64, 1, 1],
            entry,
        );

        assert_ne!(
            disk_cache_key_for(&straight),
            disk_cache_key_for(&swapped),
            "Fix: binding slots reach the generated WGSL bind-group layout, so two binding \
             layouts must not share one disk cache entry; sharing one serves a shader that \
             writes into the wrong buffer."
        );
    }

    /// Two Programs differing only in a workgroup array LENGTH must not share a
    /// WGSL disk cache key.
    ///
    /// `MemoryKind::Shared` is the one memory class whose `element_count` the naga
    /// emitter bakes into shader text, as `var<workgroup> tile: array<u32, N>`.
    /// Sharing a cache entry across two values of N returns a shader whose workgroup
    /// array is the wrong size, so indexing runs past it.
    #[test]
    fn wgsl_disk_cache_key_separates_workgroup_array_lengths() {
        let build = |shared_len: u32| {
            Program::wrapped(
                vec![
                    vyre_foundation::ir::BufferDecl::output(
                        "out",
                        0,
                        vyre_foundation::ir::DataType::U32,
                    )
                    .with_count(64),
                    vyre_foundation::ir::BufferDecl::workgroup(
                        "tile",
                        shared_len,
                        vyre_foundation::ir::DataType::U32,
                    ),
                ],
                [64, 1, 1],
                vec![vyre_foundation::ir::Node::store(
                    "out",
                    vyre_foundation::ir::Expr::u32(0),
                    vyre_foundation::ir::Expr::u32(1),
                )],
            )
        };

        assert_ne!(
            disk_cache_key_for(&build(64)),
            disk_cache_key_for(&build(128)),
            "Fix: a workgroup array length is baked into WGSL text, so two lengths must not \
             share one disk cache entry."
        );
    }

    /// Resizing a RUNTIME storage buffer must NOT change the WGSL disk cache key.
    ///
    /// The incidental-protection twin of the two tests above, and the reason the
    /// digest keys `count` conditionally instead of always. Storage lengths are
    /// erased in WGSL (`ArraySize::Dynamic`), so a key that changed on resize would
    /// force a full naga recompile for every new input size, which costs far more
    /// than the dispatch it precedes. Together with the tests above this pins the
    /// key to discriminate exactly what reaches shader text and nothing more.
    #[test]
    fn wgsl_disk_cache_key_survives_runtime_storage_resize() {
        let build = |count: u32| {
            Program::wrapped(
                vec![vyre_foundation::ir::BufferDecl::output(
                    "out",
                    0,
                    vyre_foundation::ir::DataType::U32,
                )
                .with_count(count)],
                [64, 1, 1],
                vec![vyre_foundation::ir::Node::store(
                    "out",
                    vyre_foundation::ir::Expr::u32(0),
                    vyre_foundation::ir::Expr::u32(1),
                )],
            )
        };

        assert_eq!(
            disk_cache_key_for(&build(1024)),
            disk_cache_key_for(&build(1_048_576)),
            "Fix: runtime storage lengths are erased in WGSL, so a resize must reuse the \
             cached shader instead of forcing a naga recompile."
        );
    }
}

mod cache_miss_tracing {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn cache_misses_are_traced_on_fresh_temp_dir() {
        let dir = tempfile::tempdir().unwrap();
        let _lock = env_lock(Some(dir.path().to_path_buf()));

        #[derive(Clone)]
        struct StringWriter(Arc<std::sync::Mutex<String>>);
        impl std::io::Write for StringWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if let Ok(mut s) = self.0.lock() {
                    s.push_str(std::str::from_utf8(buf).unwrap_or_default());
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let captured = Arc::new(std::sync::Mutex::new(String::new()));
        let writer = StringWriter(captured.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_level(true)
            .with_target(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        let adapter_info = wgpu::AdapterInfo {
            name: "test-adapter".to_string(),
            vendor: 0x1234,
            device: 0x5678,
            device_type: wgpu::DeviceType::Other,
            driver: "test-driver".to_string(),
            driver_info: "1.0".to_string(),
            backend: wgpu::Backend::Noop,
        };

        let program = Program::wrapped(
            vec![vyre_foundation::ir::BufferDecl::output(
                "out",
                0,
                vyre_foundation::ir::DataType::U32,
            )
            .with_count(1)],
            [1, 1, 1],
            vec![vyre_foundation::ir::Node::store(
                "out",
                vyre_foundation::ir::Expr::u32(0),
                vyre_foundation::ir::Expr::u32(42),
            )],
        );

        let enabled_features = crate::runtime::device::EnabledFeatures::default();
        let wgsl = load_or_compile_disk_wgsl(
            &program,
            &adapter_info,
            &DispatchConfig::default(),
            &enabled_features,
        )
        .expect("Fix: lowering must succeed on a trivial program; restore this invariant before continuing.");
        let key = compiled_pipeline_cache_key(&adapter_info, &wgsl);
        let blob = load_compiled_pipeline_blob(&key)
            .expect("Fix: blob load must not error; restore this invariant before continuing.");
        assert!(
            blob.is_none(),
            "fresh temp dir must miss compiled pipeline cache"
        );

        let logs = captured
            .lock()
            .expect("Fix: log capture lock must not be poisoned");
        assert!(
            logs.contains("WGSL cache miss"),
            "expected WGSL cache miss info log, got:\n{logs}"
        );
        assert!(
            logs.contains("compiled-pipeline cache miss"),
            "expected compiled-pipeline cache miss warn log, got:\n{logs}"
        );
    }
}
