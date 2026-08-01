//! Organization-level contract tests for the vyre-driver-wgpu crate.
//!
//! These tests enforce long-term structural contracts without relying on
//! brittle message wording. They may fail when code violates a contract.

use std::collections::HashSet;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// 1. Inline test modules are baselined in vyre-driver-wgpu/src
// ---------------------------------------------------------------------------

/// Files in `vyre-driver-wgpu/src` that carry an inline `#[cfg(test)]` module.
///
/// The organization contract is that new tests belong under `tests/`, where
/// they exercise the crate through its public surface. Inline modules are
/// allowed only where a test genuinely needs a private item, and every one of
/// them is listed here so adding another is a deliberate act.
const BASELINED_INLINE_TEST_MODULES: &[&str] = &[
    "src/async_dispatch.rs",
    "src/backend_impl.rs",
    "src/buffer/handle.rs",
    "src/buffer/pool.rs",
    "src/device_buffer.rs",
    "src/emit/descriptor_gate.rs",
    "src/emit/mod.rs",
    "src/engine/dispatch_scratch.rs",
    "src/engine/multi_gpu.rs",
    "src/engine/multi_gpu/partition.rs",
    "src/engine/multi_gpu/stream_shard.rs",
    "src/engine/persistent.rs",
    "src/engine/record_and_readback/binding_lookup.rs",
    "src/engine/streaming/async_copy.rs",
    "src/ext.rs",
    "src/megakernel.rs",
    "src/megakernel/batch.rs",
    "src/megakernel/dispatch_plan.rs",
    "src/megakernel/dispatcher.rs",
    "src/megakernel/segmentation.rs",
    "src/numeric.rs",
    "src/parity_probe.rs",
    "src/pipeline.rs",
    "src/pipeline/binding.rs",
    "src/pipeline/bindings_reflection.rs",
    "src/pipeline/compiled_dispatch.rs",
    "src/pipeline/compound.rs",
    "src/pipeline/descriptor_metadata.rs",
    "src/pipeline/disk_cache.rs",
    "src/pipeline/disk_cache_invalidation.rs",
    "src/pipeline/output_slots.rs",
    "src/pipeline/persistent.rs",
    "src/pipeline/tests/layout_config_contracts.rs",
    "src/runtime/adapter_caps_probe.rs",
    "src/runtime/cache.rs",
    "src/runtime/cache/lru.rs",
    "src/runtime/cache/pipeline.rs",
    "src/runtime/cache/tiered_cache.rs",
    "src/runtime/device/device.rs",
    "src/runtime/device/selector.rs",
    "src/runtime/indirect.rs",
    "src/runtime/readback_ring.rs",
    "src/runtime/router.rs",
    "src/runtime/serializer/decode_parts.rs",
    "src/runtime/serializer/encode_parts.rs",
    "src/spirv_backend.rs",
    "src/staging_reserve.rs",
    "src/wait_backoff.rs",
];

/// Collect every `src` file carrying an inline `#[cfg(test)]` module.
fn inline_test_modules_in_src() -> HashSet<String> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut found = HashSet::new();
    let mut stack = vec![manifest.join("src")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                let content = std::fs::read_to_string(&path).unwrap();
                if content.contains("#[cfg(test)]") {
                    let rel = path.strip_prefix(&manifest).unwrap_or(&path);
                    found.insert(rel.display().to_string());
                }
            }
        }
    }
    found
}

/// Organization contract: new tests must live in `tests/`, not inline source
/// modules.
///
/// The ratchet only holds if it is checked. This one had drifted by 26 files:
/// inline `#[cfg(test)]` modules were added across the crate and nothing
/// stopped them, because `cargo test --workspace --all-features` could not run
/// to completion, so the assertion never executed. The list has been refreshed
/// to what the crate actually contains; every entry above is a deliberate
/// allowance, and a new one is a violation.
#[test]
fn driver_wgpu_inline_test_modules_are_baselined() {
    let known: HashSet<String> = BASELINED_INLINE_TEST_MODULES
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    let mut new_violations: Vec<String> = inline_test_modules_in_src()
        .into_iter()
        .filter(|v| !known.contains(v))
        .collect();
    new_violations.sort();

    assert!(
        new_violations.is_empty(),
        "new inline test modules (#[cfg(test)]) are forbidden in vyre-driver-wgpu/src. \
         Add integration tests under tests/ instead. New violations:\n{}",
        new_violations.join("\n")
    );
}

/// The baseline may not name a file that no longer has an inline test module.
///
/// A one-directional ratchet rots silently: when a file's inline tests move to
/// `tests/`, its baseline entry becomes a standing permission for someone to
/// put them back. Five entries had gone stale that way (`src/lib.rs`, the three
/// `lowering/naga_emit` files, and `src/runtime/cache/buffer_pool.rs`). Failing
/// on a stale entry keeps the list an exact description of the crate rather
/// than a historical high-water mark, and makes the allowance shrink over time
/// as inline tests are migrated out.
#[test]
fn the_inline_test_baseline_contains_no_stale_entries() {
    let found = inline_test_modules_in_src();
    let mut stale: Vec<&str> = BASELINED_INLINE_TEST_MODULES
        .iter()
        .copied()
        .filter(|entry| !found.contains(*entry))
        .collect();
    stale.sort_unstable();

    assert!(
        stale.is_empty(),
        "BASELINED_INLINE_TEST_MODULES names files that no longer carry an inline \
         #[cfg(test)] module. Remove them so the allowance cannot be silently reclaimed:\n{}",
        stale.join("\n")
    );
}

/// The baseline is sorted and free of duplicates.
///
/// Keeps the list reviewable and makes a merge that adds the same path twice
/// visible instead of harmless-looking.
#[test]
fn the_inline_test_baseline_is_sorted_and_unique() {
    let mut sorted = BASELINED_INLINE_TEST_MODULES.to_vec();
    sorted.sort_unstable();
    assert_eq!(
        sorted, BASELINED_INLINE_TEST_MODULES,
        "BASELINED_INLINE_TEST_MODULES must stay in sorted order"
    );

    let unique: HashSet<&str> = BASELINED_INLINE_TEST_MODULES.iter().copied().collect();
    assert_eq!(
        unique.len(),
        BASELINED_INLINE_TEST_MODULES.len(),
        "BASELINED_INLINE_TEST_MODULES must not repeat a path"
    );
}

// ---------------------------------------------------------------------------
// 2. Wildcard pub-use surface is baselined
// ---------------------------------------------------------------------------

/// Scan vyre-driver-wgpu/src for `pub use ...::*` and baseline them.
/// New wildcard re-exports expand API surface unpredictably and are forbidden
/// without explicit approval.
#[test]
fn driver_wgpu_wildcard_pub_use_is_baselined() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = manifest.join("src");
    let mut found = Vec::new();

    let mut stack = vec![src];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                let content = std::fs::read_to_string(&path).unwrap();
                for (line_no, line) in content.lines().enumerate() {
                    let t = line.trim();
                    if t.starts_with("pub use") && t.ends_with("::*;") {
                        let rel = path.strip_prefix(&manifest).unwrap_or(&path);
                        found.push(format!("{}:{} {}", rel.display(), line_no + 1, t));
                    }
                }
            }
        }
    }

    // Currently zero wildcards in vyre-driver-wgpu/src.
    let known: HashSet<String> = HashSet::new();

    let new_violations: Vec<String> = found.into_iter().filter(|v| !known.contains(v)).collect();

    assert!(
        new_violations.is_empty(),
        "new wildcard pub re-exports are forbidden in vyre-driver-wgpu. Violations:\n{}",
        new_violations.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 3. Agent/skills artifacts stay out of production crate dirs
// ---------------------------------------------------------------------------

/// Organization contract: AGENTS.md, SKILL.md, and .kimi/ directories must not
/// appear in vyre-driver-wgpu production directories (src/ or crate root).
#[test]
fn driver_wgpu_agent_skills_artifacts_stay_out_of_production_dirs() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut found = Vec::new();

    // Scan src/ directory
    let src_dir = manifest.join("src");
    if src_dir.is_dir() {
        let mut stack = vec![src_dir];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let fname = path.file_name().and_then(|s| s.to_str());
                    if fname == Some("AGENTS.md") || fname == Some("SKILL.md") {
                        let rel = path.strip_prefix(&manifest).unwrap_or(&path);
                        found.push(rel.display().to_string());
                    }
                }
            }
        }
    }

    // Check crate root
    for name in ["AGENTS.md", "SKILL.md"] {
        let path = manifest.join(name);
        if path.exists() {
            let rel = path.strip_prefix(&manifest).unwrap_or(&path);
            found.push(rel.display().to_string());
        }
    }

    // Check for .kimi/ anywhere, excluding tests/benches/examples/target/.internals
    let mut kstack = vec![manifest.clone()];
    while let Some(dir) = kstack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if path.file_name().and_then(|s| s.to_str()) == Some(".kimi") {
                let rel = path.strip_prefix(&manifest).unwrap_or(&path);
                found.push(rel.display().to_string());
            } else {
                let fname = path.file_name().unwrap().to_string_lossy();
                if fname != "target"
                    && fname != "tests"
                    && fname != "benches"
                    && fname != "examples"
                    && fname != ".internals"
                    && !fname.starts_with('.')
                {
                    kstack.push(path);
                }
            }
        }
    }

    found.sort();

    assert!(
        found.is_empty(),
        "agent/skills artifacts (AGENTS.md, SKILL.md, .kimi/) are forbidden in production dirs. \
         Violations:\n{}",
        found.join("\n")
    );
}
