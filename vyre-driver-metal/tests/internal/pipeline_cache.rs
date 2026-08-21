//! Compiled pipeline reuse, policy partitioning, and cache invalidation on
//! shutdown.

use crate::*;

use super::fixtures::stores_word;
use vyre_driver::DispatchConfig;

#[test]
fn apple_dispatch_reuses_pipeline_cache_for_identical_program_and_policy() {
    let program = stores_word(99);

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before cache testing.",
    );
    let before = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose honest pipeline cache counters.");
    let first = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: first Metal dispatch must compile and execute the program.");
    let after_first = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose cache counters after compile.");
    let second = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: second Metal dispatch must reuse the compiled pipeline and execute.");
    let after_second = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose cache counters after a cache hit.");

    assert_eq!(first, vec![99u32.to_le_bytes().to_vec()]);
    assert_eq!(second, first);
    assert_eq!(
        after_first.misses,
        before.misses + 1,
        "Fix: first identical Metal dispatch should record exactly one pipeline cache miss."
    );
    assert_eq!(
        after_first.hits, before.hits,
        "Fix: first Metal dispatch must not claim a cache hit before the pipeline exists."
    );
    assert_eq!(
        after_second.hits,
        after_first.hits + 1,
        "Fix: second identical Metal dispatch must reuse the compiled pipeline cache."
    );
    assert_eq!(
        after_second.misses, after_first.misses,
        "Fix: cache hit dispatch must not increment Metal pipeline miss counters."
    );
}

#[test]
fn apple_pipeline_cache_partitions_workgroup_policy_changes() {
    let program = stores_word(88);

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before cache policy testing.",
    );
    let before = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose honest pipeline cache counters.");
    let default_config = DispatchConfig::default();
    let default_output = backend
        .dispatch(&program, &[], &default_config)
        .expect("Fix: first Metal dispatch must compile the default policy pipeline.");
    let after_default = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose cache counters after default policy compile.");
    let mut workgroup_policy = DispatchConfig::default();
    workgroup_policy.workgroup_override = Some([1, 1, 1]);
    let policy_output = backend
        .dispatch(&program, &[], &workgroup_policy)
        .expect("Fix: Metal dispatch must compile a distinct workgroup-policy pipeline.");
    let after_policy = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose cache counters after policy-change compile.");
    let policy_hit_output = backend
        .dispatch(&program, &[], &workgroup_policy)
        .expect("Fix: repeated Metal workgroup-policy dispatch must hit the policy cache entry.");
    let after_policy_hit = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose cache counters after policy hit.");

    assert_eq!(default_output, vec![88u32.to_le_bytes().to_vec()]);
    assert_eq!(policy_output, default_output);
    assert_eq!(policy_hit_output, default_output);
    assert_eq!(
        after_default.misses,
        before.misses + 1,
        "Fix: first default-policy Metal dispatch must record one pipeline cache miss."
    );
    assert_eq!(
        after_policy.misses,
        after_default.misses + 1,
        "Fix: changing Metal workgroup policy must compile a distinct cache entry."
    );
    assert_eq!(
        after_policy.hits, after_default.hits,
        "Fix: first dispatch for a changed workgroup policy must not claim a cache hit."
    );
    assert_eq!(
        after_policy_hit.hits,
        after_policy.hits + 1,
        "Fix: repeated dispatch for the same changed workgroup policy must hit the Metal pipeline cache."
    );
    assert_eq!(
        after_policy_hit.misses, after_policy.misses,
        "Fix: repeated dispatch for the same changed workgroup policy must not add another miss."
    );
}

#[test]
fn apple_shutdown_invalidates_pipeline_cache_entries() {
    let program = stores_word(144);

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before lifecycle cache testing.",
    );
    let before = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose honest pipeline cache counters.");
    let first = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: first Metal dispatch must compile the lifecycle cache probe.");
    let after_first = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose counters after first lifecycle cache dispatch.");
    let second = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: second Metal dispatch must hit the lifecycle cache probe.");
    let after_hit = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose counters after lifecycle cache hit.");

    backend
        .shutdown()
        .expect("Fix: native Metal shutdown must invalidate backend-owned caches.");
    let after_shutdown = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must keep cache counters observable after shutdown.");
    let third = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: native Metal dispatch must recover after shutdown cache invalidation.");
    let after_recompile = backend
        .pipeline_cache_snapshot()
        .expect("Fix: native Metal must expose counters after post-shutdown recompile.");

    assert_eq!(first, vec![144u32.to_le_bytes().to_vec()]);
    assert_eq!(second, first);
    assert_eq!(third, first);
    assert_eq!(
        after_first.misses,
        before.misses + 1,
        "Fix: first lifecycle cache probe dispatch must record one miss."
    );
    assert_eq!(
        after_hit.hits,
        after_first.hits + 1,
        "Fix: second lifecycle cache probe dispatch must hit the compiled pipeline cache."
    );
    assert_eq!(
        after_shutdown, after_hit,
        "Fix: Metal shutdown must invalidate cache entries without rewriting historical hit/miss counters."
    );
    assert_eq!(
        after_recompile.misses,
        after_hit.misses + 1,
        "Fix: dispatch after Metal shutdown must recompile instead of reusing stale cached pipeline state."
    );
    assert_eq!(
        after_recompile.hits, after_hit.hits,
        "Fix: post-shutdown recompile must not be counted as a cache hit."
    );
}
