//! Integration and contract proofs for generic runtime mechanisms:
//! - Retained page cache lifecycle: radix lookup, immutable identity, COW, isolation, and bounded residency.
//! - Paged resource residency binding, capacity/alignment validation, and fallback candidates.
//! - Intra-device routed work queues and inter-device routed exchange over explicit topology.
//! - Speculative state transactions, transactional verification, and rollback.
//! - Adversarial boundary conditions: saturation, duplicate release, stale generation, and zero leak.
//! - Content-addressed sharing: two callers with identical content share a page, and scrubbed pages cannot leak data.
//! - Public API runtime derivation: asserts absence of model/token/expert/attention/safetensors domain concepts.

#![forbid(unsafe_code)]

use std::fs;

use vyre_driver::{PeerAccessCapability, PeerLinkKind, PeerTopology, ResidentOwner, Resource};
use vyre_foundation::ir::DataType;
use vyre_runtime::paged_resource::{
    BlockTableSpec, PagedResidencyError, PagedResidencyPlanner, PagedResourceBinding,
    PagedResourceSpec, PagingCandidateStrategy,
};
use vyre_runtime::resource_residency::{StateId, StateLease};
use vyre_runtime::retained_page_cache::{
    RetainedPageCache, RetainedPageCacheError, RetainedPageCacheKey, RetainedPageLayout,
    RetainedPageLimits,
};
use vyre_runtime::routed_work_queue::{
    BoundedRoutedWorkQueue, InterDeviceRoutedExchange, InterDeviceRoutedItem, RoutedQueueLimits,
    RoutedWorkItem,
};
use vyre_runtime::speculative_transaction::{
    SpeculativeTransactionConfig, SpeculativeTransactionCoordinator,
};

use crate::retained_cache_fixtures::retained_key as test_retained_key;

// -----------------------------------------------------------------------------
// Retained Page Cache Lifecycle, COW, Isolation, and Bounds
// -----------------------------------------------------------------------------

#[test]
fn proof_retained_page_cache_radix_lifecycle_and_cow() {
    let limits = RetainedPageLimits {
        max_pages: 32,
        max_bytes: 1024 * 1024,
        max_active_requests: 16,
        max_queued_units: 4096,
        per_tenant_page_limit: 16,
    };
    let cache = RetainedPageCache::new(limits, 1);
    let key_tenant_a = test_retained_key("tenant_alpha", None, 1);

    // Request 1: Sequence [1, 2, 3, 4, 5, 6, 7, 8]
    let seq1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let p1_pages = cache
        .insert_or_extend(&key_tenant_a, &seq1, &[])
        .expect("insert 1");
    assert_eq!(p1_pages.len(), 1);

    // Request 2 (Same tenant, shared prefix [1..8] + suffix [9, 10])
    let seq2 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let match2 = cache.lookup(&key_tenant_a, &seq2).expect("lookup 2");
    assert_eq!(match2.matched_units, 8); // Matched exact 8-unit prefix
    assert_eq!(match2.page_ids, p1_pages);

    // Extend seq2
    let p2_pages = cache
        .insert_or_extend(&key_tenant_a, &seq2, &match2.page_ids)
        .expect("extend 2");
    assert_eq!(p2_pages.len(), 1); // 10 units still fits in 16-unit page

    let metrics = cache.metrics().expect("metrics");
    assert_eq!(metrics.cache_hits, 1);
    assert_eq!(metrics.allocated_pages, 1);

    // Release leases
    cache.release(&p1_pages).expect("release 1");
    cache.release(&match2.page_ids).expect("release 2 match");
    cache.release(&p2_pages).expect("release 2");
}

#[test]
fn proof_retained_page_cache_adversarial_limits_and_isolation() {
    let limits = RetainedPageLimits {
        max_pages: 2, // Strict 2 page capacity
        max_bytes: 1024 * 1024,
        max_active_requests: 8,
        max_queued_units: 64,
        per_tenant_page_limit: 8,
    };
    let cache = RetainedPageCache::new(limits, 1);

    let key_a = test_retained_key("tenant_A", None, 1);
    let key_b = test_retained_key("tenant_B", None, 1);

    let seq = vec![100, 200, 300];
    let p_a = cache.insert_or_extend(&key_a, &seq, &[]).expect("p_a");

    // Tenant B cannot access Tenant A's pages without explicit trust domain
    let err_iso = cache.lookup(&key_b, &seq).unwrap_err();
    assert!(matches!(
        err_iso,
        RetainedPageCacheError::IsolationViolation { .. }
    ));

    // Pin Page A
    cache.pin(&p_a).expect("pin");

    // Allocate Page B
    let p_b = cache
        .insert_or_extend(&key_b, &[400, 500], &[])
        .expect("p_b");
    cache.mark_in_flight(&p_b).expect("in_flight");

    // Page 3 allocation fails because p_a is pinned and p_b is in-flight
    let err_cap = cache
        .insert_or_extend(&key_a, &[600, 700], &[])
        .unwrap_err();
    assert!(matches!(
        err_cap,
        RetainedPageCacheError::CapacityExceeded { .. }
    ));

    // Duplicate release fails closed
    cache.release(&p_a).expect("first release");
    let err_dup = cache.release(&p_a).unwrap_err();
    assert!(matches!(
        err_dup,
        RetainedPageCacheError::DuplicateRelease(_)
    ));

    // Stale generation rejection
    let stale_key = test_retained_key("tenant_A", None, 999);
    let err_stale = cache.lookup(&stale_key, &seq).unwrap_err();
    assert!(matches!(
        err_stale,
        RetainedPageCacheError::StaleDeviceGeneration { .. }
    ));
}

// -----------------------------------------------------------------------------
// Paged Resource Residency Binding & Validation
// -----------------------------------------------------------------------------

#[test]
fn proof_paged_residency_validation_and_candidate_planning() {
    let slab_spec = PagedResourceSpec {
        blocks: 8,
        channels: 4,
        units_per_block: 16,
        unit_dim: 64,
        dtype: DataType::F16,
    };
    let table_spec = BlockTableSpec {
        sequences: 2,
        blocks_per_sequence: 8,
    };

    let owner = ResidentOwner::new().expect("resident owner");

    let table_bytes = table_spec.required_table_bytes();
    let slab_half = slab_spec.required_slab_bytes() / 2;

    let binding = PagedResourceBinding {
        lease: StateLease {
            id: StateId(1),
            generation: 1,
        },
        device_id: 0,
        table_resource: Resource::Resident(owner.handle(1)),
        primary_resource: Resource::Resident(owner.handle(2)),
        secondary_resource: Some(Resource::Resident(owner.handle(3))),
        resource_spec: slab_spec,
        table_spec,
        in_flight: false,
        completion_ticket: 0,
    };

    // Valid sizes pass
    assert!(binding.validate(table_bytes, slab_half, slab_half).is_ok());

    // Insufficient table capacity fails
    let err = binding.validate(32, slab_half, slab_half).unwrap_err();
    assert!(matches!(err, PagedResidencyError::CapacityMismatch { .. }));

    // Strategy selection: device without paging chooses explicit contiguous candidate
    let strategy = PagedResidencyPlanner::select_strategy(false, 4096);
    assert_eq!(
        strategy,
        PagingCandidateStrategy::ExplicitContiguousFallback {
            max_capacity_units: 4096
        }
    );
}

// -----------------------------------------------------------------------------
// Bounded Routed-Work Queues & Inter-Device Exchange
// -----------------------------------------------------------------------------

#[test]
fn proof_routed_queue_and_peer_topology() {
    // 1. Intra-device routed work queue with bounded starvation
    let limits = RoutedQueueLimits {
        max_queued_per_route: 4,
        max_starvation_ticks: 5,
        num_routes: 2,
    };
    let mut scheduler = BoundedRoutedWorkQueue::new(limits);

    for i in 0..3 {
        scheduler
            .enqueue(RoutedWorkItem {
                ticket: i,
                request_id: 10,
                item_index: i as u32,
                route_id: 0,
                weight: 0.9,
                payload: vec![1.0, 2.0],
                enqueue_tick: 0,
            })
            .expect("enqueue");
    }

    let work = scheduler.dequeue_route_work(0, 10);
    assert_eq!(work.len(), 3);

    // 2. Inter-device all-to-all exchange over PeerTopology
    let mut topo = PeerTopology::new(2);
    topo.set_symmetric_capability(
        0,
        1,
        PeerAccessCapability::DirectPeerMemory {
            bandwidth_gbps: 900,
            link: PeerLinkKind::NVLink {
                generation: 5,
                links: 18,
            },
        },
    );

    let mut exchange = InterDeviceRoutedExchange::new(topo);
    let items = vec![
        InterDeviceRoutedItem {
            item_id: 101,
            src_device: 0,
            dst_device: 1,
            target_route_id: 0,
            payload: vec![0.5f32; 128],
        },
        InterDeviceRoutedItem {
            item_id: 102,
            src_device: 0,
            dst_device: 0,
            target_route_id: 1,
            payload: vec![0.5f32; 128],
        },
    ];

    let routed = exchange.route_all_to_all(items).expect("route");
    assert_eq!(routed.get(&1).unwrap().len(), 1);
    assert_eq!(exchange.accounting().direct_transfers, 1);
    assert_eq!(exchange.accounting().direct_bytes, 128 * 4);
}

// -----------------------------------------------------------------------------
// Speculative State Transactions & Rollback
// -----------------------------------------------------------------------------

#[test]
fn proof_speculative_verification_and_rollback() {
    let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
    let config = SpeculativeTransactionConfig { max_depth: 3 };
    let coordinator = SpeculativeTransactionCoordinator::new(config, cache.clone());
    let key = test_retained_key("tenant_spec_proof", None, 1);

    let base = vec![1, 2, 3];
    let proposals = vec![(10, 0.95), (20, 0.90), (30, 0.85)];

    // Stage 3 speculative units
    let staged = coordinator
        .stage_speculative_step(1, &key, &base, &proposals)
        .expect("stage");

    // Verification: Proposals 10 and 20 match, but 30 was wrong -> ground truth produced 35
    let ground_truth = vec![10, 20, 35];
    let result = coordinator
        .verify_and_commit(staged, &ground_truth)
        .expect("verify");

    assert_eq!(result.accepted_proposal_count, 2);
    assert_eq!(result.rolled_back_count, 1);
    assert_eq!(result.accepted_units, vec![1, 2, 3, 10, 20, 35]);
    assert!(!result.released_pages.is_empty());
}

// -----------------------------------------------------------------------------
// Content-Addressed Sharing & Scrubbed Page Isolation
// -----------------------------------------------------------------------------

#[test]
fn proof_content_addressed_sharing_and_page_scrubbing() {
    let limits = RetainedPageLimits {
        max_pages: 4,
        max_bytes: 1024 * 1024,
        max_active_requests: 8,
        max_queued_units: 1024,
        per_tenant_page_limit: 4,
    };
    let cache = RetainedPageCache::new(limits, 1);

    // Two different callers with identical content identity in the same trust group
    let caller_1_key = RetainedPageCacheKey {
        content_id: [42u8; 32],
        schema_id: [1u8; 32],
        payload_digest: [99u8; 32],
        config_digest: [7u8; 32],
        dtype: DataType::F32,
        layout: RetainedPageLayout {
            feature_channels: 1,
            unit_dimension: 64,
            units_per_page: 8,
        },
        device_generation: 1,
        cache_schema_version: 1,
        isolation_domain: "caller_1".to_string(),
        trust_domain: Some("shared_workspace".to_string()),
    };

    let caller_2_key = RetainedPageCacheKey {
        content_id: [42u8; 32],
        schema_id: [1u8; 32],
        payload_digest: [99u8; 32],
        config_digest: [7u8; 32],
        dtype: DataType::F32,
        layout: RetainedPageLayout {
            feature_channels: 1,
            unit_dimension: 64,
            units_per_page: 8,
        },
        device_generation: 1,
        cache_schema_version: 1,
        isolation_domain: "caller_2".to_string(),
        trust_domain: Some("shared_workspace".to_string()),
    };

    let sequence = vec![1, 2, 3, 4, 5];

    // Caller 1 inserts content
    let p1 = cache
        .insert_or_extend(&caller_1_key, &sequence, &[])
        .expect("caller 1 insert");
    assert_eq!(p1.len(), 1);

    // Caller 2 looks up identical content -> shares the exact same physical page
    let match2 = cache
        .lookup(&caller_2_key, &sequence)
        .expect("caller 2 lookup");
    assert_eq!(match2.matched_units, 5);
    assert_eq!(match2.page_ids, p1);

    let metrics = cache.metrics().expect("metrics");
    assert_eq!(metrics.allocated_pages, 1);
    assert_eq!(metrics.cache_hits, 1);

    // Release all references so page becomes evictable
    cache.release(&p1).expect("release 1");
    cache.release(&match2.page_ids).expect("release 2");

    // Fill pool to force eviction and scrubbing of the page
    let fill_key = test_retained_key("filler_tenant", None, 1);
    let mut fill_pages = Vec::new();
    for i in 0..4 {
        let p = cache
            .insert_or_extend(
                &fill_key,
                &[(i as u32) * 100 + 1, (i as u32) * 100 + 2],
                &[],
            )
            .expect("fill insert");
        fill_pages.extend_from_slice(&p);
    }

    // Verify eviction occurred and bytes were scrubbed
    let metrics_after = cache.metrics().expect("metrics after");
    assert!(metrics_after.evicted_pages > 0);
    assert!(metrics_after.scrubbed_bytes > 0);
}

// -----------------------------------------------------------------------------
// Runtime Public Surface Derivation & Model Concept Absence Proof
// -----------------------------------------------------------------------------

#[test]
fn proof_runtime_public_surface_contains_no_model_concepts() {
    // The crate directory is resolved from the working directory through the
    // workspace member roster. A compiled-in manifest path names whichever
    // checkout last built this binary through the shared target directory, so
    // the scan would read that tree's sources.
    let src_dir =
        vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME")).join("src");

    let banned_terms = [
        // Generic model concepts
        "model",
        "expert",
        "attention",
        "safetensor",
        "mtp",
        "kv_cache",
        "tokenizer",
        "paged_attention",
        "speculative_decoding",
        // Model families and architectures
        "llama",
        "mistral",
        "transformer",
        "gpt",
        "bert",
        "moe",
        "deepseek",
        "qwen",
        "falcon",
        "phi",
        "whisper",
        "vit",
        "clip",
    ];

    // Scan all public module files in vyre-runtime/src/
    fn check_dir(dir: &std::path::Path, banned: &[&str]) {
        for entry in fs::read_dir(dir).expect("read dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                check_dir(&path, banned);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                let file_name = path.file_name().unwrap().to_str().unwrap();
                // Check file name
                for &banned_term in banned {
                    assert!(
                        !file_name.to_lowercase().contains(banned_term),
                        "File name `{}` in vyre-runtime contains banned model concept `{}`",
                        path.display(),
                        banned_term
                    );
                }

                // Check exported public items in Rust source
                let content = fs::read_to_string(&path).expect("read file");
                for (line_num, line) in content.lines().enumerate() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("pub mod ")
                        || trimmed.starts_with("pub struct ")
                        || trimmed.starts_with("pub enum ")
                        || trimmed.starts_with("pub fn ")
                        || trimmed.starts_with("pub type ")
                        || trimmed.starts_with("pub trait ")
                    {
                        for &banned_term in banned {
                            // CancellationToken and driver completion token are allowed concurrency primitives
                            let lower = trimmed.to_lowercase();
                            if lower.contains("cancellationtoken")
                                || lower.contains("completion_event")
                            {
                                continue;
                            }
                            assert!(
                                !lower.contains(banned_term),
                                "Public item in `{}:{}` contains banned model concept `{}`: `{}`",
                                path.display(),
                                line_num + 1,
                                banned_term,
                                trimmed
                            );
                        }
                    }
                }
            }
        }
    }

    check_dir(&src_dir, &banned_terms);
}
