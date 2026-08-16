//! Multi-tenant megakernel multiplexing.
//!
//! A single persistent megakernel per GPU can service many producer
//! tools without each one paying the dispatch-setup cost. The
//! `tenant_id` field already lives in the ring-slot protocol
//! (`protocol::TENANT_WORD`); this module owns the host-side
//! bookkeeping that hands each producer a stable id, reserves an
//! opcode-range per producer, and gates publish operations against a
//! per-tenant mask so one producer cannot accidentally drive another
//! producer's opcodes.
//!
//! ## Tenants and opcodes
//!
//! Every tenant owns an opcode range `[base, base + cap)` where the
//! whole range sits inside the user-extension space reserved by
//! `vyre_runtime::megakernel::protocol::opcode` (≥ `0x4000_0000`).
//! When [`crate::tenant::TenantRegistry::register`] returns a [`crate::tenant::TenantHandle`],
//! callers publish into slot args `[rule_local_opcode, ...]` and
//! the registry maps that to `(tenant_base + rule_local_opcode)`
//! before writing into the ring. A tenant that tries to publish an
//! opcode outside its own range fails with a structured error.
//!
//! ## Draining
//!
//! Unregistering a tenant revokes future publishes but does NOT
//! revoke in-flight slots  -  the GPU is still going to execute any
//! slot it already CAS-claimed. Callers that need hard draining
//! drive [`crate::tenant::TenantHandle::quiesce`] which spins on the megakernel
//! DONE_COUNT until every slot the tenant published has been
//! acknowledged.
//!
//! ## Daemon surface
//!
//! The registry is the reusable piece. A full `MegakernelDaemon`
//! (listening on a Unix socket, vending handles over RPC) is a thin
//! wrapper that we can ship alongside the runtime  -  the registry
//! here already handles the interesting concurrency.

mod error;
mod handle;
mod quota;
mod registry;
pub use error::TenantError;
pub use handle::{TenantHandle, TenantQuotaCounters, TenantRuntimeCounters};
pub use quota::TenantQuota;
pub use registry::{TenantRegistry, TenantSelectionScratch};

/// First opcode the tenant registry hands out. Sits inside the
/// user-extension range reserved by the megakernel protocol so fused
/// rule documents compose with tenant allocation without colliding
/// with built-in opcodes.
pub const TENANT_OPCODE_BASE: u32 = 0x4000_0000;

/// Upper bound on the tenant-id space. `tenant_id == TENANT_ID_MAX`
/// is reserved as an invalid / revoked sentinel.
pub const TENANT_ID_MAX: u32 = u32::MAX - 1;

/// Size of the opcode window reserved per tenant. 1 << 20 = 1 MiB
/// of opcodes  -  well over any realistic rule count per producer
/// while still allowing ~4094 simultaneous tenants inside the u32
/// opcode range.
pub const OPCODE_RANGE_PER_TENANT: u32 = 1 << 20;

// Inline: covers the crate-private `handle` module and its `pub(super)`
// quiesce parameters (`quiesce_backoff_duration`, `tenant_registry_retry_idle`,
// `QUIESCE_MIN_PARK`, `QUIESCE_MAX_PARK`, `QUIESCE_SPIN_POLLS`), which no
// integration test can reach.
#[cfg(test)]
mod tests {
    use super::handle::{
        quiesce_backoff_duration, tenant_registry_retry_idle, QUIESCE_MAX_PARK, QUIESCE_MIN_PARK,
        QUIESCE_SPIN_POLLS,
    };
    use std::sync::Arc;

    use super::*;
    use crate::resident_work_queue::ResidentWorkQueue;

    #[test]
    fn two_tenants_get_distinct_id_and_opcode_ranges() {
        let reg = TenantRegistry::new();
        let a = reg
            .register("scanner-a")
            .expect("Fix: register a; restore this invariant before continuing.");
        let b = reg
            .register("scanner-b")
            .expect("Fix: register b; restore this invariant before continuing.");
        assert_ne!(a.id(), b.id());
        assert!(a.base_opcode() + OPCODE_RANGE_PER_TENANT <= b.base_opcode());
        assert_eq!(a.label(), "scanner-a");
        assert_eq!(b.label(), "scanner-b");
    }

    #[test]
    fn global_opcode_rejects_out_of_range_local() {
        let reg = TenantRegistry::new();
        let t = reg.register("soleno").unwrap();
        let err = t
            .global_opcode(OPCODE_RANGE_PER_TENANT)
            .expect_err("oversized local opcode must reject");
        assert!(matches!(err, TenantError::OpcodeOutOfRange { .. }));

        let ok = t
            .global_opcode(42)
            .expect("Fix: 42 < cap; restore this invariant before continuing.");
        assert_eq!(ok, t.base_opcode() + 42);
    }

    #[test]
    fn publish_slot_writes_with_tenant_id_and_bumps_counter() {
        let reg = TenantRegistry::new();
        let t = reg.register("warpscan").unwrap();
        let mut ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();

        t.publish_slot(
            &mut ring,
            /* slot = */ 0,
            /* local = */ 7,
            &[1, 2, 3],
        )
        .expect("Fix: publish; restore this invariant before continuing.");
        assert_eq!(t.published_count(), 1);

        // Slot 0 should carry tenant=t.id(), opcode=t.base_opcode()+7.
        let tenant_off = super::super::resident_work_queue::protocol::TENANT_WORD as usize * 4;
        let opcode_off = super::super::resident_work_queue::protocol::OPCODE_WORD as usize * 4;
        let stored_tenant =
            u32::from_le_bytes(ring[tenant_off..tenant_off + 4].try_into().unwrap());
        let stored_opcode =
            u32::from_le_bytes(ring[opcode_off..opcode_off + 4].try_into().unwrap());
        assert_eq!(stored_tenant, t.id());
        assert_eq!(stored_opcode, t.base_opcode() + 7);
    }

    #[test]
    fn unregister_blocks_future_publishes() {
        let reg = TenantRegistry::new();
        let t = reg.register("vein").unwrap();
        let tenant_id = t.id();
        let mut ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();
        t.publish_slot(&mut ring, 0, 0, &[0, 0, 0])
            .expect("Fix: first publish ok; restore this invariant before continuing.");
        reg.unregister(tenant_id)
            .expect("Fix: unregister; restore this invariant before continuing.");
        let err = t
            .publish_slot(&mut ring, 1, 0, &[0, 0, 0])
            .expect_err("publish after unregister must reject");
        assert!(matches!(err, TenantError::Revoked { .. }));
        assert!(reg.lookup(tenant_id).is_none());
    }

    #[test]
    fn quiesce_returns_when_drained_catches_up() {
        let reg = TenantRegistry::new();
        let t = reg.register("t1").unwrap();
        let mut ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();
        t.publish_slot(&mut ring, 0, 0, &[1, 2, 3]).unwrap();
        t.publish_slot(&mut ring, 1, 0, &[4, 5, 6]).unwrap();
        assert_eq!(t.published_count(), 2);
        t.note_drained(2);
        t.quiesce(1).expect(
            "Fix: drained == published after note_drained; restore this invariant before continuing.",
        );
        let counters = t.runtime_counters();
        assert_eq!(counters.published_count, 2);
        assert_eq!(counters.drained_count, 2);
        assert_eq!(counters.outstanding_slots, 0);
        assert_eq!(counters.quiesce_calls, 1);
        assert_eq!(counters.quiesce_timeouts, 0);
    }

    #[test]
    fn quiesce_times_out_when_drain_stalled() {
        let reg = TenantRegistry::new();
        let t = reg.register("t2").unwrap();
        let mut ring = ResidentWorkQueue::try_encode_empty_ring(1).unwrap();
        t.publish_slot(&mut ring, 0, 0, &[0, 0, 0]).unwrap();
        // Never note_drained → quiesce must time out.
        let err = t.quiesce(4).expect_err("stalled quiesce must time out");
        assert!(matches!(
            err,
            TenantError::QuiesceTimeout { outstanding: 1, .. }
        ));
        let counters = t.runtime_counters();
        assert_eq!(counters.outstanding_slots, 1);
        assert_eq!(counters.quiesce_calls, 1);
        assert_eq!(counters.quiesce_timeouts, 1);
    }

    #[test]
    fn bounded_tenant_backpressure_rejects_unbounded_publish_backlog() {
        let reg = TenantRegistry::new();
        let t = reg.register_with_backpressure("bounded", 2).unwrap();
        let mut ring = ResidentWorkQueue::try_encode_empty_ring(4).unwrap();

        t.publish_slot(&mut ring, 0, 0, &[1]).unwrap();
        t.publish_slot(&mut ring, 1, 0, &[2]).unwrap();
        let err = t
            .publish_slot(&mut ring, 2, 0, &[3])
            .expect_err("third outstanding publish must hit tenant backpressure");
        assert!(matches!(
            err,
            TenantError::Backpressure {
                outstanding: 2,
                cap: 2,
                ..
            }
        ));
        assert_eq!(t.published_count(), 2);
        let counters = t.runtime_counters();
        assert_eq!(counters.max_outstanding_slots, 2);
        assert_eq!(counters.outstanding_slots, 2);
    }

    #[test]
    fn tenant_backpressure_reopens_after_drain_progress() {
        let reg = TenantRegistry::new();
        let t = reg.register_with_backpressure("bounded", 1).unwrap();
        let mut ring = ResidentWorkQueue::try_encode_empty_ring(2).unwrap();

        t.publish_slot(&mut ring, 0, 0, &[1]).unwrap();
        assert!(matches!(
            t.publish_slot(&mut ring, 1, 0, &[2]).unwrap_err(),
            TenantError::Backpressure { .. }
        ));
        t.note_drained(1);
        t.publish_slot(&mut ring, 1, 0, &[2])
            .expect("Fix: drain progress must reopen the bounded tenant queue; restore this invariant before continuing.");
        assert_eq!(t.published_count(), 2);
        assert_eq!(t.runtime_counters().outstanding_slots, 1);
    }

    #[test]
    fn tenant_resource_quotas_reject_overcommit_and_cleanup_on_unregister() {
        let reg = TenantRegistry::new();
        let t = reg
            .register_with_quotas("quota", TenantQuota::bounded(2, 16, 1))
            .unwrap();

        t.reserve_staging_bytes(8).unwrap();
        let staging_error = t
            .reserve_staging_bytes(9)
            .expect_err("staging byte quota must reject overcommit");
        assert!(matches!(
            staging_error,
            TenantError::StagingBackpressure {
                requested: 9,
                cap: 16,
                ..
            }
        ));
        assert_eq!(t.quota_counters().staging_bytes, 8);

        t.release_staging_bytes(4).unwrap();
        t.reserve_staging_bytes(12).unwrap();
        assert_eq!(t.quota_counters().staging_bytes, 16);
        let underflow = t
            .release_staging_bytes(17)
            .expect_err("staging release must reject underflow");
        assert!(matches!(
            underflow,
            TenantError::ResourceUnderflow {
                resource: "staging bytes",
                requested: 17,
                used: 16,
                ..
            }
        ));

        t.reserve_resident_handles(1).unwrap();
        let handle_error = t
            .reserve_resident_handles(1)
            .expect_err("resident handle quota must reject overcommit");
        assert!(matches!(
            handle_error,
            TenantError::ResidentHandleBackpressure {
                requested: 1,
                cap: 1,
                ..
            }
        ));
        assert_eq!(t.quota_counters().resident_handles, 1);

        let removed = reg.unregister(t.id()).unwrap();
        assert_eq!(removed.quota_counters().staging_bytes, 0);
        assert_eq!(removed.quota_counters().resident_handles, 0);
        assert!(matches!(
            t.reserve_staging_bytes(1).unwrap_err(),
            TenantError::Revoked { .. }
        ));
        assert!(matches!(
            t.reserve_resident_handles(1).unwrap_err(),
            TenantError::Revoked { .. }
        ));
    }

    #[test]
    fn tenant_registry_registration_retry_uses_adaptive_idle_not_unbounded_spin() {
        for retry in [0, 1, 2, QUIESCE_SPIN_POLLS - 1, QUIESCE_SPIN_POLLS] {
            tenant_registry_retry_idle(retry);
        }
        assert_eq!(
            quiesce_backoff_duration(QUIESCE_SPIN_POLLS),
            QUIESCE_MIN_PARK
        );
        assert_eq!(quiesce_backoff_duration(u64::MAX), QUIESCE_MAX_PARK);
    }

    #[test]
    fn quiesce_backoff_is_bounded_and_monotonic() {
        let samples = [
            quiesce_backoff_duration(0),
            quiesce_backoff_duration(1),
            quiesce_backoff_duration(2),
            quiesce_backoff_duration(8),
            quiesce_backoff_duration(64),
        ];
        assert_eq!(samples[0], QUIESCE_MIN_PARK);
        for pair in samples.windows(2) {
            assert!(pair[0] <= pair[1], "quiesce backoff must not shrink");
            assert!(pair[1] <= QUIESCE_MAX_PARK, "quiesce backoff must cap");
        }
        assert_eq!(quiesce_backoff_duration(u64::MAX), QUIESCE_MAX_PARK);
    }

    #[test]
    fn active_tenants_tracks_registrations() {
        let reg = TenantRegistry::new();
        let a = reg.register("a").unwrap();
        let b = reg.register("b").unwrap();
        let active: Vec<u32> = reg.active_tenants().iter().map(|t| t.id()).collect();
        assert!(active.contains(&a.id()));
        assert!(active.contains(&b.id()));
        reg.unregister(a.id());
        let after: Vec<u32> = reg.active_tenants().iter().map(|t| t.id()).collect();
        assert!(!after.contains(&a.id()));
        assert!(after.contains(&b.id()));
        let counters: Vec<u32> = reg
            .runtime_counters()
            .iter()
            .map(|tenant| tenant.tenant_id)
            .collect();
        assert_eq!(counters, vec![b.id()]);
    }

    #[test]
    fn tenant_snapshots_reuse_caller_storage() {
        let reg = TenantRegistry::new();
        let a = reg.register("a").unwrap();
        let b = reg.register("b").unwrap();
        let mut active = Vec::with_capacity(2);
        let mut counters = Vec::with_capacity(2);

        reg.active_tenants_into(&mut active);
        reg.runtime_counters_into(&mut counters);
        let active_ptr = active.as_ptr();
        let counters_ptr = counters.as_ptr();
        reg.active_tenants_into(&mut active);
        reg.runtime_counters_into(&mut counters);

        assert_eq!(active.as_ptr(), active_ptr);
        assert_eq!(counters.as_ptr(), counters_ptr);
        assert!(active.iter().any(|tenant| tenant.id() == a.id()));
        assert!(active.iter().any(|tenant| tenant.id() == b.id()));
        assert!(counters.iter().any(|tenant| tenant.tenant_id == a.id()));
        assert!(counters.iter().any(|tenant| tenant.tenant_id == b.id()));
    }

    #[test]
    fn concurrent_tenant_selection_reuses_scratch_and_output() {
        let reg = TenantRegistry::new();
        let a = reg.register("a").unwrap();
        let b = reg.register("b").unwrap();
        let c = reg.register("c").unwrap();
        let n = 3;
        let mut conflicts = vec![0_u32; n * n];
        conflicts[0 * n + 1] = 1;
        conflicts[1 * n + 0] = 1;
        let mut out = Vec::with_capacity(3);
        let mut scratch = TenantSelectionScratch::new();

        reg.select_concurrent_tenants_into(&conflicts, &mut out, &mut scratch);
        let out_ptr = out.as_ptr();
        let active_ids_ptr = scratch.active_ids.as_ptr();
        let selected_ptr = scratch.selected_indices.as_ptr();
        reg.select_concurrent_tenants_into(&conflicts, &mut out, &mut scratch);

        assert_eq!(out.as_ptr(), out_ptr);
        assert_eq!(scratch.active_ids.as_ptr(), active_ids_ptr);
        assert_eq!(scratch.selected_indices.as_ptr(), selected_ptr);
        assert!(out.contains(&a.id()) || out.contains(&b.id()));
        assert!(!(out.contains(&a.id()) && out.contains(&b.id())));
        assert!(out.contains(&c.id()));
    }

    #[test]
    fn concurrent_tenant_selection_fast_paths_all_zero_conflicts() {
        let reg = TenantRegistry::new();
        let a = reg.register("a").unwrap();
        let b = reg.register("b").unwrap();
        let c = reg.register("c").unwrap();
        let mut out = Vec::with_capacity(8);
        let mut scratch = TenantSelectionScratch::new();
        let conflicts = vec![0_u32; 9];
        let out_ptr = out.as_ptr();

        reg.select_concurrent_tenants_into(&conflicts, &mut out, &mut scratch);

        assert_eq!(out, vec![a.id(), b.id(), c.id()]);
        assert_eq!(
            out.as_ptr(),
            out_ptr,
            "all-zero conflict fast path must reuse caller-owned output storage"
        );
        assert!(
            scratch.selected_indices.is_empty(),
            "all-zero conflict fast path must not populate pairwise selection scratch"
        );
    }

    #[test]
    fn concurrent_tenant_selection_respects_conflicts() {
        let reg = TenantRegistry::new();
        let a = reg.register("a").unwrap();
        let b = reg.register("b").unwrap();
        let c = reg.register("c").unwrap();
        let n = 3;
        let mut conflicts = vec![0_u32; n * n];
        conflicts[0 * n + 1] = 1;
        conflicts[1 * n + 0] = 1;

        let selected = reg.select_concurrent_tenants(&conflicts);

        assert!(selected.contains(&a.id()) || selected.contains(&b.id()));
        assert!(!(selected.contains(&a.id()) && selected.contains(&b.id())));
        assert!(selected.contains(&c.id()));
    }

    #[test]
    fn concurrent_registration_assigns_unique_ids() {
        use std::thread;
        let reg = Arc::new(TenantRegistry::new());
        let mut handles = Vec::new();
        for i in 0..32 {
            let reg = Arc::clone(&reg);
            handles.push(thread::spawn(move || {
                reg.register(format!("t{i}")).unwrap().id()
            }));
        }
        let ids: Vec<u32> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "concurrent ids must be unique");
    }
    #[test]
    fn unregister_reclaims_slot_and_advances_generation_without_aba() {
        let reg = TenantRegistry::new();
        let h1 = reg.register("first").unwrap();
        let id1 = h1.id();
        let gen1 = h1.generation();
        assert_eq!(gen1, 1);

        // Unregister releases id
        reg.unregister(id1).expect("unregister must succeed");

        // Register again reclaims the recycled ID and increments generation
        let h2 = reg.register("second").unwrap();
        assert_eq!(h2.id(), id1, "recycled ID must match unregistered ID");
        assert_eq!(h2.generation(), gen1 + 1, "generation must advance on slot recycling (ABA prevention)");

        // Old handle is revoked and cannot publish
        let mut ring = vec![0u8; 1024];
        let err = h1.publish_slot(&mut ring, 0, 0, &[]).unwrap_err();
        assert!(matches!(err, TenantError::Revoked { .. }));
    }
}
