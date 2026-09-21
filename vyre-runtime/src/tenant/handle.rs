//! One tenant's accounting state and the handle its owner publishes through.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::resident_work_queue::ResidentWorkQueue;
use crate::{CounterArithmetic, CounterScope, PipelineError};

use super::counters::TenantRuntimeCounters;
use super::error::TenantError;
use super::quiesce::quiesce_idle;
use super::quota::{saturating_atomic_add_u64, saturating_atomic_sub_u64};

/// One tenant's accounting state. Lives inside an `Arc` so handles
/// stay valid after the registry borrow drops.
pub(super) struct TenantState {
    pub(super) id: u32,
    pub(super) generation: u32,
    pub(super) base_opcode: u32,
    pub(super) opcode_cap: u32,
    /// Number of slots this tenant has ever published.
    pub(super) published_count: AtomicU64,
    /// Maximum host-visible slots this tenant may keep outstanding.
    pub(super) max_outstanding_slots: u64,
    /// Number of staging bytes currently reserved by this tenant.
    pub(super) staging_bytes: AtomicU64,
    /// Maximum staging bytes this tenant may reserve.
    pub(super) max_staging_bytes: u64,
    /// Number of resident handles currently reserved by this tenant.
    pub(super) resident_handles: AtomicU64,
    /// Maximum resident handles this tenant may reserve.
    pub(super) max_resident_handles: u64,
    /// Number of slots the GPU has reported DONE for this tenant.
    /// Advanced by [`TenantHandle::note_drained`].
    pub(super) drained_count: AtomicU64,
    /// Number of quiesce calls completed or timed out for this tenant.
    pub(super) quiesce_calls: AtomicU64,
    /// Number of quiesce calls that timed out before the tenant drained.
    pub(super) quiesce_timeouts: AtomicU64,
    /// Cumulative host-observed drain wait across quiesce calls.
    pub(super) quiesce_wait_ns: AtomicU64,
    /// Set to 1 on `unregister`; publishes reject afterwards.
    pub(super) revoked: AtomicU32,
    /// Stable label for diagnostics (for example, `"scanner-a"`, `"scanner-b"`).
    pub(super) label: String,
}

/// Stable handle returned by [`crate::tenant::TenantRegistry::register`]. Clones
/// share the same underlying state, so multiple producer threads
/// inside one tenant can publish through their own handles.
#[derive(Clone)]
pub struct TenantHandle {
    pub(super) state: Arc<TenantState>,
}

impl std::fmt::Debug for TenantHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantHandle")
            .field("id", &self.state.id)
            .field("label", &self.state.label)
            .field("base_opcode", &self.state.base_opcode)
            .field(
                "published_count",
                &self.state.published_count.load(Ordering::Relaxed),
            )
            .field("max_outstanding_slots", &self.state.max_outstanding_slots)
            .field(
                "staging_bytes",
                &self.state.staging_bytes.load(Ordering::Relaxed),
            )
            .field("max_staging_bytes", &self.state.max_staging_bytes)
            .field(
                "resident_handles",
                &self.state.resident_handles.load(Ordering::Relaxed),
            )
            .field("max_resident_handles", &self.state.max_resident_handles)
            .field(
                "drained_count",
                &self.state.drained_count.load(Ordering::Relaxed),
            )
            .field(
                "revoked",
                &(self.state.revoked.load(Ordering::Acquire) != 0),
            )
            .finish()
    }
}

impl TenantHandle {
    /// Stable tenant id; maps onto the ring-slot `TENANT_WORD`.
    #[must_use]
    pub fn id(&self) -> u32 {
        self.state.id
    }

    /// Epoch/generation allocated for this tenant identity.
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.state.generation
    }

    /// Human-readable label supplied at registration time.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.state.label
    }
    /// Return the 128-bit typed TenantId representation.
    #[must_use]
    pub fn tenant_id_128(&self) -> vyre_foundation::security::TenantId {
        vyre_foundation::security::TenantId::new(self.state.id as u128)
    }

    /// Issue an unforgeable capability handle for a given resource.
    pub fn issue_capability(
        &self,
        authenticator: &vyre_foundation::security::CapabilityAuthenticator,
        device_id: u64,
        resource_id: u64,
        permissions: std::collections::BTreeSet<vyre_foundation::security::Permission>,
    ) -> vyre_foundation::security::UnforgeableCapability {
        authenticator.issue_handle(
            self.tenant_id_128(),
            device_id,
            resource_id,
            vyre_foundation::security::GenerationId::new(self.state.generation as u64),
            permissions,
        )
    }

    /// Validate an unforgeable capability handle presented to this tenant.
    pub fn validate_capability(
        &self,
        authenticator: &vyre_foundation::security::CapabilityAuthenticator,
        handle: &vyre_foundation::security::UnforgeableCapability,
        device_id: u64,
        required_permission: &vyre_foundation::security::Permission,
    ) -> Result<(), vyre_foundation::security::SecurityError> {
        authenticator.validate_handle(
            handle,
            self.tenant_id_128(),
            device_id,
            vyre_foundation::security::GenerationId::new(self.state.generation as u64),
            required_permission,
        )
    }

    /// First opcode this tenant owns.
    #[must_use]
    pub fn base_opcode(&self) -> u32 {
        self.state.base_opcode
    }

    /// Convert a tenant-local opcode to the global opcode used in
    /// the ring slot. Caller enforces `local < opcode_cap()`.
    ///
    /// # Errors
    ///
    /// Returns [`TenantError::OpcodeOutOfRange`] when the local
    /// value is outside the reserved window.
    pub fn global_opcode(&self, local: u32) -> Result<u32, TenantError> {
        self.ensure_not_revoked()?;
        if local >= self.state.opcode_cap {
            return Err(TenantError::OpcodeOutOfRange {
                tenant_id: self.id(),
                local_opcode: local,
                cap: self.state.opcode_cap,
            });
        }
        let global = self.state.base_opcode + local;
        if crate::resident_work_queue::protocol::opcode::validate_user_opcode(global).is_err() {
            return Err(TenantError::Pipeline(PipelineError::ReservedOpcode {
                tenant_id: self.id(),
                local_opcode: local,
                global_opcode: global,
                fix: "repair the tenant opcode window allocation so the window does not overlap the reserved system range, then publish again",
            }));
        }
        Ok(global)
    }

    /// Publish a tenant-local opcode through [`ResidentWorkQueue::publish_slot`].
    ///
    /// # Errors
    ///
    /// - [`TenantError::Revoked`] if the tenant was unregistered.
    /// - [`TenantError::OpcodeOutOfRange`] if `local_opcode` is
    ///   outside the tenant's window.
    /// - [`TenantError::Backpressure`] if the tenant already holds its cap of
    ///   outstanding slots.
    /// - [`TenantError::Pipeline`] when the underlying `publish_slot` rejects,
    ///   or when this tenant's own slot accounting is inconsistent.
    pub fn publish_slot(
        &self,
        ring_bytes: &mut [u8],
        slot_idx: u32,
        local_opcode: u32,
        args: &[u32],
    ) -> Result<(), TenantError> {
        self.ensure_not_revoked()?;
        let global = self.global_opcode(local_opcode)?;
        self.reserve_publish_slot()?;
        if let Err(error) =
            ResidentWorkQueue::publish_slot(ring_bytes, slot_idx, self.state.id, global, args)
        {
            saturating_atomic_sub_u64(&self.state.published_count, 1, "tenant published rollback");
            return Err(error.into());
        }
        Ok(())
    }

    pub(super) fn ensure_not_revoked(&self) -> Result<(), TenantError> {
        if self.state.revoked.load(Ordering::Acquire) != 0 {
            return Err(TenantError::Revoked {
                tenant_id: self.state.id,
            });
        }
        Ok(())
    }

    fn reserve_publish_slot(&self) -> Result<(), TenantError> {
        let cap = self.state.max_outstanding_slots;
        vyre_driver::accounting::checked_atomic_update_u64_with_order(
            &self.state.published_count,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |published| {
                let drained = self.state.drained_count.load(Ordering::Acquire);
                let outstanding = vyre_driver::accounting::checked_sub_u64_lazy(
                    published,
                    drained,
                    || {
                        TenantError::Pipeline(PipelineError::CounterOrder {
                            scope: CounterScope::Tenant(self.state.id),
                            produced_counter: "published_count",
                            produced: published,
                            consumed_counter: "drained_count",
                            consumed: drained,
                            fix: "rebuild this tenant's slot accounting; note_drained ran for slots the tenant never published",
                        })
                    },
                )?;
                if outstanding >= cap {
                    return Err(TenantError::Backpressure {
                        tenant_id: self.state.id,
                        outstanding,
                        cap,
                    });
                }
                vyre_driver::accounting::checked_add_u64_lazy(published, 1, || {
                    TenantError::Pipeline(PipelineError::CounterOverflow {
                        scope: CounterScope::Tenant(self.state.id),
                        counter: "published_count",
                        arithmetic: CounterArithmetic::Sum,
                        lhs: published,
                        rhs: 1,
                        bits: 64,
                        fix: "quiesce or recreate the tenant before publishing more slots",
                    })
                })
            },
            |_, _| Ok(()),
        )?;
        Ok(())
    }

    /// Number of slots this tenant has ever published.
    #[must_use]
    pub fn published_count(&self) -> u64 {
        self.state.published_count.load(Ordering::Relaxed)
    }

    /// Number of slots this tenant has observed drained (via
    /// [`note_drained`](Self::note_drained)).
    #[must_use]
    pub fn drained_count(&self) -> u64 {
        self.state.drained_count.load(Ordering::Relaxed)
    }

    /// Maximum host-visible slots this tenant may keep outstanding.
    #[must_use]
    pub fn max_outstanding_slots(&self) -> u64 {
        self.state.max_outstanding_slots
    }

    /// Snapshot host-visible runtime counters for this tenant.
    #[must_use]
    pub fn runtime_counters(&self) -> TenantRuntimeCounters {
        let published_count = self.state.published_count.load(Ordering::Acquire);
        let drained_count = self.state.drained_count.load(Ordering::Acquire);
        TenantRuntimeCounters {
            tenant_id: self.state.id,
            published_count,
            drained_count,
            outstanding_slots: published_count.saturating_sub(drained_count),
            max_outstanding_slots: self.state.max_outstanding_slots,
            quiesce_calls: self.state.quiesce_calls.load(Ordering::Acquire),
            quiesce_timeouts: self.state.quiesce_timeouts.load(Ordering::Acquire),
            quiesce_wait_ns: self.state.quiesce_wait_ns.load(Ordering::Acquire),
        }
    }

    /// Mark `count` slots as drained. The host pump that observes
    /// DONE_COUNT calls this when it sees the global counter
    /// advance past the tenant's last-published cursor.
    pub fn note_drained(&self, count: u64) {
        saturating_atomic_add_u64(&self.state.drained_count, count, "tenant drained_count");
    }

    /// Block-style quiesce: bounded backoff until every published
    /// slot has been drained or `max_spins` polls elapse.
    ///
    /// # Errors
    ///
    /// Returns [`TenantError::QuiesceTimeout`] when `max_spins`
    /// iterations pass without full drain. The outstanding count
    /// at timeout is included for diagnostics.
    pub fn quiesce(&self, max_spins: u64) -> Result<(), TenantError> {
        let started = Instant::now();
        for poll in 0..max_spins {
            let pub_count = self.state.published_count.load(Ordering::Acquire);
            let drained = self.state.drained_count.load(Ordering::Acquire);
            if drained >= pub_count {
                self.record_quiesce(started, false);
                return Ok(());
            }
            quiesce_idle(poll);
        }
        let pub_count = self.state.published_count.load(Ordering::Acquire);
        let drained = self.state.drained_count.load(Ordering::Acquire);
        if drained >= pub_count {
            self.record_quiesce(started, false);
            return Ok(());
        }
        self.record_quiesce(started, true);
        Err(TenantError::QuiesceTimeout {
            tenant_id: self.state.id,
            outstanding: pub_count.saturating_sub(drained),
        })
    }

    fn record_quiesce(&self, started: Instant, timed_out: bool) {
        saturating_atomic_add_u64(&self.state.quiesce_calls, 1, "tenant quiesce_calls");
        if timed_out {
            saturating_atomic_add_u64(&self.state.quiesce_timeouts, 1, "tenant quiesce_timeouts");
        }
        let elapsed_ns = match u64::try_from(started.elapsed().as_nanos()) {
            Ok(elapsed_ns) => elapsed_ns,
            Err(_) => u64::MAX,
        };
        saturating_atomic_add_u64(
            &self.state.quiesce_wait_ns,
            elapsed_ns,
            "tenant quiesce_wait_ns",
        );
    }
}

/// WHY: `global_opcode` guards an internal invariant, that a tenant's opcode
/// window stays clear of the range the megakernel reserves. The registry
/// allocates windows from `TENANT_OPCODE_BASE` and refuses an id past
/// `MAX_TENANT_OPCODE_WINDOWS`, so no window it hands out can reach the
/// reserved range and the guard is unreachable through the public registry.
/// That is exactly why it needs a test: the fault it reports is otherwise
/// asserted by nothing, and it reported an untyped backend-error string until
/// `PipelineError::ReservedOpcode` existed. The state fields are `pub(super)`,
/// so this is the only scope that can build the broken allocation the guard
/// exists for.
///
/// Does not catch: a change to `is_system` that widens the reserved range under
/// a window the registry does hand out. `validate_user_opcode` owns that
/// boundary and is tested beside it.
#[cfg(test)]
mod reserved_opcode_tests {
    use super::{TenantHandle, TenantState};
    use crate::tenant::error::TenantError;
    use crate::PipelineError;
    use std::sync::atomic::{AtomicU32, AtomicU64};
    use std::sync::Arc;

    fn handle_with_window(base_opcode: u32, opcode_cap: u32) -> TenantHandle {
        TenantHandle {
            state: Arc::new(TenantState {
                id: 7,
                generation: 1,
                base_opcode,
                opcode_cap,
                published_count: AtomicU64::new(0),
                max_outstanding_slots: 1,
                staging_bytes: AtomicU64::new(0),
                max_staging_bytes: 1,
                resident_handles: AtomicU64::new(0),
                max_resident_handles: 1,
                drained_count: AtomicU64::new(0),
                quiesce_calls: AtomicU64::new(0),
                quiesce_timeouts: AtomicU64::new(0),
                quiesce_wait_ns: AtomicU64::new(0),
                revoked: AtomicU32::new(0),
                label: "reserved-window".to_string(),
            }),
        }
    }

    #[test]
    fn a_window_reaching_the_system_range_reports_the_opcode_it_produced() {
        let handle = handle_with_window(0x8000_0000, 4);

        let error = handle
            .global_opcode(2)
            .expect_err("Fix: a global opcode with the system bit set must be refused.");

        let TenantError::Pipeline(PipelineError::ReservedOpcode {
            tenant_id,
            local_opcode,
            global_opcode,
            ..
        }) = &error
        else {
            panic!("Fix: a reserved global opcode must report ReservedOpcode, got {error:?}");
        };
        assert_eq!(*tenant_id, 7);
        assert_eq!(*local_opcode, 2);
        assert_eq!(*global_opcode, 0x8000_0002);
    }

    #[test]
    fn a_window_clear_of_the_system_range_maps_the_opcode() {
        let handle = handle_with_window(0x4010_0000, 4);

        assert_eq!(
            handle
                .global_opcode(2)
                .expect("Fix: a window clear of the reserved range must map its local opcodes."),
            0x4010_0002
        );
    }
}
