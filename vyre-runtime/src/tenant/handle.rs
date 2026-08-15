//! One tenant's accounting state and the handle its owner publishes through.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::resident_work_queue::ResidentWorkQueue;
use crate::PipelineError;

use super::error::TenantError;
use super::quota::{
    release_resource_quota, reserve_resource_quota, saturating_atomic_add_u64,
    saturating_atomic_sub_u64,
};

pub(super) const QUIESCE_SPIN_POLLS: u64 = 64;
pub(super) const QUIESCE_MIN_PARK: Duration = Duration::from_micros(2);
pub(super) const QUIESCE_MAX_PARK: Duration = Duration::from_micros(50);
pub(super) const QUIESCE_BACKOFF_SHIFT_CAP: u64 = 5;

#[allow(clippy::unnecessary_min_or_max)]
pub(super) fn quiesce_backoff_duration(poll: u64) -> Duration {
    let parked_poll = poll.saturating_sub(QUIESCE_SPIN_POLLS);
    let shift = parked_poll.min(QUIESCE_BACKOFF_SHIFT_CAP) as u32;
    let multiplier = 1_u32 << shift;
    QUIESCE_MIN_PARK
        .checked_mul(multiplier)
        .unwrap_or(QUIESCE_MAX_PARK)
        .min(QUIESCE_MAX_PARK)
}

fn quiesce_idle(poll: u64) {
    if poll < QUIESCE_SPIN_POLLS {
        std::hint::spin_loop();
    } else {
        std::thread::park_timeout(quiesce_backoff_duration(poll));
    }
}

pub(super) fn tenant_registry_retry_idle(retry: u64) {
    if retry < QUIESCE_SPIN_POLLS {
        std::hint::spin_loop();
    } else {
        std::thread::park_timeout(quiesce_backoff_duration(retry));
    }
}

/// One tenant's accounting state. Lives inside an `Arc` so handles
/// stay valid after the registry borrow drops.
pub(super) struct TenantState {
    pub(super) id: u32,
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

/// Stable handle returned by [`TenantRegistry::register`]. Clones
/// share the same underlying state, so multiple producer threads
/// inside one tenant can publish through their own handles.
#[derive(Clone)]
pub struct TenantHandle {
    pub(super) state: Arc<TenantState>,
}

/// Host-visible tenant runtime counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantRuntimeCounters {
    /// Tenant id.
    pub tenant_id: u32,
    /// Number of slots ever published by this tenant.
    pub published_count: u64,
    /// Number of slots observed drained for this tenant.
    pub drained_count: u64,
    /// Current host-visible backlog (`published_count - drained_count`).
    pub outstanding_slots: u64,
    /// Configured outstanding-slot cap for this tenant.
    pub max_outstanding_slots: u64,
    /// Number of quiesce calls recorded for this tenant.
    pub quiesce_calls: u64,
    /// Number of quiesce calls that timed out.
    pub quiesce_timeouts: u64,
    /// Cumulative nanoseconds spent waiting for this tenant to drain.
    pub quiesce_wait_ns: u64,
}

/// Host-visible tenant quota counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantQuotaCounters {
    /// Tenant id.
    pub tenant_id: u32,
    /// Current reserved staging bytes.
    pub staging_bytes: u64,
    /// Configured staging byte cap.
    pub max_staging_bytes: u64,
    /// Current reserved resident handle count.
    pub resident_handles: u64,
    /// Configured resident handle cap.
    pub max_resident_handles: u64,
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

    /// Human-readable label supplied at registration time.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.state.label
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
        if let Err(e) = crate::resident_work_queue::protocol::opcode::validate_user_opcode(global) {
            return Err(TenantError::Pipeline(PipelineError::Backend(format!(
                "tenant registry produced invalid global opcode {global}: {e}. Fix: repair tenant opcode window allocation before publishing."
            ))));
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
    /// - [`TenantError::Pipeline`] when the underlying
    ///   `publish_slot` rejects (e.g., slot still in-flight).
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

    fn ensure_not_revoked(&self) -> Result<(), TenantError> {
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
                        TenantError::Pipeline(PipelineError::QueueFull {
                            queue: "tenant",
                            fix: "tenant drained_count exceeded published_count; rebuild tenant accounting state",
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
                    TenantError::Pipeline(PipelineError::QueueFull {
                        queue: "tenant",
                        fix: "tenant published_count overflowed u64; quiesce or recreate the tenant before publishing more slots",
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

    /// Reserve staging bytes against this tenant's quota.
    pub fn reserve_staging_bytes(&self, byte_count: u64) -> Result<(), TenantError> {
        self.ensure_not_revoked()?;
        reserve_resource_quota(
            &self.state.staging_bytes,
            byte_count,
            self.state.max_staging_bytes,
            || {
                TenantError::StagingBackpressure {
                    tenant_id: self.state.id,
                    requested: byte_count,
                    used: self.state.staging_bytes.load(Ordering::Acquire),
                    cap: self.state.max_staging_bytes,
                }
            },
            "tenant staging byte reservation overflowed u64; release staging reservations or recreate the tenant before reserving more bytes",
        )
    }

    /// Release staging bytes previously reserved by this tenant.
    pub fn release_staging_bytes(&self, byte_count: u64) -> Result<(), TenantError> {
        release_resource_quota(
            &self.state.staging_bytes,
            byte_count,
            self.state.id,
            "staging bytes",
        )
    }

    /// Reserve resident handles against this tenant's quota.
    pub fn reserve_resident_handles(&self, handle_count: u64) -> Result<(), TenantError> {
        self.ensure_not_revoked()?;
        reserve_resource_quota(
            &self.state.resident_handles,
            handle_count,
            self.state.max_resident_handles,
            || {
                TenantError::ResidentHandleBackpressure {
                    tenant_id: self.state.id,
                    requested: handle_count,
                    used: self.state.resident_handles.load(Ordering::Acquire),
                    cap: self.state.max_resident_handles,
                }
            },
            "tenant resident handle reservation overflowed u64; release resident handles or recreate the tenant before reserving more handles",
        )
    }

    /// Release resident handles previously reserved by this tenant.
    pub fn release_resident_handles(&self, handle_count: u64) -> Result<(), TenantError> {
        release_resource_quota(
            &self.state.resident_handles,
            handle_count,
            self.state.id,
            "resident handles",
        )
    }

    /// Snapshot quota counters for this tenant.
    #[must_use]
    pub fn quota_counters(&self) -> TenantQuotaCounters {
        TenantQuotaCounters {
            tenant_id: self.state.id,
            staging_bytes: self.state.staging_bytes.load(Ordering::Acquire),
            max_staging_bytes: self.state.max_staging_bytes,
            resident_handles: self.state.resident_handles.load(Ordering::Acquire),
            max_resident_handles: self.state.max_resident_handles,
        }
    }

    pub(super) fn release_all_resource_reservations(&self) {
        self.state.staging_bytes.store(0, Ordering::Release);
        self.state.resident_handles.store(0, Ordering::Release);
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
        self.record_quiesce(started, true);
        Err(TenantError::QuiesceTimeout {
            tenant_id: self.state.id,
            outstanding: vyre_driver::accounting::checked_sub_u64_lazy(pub_count, drained, || {
                TenantError::Pipeline(PipelineError::QueueFull {
                    queue: "tenant",
                    fix: "tenant drained_count exceeded published_count during quiesce; rebuild tenant accounting state",
                })
            })?,
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
