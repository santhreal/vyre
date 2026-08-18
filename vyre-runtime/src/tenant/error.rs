//! Structured failures the tenant registry and its handles return.

use crate::PipelineError;

/// Errors surfaced by the tenant registry.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TenantError {
    /// The registry has reached its maximum tenant capacity.
    #[error("tenant registry capacity exceeded (max {cap} tenants). Fix: unregister unused tenants or raise max tenant limit.")]
    CapacityExceeded {
        /// Configured maximum tenant count.
        cap: u32,
    },
    /// Tenant with the requested id was not found in the registry.
    #[error("tenant {tenant_id} not found. Fix: {fix}")]
    NotFound {
        /// Missing tenant id.
        tenant_id: u32,
        /// Corrective action for the caller.
        fix: &'static str,
    },
    /// The registry ran out of tenant ids. Unregister unused tenants
    /// or raise the range per tenant.
    #[error("tenant registry exhausted after {issued} registrations. Fix: shrink OPCODE_RANGE_PER_TENANT or recycle tenants.")]
    RegistryFull {
        /// Number of tenants already issued when exhaustion hit.
        issued: u32,
    },
    /// Tried to publish an opcode outside the tenant's reserved
    /// range. Almost always a caller bug.
    #[error(
        "tenant {tenant_id} published local opcode {local_opcode}; out of range [0, {cap}). \
         Fix: caller must stay inside the opcode window returned by `register()`."
    )]
    OpcodeOutOfRange {
        /// Tenant id that tripped.
        tenant_id: u32,
        /// Local opcode the caller supplied.
        local_opcode: u32,
        /// Cap on the tenant's local opcode range.
        cap: u32,
    },
    /// Tenant was unregistered concurrently; its handle is stale.
    #[error("tenant {tenant_id} was revoked; handle is stale. Fix: acquire a fresh handle from the registry.")]
    Revoked {
        /// Tenant id that was revoked.
        tenant_id: u32,
    },
    /// Quiesce timed out with inflight slots still outstanding.
    #[error(
        "tenant {tenant_id} quiesce timed out with {outstanding} inflight slots. \
         Fix: ensure the megakernel is making progress (check DONE_COUNT) or raise the timeout."
    )]
    QuiesceTimeout {
        /// Tenant id whose quiesce tripped.
        tenant_id: u32,
        /// Number of slots still inflight at timeout.
        outstanding: u64,
    },
    /// Tenant has reached its configured outstanding-slot cap.
    #[error(
        "tenant {tenant_id} has {outstanding} outstanding slots, cap {cap}. \
         Fix: wait for drain progress or register the tenant with a larger bounded backlog."
    )]
    Backpressure {
        /// Tenant id whose backlog is full.
        tenant_id: u32,
        /// Current host-visible outstanding slots.
        outstanding: u64,
        /// Configured outstanding-slot cap.
        cap: u64,
    },
    /// Tenant has reached its configured staging-byte cap.
    #[error(
        "tenant {tenant_id} requested {requested} staging bytes with {used} already reserved, cap {cap}. \
         Fix: release staging reservations after publish/readback progress or register the tenant with a larger bounded staging budget."
    )]
    StagingBackpressure {
        /// Tenant id whose staging byte budget is full.
        tenant_id: u32,
        /// New bytes requested.
        requested: u64,
        /// Current reserved staging bytes.
        used: u64,
        /// Configured staging byte cap.
        cap: u64,
    },
    /// Tenant has reached its configured resident-handle cap.
    #[error(
        "tenant {tenant_id} requested {requested} resident handles with {used} already reserved, cap {cap}. \
         Fix: release resident handles when backend ownership ends or register the tenant with a larger bounded resident-handle budget."
    )]
    ResidentHandleBackpressure {
        /// Tenant id whose resident handle budget is full.
        tenant_id: u32,
        /// New handles requested.
        requested: u64,
        /// Current reserved resident handles.
        used: u64,
        /// Configured resident handle cap.
        cap: u64,
    },
    /// Tenant resource accounting would underflow.
    #[error(
        "tenant {tenant_id} released {requested} {resource} with only {used} reserved. \
         Fix: pair every tenant resource release with a successful reservation."
    )]
    ResourceUnderflow {
        /// Tenant id whose counter would underflow.
        tenant_id: u32,
        /// Resource counter being released.
        resource: &'static str,
        /// Release count requested.
        requested: u64,
        /// Current reserved count.
        used: u64,
    },
    /// Protocol error bubbled up from [`crate::resident_work_queue::ResidentWorkQueue::publish_slot`].
    #[error("{0}")]
    Pipeline(#[from] PipelineError),
}
