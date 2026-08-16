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
#[cfg(test)]
mod tests;

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
