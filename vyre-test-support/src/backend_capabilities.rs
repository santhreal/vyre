//! A capability record that grants everything, for suites proving the wide path.
//!
//! Lowering asks what a device can do, so a suite that wants the widest legal
//! path states a record with every flag granted. Written per suite, that record
//! is eleven boolean lines and four numeric ones, and a new capability field
//! lands in whichever copies its author noticed. Stated here, a suite names the
//! one field it varies and inherits the rest.

use vyre_foundation::validate::BackendCapabilities;

/// Every capability granted, at 64-bit native integer width.
///
/// A suite that proves a narrower device overrides the fields it narrows:
/// `BackendCapabilities { max_native_int_width: 32, ..all_granted() }`.
#[must_use]
pub fn all_granted() -> BackendCapabilities {
    BackendCapabilities {
        supports_subgroup_ops: true,
        supports_indirect_dispatch: true,
        supports_specialization_constants: true,
        supports_distributed_collectives: true,
        has_mul_high: true,
        has_dual_issue_fp32_int32: true,
        has_tensor_core_int: true,
        has_native_f16: true,
        has_warp_shuffle: true,
        has_shared_memory: true,
        has_transcendental_polynomial_emit: true,
        max_native_int_width: 64,
        max_shared_memory_bytes: 64 * 1024,
        regs_per_thread_max: 255,
        subgroup_size: 32,
        supports_tensor_cores: true,
    }
}
