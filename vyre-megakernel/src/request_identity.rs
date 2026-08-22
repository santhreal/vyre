//! Every fact that makes one compilation of one graph produce one artifact.

use std::collections::BTreeMap;

use serde::Serialize;
use vyre_foundation::validate::BackendCapabilities;

use crate::identity::{domain_digest, Digest};
use crate::request::{SearchBudget, ValidatedCompileRequest};

pub(crate) const SOURCE_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-source-v2\0";
pub(crate) const REQUEST_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-request-v3\0";
pub(crate) const REPRESENTATIVE_INPUT_DOMAIN: &[u8] = b"vyre-megakernel-representative-input-v1\0";
/// Every fact that makes one compilation of one graph produce one artifact.
///
/// Device facts belong here because the plan is selected against them: the same
/// graph compiled for a device with a different capability snapshot, invocation
/// limit, occupancy budget, or launch cost is a different compilation and must
/// not reuse a cached artifact.
#[derive(Serialize)]
pub(crate) struct RequestIdentity<'a> {
    configuration_digest: Digest,
    symbolic_bindings: &'a BTreeMap<String, u64>,
    constant_identities: Vec<(u32, Digest)>,
    representative_inputs: Vec<(u32, Digest, u64)>,
    expected_launch_batch: u32,
    search_budget: SearchBudget,
    device_capabilities: DeviceCapabilityIdentity,
    device_cooperative_launch: bool,
    device_timestamps: bool,
    device_spatial_partitioning: bool,
    device_compute_units: u32,
    device_concurrent_queues: u32,
    device_max_invocations_per_workgroup: u32,
    device_registers_per_invocation: u32,
    device_shared_scratch_bytes_per_workgroup: u32,
    device_per_launch_overhead_ns: u64,
    device_persistent_setup_overhead_ns: u64,
}

/// Serializable projection of the live capability snapshot.
///
/// [`BackendCapabilities`] is owned by the foundation validator and carries no
/// serialization, so artifact identity projects every field it exposes. A field
/// added there and not added here would silently share one artifact identity
/// between two devices that disagree.
#[derive(Serialize)]
struct DeviceCapabilityIdentity {
    supports_subgroup_ops: bool,
    supports_indirect_dispatch: bool,
    supports_specialization_constants: bool,
    supports_distributed_collectives: bool,
    has_mul_high: bool,
    has_dual_issue_fp32_int32: bool,
    has_tensor_core_int: bool,
    has_native_f16: bool,
    has_warp_shuffle: bool,
    has_shared_memory: bool,
    has_transcendental_polynomial_emit: bool,
    max_native_int_width: u32,
}

impl From<BackendCapabilities> for DeviceCapabilityIdentity {
    fn from(capabilities: BackendCapabilities) -> Self {
        let BackendCapabilities {
            supports_subgroup_ops,
            supports_indirect_dispatch,
            supports_specialization_constants,
            supports_distributed_collectives,
            has_mul_high,
            has_dual_issue_fp32_int32,
            has_tensor_core_int,
            has_native_f16,
            has_warp_shuffle,
            has_shared_memory,
            has_transcendental_polynomial_emit,
            max_native_int_width,
            supports_tensor_cores: _,
            max_shared_memory_bytes: _,
            regs_per_thread_max: _,
            subgroup_size: _,
        } = capabilities;
        Self {
            supports_subgroup_ops,
            supports_indirect_dispatch,
            supports_specialization_constants,
            supports_distributed_collectives,
            has_mul_high,
            has_dual_issue_fp32_int32,
            has_tensor_core_int,
            has_native_f16,
            has_warp_shuffle,
            has_shared_memory,
            has_transcendental_polynomial_emit,
            max_native_int_width,
        }
    }
}

impl<'a> From<&'a ValidatedCompileRequest> for RequestIdentity<'a> {
    fn from(request: &'a ValidatedCompileRequest) -> Self {
        Self {
            configuration_digest: request.facts.configuration_digest,
            symbolic_bindings: &request.facts.symbolic_bindings,
            constant_identities: request
                .facts
                .constant_identities
                .iter()
                .map(|(id, digest)| (id.0, *digest))
                .collect(),
            representative_inputs: request
                .representative_inputs()
                .iter()
                .map(|(id, bytes)| {
                    (
                        id.0,
                        domain_digest(REPRESENTATIVE_INPUT_DOMAIN, bytes),
                        bytes.len() as u64,
                    )
                })
                .collect(),
            expected_launch_batch: request.facts.expected_launch_batch,
            search_budget: request.search_budget,
            device_capabilities: request.device.capabilities().into(),
            device_cooperative_launch: request.device.supports_cooperative_launch(),
            device_timestamps: request.device.supports_device_timestamps(),
            device_spatial_partitioning: request.device.supports_spatial_partitioning(),
            device_compute_units: request.device.compute_units(),
            device_concurrent_queues: request.device.concurrent_queues(),
            device_max_invocations_per_workgroup: request.device.max_invocations_per_workgroup(),
            device_registers_per_invocation: request.device.registers_per_invocation(),
            device_shared_scratch_bytes_per_workgroup: request
                .device
                .shared_scratch_bytes_per_workgroup(),
            device_per_launch_overhead_ns: request.device.per_launch_overhead_ns(),
            device_persistent_setup_overhead_ns: request.device.persistent_setup_overhead_ns(),
        }
    }
}
