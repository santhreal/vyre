//! Every fact that makes one compilation of one graph produce one artifact.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::device_facts::DeviceIdentity;
use crate::identity::{domain_digest, Digest};
use crate::objective::CompileObjective;
use crate::request::{SearchBudget, ValidatedCompileRequest};
use crate::DeviceFacts;

pub(crate) const SOURCE_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-source-v2\0";
pub(crate) const SEMANTIC_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-semantic-graph-v1\0";
pub(crate) const REQUEST_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-request-v7\0";
pub(crate) const REPRESENTATIVE_INPUT_DOMAIN: &[u8] = b"vyre-megakernel-representative-input-v1\0";
const TARGET_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-target-v1\0";
/// Every fact that makes one compilation of one graph produce one artifact.
///
/// Device facts belong here because the plan is selected against them: the same
/// graph compiled for a device with a different capability snapshot, invocation
/// limit, occupancy budget, launch cost, or calibration version is a different
/// compilation and must not reuse a cached artifact.
#[derive(Serialize)]
pub(crate) struct RequestIdentity<'a> {
    configuration_digest: Digest,
    symbolic_bindings: &'a BTreeMap<String, u64>,
    constant_identities: Vec<(u32, Digest)>,
    representative_inputs: Vec<(u32, Digest, u64)>,
    expected_launch_batch: u32,
    objective: CompileObjective,
    search_budget: SearchBudget,
    device: DeviceIdentity,
    mesh: Digest,
}

/// The identity of one target a guarded artifact set was compiled for.
///
/// A consumer holding a set has to establish that the device in front of it is
/// the device the set was compiled for before it evaluates a single guard: a
/// variant selected on a capability the device does not have is a launch that
/// fails, and one selected on a stale calibration is a launch priced against a
/// device that no longer exists. The objective participates because two sets
/// compiled for one device under different objectives answer different
/// questions.
#[must_use]
pub fn target_identity(device: DeviceFacts, objective: &CompileObjective) -> Digest {
    #[derive(Serialize)]
    struct TargetIdentity {
        device: DeviceIdentity,
        objective: CompileObjective,
        compiler_version: &'static str,
    }
    let body = serde_json::to_vec(&TargetIdentity {
        device: device.into(),
        objective: *objective,
        compiler_version: env!("CARGO_PKG_VERSION"),
    })
    .expect(
        "Fix: keep every target identity field an integer, string, or fixed array; a map with \
         non-string keys or a floating-point field is what makes this encoding fail.",
    );
    domain_digest(TARGET_DIGEST_DOMAIN, &body)
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
            objective: request.objective,
            search_budget: request.search_budget,
            device: request.device.into(),
            mesh: request.mesh().authentication(),
        }
    }
}
