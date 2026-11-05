//! Cross-dispatch fusion decisions shared by concrete backends.

use crate::specialization::SpecMap;

/// One dispatch's pre-fusion description.
#[derive(Debug, Clone)]
pub struct DispatchShape {
    /// Stable id for this dispatch inside the containing program.
    pub id: &'static str,
    /// Workgroup size `[x, y, z]`.
    pub workgroup_size: [u32; 3],
    /// Per-dispatch shared memory bytes.
    pub shared_memory_bytes: u32,
    /// Buffers this dispatch reads.
    pub inputs: Vec<&'static str>,
    /// Buffers this dispatch writes.
    pub outputs: Vec<&'static str>,
    /// Specialization constants baked into this dispatch.
    pub specs: SpecMap,
}

/// Adapter caps honored by the generic fusion pass.
#[derive(Debug, Clone, Copy)]
pub struct FusionCaps {
    /// Maximum workgroup-shared memory the adapter can serve.
    pub max_shared_memory_bytes: u32,
    /// Maximum workgroup invocation count.
    pub max_invocations_per_workgroup: u32,
}

impl Default for FusionCaps {
    fn default() -> Self {
        Self {
            max_shared_memory_bytes: 16 * 1024,
            max_invocations_per_workgroup: 256,
        }
    }
}

impl FusionCaps {
    /// High-end profile for tests and capability probes.
    #[must_use]
    pub const fn high_end() -> Self {
        Self {
            max_shared_memory_bytes: 128 * 1024,
            max_invocations_per_workgroup: 1024,
        }
    }
}

/// Why the fusion pass accepted or rejected a pair.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FusionDecision {
    /// Fusion is legal; the concrete backend may stitch its target modules.
    Accept,
    /// Upstream and downstream workgroup sizes differ.
    WorkgroupSizeMismatch {
        /// Upstream size.
        upstream: [u32; 3],
        /// Downstream size.
        downstream: [u32; 3],
    },
    /// Combined workgroup invocations exceed the adapter cap.
    InvocationBudgetExceeded {
        /// Workgroup shape whose product exceeds the cap.
        workgroup: [u32; 3],
        /// Computed invocation product (saturated to `u64`).
        invocations: u64,
        /// Adapter cap.
        cap: u32,
    },
    /// Shared-memory budget would exceed adapter caps.
    SharedMemoryBudget {
        /// Combined bytes the fused kernel would request.
        needed: u64,
        /// Adapter cap.
        cap: u32,
    },
    /// A flow-through output is still consumed by a third dispatch.
    OutputConsumedElsewhere,
    /// No buffer flows from upstream outputs to downstream inputs.
    NoPipelineDependency,
}

/// Pure cross-dispatch fusion analysis.
pub struct FusionPass;

impl FusionPass {
    /// Decide whether `upstream` -> `downstream` is legal to fuse.
    #[must_use]
    pub fn decide(
        upstream: &DispatchShape,
        downstream: &DispatchShape,
        caps: FusionCaps,
        other_consumers: &[&str],
    ) -> FusionDecision {
        if upstream.workgroup_size != downstream.workgroup_size {
            return FusionDecision::WorkgroupSizeMismatch {
                upstream: upstream.workgroup_size,
                downstream: downstream.workgroup_size,
            };
        }
        let invocations = u128::from(upstream.workgroup_size[0])
            * u128::from(upstream.workgroup_size[1])
            * u128::from(upstream.workgroup_size[2]);
        if invocations > u128::from(caps.max_invocations_per_workgroup) {
            return FusionDecision::InvocationBudgetExceeded {
                workgroup: upstream.workgroup_size,
                invocations: u64::try_from(invocations).unwrap_or(u64::MAX),
                cap: caps.max_invocations_per_workgroup,
            };
        }
        let needed =
            u64::from(upstream.shared_memory_bytes) + u64::from(downstream.shared_memory_bytes);
        if needed > u64::from(caps.max_shared_memory_bytes) {
            return FusionDecision::SharedMemoryBudget {
                needed,
                cap: caps.max_shared_memory_bytes,
            };
        }

        let mut has_pipeline_dependency = false;
        for output in &upstream.outputs {
            if !downstream.inputs.iter().any(|input| input == output) {
                continue;
            }
            has_pipeline_dependency = true;
            if other_consumers.iter().any(|consumer| consumer == output) {
                return FusionDecision::OutputConsumedElsewhere;
            }
        }
        if !has_pipeline_dependency {
            return FusionDecision::NoPipelineDependency;
        }
        FusionDecision::Accept
    }
}
