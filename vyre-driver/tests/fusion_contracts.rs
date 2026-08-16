//! Contracts for `vyre_driver::fusion`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::fusion::{DispatchShape, FusionCaps, FusionDecision, FusionPass};
use vyre_driver::specialization::SpecMap;

fn dispatch(
    id: &'static str,
    inputs: &[&'static str],
    outputs: &[&'static str],
) -> DispatchShape {
    DispatchShape {
        id,
        workgroup_size: [64, 1, 1],
        shared_memory_bytes: 1024,
        inputs: inputs.to_vec(),
        outputs: outputs.to_vec(),
        specs: SpecMap::new(),
    }
}

#[test]
fn straight_producer_consumer_fuses() {
    let up = dispatch("load", &["in"], &["stage"]);
    let down = dispatch("xor", &["stage"], &["out"]);
    assert_eq!(
        FusionPass::decide(&up, &down, FusionCaps::high_end(), &[]),
        FusionDecision::Accept
    );
}

#[test]
fn third_consumer_rejects() {
    let up = dispatch("a", &[], &["x"]);
    let down = dispatch("b", &["x"], &[]);
    assert_eq!(
        FusionPass::decide(&up, &down, FusionCaps::high_end(), &["x"]),
        FusionDecision::OutputConsumedElsewhere
    );
}

#[test]
fn workgroup_invocation_overflow_rejects_instead_of_wrapping_or_clamping() {
    let mut up = dispatch("wide-a", &["in"], &["stage"]);
    up.workgroup_size = [u32::MAX, u32::MAX, 2];
    let mut down = dispatch("wide-b", &["stage"], &["out"]);
    down.workgroup_size = up.workgroup_size;
    assert_eq!(
        FusionPass::decide(&up, &down, FusionCaps::high_end(), &[]),
        FusionDecision::InvocationBudgetExceeded {
            workgroup: up.workgroup_size,
            invocations: u64::MAX,
            cap: FusionCaps::high_end().max_invocations_per_workgroup,
        }
    );
}

#[test]
fn shared_memory_overflow_rejects_instead_of_appearing_under_cap() {
    let mut up = dispatch("smem-a", &["in"], &["stage"]);
    up.shared_memory_bytes = u32::MAX;
    let mut down = dispatch("smem-b", &["stage"], &["out"]);
    down.shared_memory_bytes = 1;
    assert_eq!(
        FusionPass::decide(&up, &down, FusionCaps::high_end(), &[]),
        FusionDecision::SharedMemoryBudget {
            needed: u64::from(u32::MAX) + 1,
            cap: FusionCaps::high_end().max_shared_memory_bytes,
        }
    );
}

// Reproducing test for: fusion-invocation-overflow-wrong-variant
// Before fix: FusionPass returned WorkgroupSizeMismatch{upstream==downstream} when the
// real failure was invocations > cap, misreporting the rejection reason to callers.
// After fix: returns InvocationBudgetExceeded{workgroup, invocations, cap} instead.
#[test]
fn invocation_budget_exceeded_returns_distinct_variant_not_workgroup_size_mismatch() {
    // Workgroup sizes must match (passing the size-mismatch gate) but product > cap.
    let caps = FusionCaps {
        max_shared_memory_bytes: 128 * 1024,
        max_invocations_per_workgroup: 64,
    };
    let mut up = dispatch("overinvoke-a", &["in"], &["stage"]);
    up.workgroup_size = [32, 4, 1]; // 128 invocations > cap 64
    let mut down = dispatch("overinvoke-b", &["stage"], &["out"]);
    down.workgroup_size = up.workgroup_size; // sizes are equal, not a mismatch

    let decision = FusionPass::decide(&up, &down, caps, &[]);

    // Must NOT be WorkgroupSizeMismatch (wrong variant from the old code).
    assert_ne!(
        decision,
        FusionDecision::WorkgroupSizeMismatch {
            upstream: up.workgroup_size,
            downstream: down.workgroup_size,
        },
        "Fix: when invocations exceed the cap and sizes are equal, the decision must not be WorkgroupSizeMismatch"
    );
    // Must be the correct InvocationBudgetExceeded variant with exact fields.
    assert_eq!(
        decision,
        FusionDecision::InvocationBudgetExceeded {
            workgroup: [32, 4, 1],
            invocations: 128,
            cap: 64,
        },
        "Fix: FusionPass must return InvocationBudgetExceeded{{workgroup=[32,4,1], invocations=128, cap=64}} when invocations > cap"
    );
}

#[test]
fn workgroup_size_mismatch_variant_is_only_returned_when_sizes_actually_differ() {
    // Mismatch case (must still work correctly).
    let mut up = dispatch("mismatch-a", &["in"], &["mid"]);
    up.workgroup_size = [32, 1, 1];
    let mut down = dispatch("mismatch-b", &["mid"], &["out"]);
    down.workgroup_size = [64, 1, 1];
    assert_eq!(
        FusionPass::decide(&up, &down, FusionCaps::high_end(), &[]),
        FusionDecision::WorkgroupSizeMismatch {
            upstream: [32, 1, 1],
            downstream: [64, 1, 1],
        },
        "Fix: WorkgroupSizeMismatch must carry the actual differing sizes"
    );
}
