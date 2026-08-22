//! Contracts for `vyre_runtime::routing::standard_policy`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_foundation::execution_plan::{
    AccuracyPlan, AutotunePlan, AutotuneStrategy, ConformanceStrength, DispatchStrategy,
    ExecutionPlan, FusionPlan, FusionStrategy, LayoutStrategy, MemoryPlan, ProvenancePlan,
    ProvenanceStrategy, ReadbackStrategy, StrategyPlan,
};
use vyre_foundation::program_caps::RequiredCapabilities;
use vyre_runtime::routing::standard_policy::StandardPolicy;
use vyre_runtime::routing::{RoutingDecision, RoutingPolicy};

fn plan(node_count: usize, static_bytes: u64) -> ExecutionPlan {
    ExecutionPlan {
        program_fingerprint: [0; 32],
        required_capabilities: RequiredCapabilities::default(),
        fusion: FusionPlan {
            entry_op_id: None,
            top_level_regions: 1,
            node_count,
            batch_fusion_candidate: false,
        },
        memory: MemoryPlan {
            buffers: Vec::new(),
            static_bytes,
            dynamic_buffers: 0,
            visible_readback_bytes: 0,
            avoided_readback_bytes: 0,
        },
        provenance: ProvenancePlan {
            top_level_region_wrapped: true,
            region_count: 1,
            emit_region_trace: false,
        },
        accuracy: AccuracyPlan {
            exhaustive_conformance_required: false,
            reason: "test fixture",
        },
        autotune: AutotunePlan {
            recommended: false,
            parallel_region_size: [1, 1, 1],
            recommended_workgroup_size: [1, 1, 1],
            recommended_tile: [1, 1, 1],
            recommended_vector_pack_bits: 32,
            recommended_unroll_depth: 1,
            reason: "test fixture",
        },
        strategy: StrategyPlan {
            fusion: FusionStrategy::Isolated,
            dispatch: DispatchStrategy::PersistentRuntime,
            conformance: ConformanceStrength::Standard,
            autotune: AutotuneStrategy::DeclaredShape,
            provenance: ProvenanceStrategy::Minimal,
            layout: LayoutStrategy::Empty,
            readback: ReadbackStrategy::Full { bytes: 0 },
        },
        tracks: Vec::new(),
    }
}

#[test]
fn standard_policy_explains_persistent_megakernel_override() {
    let policy = StandardPolicy;
    let explanation = policy.route_with_explanation(&plan(1, 1));

    assert_eq!(explanation.policy, "standard-megakernel-first");
    assert_eq!(explanation.decision, RoutingDecision::PersistentMegakernel);
    assert!(
        explanation.reason.contains("persistent megakernel"),
        "Fix: routing explanation must expose why persistent execution was selected: {explanation:?}"
    );
}

#[test]
fn routing_engine_exposes_policy_explanation() {
    let engine = vyre_runtime::routing::RoutingEngine::new(StandardPolicy);
    let explanation = engine.route_with_explanation(&plan(128, 1 << 20));

    assert_eq!(explanation.policy, "standard-megakernel-first");
    assert_eq!(explanation.decision, RoutingDecision::PersistentMegakernel);
    assert!(!explanation.reason.is_empty());
}
