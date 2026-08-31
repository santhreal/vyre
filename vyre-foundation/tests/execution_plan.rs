//! Execution-planning contract tests.

use vyre_foundation::execution_plan::{
    plan, plan_for_adapter, plan_with_options, AutotuneStrategy, ConformanceStrength,
    DispatchStrategy, FusionStrategy, InnovationTrack, LayoutStrategy, PlanError,
    ProvenanceStrategy, ReadbackStrategy, SchedulingPolicy,
};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::AdapterCaps;
use vyre_foundation::validate::{BackendCapabilities, ValidationOptions};

fn ranged_output_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(1024)
            .with_output_byte_range(4..12)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    )
}

#[test]
fn plan_tracks_readback_minimization() {
    let plan = plan(&ranged_output_program()).expect("canonical ranged output program must plan");
    assert_eq!(plan.memory.visible_readback_bytes, 8);
    assert_eq!(plan.memory.avoided_readback_bytes, 4088);
    assert!(plan.track_active(InnovationTrack::ReadbackMinimization));
}

#[test]
fn plan_marks_subgroup_program_accuracy_sensitive() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::let_bind("x", Expr::subgroup_add(Expr::u32(1)))],
    );
    let options = ValidationOptions::default().with_backend_capabilities(BackendCapabilities {
        supports_subgroup_ops: true,
        ..BackendCapabilities::default()
    });
    let plan = plan_with_options(&program, options).expect("subgroup-capable backend must plan");
    assert!(plan.required_capabilities.subgroup_ops);
    assert!(plan.track_active(InnovationTrack::DifferentialAccuracy));
}

#[test]
fn plan_rejects_subgroup_program_without_capability_context() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::let_bind("x", Expr::subgroup_add(Expr::u32(1)))],
    );
    let err = plan(&program).expect_err("subgroup program needs backend capability context");
    assert!(
        err.to_string().contains("subgroup-ops support"),
        "capability-sensitive rejection must name subgroup support, got {err}"
    );
}

#[test]
fn plan_marks_wrapped_program_fusion_candidate() {
    let plan = plan(&ranged_output_program()).expect("canonical ranged output program must plan");
    assert!(plan.fusion.batch_fusion_candidate);
    assert!(plan.track_active(InnovationTrack::WholeProgramFusion));
    assert!(plan.provenance.top_level_region_wrapped);
}

#[test]
fn strategy_encodes_all_seven_tracks_for_small_trimmed_program() {
    let plan = plan(&ranged_output_program()).expect("canonical ranged output program must plan");
    assert_eq!(plan.strategy.fusion, FusionStrategy::Candidate);
    assert_eq!(plan.strategy.dispatch, DispatchStrategy::PersistentRuntime);
    assert_eq!(plan.strategy.conformance, ConformanceStrength::Standard);
    assert_eq!(plan.strategy.autotune, AutotuneStrategy::DeclaredShape);
    assert_eq!(plan.strategy.provenance, ProvenanceStrategy::GpuTrace);
    assert_eq!(plan.strategy.layout, LayoutStrategy::Static);
    assert_eq!(
        plan.strategy.readback,
        ReadbackStrategy::Trimmed {
            visible_bytes: 8,
            avoided_bytes: 4088,
        }
    );
}

/// WHY: measuring variants is a fact about the target, not about program size.
/// The plan used to report `MeasureVariants` once a program passed a node count
/// nobody measured, which is a threshold masquerading as a fact.
#[test]
fn measuring_variants_is_a_target_fact_and_not_a_node_count() {
    let large_body: Vec<Node> = (0..65)
        .map(|idx| Node::store("out", Expr::u32(idx), Expr::u32(idx)))
        .collect();
    let large = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(128)],
        [128, 1, 1],
        large_body,
    );
    let small = ranged_output_program();

    let bare = AdapterCaps {
        ideal_unroll_depth: 0,
        ideal_vector_pack_bits: 0,
        ideal_workgroup_tile: [0, 0, 0],
        ..AdapterCaps::conservative()
    };
    let declares_shapes = AdapterCaps {
        ideal_unroll_depth: 4,
        ideal_vector_pack_bits: 128,
        ideal_workgroup_tile: [16, 16, 1],
        ..AdapterCaps::conservative()
    };

    for program in [&large, &small] {
        let bare_plan = plan_for_adapter(program, &bare).expect("static program must plan");
        assert_eq!(
            bare_plan.strategy.autotune,
            AutotuneStrategy::DeclaredShape,
            "a target that declares no shape has nothing to measure, whatever the node count"
        );
        assert_eq!(
            bare_plan.strategy.dispatch,
            DispatchStrategy::PersistentRuntime
        );

        let measured =
            plan_for_adapter(program, &declares_shapes).expect("static program must plan");
        assert_eq!(
            measured.strategy.autotune,
            AutotuneStrategy::MeasureVariants,
            "a target that declares shapes states there is something to measure"
        );
    }
}

/// WHY: the shared policy answers legality and ring arithmetic, and nothing
/// it answers depends on a node count. It used to route on one: three
/// predicates ignored their argument and answered the same value for every
/// program, which read as a decision and was a constant. Selecting a schedule
/// is `vyre-megakernel`'s.
#[test]
fn the_shared_policy_answers_legality_and_ring_arithmetic() {
    let policy = SchedulingPolicy::standard();
    let multiplier = policy.fused_over_dispatch_multiplier();
    assert!(policy.allow_fused_threads(100 * multiplier, 100));
    assert!(!policy.allow_fused_threads(100 * multiplier + 1, 100));
    assert_eq!(policy.worker_workgroup_size(512, 256), 256);
    assert_eq!(policy.padded_slot_count(65, 64), 128);
}

#[test]
fn runtime_sized_storage_buffers_remain_dynamic_layout() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );
    let plan = plan(&program).expect("runtime-sized input storage must be wire-roundtrippable");
    assert_eq!(plan.strategy.layout, LayoutStrategy::Dynamic);
    assert_eq!(plan.memory.dynamic_buffers, 1);
}

#[test]
fn zero_count_output_is_rejected_before_strategy() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );
    let err = plan(&program).expect_err("dynamic output must not plan without a concrete size");
    match err {
        PlanError::Validation { issues } => {
            assert_eq!(issues.len(), 1);
            assert_eq!(issues[0].code().as_str(), "V130");
            assert_eq!(
                issues[0].phase(),
                vyre_foundation::validate::ValidationPhase::Program
            );
        }
        other => panic!("invalid Program must retain structured validation issues, got {other:?}"),
    }
}

#[test]
fn inverted_output_byte_range_is_rejected_with_named_error() {
    let inverted = std::ops::Range { start: 12, end: 4 };
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(1024)
            .with_output_byte_range(inverted)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );
    let err = plan(&program).expect_err("inverted range must fail planning");
    assert!(
        matches!(err, PlanError::InvalidOutputRange { ref name, start: 12, end: 4, .. } if name == "out"),
        "expected InvalidOutputRange for inverted range, got {err:?}"
    );
    let msg = err.to_string();
    assert!(msg.contains("out"), "error must name the buffer: {msg}");
    assert!(msg.contains("12"), "error must name the start: {msg}");
    assert!(msg.contains("4"), "error must name the end: {msg}");
}

#[test]
fn output_byte_range_past_end_is_rejected_with_named_error() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(4)
            .with_output_byte_range(0..64)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );
    let err = plan(&program).expect_err("range past end must fail planning");
    assert!(
        matches!(err, PlanError::InvalidOutputRange { ref name, start: 0, end: 64, .. } if name == "out"),
        "expected InvalidOutputRange for range past end, got {err:?}"
    );
}

#[test]
fn unwrapped_raw_program_is_rejected_by_plan() {
    let program = Program::from_raw_parts(vec![], [1, 1, 1], vec![Node::Return]);
    let err = plan(&program).expect_err("unwrapped program must not plan");
    match err {
        PlanError::Validation { issues } => {
            assert_eq!(issues.len(), 1);
            assert_eq!(issues[0].code().as_str(), "V105");
            assert_eq!(
                issues[0].corrective_action(),
                "construct runnable programs with `Program::wrapped(...)` or wrap the body in `Node::Region` before validation, interpretation, or dispatch"
            );
        }
        other => panic!("expected structured validation issues, got {other:?}"),
    }
}
