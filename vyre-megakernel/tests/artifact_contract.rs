//! Whole-program request, identity, ABI, artifact, and corruption contracts.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    compile,
    legality::{analyze_fusion_pair, FusionDecision, FusionRejectionReason},
    Artifact, ArtifactNodeId, ArtifactValueId, CompileError, CompileRequest, DependencyKind,
    DeviceFacts, Digest, ExternalFacts, SearchBudget,
};

const LIMIT: u64 = 1_000_000;

fn diagnostic_path(error: &CompileError) -> Option<&str> {
    error
        .diagnostic
        .location
        .as_ref()
        .and_then(|location| location.path.as_deref())
}

fn contract(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Symbol("items".into())],
        access,
        lifetime,
    }
}

fn copy_program(input: &str, output: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            output,
            Expr::u32(0),
            Expr::load(input, Expr::u32(0)),
        )],
    )
}

fn add_program(left: &str, right: &str, output: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(left, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(right, 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(output, 2, BufferAccess::ReadWrite, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            output,
            Expr::u32(0),
            Expr::add(
                Expr::load(left, Expr::u32(0)),
                Expr::load(right, Expr::u32(0)),
            ),
        )],
    )
}

fn retained_program(input: &str, retained: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(retained, 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            retained,
            Expr::u32(0),
            Expr::load(input, Expr::u32(0)),
        )],
    )
}

fn whole_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value(
            "input",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .unwrap();
    let constant = graph
        .add_external_value(
            "constant",
            contract(BufferAccess::ReadOnly, ValueLifetime::Constant),
        )
        .unwrap();
    let retained = graph
        .add_external_value(
            "retained",
            contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
        )
        .unwrap();
    let (_, alpha_outputs) = graph
        .add_node(
            "zeta",
            add_program("input", "constant", "intermediate"),
            vec![
                GraphInput {
                    buffer: "input".into(),
                    value: input,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                },
                GraphInput {
                    buffer: "constant".into(),
                    value: constant,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Constant),
                },
            ],
            vec![GraphOutput {
                buffer: "intermediate".into(),
                name: "intermediate".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    let (_, beta_outputs) = graph
        .add_node(
            "alpha",
            retained_program("intermediate", "retained"),
            vec![
                GraphInput {
                    buffer: "intermediate".into(),
                    value: alpha_outputs[0],
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
                },
                GraphInput {
                    buffer: "retained".into(),
                    value: retained,
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                },
            ],
            vec![GraphOutput {
                buffer: "retained".into(),
                name: "retained.next".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                retained_successor_of: Some(retained),
            }],
        )
        .unwrap();
    graph
        .add_node(
            "omega",
            copy_program("retained.next", "result"),
            vec![GraphInput {
                buffer: "retained.next".into(),
                value: beta_outputs[0],
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            }],
            vec![GraphOutput {
                buffer: "result".into(),
                name: "result".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
}
fn fusion_pair_graph(
    producer_workgroup: [u32; 3],
    consumer_workgroup: [u32; 3],
    producer_barrier: bool,
) -> ProgramGraph {
    fn pair_program(input: &str, output: &str, workgroup: [u32; 3], barrier: bool) -> Program {
        let mut body = Vec::new();
        if barrier {
            body.push(Node::barrier());
        }
        body.push(Node::store(
            output,
            Expr::u32(0),
            Expr::load(input, Expr::u32(0)),
        ));
        Program::wrapped(
            vec![
                BufferDecl::storage(input, 0, BufferAccess::ReadWrite, DataType::U32),
                BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32),
            ],
            workgroup,
            body,
        )
    }

    let invocation = contract(BufferAccess::ReadWrite, ValueLifetime::Invocation);
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value("input", invocation.clone())
        .unwrap();
    let (_, intermediate) = graph
        .add_node(
            "producer",
            pair_program(
                "input",
                "intermediate",
                producer_workgroup,
                producer_barrier,
            ),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: invocation.clone(),
            }],
            vec![GraphOutput {
                buffer: "intermediate".into(),
                name: "intermediate".into(),
                contract: invocation.clone(),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
        .add_node(
            "consumer",
            pair_program("intermediate", "output", consumer_workgroup, false),
            vec![GraphInput {
                buffer: "intermediate".into(),
                value: intermediate[0],
                contract: invocation,
            }],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
}

fn budget() -> SearchBudget {
    SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000)
}

fn facts() -> ExternalFacts {
    let mut facts = ExternalFacts::new(Digest([0xA5; 32]), BTreeMap::from([("items".into(), 17)]));
    facts
        .constant_identities
        .insert(vyre_foundation::ir::GraphValueId(1), Digest([0x5A; 32]));
    facts
}

fn request_with(
    facts: ExternalFacts,
    budget: SearchBudget,
    max_artifact_bytes: u64,
) -> vyre_megakernel::ValidatedCompileRequest {
    CompileRequest::new(
        whole_graph(),
        facts,
        DeviceFacts::unknown(),
        budget,
        max_artifact_bytes,
    )
    .validate()
    .expect("fixture request must validate")
}

fn request(max_artifact_bytes: u64) -> vyre_megakernel::ValidatedCompileRequest {
    request_with(facts(), budget(), max_artifact_bytes)
}

/// WHY: artifact encoding must preserve typed graph identities across every stage.
#[test]
fn round_trip_preserves_typed_ids_abi_plan_and_digest() {
    let artifact = compile(&request(LIMIT)).expect("valid request must compile");
    let bytes = artifact.to_bytes().expect("artifact must encode");
    let decoded = Artifact::from_bytes(&bytes).expect("canonical artifact must decode");

    assert_eq!(decoded, artifact);
    assert_eq!(decoded.digest(), artifact.digest());
    assert_eq!(
        decoded
            .nodes()
            .iter()
            .map(|node| node.id.0)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        decoded
            .abi()
            .resources
            .iter()
            .map(|resource| (resource.slot, resource.value.0))
            .collect::<Vec<_>>(),
        vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    );
    assert_eq!(decoded.abi().entries[0].inputs[0].0, 0);
    assert_eq!(decoded.abi().entries[0].inputs[1].0, 1);
    assert_eq!(decoded.abi().entries[0].outputs[0].0, 3);
    assert_eq!(decoded.selected_plan().candidates_explored, 2);
    assert_eq!(decoded.selected_plan().search_budget, budget());
    assert_eq!(decoded.selected_plan().search_work.candidates_explored, 2);
    assert_eq!(decoded.selected_plan().search_work.target_compilations, 0);
    assert_eq!(decoded.selected_plan().search_work.measurements, 0);
    assert!(decoded.selected_plan().search_work.cpu_work <= budget().max_cpu_work);
    assert_eq!(decoded.fusion().len(), 2);
    assert_eq!(decoded.fusion()[0].members.len(), 2);
    assert_eq!(decoded.fusion()[1].members.len(), 1);
    assert!(decoded
        .dependencies()
        .iter()
        .any(|edge| edge.kind == DependencyKind::Retained));
}
/// WHY: the compiler must select profitable legal fusion while preserving lifecycle boundaries.
#[test]
fn planner_fuses_invocation_dataflow_and_prunes_retained_dataflow() {
    let graph = whole_graph();
    assert_eq!(
        analyze_fusion_pair(
            &graph,
            ArtifactNodeId(0),
            ArtifactNodeId(1),
            ArtifactValueId(3),
        ),
        FusionDecision::Legal
    );
    assert_eq!(
        analyze_fusion_pair(
            &graph,
            ArtifactNodeId(1),
            ArtifactNodeId(2),
            ArtifactValueId(4),
        ),
        FusionDecision::Rejected(FusionRejectionReason::LifecycleBoundary)
    );

    let artifact = compile(&request(LIMIT)).unwrap();
    assert_eq!(
        artifact.fusion()[0].members,
        [ArtifactNodeId(0), ArtifactNodeId(1)]
    );
    assert_eq!(artifact.selected_plan().pruned_fusions.len(), 1);
    assert_eq!(
        artifact.selected_plan().pruned_fusions[0].reason,
        FusionRejectionReason::LifecycleBoundary
    );
    assert_eq!(artifact.selected_plan().selection_cost.launches, 2);
    assert_eq!(artifact.selected_plan().selection_cost.materializations, 1);
}

/// WHY: the baseline remains a valid result when the explicit candidate budget forbids alternatives.
#[test]
fn candidate_bound_terminates_search_with_best_explored_plan() {
    let bounded = SearchBudget::new(1, 1_000_000, 0, 0, 1_000_000_000);
    let artifact = compile(&request_with(facts(), bounded, LIMIT)).unwrap();
    assert_eq!(artifact.selected_plan().search_work.candidates_explored, 1);
    assert_eq!(artifact.fusion().len(), 3);
    assert!(artifact
        .fusion()
        .iter()
        .all(|group| group.members.len() == 1));
}
/// WHY: target geometry and explicit synchronization are hard legality boundaries, not costs.
#[test]
fn fusion_legality_reasons_are_stable_for_geometry_and_synchronization() {
    let geometry = fusion_pair_graph([32, 1, 1], [64, 1, 1], false);
    assert_eq!(
        analyze_fusion_pair(
            &geometry,
            ArtifactNodeId(0),
            ArtifactNodeId(1),
            ArtifactValueId(1),
        ),
        FusionDecision::Rejected(FusionRejectionReason::WorkgroupMismatch)
    );
    assert_eq!(
        FusionRejectionReason::WorkgroupMismatch.code(),
        "MKL005_WORKGROUP_MISMATCH"
    );

    let synchronization = fusion_pair_graph([32, 1, 1], [32, 1, 1], true);
    assert_eq!(
        analyze_fusion_pair(
            &synchronization,
            ArtifactNodeId(0),
            ArtifactNodeId(1),
            ArtifactValueId(1),
        ),
        FusionDecision::Rejected(FusionRejectionReason::SynchronizationBoundary)
    );
    assert_eq!(
        FusionRejectionReason::SynchronizationBoundary.code(),
        "MKL006_SYNCHRONIZATION_BOUNDARY"
    );
}

/// WHY: names are diagnostic metadata and must not remap typed graph identities.
#[test]
fn lexical_name_order_does_not_reassign_graph_ids() {
    let artifact = compile(&request(LIMIT)).unwrap();
    assert_eq!(artifact.nodes()[0].name, "zeta");
    assert_eq!(artifact.nodes()[0].id.0, 0);
    assert_eq!(artifact.nodes()[1].name, "alpha");
    assert_eq!(artifact.nodes()[1].id.0, 1);
}

/// WHY: admission policy bounds may reject an artifact but cannot rename accepted bytes.
#[test]
fn artifact_admission_bound_is_not_artifact_identity() {
    let first = compile(&request(LIMIT)).unwrap();
    let second = compile(&request(LIMIT * 2)).unwrap();
    assert_eq!(first.digest(), second.digest());
    assert_eq!(first.to_bytes().unwrap(), second.to_bytes().unwrap());
}
/// WHY: every semantic fact and search bound that can alter selection is authenticated.
#[test]
fn external_facts_and_search_budget_change_artifact_identity() {
    let baseline = compile(&request(LIMIT)).unwrap();

    let mut changed_configuration = facts();
    changed_configuration.configuration_digest = Digest([0x11; 32]);
    let configured = compile(&request_with(changed_configuration, budget(), LIMIT)).unwrap();
    assert_ne!(baseline.digest(), configured.digest());

    let mut changed_constant = facts();
    changed_constant
        .constant_identities
        .insert(vyre_foundation::ir::GraphValueId(1), Digest([0x22; 32]));
    let constant = compile(&request_with(changed_constant, budget(), LIMIT)).unwrap();
    assert_ne!(baseline.digest(), constant.digest());

    let searched = compile(&request_with(
        facts(),
        SearchBudget::new(64, 1_000_000, 8, 0, 1_000_000_000),
        LIMIT,
    ))
    .unwrap();
    assert_ne!(baseline.digest(), searched.digest());
}

/// WHY: missing external semantic facts must fail before artifact construction.
#[test]
fn missing_symbol_and_constant_identity_have_stable_diagnostics() {
    let mut missing_symbol = facts();
    missing_symbol.symbolic_bindings.clear();
    let error = CompileRequest::new(
        whole_graph(),
        missing_symbol,
        DeviceFacts::unknown(),
        budget(),
        LIMIT,
    )
    .validate()
    .err()
    .expect("missing symbol must fail");
    assert_eq!(error.diagnostic.code.as_str(), "MKC002_MISSING_SYMBOL");
    assert_eq!(
        diagnostic_path(&error),
        Some("request.facts.symbolic_bindings.items")
    );

    let mut missing_constant = facts();
    missing_constant.constant_identities.clear();
    let error = CompileRequest::new(
        whole_graph(),
        missing_constant,
        DeviceFacts::unknown(),
        budget(),
        LIMIT,
    )
    .validate()
    .err()
    .expect("missing constant identity must fail");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC023_MISSING_CONSTANT_IDENTITY"
    );
    assert_eq!(
        diagnostic_path(&error),
        Some("request.facts.constant_identities.1")
    );
}

/// WHY: an unbounded or disabled mandatory search dimension is not a valid request.
#[test]
fn zero_mandatory_search_bound_is_rejected() {
    let error = CompileRequest::new(
        whole_graph(),
        facts(),
        DeviceFacts::unknown(),
        SearchBudget::new(0, 1, 0, 0, 1),
        LIMIT,
    )
    .validate()
    .err()
    .expect("zero candidate bound must fail");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC022_INVALID_SEARCH_BUDGET"
    );
    assert_eq!(diagnostic_path(&error), Some("request.search_budget"));
}

/// WHY: exact artifact limits are inclusive and smaller limits fail closed.
#[test]
fn artifact_byte_limit_is_inclusive_and_checked() {
    let len = compile(&request(LIMIT)).unwrap().to_bytes().unwrap().len() as u64;
    compile(&request(len)).expect("exact byte limit is inclusive");
    let error = compile(&request(len - 1)).expect_err("one-byte-short limit must fail");
    assert_eq!(error.diagnostic.code.as_str(), "MKC013_ARTIFACT_LIMIT");
}

/// WHY: neutral whole-program compilation validates subgroup semantics without
/// pretending that target-independent validation is a backend capability decision.
#[test]
fn compile_request_accepts_semantically_valid_subgroup_ir() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
        [32, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::subgroup_add(Expr::u32(1)),
        )],
    );
    let graph = ProgramGraph::from_program("subgroup", program).unwrap();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device_with(|capabilities| capabilities.supports_subgroup_ops = true),
        budget(),
        LIMIT,
    )
    .validate()
    .expect("compiler validation must accept subgroup semantics on a subgroup device");

    compile(&request).expect("semantically valid subgroup IR must compile on a subgroup device");
}

/// WHY: 150.11. Validation reads the live capability snapshot, so a program that
/// needs a capability the device lacks fails at compile with the device-support
/// diagnostic. The pre-change compiler validated against a constant whose every
/// flag was true, which admitted this program on every device.
///
/// The capability axis is derived from the program: `program_caps::scan` reports
/// what the IR needs, and the test flips exactly the reported flag off.
#[test]
fn device_without_a_needed_capability_rejects_the_program() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
        [32, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::subgroup_add(Expr::u32(1)),
        )],
    );
    let required = vyre_foundation::program_caps::scan(&program);
    assert!(
        required.subgroup_ops,
        "fixture must need the capability the device denies"
    );
    let graph = ProgramGraph::from_program("subgroup", program).unwrap();
    let error = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device_with(|capabilities| capabilities.supports_subgroup_ops = false),
        budget(),
        LIMIT,
    )
    .validate()
    .err()
    .expect("a device without the needed capability must fail validation");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "V041",
        "the live capability snapshot must reach IR validation"
    );
}

/// WHY: 150.11. A whole-grid fence runs in one kernel only under a cooperative
/// launch. Without one the plan is unrunnable, so the compiler rejects it instead
/// of leaving the target emitter to fail later.
#[test]
fn grid_sync_without_cooperative_launch_fails_at_compile() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
        [32, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::u32(1)),
            Node::Barrier {
                ordering: vyre_foundation::ir::MemoryOrdering::GridSync,
            },
        ],
    );
    assert!(
        vyre_megakernel::grid_sync::requires_grid_sync(&program),
        "fixture must carry a whole-grid fence"
    );
    let graph = ProgramGraph::from_program("fence", program).unwrap();
    let error = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device_with(|_| ()),
        budget(),
        LIMIT,
    )
    .validate()
    .err()
    .expect("a device without cooperative launch must reject a whole-grid fence");
    assert_eq!(error.diagnostic.code.as_str(), "MKC001_INVALID_PROGRAM");
}

/// Device facts stating one capability decision, with every other capability
/// present and generous budgets, so a test isolates the flag it flips.
fn device_with(
    edit: impl FnOnce(&mut vyre_foundation::validate::BackendCapabilities),
) -> DeviceFacts {
    let mut capabilities = vyre_foundation::validate::BackendCapabilities {
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
        ..Default::default()
    };
    edit(&mut capabilities);
    DeviceFacts::new(capabilities, 1024)
}

/// WHY: resource arithmetic must never wrap.
#[test]
fn resource_shape_overflow_has_stable_diagnostic() {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "huge",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(u64::MAX), ShapeDim::Known(2)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .unwrap();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        budget(),
        LIMIT,
    )
    .validate()
    .unwrap();
    let error = compile(&request).expect_err("overflow must fail closed");
    assert_eq!(error.diagnostic.code.as_str(), "MKC011_RESOURCE_OVERFLOW");
}

/// WHY: persisted v3 artifacts must be rejected after the v4 target-seam cutover.
#[test]
fn stale_artifact_version_is_rejected_before_body_decode() {
    let artifact = compile(&request(LIMIT)).unwrap();
    let mut bytes = artifact.to_bytes().unwrap();
    bytes[4..6].copy_from_slice(&3u16.to_le_bytes());
    let error = Artifact::from_bytes(&bytes).expect_err("stale schema must fail");
    assert_eq!(error.diagnostic.code.as_str(), "MKC015_VERSION_SKEW");
    assert_eq!(diagnostic_path(&error), Some("artifact.schema_version"));
}

/// WHY: authenticated canonical artifacts must reject body mutation.
#[test]
fn artifact_body_tampering_is_detected() {
    let artifact = compile(&request(LIMIT)).unwrap();
    let mut bytes = artifact.to_bytes().unwrap();
    bytes[10] ^= 1;
    let error = Artifact::from_bytes(&bytes).expect_err("tampered body must fail");
    assert_eq!(error.diagnostic.code.as_str(), "MKC016_DIGEST_MISMATCH");
}
