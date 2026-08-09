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
    selected_modules, Artifact, ArtifactNodeId, ArtifactValueId, CompileRequest, DependencyKind,
    DiagnosticCode, Digest, ExternalFacts, SearchBudget,
};

const LIMIT: u64 = 1_000_000;

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
    SearchBudget::new(128, 1_000_000, 8, 4, 1_000_000_000)
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
    CompileRequest::new(whole_graph(), facts, budget, max_artifact_bytes)
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
/// WHY: every target compiler must receive the same selected modules and dependency stages.
#[test]
fn selected_modules_decode_exact_planner_groups() {
    let artifact = compile(&request(LIMIT)).unwrap();
    let modules = selected_modules(&artifact).expect("selected modules must decode");
    assert_eq!(modules.len(), 2);
    assert_eq!(modules[0].stage, 0);
    assert_eq!(modules[0].nodes, [ArtifactNodeId(0), ArtifactNodeId(1)]);
    assert_eq!(modules[0].programs.len(), 2);
    assert_eq!(modules[1].stage, 1);
    assert_eq!(modules[1].nodes, [ArtifactNodeId(2)]);
    assert_eq!(
        modules[0].programs[0].canonical_wire_bytes().unwrap(),
        artifact.nodes()[0].program
    );
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
        SearchBudget::new(64, 1_000_000, 8, 4, 1_000_000_000),
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
    let error = CompileRequest::new(whole_graph(), missing_symbol, budget(), LIMIT)
        .validate()
        .err()
        .expect("missing symbol must fail");
    assert_eq!(error.diagnostic.code, DiagnosticCode::MissingSymbol);
    assert_eq!(
        error.diagnostic.path,
        "request.facts.symbolic_bindings.items"
    );

    let mut missing_constant = facts();
    missing_constant.constant_identities.clear();
    let error = CompileRequest::new(whole_graph(), missing_constant, budget(), LIMIT)
        .validate()
        .err()
        .expect("missing constant identity must fail");
    assert_eq!(
        error.diagnostic.code,
        DiagnosticCode::MissingConstantIdentity
    );
    assert_eq!(error.diagnostic.path, "request.facts.constant_identities.1");
}

/// WHY: an unbounded or disabled mandatory search dimension is not a valid request.
#[test]
fn zero_mandatory_search_bound_is_rejected() {
    let error = CompileRequest::new(
        whole_graph(),
        facts(),
        SearchBudget::new(0, 1, 0, 0, 1),
        LIMIT,
    )
    .validate()
    .err()
    .expect("zero candidate bound must fail");
    assert_eq!(error.diagnostic.code, DiagnosticCode::InvalidSearchBudget);
    assert_eq!(error.diagnostic.path, "request.search_budget");
}

/// WHY: exact artifact limits are inclusive and smaller limits fail closed.
#[test]
fn artifact_byte_limit_is_inclusive_and_checked() {
    let len = compile(&request(LIMIT)).unwrap().to_bytes().unwrap().len() as u64;
    compile(&request(len)).expect("exact byte limit is inclusive");
    let error = compile(&request(len - 1)).expect_err("one-byte-short limit must fail");
    assert_eq!(error.diagnostic.code, DiagnosticCode::ArtifactLimit);
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
        budget(),
        LIMIT,
    )
    .validate()
    .unwrap();
    let error = compile(&request).expect_err("overflow must fail closed");
    assert_eq!(error.diagnostic.code, DiagnosticCode::ResourceOverflow);
}

/// WHY: persisted v3 artifacts must be rejected after the v4 target-seam cutover.
#[test]
fn stale_artifact_version_is_rejected_before_body_decode() {
    let artifact = compile(&request(LIMIT)).unwrap();
    let mut bytes = artifact.to_bytes().unwrap();
    bytes[4..6].copy_from_slice(&3u16.to_le_bytes());
    let error = Artifact::from_bytes(&bytes).expect_err("stale schema must fail");
    assert_eq!(error.diagnostic.code, DiagnosticCode::VersionSkew);
    assert_eq!(error.diagnostic.path, "artifact.schema_version");
}

/// WHY: authenticated canonical artifacts must reject body mutation.
#[test]
fn artifact_body_tampering_is_detected() {
    let artifact = compile(&request(LIMIT)).unwrap();
    let mut bytes = artifact.to_bytes().unwrap();
    bytes[10] ^= 1;
    let error = Artifact::from_bytes(&bytes).expect_err("tampered body must fail");
    assert_eq!(error.diagnostic.code, DiagnosticCode::DigestMismatch);
}
