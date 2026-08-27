//! Whole-program request, identity, ABI, artifact, and corruption contracts.
//!
//! Fence admission is judged on the graph the planner produces, so which fence
//! shapes reach that decision is a property of the fixture. `fenced_program`
//! states it.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    compile,
    legality::{analyze_fusion_pair, FusionDecision, FusionRejectionReason},
    Artifact, ArtifactNodeId, ArtifactValueId, CompileError, CompileObjective, CompileRequest,
    DependencyKind, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric, SearchBudget,
};

use vyre_test_support::graph_values::{graph_output, u32_symbolic};

use vyre_test_support::pass_programs::{add_program, copy_program};

#[path = "graph_fixtures/mod.rs"]
mod graph_fixtures;

const LIMIT: u64 = 1_000_000;

fn diagnostic_path(error: &CompileError) -> Option<&str> {
    error
        .diagnostic
        .location
        .as_ref()
        .and_then(|location| location.path.as_deref())
}

fn contract(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    u32_symbolic(access, lifetime)
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
            graph_fixtures::value_and_constant_ports(input, constant),
            vec![graph_output(
                "intermediate",
                contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
            )],
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
            vec![graph_output(
                "result",
                contract(BufferAccess::ReadWrite, ValueLifetime::Output),
            )],
        )
        .unwrap();
    graph
}
#[derive(Clone, Copy)]
enum GeometryPin {
    None,
    Barrier,
    WorkgroupScratch,
}

fn fusion_pair_graph(
    producer_workgroup: [u32; 3],
    consumer_workgroup: [u32; 3],
    producer_pin: GeometryPin,
) -> ProgramGraph {
    fn pair_program(input: &str, output: &str, workgroup: [u32; 3], pin: GeometryPin) -> Program {
        let mut buffers = vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32),
        ];
        let mut body = Vec::new();
        match pin {
            GeometryPin::None => {}
            GeometryPin::Barrier => body.push(Node::barrier()),
            GeometryPin::WorkgroupScratch => {
                buffers.push(BufferDecl::workgroup("tile", 8, DataType::U32));
                body.push(Node::store("tile", Expr::u32(0), Expr::u32(1)));
            }
        }
        body.push(Node::store(
            output,
            Expr::u32(0),
            Expr::load(input, Expr::u32(0)),
        ));
        Program::wrapped(buffers, workgroup, body)
    }

    let invocation = contract(BufferAccess::ReadWrite, ValueLifetime::Invocation);
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value("input", invocation.clone())
        .unwrap();
    let (_, intermediate) = graph
        .add_node(
            "producer",
            pair_program("input", "intermediate", producer_workgroup, producer_pin),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: invocation.clone(),
            }],
            vec![graph_output("intermediate", invocation.clone())],
        )
        .unwrap();
    graph
        .add_node(
            "consumer",
            pair_program(
                "intermediate",
                "output",
                consumer_workgroup,
                GeometryPin::None,
            ),
            vec![GraphInput {
                buffer: "intermediate".into(),
                value: intermediate[0],
                contract: invocation,
            }],
            vec![graph_output(
                "output",
                contract(BufferAccess::ReadWrite, ValueLifetime::Output),
            )],
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
        CompileObjective::minimize_latency()
            .with_bound(ObjectiveMetric::ArtifactBytes, max_artifact_bytes),
    )
    .validate()
    .expect("fixture request must validate")
}

fn request(max_artifact_bytes: u64) -> vyre_megakernel::ValidatedCompileRequest {
    request_with(facts(), budget(), max_artifact_bytes)
}

fn request_with_representative_inputs(
    representative_inputs: BTreeMap<vyre_foundation::ir::GraphValueId, Vec<u8>>,
) -> vyre_megakernel::ValidatedCompileRequest {
    CompileRequest::new(
        whole_graph(),
        facts(),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .with_representative_inputs(representative_inputs)
    .validate()
    .expect("fixture request with representative inputs must validate")
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
    let plan = decoded.selected_plan();
    assert_eq!(
        plan.candidates_explored,
        plan.search_work.candidates_explored
    );
    assert!(plan.candidates_explored >= 1);
    assert!(plan.candidates_explored <= budget().max_candidates);
    assert_eq!(plan.search_budget, budget());
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
/// WHY: workgroup geometry is a hard legality boundary and synchronization is only a
/// boundary when the geometries already differ. A barrier at one shared geometry fuses,
/// which is the fused attention shape over a workgroup tile.
#[test]
fn fusion_legality_reasons_are_stable_for_geometry_and_synchronization() {
    fn decide(graph: &ProgramGraph) -> FusionDecision {
        analyze_fusion_pair(
            graph,
            ArtifactNodeId(0),
            ArtifactNodeId(1),
            ArtifactValueId(1),
        )
    }

    let geometry = fusion_pair_graph([32, 1, 1], [64, 1, 1], GeometryPin::None);
    assert_eq!(
        decide(&geometry),
        FusionDecision::Rejected(FusionRejectionReason::WorkgroupMismatch)
    );
    assert_eq!(
        FusionRejectionReason::WorkgroupMismatch.code(),
        "MKL005_WORKGROUP_MISMATCH"
    );

    for pin in [GeometryPin::Barrier, GeometryPin::WorkgroupScratch] {
        let widened = fusion_pair_graph([32, 1, 1], [64, 1, 1], pin);
        assert_eq!(
            decide(&widened),
            FusionDecision::Rejected(FusionRejectionReason::SynchronizationBoundary)
        );

        let shared = fusion_pair_graph([32, 1, 1], [32, 1, 1], pin);
        assert_eq!(decide(&shared), FusionDecision::Legal);
    }
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

/// WHY: the artifact byte bound is a stated hard constraint, so it is part of
/// what a schedule was selected under. A cache keyed on identity that ignored
/// it would serve a plan chosen under a looser bound to a caller that cannot
/// retain it. Recompiling under the same bound still reproduces one artifact.
#[test]
fn the_retained_artifact_bound_is_part_of_artifact_identity() {
    let first = compile(&request(LIMIT)).unwrap();
    let again = compile(&request(LIMIT)).unwrap();
    assert_eq!(first.digest(), again.digest());
    assert_eq!(first.to_bytes().unwrap(), again.to_bytes().unwrap());

    let looser = compile(&request(LIMIT * 2)).unwrap();
    assert_ne!(
        first.digest(),
        looser.digest(),
        "Fix: a different retained-bytes bound is a different constraint, so it \
         must reach the request identity the artifact records"
    );
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

    let with_representative =
        BTreeMap::from([(vyre_foundation::ir::GraphValueId(0), vec![1u8; 68])]);
    let representative_a = compile(&request_with_representative_inputs(
        with_representative.clone(),
    ))
    .unwrap();
    assert_ne!(baseline.digest(), representative_a.digest());

    let with_different_bytes =
        BTreeMap::from([(vyre_foundation::ir::GraphValueId(0), vec![2u8; 68])]);
    let representative_b =
        compile(&request_with_representative_inputs(with_different_bytes)).unwrap();
    assert_ne!(representative_a.digest(), representative_b.digest());
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
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
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
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
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

/// WHY: unknown representative input graph values and byte length mismatches fail validation.
#[test]
fn unknown_and_mismatched_representative_inputs_have_stable_diagnostics() {
    let unknown_value = BTreeMap::from([(vyre_foundation::ir::GraphValueId(999), vec![0u8; 68])]);
    let error = CompileRequest::new(
        whole_graph(),
        facts(),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .with_representative_inputs(unknown_value)
    .validate()
    .err()
    .expect("unknown representative value must fail validation");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC027_UNKNOWN_REPRESENTATIVE_INPUT"
    );
    assert_eq!(
        diagnostic_path(&error),
        Some("request.representative_inputs.999")
    );

    let produced_value = BTreeMap::from([(vyre_foundation::ir::GraphValueId(3), vec![0u8; 68])]);
    let error = CompileRequest::new(
        whole_graph(),
        facts(),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .with_representative_inputs(produced_value)
    .validate()
    .err()
    .expect("graph-produced representative input must fail validation");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC027_UNKNOWN_REPRESENTATIVE_INPUT"
    );
    assert_eq!(
        diagnostic_path(&error),
        Some("request.representative_inputs.3")
    );

    let mismatched_length = BTreeMap::from([(vyre_foundation::ir::GraphValueId(0), vec![0u8; 12])]);
    let error = CompileRequest::new(
        whole_graph(),
        facts(),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .with_representative_inputs(mismatched_length)
    .validate()
    .err()
    .expect("representative input length mismatch must fail validation");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC028_REPRESENTATIVE_INPUT_LENGTH_MISMATCH"
    );
    assert_eq!(
        diagnostic_path(&error),
        Some("request.representative_inputs.0")
    );
}

/// WHY: constant resources are caller-supplied host inputs during measurement,
/// so their authenticated identity does not replace their representative bytes.
#[test]
fn constant_representative_inputs_are_accepted() {
    let constant_input = BTreeMap::from([(vyre_foundation::ir::GraphValueId(1), vec![0xCD; 68])]);
    let request = request_with_representative_inputs(constant_input);
    assert_eq!(
        request
            .representative_inputs()
            .get(&vyre_foundation::ir::GraphValueId(1))
            .map(Vec::as_slice),
        Some(&[0xCD; 68][..])
    );
}

/// WHY: representative-input validation must propagate static-size failures
/// instead of treating an unaddressable input as if no size contract existed.
#[test]
fn representative_input_size_overflow_fails_closed() {
    let mut oversized_facts = facts();
    oversized_facts
        .symbolic_bindings
        .insert("items".to_string(), u64::MAX);
    let representative_inputs = BTreeMap::from([(vyre_foundation::ir::GraphValueId(0), vec![0u8])]);
    let error = CompileRequest::new(
        whole_graph(),
        oversized_facts,
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .with_representative_inputs(representative_inputs)
    .validate()
    .err()
    .expect("unaddressable representative input size must fail validation");
    assert_eq!(error.diagnostic.code.as_str(), "MKC011_RESOURCE_OVERFLOW");
}

/// WHY: an unbounded or disabled mandatory search dimension is not a valid request.
#[test]
fn zero_mandatory_search_bound_is_rejected() {
    let error = CompileRequest::new(
        whole_graph(),
        facts(),
        DeviceFacts::unknown(),
        SearchBudget::new(0, 1, 0, 0, 1),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
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
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
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
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
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

/// A fence the planner can cut needs no cooperative launch.
///
/// The cut turns the fenced node into one node per segment ordered through
/// retained state, so the segments run as separate launches and the whole-grid
/// fence is satisfied by the launch boundary. Refusing the submitted graph would
/// refuse work the compiler completes, so admission is judged on the graph the
/// planner produces and this case has to compile.
#[test]
fn a_cuttable_grid_fence_compiles_without_cooperative_launch() {
    let graph = ProgramGraph::from_program("fence", fenced_program(false)).unwrap();
    let validated = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device_with(|_| ()),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .validate()
    .expect("Fix: a fence the planner cuts must compile on a device without cooperative launch");
    let nodes = validated.graph().nodes();
    assert_eq!(
        nodes.len(),
        2,
        "Fix: the fence must become one node per segment"
    );
    for node in nodes {
        assert!(
            !vyre_megakernel::grid_sync::requires_grid_sync(&node.program),
            "Fix: no segment may carry the fence it was split at"
        );
    }
}

/// WHY: a fence split must carry every mutable buffer whose first write occurs
/// before a later segment reads it. Ordering the launches through one retained
/// buffer is insufficient when a sibling backend-allocated carrier crosses the
/// same fence.
#[test]
fn grid_fence_split_carries_every_mutable_sibling_value() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read_write("state", 0, DataType::U32).with_count(1),
            BufferDecl::read_write("scratch", 1, DataType::U32).with_count(1),
        ],
        [32, 1, 1],
        vec![
            Node::store("scratch", Expr::u32(0), Expr::u32(2)),
            Node::barrier_with_ordering(vyre_foundation::ir::MemoryOrdering::GridSync),
            Node::store(
                "state",
                Expr::u32(0),
                Expr::bitor(
                    Expr::load("state", Expr::u32(0)),
                    Expr::load("scratch", Expr::u32(0)),
                ),
            ),
        ],
    );
    let graph = ProgramGraph::from_program("sibling_carriers", program).unwrap();
    let validated = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device_with(|_| ()),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .validate()
    .expect("every mutable sibling carrier must survive the launch split");
    let nodes = validated.graph().nodes();
    assert_eq!(nodes.len(), 2);
    let produced = nodes[0]
        .output_ports
        .iter()
        .position(|port| port.buffer == "scratch")
        .and_then(|position| nodes[0].outputs.get(position))
        .copied()
        .expect("the producing segment must publish scratch");
    let consumed = nodes[1]
        .inputs
        .iter()
        .find(|input| input.buffer == "scratch")
        .map(|input| input.value)
        .expect("the consuming segment must bind scratch");
    assert_eq!(consumed, produced);
    assert_eq!(
        nodes[1]
            .output_ports
            .iter()
            .find(|port| port.buffer == "scratch")
            .map(|port| port.contract.lifetime),
        Some(vyre_foundation::ir::ValueLifetime::Retained),
        "an internal sibling carrier must remain retained rather than become a caller output"
    );
}

/// WHY: the first segment can publish the value that orders later launches.
/// Requiring pre-existing retained input state rejects valid producer-consumer
/// pipelines whose only crossing value is backend allocated.
#[test]
fn grid_fence_split_accepts_a_first_write_carrier_without_retained_input() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(1),
            BufferDecl::read_write("intermediate", 1, DataType::U32)
                .with_count(1)
                .with_pipeline_live_out(true),
            BufferDecl::output("out", 2, DataType::U32).with_count(1),
        ],
        [32, 1, 1],
        vec![
            Node::store(
                "intermediate",
                Expr::u32(0),
                Expr::load("input", Expr::u32(0)),
            ),
            Node::barrier_with_ordering(vyre_foundation::ir::MemoryOrdering::GridSync),
            Node::store(
                "out",
                Expr::u32(0),
                Expr::load("intermediate", Expr::u32(0)),
            ),
        ],
    );
    let graph = ProgramGraph::from_program("first_write_carrier", program).unwrap();
    let validated = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device_with(|_| ()),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .validate()
    .expect("a first-write carrier must order split launches");
    let nodes = validated.graph().nodes();
    assert_eq!(nodes.len(), 2);
    let produced = nodes[0]
        .output_ports
        .iter()
        .position(|port| port.buffer == "intermediate")
        .and_then(|position| nodes[0].outputs.get(position))
        .copied()
        .expect("the first segment must publish the carrier");
    assert_eq!(
        nodes[1]
            .inputs
            .iter()
            .find(|input| input.buffer == "intermediate")
            .map(|input| input.value),
        Some(produced)
    );
}

/// WHY: a caller-visible output can be written before a whole-grid fence and
/// read after it. Intermediate segments must carry its bytes as retained state,
/// while the final segment must preserve the public Output lifetime.
#[test]
fn grid_fence_split_preserves_a_crossing_caller_output() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read_write("state", 0, DataType::U32).with_count(1),
            BufferDecl::output("result", 1, DataType::U32).with_count(1),
        ],
        [32, 1, 1],
        vec![
            Node::store("result", Expr::u32(0), Expr::u32(2)),
            Node::barrier_with_ordering(vyre_foundation::ir::MemoryOrdering::GridSync),
            Node::store(
                "state",
                Expr::u32(0),
                Expr::bitor(
                    Expr::load("state", Expr::u32(0)),
                    Expr::load("result", Expr::u32(0)),
                ),
            ),
        ],
    );
    let graph = ProgramGraph::from_program("caller_output_carrier", program).unwrap();
    let validated = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device_with(|_| ()),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .validate()
    .expect("a caller output crossing a fence must survive the launch split");
    let nodes = validated.graph().nodes();
    assert_eq!(nodes.len(), 2);
    let produced = nodes[0]
        .output_ports
        .iter()
        .position(|port| port.buffer == "result")
        .and_then(|position| nodes[0].outputs.get(position))
        .copied()
        .expect("the producing segment must publish the output carrier");
    assert_eq!(
        nodes[1]
            .inputs
            .iter()
            .find(|input| input.buffer == "result")
            .map(|input| input.value),
        Some(produced)
    );
    let final_port = nodes[1]
        .output_ports
        .iter()
        .find(|port| port.buffer == "result")
        .expect("the final segment must publish the caller output");
    assert_eq!(
        final_port.contract.lifetime,
        vyre_foundation::ir::ValueLifetime::Output
    );
    assert_eq!(final_port.retained_successor_of, Some(produced));
}

/// WHY: cooperative dispatch already enforces a whole-grid fence in one kernel.
/// Splitting that kernel would turn backend-allocated in-place pipeline storage
/// into a retained host input and change the artifact ABI.
#[test]
fn a_cooperative_device_preserves_a_cuttable_grid_fence() {
    let graph = ProgramGraph::from_program("fence", fenced_program(false)).unwrap();
    let validated = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device_with(|_| ()).with_cooperative_launch(true),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .validate()
    .expect("Fix: a cooperative device must keep and admit the fenced kernel");
    let nodes = validated.graph().nodes();
    assert_eq!(
        nodes.len(),
        1,
        "cooperative execution needs no launch split"
    );
    assert!(
        vyre_megakernel::grid_sync::requires_grid_sync(&nodes[0].program),
        "the target must still see the fence and select cooperative direct dispatch"
    );
}

/// WHY: 150.11. A whole-grid fence runs in one kernel only under a cooperative
/// launch. A fence the planner cannot cut is still in the graph compilation
/// consumes, so the compiler rejects it instead of leaving the target emitter to
/// fail later.
///
/// A conditional fence is the uncuttable case: hoisting it out of the branch
/// would change which invocations reach it, so the planner copies it verbatim
/// and it survives into a segment.
#[test]
fn an_uncuttable_grid_fence_fails_without_cooperative_launch() {
    let graph = ProgramGraph::from_program("fence", fenced_program(true)).unwrap();
    let error = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device_with(|_| ()),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .validate()
    .err()
    .expect("a device without cooperative launch must reject a whole-grid fence");
    assert_eq!(error.diagnostic.code.as_str(), "MKC001_INVALID_PROGRAM");
    assert!(
        error
            .diagnostic
            .message
            .contains("cannot launch a cooperative grid"),
        "Fix: the refusal must name the device fact it read: {}",
        error.diagnostic.message
    );
}

/// The same uncuttable fence compiles once the device reports the launch it
/// needs, so the refusal reads the device fact and not the fence alone.
#[test]
fn an_uncuttable_grid_fence_compiles_with_cooperative_launch() {
    let graph = ProgramGraph::from_program("fence", fenced_program(true)).unwrap();
    CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device_with(|_| ()).with_cooperative_launch(true),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .validate()
    .expect("Fix: a cooperative device must accept a whole-grid fence");
}

/// A program carrying a whole-grid fence, inside a branch when
/// `under_a_uniform_branch`.
///
/// An unconditional fence at the top of the entry sequence is a cut point. A
/// fence inside a branch is not, because the branch decides where in the
/// sequence it sits and hoisting it out would move it.
///
/// The branch condition is `true`, which every lane of a workgroup takes
/// together. That is deliberate: a fence under a divergent condition is refused
/// by IR validation as V010 and never reaches device admission, which is the
/// decision these cases read. Uncuttable here means the planner cannot lift the
/// fence, not that the fence is invalid.
fn fenced_program(under_a_uniform_branch: bool) -> Program {
    let fence = Node::Barrier {
        ordering: vyre_foundation::ir::MemoryOrdering::GridSync,
    };
    let body = if under_a_uniform_branch {
        vec![
            Node::store("out", Expr::u32(0), Expr::u32(1)),
            Node::if_then(Expr::bool(true), vec![fence]),
        ]
    } else {
        vec![Node::store("out", Expr::u32(0), Expr::u32(1)), fence]
    };
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
        [32, 1, 1],
        body,
    );
    assert!(
        vyre_megakernel::grid_sync::requires_grid_sync(&program),
        "fixture must carry a whole-grid fence"
    );
    program
}

/// Device facts stating one capability decision, with every other capability
/// present and generous budgets, so a test isolates the flag it flips.
fn device_with(
    edit: impl FnOnce(&mut vyre_foundation::validate::BackendCapabilities),
) -> DeviceFacts {
    let mut capabilities = vyre_test_support::backend_capabilities::all_granted();
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
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .validate()
    .unwrap();
    let error = compile(&request).expect_err("overflow must fail closed");
    assert_eq!(error.diagnostic.code.as_str(), "MKC011_RESOURCE_OVERFLOW");
}

/// WHY: persisted v6 artifacts must be rejected after the v7 named-resource-ABI cutover.
#[test]
fn stale_artifact_version_is_rejected_before_body_decode() {
    let artifact = compile(&request(LIMIT)).unwrap();
    let mut bytes = artifact.to_bytes().unwrap();
    bytes[4..6].copy_from_slice(&6u16.to_le_bytes());
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
/// WHY: entry ABI input and output records must preserve Program buffer order even when graph ports are declared in a different order.
#[test]
fn entry_abi_inputs_and_outputs_preserve_program_buffer_order_despite_port_reordering() {
    let mut graph = ProgramGraph::new();
    let in_contract = contract(BufferAccess::ReadOnly, ValueLifetime::Invocation);
    let val_x = graph.add_external_value("x", in_contract.clone()).unwrap();
    let val_y = graph.add_external_value("y", in_contract).unwrap();
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("first", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("second", 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("out", 2, BufferAccess::WriteOnly, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(
                Expr::load("first", Expr::u32(0)),
                Expr::load("second", Expr::u32(0)),
            ),
        )],
    );

    graph
        .add_node(
            "node_rev",
            program,
            vec![
                GraphInput {
                    buffer: "second".into(),
                    value: val_y,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                },
                GraphInput {
                    buffer: "first".into(),
                    value: val_x,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                },
            ],
            vec![GraphOutput {
                buffer: "out".into(),
                name: "res".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let mut symbols = BTreeMap::new();
    symbols.insert("items".into(), 32);
    let req = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), symbols),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .validate()
    .unwrap();

    let artifact = compile(&req).expect("compilation must succeed");
    let entry = artifact.abi().entries.first().expect("entry must exist");

    assert_eq!(entry.inputs, vec![ArtifactValueId(0), ArtifactValueId(1)]);
    assert_eq!(
        entry
            .input_bindings
            .iter()
            .map(|binding| (binding.buffer.as_str(), binding.value))
            .collect::<Vec<_>>(),
        vec![
            ("first", ArtifactValueId(0)),
            ("second", ArtifactValueId(1))
        ]
    );
    assert_eq!(
        entry
            .output_bindings
            .iter()
            .map(|binding| (binding.buffer.as_str(), binding.value))
            .collect::<Vec<_>>(),
        vec![("out", ArtifactValueId(2))]
    );
}

/// WHY: ResourceRecord retained_predecessor must serialize and round-trip through artifact framing.
#[test]
fn resource_record_retained_predecessor_round_trips() {
    let mut graph = ProgramGraph::new();
    let mut symbols = BTreeMap::new();
    symbols.insert("items".into(), 32);
    let state_init = graph
        .add_external_value(
            "state_init",
            contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
        )
        .unwrap();
    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in_s", 0, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage("out_s", 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "out_s",
            Expr::u32(0),
            Expr::load("in_s", Expr::u32(0)),
        )],
    );
    graph
        .add_node(
            "node0",
            prog,
            vec![GraphInput {
                buffer: "in_s".into(),
                value: state_init,
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            }],
            vec![GraphOutput {
                buffer: "out_s".into(),
                name: "state_final".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                retained_successor_of: Some(state_init),
            }],
        )
        .unwrap();

    let req = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), symbols),
        DeviceFacts::unknown(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, LIMIT),
    )
    .validate()
    .unwrap();

    let artifact = compile(&req).expect("compilation must succeed");
    let bytes = artifact.to_bytes().expect("artifact must encode");
    let decoded = Artifact::from_bytes(&bytes).expect("artifact must decode");

    let final_res = decoded
        .resources()
        .iter()
        .find(|r| r.name == "state_final")
        .expect("state_final resource must exist");
    assert_eq!(
        final_res.retained_predecessor,
        Some(ArtifactValueId(state_init.0))
    );
}
