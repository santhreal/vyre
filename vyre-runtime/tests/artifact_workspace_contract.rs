//! Runtime allocation and binding of the workspace an artifact recorded.
//!
//! WHY: a multi-entry artifact records one allocation region per value it
//! produces for itself, with the offset and byte count the selected schedule
//! assigned. Nothing read that plan: the runtime allocated whatever a caller
//! asked for and bound whatever a caller supplied, so a cross-entry value could
//! be sized by the caller, shared with an unrelated value, or replaced by a
//! buffer the compiler never planned for. These contracts pin the plan as the
//! only authority over that storage.

use std::sync::Arc;

use vyre_driver::{BackendRegistration, BoundResource};

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{Artifact, ArtifactEnvelope, ArtifactValueId};
use vyre_runtime::artifact_admission::ArtifactSession;

use vyre_test_support::artifact_fixtures;

use crate::artifact_session_fixtures::{
    fixture_backend_registration, fixture_target_payload, SessionFixtureMaterializer,
};

const FORMAT: &str = "workspace.target";
const MIDDLE: &str = "middle";
const OUTPUT: &str = "out";
static WORKSPACE_REGISTRATION: BackendRegistration =
    fixture_backend_registration("workspace-artifact");

/// A two-stage artifact: the first entry's output is the second entry's input.
///
/// The intermediate value is the whole point. It is produced inside the artifact
/// and read inside the artifact, so nothing outside owns it and the compiler
/// places it in an artifact-owned region.
fn two_stage_artifact() -> Artifact {
    let mut graph = ProgramGraph::new();
    let (_, produced) = graph
        .add_node(
            "first",
            Program::wrapped(
                vec![BufferDecl::output(OUTPUT, 0, DataType::U32).with_count(1)],
                [1, 1, 1],
                vec![Node::store(OUTPUT, Expr::u32(0), Expr::u32(1))],
            ),
            Vec::new(),
            vec![GraphOutput {
                buffer: OUTPUT.into(),
                name: MIDDLE.into(),
                contract: value(BufferAccess::ReadWrite, ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("the workspace fixture must accept its producer");
    let middle = *produced
        .first()
        .expect("the producer declares one output value");
    graph
        .add_node(
            "second",
            Program::wrapped(
                vec![
                    BufferDecl::storage(MIDDLE, 0, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(1),
                    BufferDecl::output(OUTPUT, 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    OUTPUT,
                    Expr::u32(0),
                    Expr::load(MIDDLE, Expr::u32(0)),
                )],
            ),
            vec![GraphInput {
                buffer: MIDDLE.into(),
                value: middle,
                contract: value(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: OUTPUT.into(),
                name: OUTPUT.into(),
                contract: value(BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("the workspace fixture must accept its consumer");
    artifact_fixtures::compile_graph(graph, 0)
}

fn value(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(1)],
        access,
        lifetime,
    }
}

/// A session over the two-stage artifact and the recording materializer.
fn session() -> (Artifact, Arc<SessionFixtureMaterializer>, ArtifactSession) {
    let artifact = two_stage_artifact();
    assert!(
        artifact.allocation().owned().next().is_some(),
        "Fix: the fixture must record at least one artifact-owned region, or every contract here \
         is vacuous."
    );
    let mut envelope = ArtifactEnvelope::new(artifact.clone());
    envelope
        .attach_target_payload(fixture_target_payload(&artifact, FORMAT, vec![1, 2, 3, 4]))
        .expect("the fixture payload must attach");
    let materializer =
        SessionFixtureMaterializer::new("workspace-artifact", "workspace-device", FORMAT);
    let session = ArtifactSession::from_envelope_with_materializer(
        &WORKSPACE_REGISTRATION,
        envelope,
        materializer.clone(),
    )
    .expect("the two-stage envelope must materialize");
    (artifact, materializer, session)
}

/// Canonical value the fixture artifact produces for itself.
fn workspace_value(artifact: &Artifact) -> ArtifactValueId {
    artifact
        .allocation()
        .owned()
        .flat_map(|region| region.placements.iter())
        .next()
        .expect("the fixture places one artifact-owned value")
        .value
}

/// WHY: the plan states one region per cross-entry value, with the byte count
/// the schedule assigned. A runtime that rounded, merged, or padded an
/// allocation binds a buffer of a size nothing compiled against.
#[test]
fn the_runtime_allocates_exactly_the_recorded_workspace() {
    let (artifact, materializer, session) = session();

    let workspace = session
        .allocate_workspace()
        .expect("the recorded workspace must allocate");

    let plan = artifact.allocation();
    assert_eq!(workspace.total_bytes(), plan.owned_bytes());
    assert_eq!(
        materializer
            .allocated
            .lock()
            .expect("the allocation log must not be poisoned")
            .as_slice(),
        plan.owned()
            .map(|region| usize::try_from(region.bytes).expect("fixture regions are small"))
            .collect::<Vec<_>>()
            .as_slice(),
        "one allocation per artifact-owned region, of exactly the recorded byte count, in recorded \
         order"
    );
    assert_eq!(workspace.buffers().len(), plan.owned().count());
    for region in plan.owned() {
        for placement in &region.placements {
            assert!(
                workspace.owns(placement.value),
                "the workspace must own every value the plan places in its own region"
            );
        }
    }
    assert_eq!(
        workspace.bindings().len(),
        plan.owned()
            .map(|region| region.placements.len())
            .sum::<usize>(),
        "every placed value binds a buffer"
    );

    let allocated = workspace.buffers().to_vec();
    session
        .free_workspace(workspace)
        .expect("the workspace must release");
    assert_eq!(
        materializer
            .freed
            .lock()
            .expect("the release log must not be poisoned")
            .as_slice(),
        allocated.as_slice(),
        "every allocated buffer is released, and nothing else"
    );
}

/// WHY: the artifact allocated its own storage for the values its entries pass
/// between themselves. A caller buffer in that place is a wrong bind, not a
/// substitution, and silently preferring either side is how the compiler stops
/// owning cross-entry storage.
#[test]
fn a_caller_cannot_rebind_a_workspace_owned_value() {
    let (artifact, _materializer, session) = session();
    let workspace = session
        .allocate_workspace()
        .expect("the recorded workspace must allocate");
    let owned = workspace_value(&artifact);
    let name = artifact
        .resources()
        .iter()
        .find(|resource| resource.value == owned)
        .expect("the workspace value is a canonical resource")
        .name
        .clone();
    let caller = session
        .allocate_resident(4)
        .expect("the fixture materializer must allocate");

    let error = session
        .resident_bindings_with_workspace(&workspace, [(name.as_str(), &caller)])
        .expect_err("a caller must not rebind a workspace-owned value");

    let text = error.to_string();
    assert!(
        text.contains("workspace-owned") && text.contains(&format!("canonical value {}", owned.0)),
        "the refusal must name the value and why it is refused; got `{text}`"
    );
}

/// WHY: binding over a workspace must still supply every value the entries
/// declare. The workspace covers what the artifact produces for itself and
/// nothing else, so a graph input or a public output left out is refused rather
/// than launched unbound.
#[test]
fn workspace_bindings_cover_the_workspace_and_demand_the_rest() {
    let (artifact, _materializer, session) = session();
    let workspace = session
        .allocate_workspace()
        .expect("the recorded workspace must allocate");
    let owned = workspace_value(&artifact);
    let caller_values = artifact
        .resources()
        .iter()
        .filter(|resource| resource.value != owned)
        .map(|resource| resource.name.clone())
        .collect::<Vec<_>>();
    assert!(
        !caller_values.is_empty(),
        "Fix: the fixture must carry a caller-owned resource beside its workspace."
    );
    let caller = session
        .allocate_resident(4)
        .expect("the fixture materializer must allocate");

    let missing = session
        .resident_bindings_with_workspace(&workspace, [])
        .expect_err("a caller-owned value must not be defaulted");
    assert!(
        missing.to_string().contains("requires resident resource"),
        "the refusal must name the unbound entry resource; got `{missing}`"
    );

    let bound = session
        .resident_bindings_with_workspace(
            &workspace,
            caller_values
                .iter()
                .map(|name| (name.as_str(), &caller))
                .collect::<Vec<_>>(),
        )
        .expect("the workspace plus every caller-owned value must bind");

    assert_eq!(
        bound.resources().get(&owned),
        workspace
            .bindings()
            .get(&owned)
            .map(|resource| BoundResource::Resident(resource.clone()))
            .as_ref(),
        "the workspace's own region must reach the binding set"
    );
    for name in &caller_values {
        let value = artifact
            .resources()
            .iter()
            .find(|resource| &resource.name == name)
            .expect("the caller value is a canonical resource")
            .value;
        assert_eq!(
            bound.resources().get(&value),
            Some(&BoundResource::Resident(caller.clone()))
        );
    }
}
