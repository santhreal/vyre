//! WHY: the allocation plan states where every value lives and which stages
//! hold it. Lowering binds storage, so a binding the plan does not authorize
//! is a device that reads bytes nobody wrote or writes storage the caller
//! owns read-only. Each case below mutates one plan field of an otherwise
//! selected artifact and requires the projection to refuse it.
//!
//! What this does NOT catch: a plan that disagrees with the resource records
//! it was derived from. Decode rejects that before a module is projected.

use std::collections::BTreeMap;

use vyre_foundation::ir::{BufferAccess, GraphInput, GraphOutput, ProgramGraph, ValueLifetime};
use vyre_test_support::graph_values::{graph_output, u32_symbolic};
use vyre_test_support::pass_programs::copy_program;

use super::{compile_selected_modules, TargetCompileError};
use crate::allocation::{AddressSpace, RegionOwner};
use crate::schema::ResourceLifetime;
use crate::{
    compile, Artifact, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget, TargetPayloadFormat, TargetProfile, TargetResourceBinding,
};

/// Two chained nodes: `middle` is written at one stage and read at the next,
/// so the plan owns one region and the caller owns the rest.
fn chained_artifact() -> Artifact {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value(
            "input",
            u32_symbolic(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("Fix: keep the fixture input value acceptable.");
    let (_, produced) = graph
        .add_node(
            "alpha",
            copy_program("input", "middle"),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: u32_symbolic(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![graph_output(
                "middle",
                u32_symbolic(BufferAccess::ReadWrite, ValueLifetime::Invocation),
            )],
        )
        .expect("Fix: keep the fixture producer node acceptable.");
    graph
        .add_node(
            "beta",
            copy_program("middle", "result"),
            vec![GraphInput {
                buffer: "middle".into(),
                value: produced[0],
                contract: u32_symbolic(BufferAccess::ReadWrite, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "result".into(),
                name: "result".into(),
                contract: u32_symbolic(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("Fix: keep the fixture consumer node acceptable.");
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0xA5; 32]), BTreeMap::from([("items".into(), 24)])),
        DeviceFacts::unknown(),
        SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("Fix: keep the fixture request validatable.");
    compile(&request).expect("Fix: keep the fixture request compilable.")
}

/// Index of the first region `owner` holds whose placement has `lifetime`.
fn region_of(artifact: &Artifact, owner: RegionOwner, lifetime: ResourceLifetime) -> usize {
    artifact
        .allocation()
        .regions
        .iter()
        .position(|region| {
            region.owner == owner
                && region
                    .placements
                    .iter()
                    .any(|placement| placement.lifetime == lifetime)
        })
        .expect("Fix: keep the fixture graph producing one such region.")
}

/// Sentinel the fixture emitter fails with once projection has happened.
const PROJECTED: &str = "fixture emitter stops once bindings are projected";

/// Projects the bindings of every selected group through the real boundary.
fn project(artifact: &Artifact) -> Result<Vec<TargetResourceBinding>, TargetCompileError> {
    let mut projected = Vec::new();
    let outcome = compile_selected_modules(
        artifact,
        TargetPayloadFormat::new("test.target-binary", 1)
            .expect("Fix: keep the fixture format valid."),
        TargetProfile::new("test.target-binary", 1, [64, 1, 1], 64, 1_024, 0)
            .expect("Fix: keep the fixture profile valid."),
        |selected, _| {
            projected.extend(selected.canonical_bindings.iter().cloned());
            Err(TargetCompileError::Unsupported(PROJECTED.into()))
        },
    );
    match outcome {
        Ok(_) => Ok(projected),
        Err(TargetCompileError::Unsupported(message)) if message == PROJECTED => Ok(projected),
        Err(other) => Err(other),
    }
}

fn refusal(artifact: &Artifact) -> String {
    project(artifact)
        .expect_err("Fix: refuse a binding the allocation plan does not authorize.")
        .to_string()
}

#[test]
fn every_projected_binding_addresses_storage_the_plan_placed() {
    let artifact = chained_artifact();
    let bindings = project(&artifact).expect("Fix: keep a consistent plan projectable.");
    assert!(!bindings.is_empty());
    for binding in &bindings {
        let placed = artifact.allocation().placement(binding.resource).is_some();
        let empty = artifact
            .resources()
            .iter()
            .find(|record| record.value == binding.resource)
            .is_none_or(|record| record.byte_count == 0);
        assert!(
            placed || empty,
            "value {} is bound with no placement",
            binding.resource.0
        );
    }
}

#[test]
fn a_value_the_plan_places_nowhere_has_no_storage_to_bind() {
    let mut artifact = chained_artifact();
    let region = region_of(
        &artifact,
        RegionOwner::Artifact,
        ResourceLifetime::Invocation,
    );
    artifact.payload.allocation.regions.remove(region);
    let message = refusal(&artifact);
    assert!(
        message.contains("the allocation plan places nowhere"),
        "the refusal must name the unplaced value: {message}"
    );
}

#[test]
fn a_group_outside_a_placement_live_range_binds_reused_bytes() {
    let mut artifact = chained_artifact();
    let last = artifact
        .allocation()
        .regions
        .iter()
        .flat_map(|region| &region.placements)
        .map(|placement| placement.last_stage)
        .max()
        .expect("Fix: keep the fixture plan placing at least one value.");
    for group in &mut artifact.payload.selected_plan.fusion {
        group.stage = last + 1;
    }
    let message = refusal(&artifact);
    assert!(
        message.contains("which the allocation plan holds over stages"),
        "the refusal must name the live range: {message}"
    );
}

#[test]
fn constant_space_storage_is_never_bound_writable() {
    let mut artifact = chained_artifact();
    let region = region_of(
        &artifact,
        RegionOwner::Artifact,
        ResourceLifetime::Invocation,
    );
    artifact.payload.allocation.regions[region].address_space = AddressSpace::Constant;
    let message = refusal(&artifact);
    assert!(
        message.contains("writable"),
        "the refusal must name the writable constant bind: {message}"
    );
}
