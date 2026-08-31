//! One selected schedule is the only launch authority a compiled artifact has.
//!
//! Each contract here derives its expectation from the selected schedule rather
//! than from a written-down shape, so a compiler that recorded a launch nothing
//! selected fails these tests instead of agreeing with them.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::ir::{
    BufferAccess, GraphInput, GraphOutput, Program, ProgramGraph, ValueContract, ValueLifetime,
};
use vyre_foundation::schedule::ScheduleTransform;
use vyre_megakernel::allocation::{
    AddressSpace, AllocationPlan, AllocationRegion, RegionOwner, REGION_ALIGNMENT,
};
use vyre_megakernel::{
    compile, Artifact, ArtifactNodeId, ArtifactValueId, CompileObjective, CompileRequest,
    DependencyEndpoint, DeviceFacts, Digest, EntryPersistence, ExecutionMode, ExternalFacts,
    GeometryRecord, ObjectiveMetric, ResourceLifetime, ResourceRecord, SearchBudget,
};

use vyre_test_support::graph_values::{graph_output, u32_symbolic};

use vyre_test_support::pass_programs::{add_program, copy_program};

#[path = "graph_fixtures/mod.rs"]
mod graph_fixtures;

fn contract(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    u32_symbolic(access, lifetime)
}

/// Two chained nodes over one caller input, one constant, and one retained value.
///
/// The chain is what makes the contracts meaningful: `middle` is produced by one
/// entry point and consumed by the next, so it is the only value the artifact
/// has to place in its own workspace.
fn chain_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value(
            "input",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("input value must be accepted");
    let constant = graph
        .add_external_value(
            "constant",
            contract(BufferAccess::ReadOnly, ValueLifetime::Constant),
        )
        .expect("constant value must be accepted");
    let (_, produced) = graph
        .add_node(
            "alpha",
            add_program("input", "constant", "middle"),
            graph_fixtures::value_and_constant_ports(input, constant),
            vec![graph_output(
                "middle",
                contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
            )],
        )
        .expect("producer node must be accepted");
    graph
        .add_node(
            "beta",
            copy_program("middle", "result"),
            vec![GraphInput {
                buffer: "middle".into(),
                value: produced[0],
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "result".into(),
                name: "result".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("consumer node must be accepted");
    graph
}

fn facts(launch_batch: u32) -> ExternalFacts {
    let mut facts = ExternalFacts::new(Digest([0xA5; 32]), BTreeMap::from([("items".into(), 24)]))
        .with_expected_launch_batch(launch_batch);
    facts
        .constant_identities
        .insert(vyre_foundation::ir::GraphValueId(1), Digest([0x5A; 32]));
    facts
}

fn artifact_for(device: DeviceFacts, launch_batch: u32) -> Artifact {
    let request = CompileRequest::new(
        chain_graph(),
        facts(launch_batch),
        device,
        SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("fixture request must validate");
    compile(&request).expect("fixture request must compile")
}

fn static_artifact() -> Artifact {
    artifact_for(DeviceFacts::unknown(), 1)
}

/// A device that measured a launch cost and can hold a resident grid, which is
/// what makes a persistent route profitable at all.
fn persistent_artifact() -> Artifact {
    artifact_for(
        DeviceFacts::unknown()
            .with_cooperative_launch(true)
            .with_launch_costs(4_000, 100),
        8,
    )
}

fn record_for(artifact: &Artifact, node: ArtifactNodeId) -> &GeometryRecord {
    artifact
        .geometry()
        .iter()
        .find(|record| record.node == node)
        .expect("every node carries one selected geometry record")
}

/// WHY: the artifact records launch geometry so no consumer computes one. Every
/// field of every record has to be the projection of the schedule phase that
/// covers the node, because a field derived any other way is a launch the
/// selected schedule never authorized.
#[test]
fn every_recorded_launch_is_the_selected_schedule_phase_that_covers_the_node() {
    let artifact = static_artifact();
    let schedule = &artifact.selected_plan().schedule;

    assert_eq!(artifact.geometry().len(), artifact.nodes().len());
    assert_eq!(
        artifact.schema_version(),
        vyre_megakernel::ARTIFACT_SCHEMA_VERSION,
        "an artifact stamps the schema its own crate states"
    );

    for node in artifact.nodes() {
        let record = record_for(&artifact, node.id);
        let phase = schedule
            .phase_for_region(node.id.0)
            .expect("the selected schedule covers every node");

        assert_eq!(record.phase, phase.id);
        assert_eq!(record.logical_coverage, phase.grid);
        assert_eq!(record.workgroup_size, phase.workgroup);
        assert_eq!(record.vector_width, phase.vector_width);
        assert_eq!(
            record.dynamic_shared_bytes,
            u32::try_from(phase.resources.shared_bytes).expect("fixture shared bytes fit u32")
        );
        assert_eq!(
            record.launch_intent.private_bytes,
            phase.resources.private_bytes
        );
        assert_eq!(
            record.launch_intent.registers_per_invocation,
            phase.resources.registers_per_invocation
        );
        assert_eq!(
            record.grid,
            GeometryRecord::covering_grid(phase.grid, phase.workgroup)
                .expect("a selected phase covers positive points"),
        );
        for axis in 0..3 {
            let covered = u64::from(record.grid[axis]) * u64::from(record.workgroup_size[axis]);
            assert!(
                covered >= record.logical_coverage[axis],
                "axis {axis} launches {covered} points for {} logical points",
                record.logical_coverage[axis]
            );
            assert!(
                covered - record.logical_coverage[axis] < u64::from(record.workgroup_size[axis])
            );
        }
    }
}

/// WHY: a consumer submits entry points, so the dependency order it needs is
/// between entry points. The record has to carry the same order the canonical
/// dependency edges state, or a submission ordered from the record runs a
/// consumer before its producer.
#[test]
fn recorded_predecessors_are_the_canonical_dependency_order() {
    let artifact = static_artifact();
    let mut expected = BTreeMap::<ArtifactNodeId, BTreeSet<ArtifactNodeId>>::new();
    for edge in artifact.dependencies() {
        if let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) = (edge.from, edge.to)
        {
            if from != to {
                expected.entry(to).or_default().insert(from);
            }
        }
    }
    assert!(
        expected.values().any(|set| !set.is_empty()),
        "the fixture chain must carry at least one entry-point dependency"
    );

    for record in artifact.geometry() {
        let recorded: BTreeSet<ArtifactNodeId> = record.predecessors.iter().copied().collect();
        assert_eq!(
            recorded,
            expected.get(&record.node).cloned().unwrap_or_default(),
            "node {} predecessors",
            record.node.0
        );
        assert!(
            !record.predecessors.contains(&record.node),
            "a node cannot wait on itself"
        );
    }
}

/// WHY: the workgroup a source program declares is an input to the search, not
/// its result. Leaving the declared shape in the artifact let target compilation
/// rewrite it during emission, so the bytes the artifact authenticated and the
/// bytes the device ran disagreed on the one field a launch cannot recover from.
#[test]
fn node_programs_are_frozen_at_the_selected_workgroup() {
    let artifact = static_artifact();
    for node in artifact.nodes() {
        let program = Program::from_wire(&node.program).expect("a recorded program decodes");
        assert_eq!(
            program.workgroup_size,
            record_for(&artifact, node.id).workgroup_size,
            "node {} program declares a shape the artifact did not select",
            node.id.0
        );
    }
}

/// WHY: the runtime allocates the recorded regions and binds their offsets
/// verbatim. A region for a value the caller owns would double-allocate it, a
/// missing placement for a produced value would leave an entry point with
/// nothing to write into, and a recorded peak the placements do not hold is the
/// figure ranking priced, so it must be the figure the artifact states.
#[test]
fn every_value_is_placed_where_its_resource_row_says_it_lives() {
    let artifact = static_artifact();
    let plan = artifact.allocation();
    let rows: BTreeMap<ArtifactValueId, &ResourceRecord> = artifact
        .resources()
        .iter()
        .map(|resource| (resource.value, resource))
        .collect();

    let mut produced = BTreeSet::new();
    for entry in &artifact.abi().entries {
        produced.extend(entry.outputs.iter().copied());
    }
    let expected: BTreeSet<ArtifactValueId> = produced
        .iter()
        .copied()
        .filter(|value| {
            rows.get(value).is_some_and(|row| {
                matches!(
                    row.lifetime,
                    ResourceLifetime::Invocation | ResourceLifetime::Retained
                ) && row.byte_count > 0
            })
        })
        .collect();
    assert!(
        !expected.is_empty(),
        "the fixture chain must produce at least one value the artifact owns"
    );
    let owned: BTreeSet<ArtifactValueId> = plan
        .owned()
        .flat_map(|region| region.placements.iter())
        .map(|placement| placement.value)
        .collect();
    assert_eq!(owned, expected);

    let mut end = 0;
    for region in plan.owned() {
        assert_eq!(region.offset % REGION_ALIGNMENT, 0);
        assert!(region.offset >= end, "artifact regions must not overlap");
        for placement in &region.placements {
            let row = rows[&placement.value];
            assert_eq!(placement.bytes, row.byte_count);
            assert_eq!(placement.lifetime, row.lifetime);
            assert_eq!(placement.first_stage, row.first_stage);
            assert_eq!(placement.last_stage, row.last_stage);
            assert!(placement.byte_offset + placement.bytes <= region.bytes);
        }
        end = region.offset + region.bytes;
    }
    for region in &plan.regions {
        if region.owner == RegionOwner::Artifact {
            continue;
        }
        assert_eq!(region.offset, 0, "a caller binds a whole buffer");
        assert_eq!(region.placements.len(), 1);
        assert!(!owned.contains(&region.placements[0].value));
    }

    let final_stage = artifact
        .resources()
        .iter()
        .map(|row| row.last_stage)
        .max()
        .unwrap_or(0);
    let independent_peak = (0..=final_stage)
        .map(|stage| {
            artifact
                .resources()
                .iter()
                .filter(|row| row.first_stage <= stage && stage <= row.last_stage)
                .map(|row| row.byte_count)
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0);
    assert_eq!(plan.aggregate_peak_bytes, independent_peak);
    assert_eq!(
        plan.owned_bytes(),
        plan.owned().map(|region| region.bytes).sum::<u64>()
    );
}

/// WHY: persistence is a property of the selected schedule, not a label beside
/// it. A recorded mode with no queue capacity left every consumer to size the
/// queue itself, and two consumers sizing it differently is a deadlock rather
/// than a slowdown.
#[test]
fn a_persistent_route_records_the_queue_the_schedule_reserved() {
    let static_artifact = static_artifact();
    assert_eq!(
        static_artifact.selected_plan().execution,
        ExecutionMode::Static
    );
    for record in static_artifact.geometry() {
        assert_eq!(record.persistence, EntryPersistence::Static);
    }
    assert!(
        !static_artifact
            .selected_plan()
            .schedule
            .transforms
            .iter()
            .any(|record| matches!(record.transform, ScheduleTransform::PersistentQueue { .. })),
        "a static route reserves no device-side queue"
    );

    let persistent = persistent_artifact();
    let ExecutionMode::Persistent { saved_ns } = persistent.selected_plan().execution else {
        panic!("a measured launch cost on a cooperative device selects a persistent route");
    };
    assert!(saved_ns > 0);

    let schedule = &persistent.selected_plan().schedule;
    for record in persistent.geometry() {
        let EntryPersistence::Persistent { queue_capacity } = record.persistence else {
            panic!("every entry of a persistent route drains the recorded queue");
        };
        assert!(queue_capacity > 0);
        let phase = schedule
            .phase_for_region(record.node.0)
            .expect("the selected schedule covers every node");
        assert_eq!(phase.resources.queue_capacity, queue_capacity);
        assert!(schedule.transforms.iter().any(|applied| matches!(
            applied.transform,
            ScheduleTransform::PersistentQueue { phase: id, capacity }
                if id == record.phase && capacity == queue_capacity
        )));
    }
}

/// WHY: the geometry set and the allocation plan are what a consumer submits, so
/// they have to survive the byte boundary exactly. A round trip that dropped or
/// reordered a field would leave the decoded artifact launchable and wrong.
#[test]
fn the_recorded_launch_survives_the_byte_boundary_exactly() {
    for artifact in [static_artifact(), persistent_artifact()] {
        let bytes = artifact.to_bytes().expect("an artifact encodes");
        let decoded = Artifact::from_bytes(&bytes).expect("canonical bytes decode");
        assert_eq!(decoded.digest(), artifact.digest());
        assert_eq!(decoded.geometry(), artifact.geometry());
        assert_eq!(decoded.allocation(), artifact.allocation());
        assert_eq!(
            decoded.to_bytes().expect("a decoded artifact re-encodes"),
            bytes
        );
    }
}

fn rejection_path(plan: &AllocationPlan, case: &str) -> String {
    plan.validate()
        .expect_err(case)
        .diagnostic
        .location
        .and_then(|location| location.path)
        .unwrap_or_default()
}

/// WHY: the space a region is addressed in decides what a backend may emit
/// against it. A constant-lifetime value addressed as device storage is emitted
/// through a writable path, and a produced value addressed as constant storage
/// is emitted through a path the device forbids writes on. The two facts are
/// derived from one another here, so a plan stating one and meaning the other
/// is refused where the plan is built rather than at whichever backend notices.
#[test]
fn the_space_addressing_a_value_agrees_with_its_lifetime() {
    let artifact = static_artifact();
    let plan = artifact.allocation();
    plan.validate().expect("a selected plan validates");

    // Regions are ordered by address space, so a mutation that keeps the order
    // ascending is the one that reaches the placement rule under test.
    for (case, space, wanted, last) in [
        (
            "a device value addressed as constant",
            AddressSpace::Constant,
            ResourceLifetime::Invocation,
            true,
        ),
        (
            "a constant value addressed as device",
            AddressSpace::Device,
            ResourceLifetime::Constant,
            false,
        ),
    ] {
        let matches = |region: &AllocationRegion| {
            region.owner == RegionOwner::Caller
                && region
                    .placements
                    .iter()
                    .all(|placement| placement.lifetime == wanted)
        };
        let index = if last {
            plan.regions.iter().rposition(matches)
        } else {
            plan.regions.iter().position(matches)
        }
        .unwrap_or_else(|| panic!("the fixture plan places {case}"));
        let mut mutated = plan.clone();
        mutated.regions[index].address_space = space;
        assert_eq!(
            rejection_path(&mutated, case),
            format!("artifact.allocation.regions[{index}].placements[0].lifetime"),
            "{case}"
        );
    }
}

/// WHY: constant storage is filled once, by the caller, before any entry runs.
/// An artifact-owned constant region states that the runtime allocates it, which
/// leaves storage no entry may write and no caller filled.
#[test]
fn the_artifact_never_allocates_constant_storage() {
    let artifact = static_artifact();
    let plan = artifact.allocation();
    let index = plan
        .regions
        .iter()
        .position(|region| region.owner == RegionOwner::Artifact)
        .expect("the fixture plan owns one region");
    let mut mutated = plan.clone();
    mutated.regions[index].address_space = AddressSpace::Constant;
    assert_eq!(
        rejection_path(&mutated, "artifact-allocated constant storage"),
        format!("artifact.allocation.regions[{index}].owner")
    );
}
