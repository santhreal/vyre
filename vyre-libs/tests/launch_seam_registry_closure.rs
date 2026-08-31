//! Registry-derived closure over the compiler-owned launch seam.
//!
//! WHY: every launch geometry and every sequential/persistent route decision
//! belongs to schedule search, not to the caller that submits work. The class
//! that regresses is a library wrapper that reintroduces a caller-chosen grid
//! for one operation, or a compiler that substitutes a unit launch when
//! geometry is unresolved. The roster is the live operation registry, so an
//! operation registered tomorrow is covered without editing this file, and the
//! `ExecutionMode`/`ExecutionTopology` matches carry no catch-all arm, so a new
//! route variant turns this suite red until a decision is recorded for it.
//!
//! This file does not check numeric results; per-operation reference parity
//! suites own that.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::ir::{ProgramGraph, ShapeDim, ValueLifetime};
use vyre_foundation::operation::OperationRegistry;
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    compile, Artifact, CompileObjective, CompileRequest, DeviceFacts, Digest, ExecutionMode,
    ExecutionTopology, ExternalFacts, ObjectiveMetric, SearchBudget,
};

/// Exact extent bound to every symbolic graph dimension.
const SYMBOLIC_EXTENT: u64 = 64;

/// Ranking budget wide enough to reach a persistent candidate and bounded
/// enough to compile the whole registry.
fn budget() -> SearchBudget {
    SearchBudget::new(32, 1 << 20, 1, 0, 5_000_000_000)
}

/// Target facts that grant every gated capability, so legality never hides a
/// route the ranker would otherwise consider.
fn target_facts() -> DeviceFacts {
    DeviceFacts::new(
        BackendCapabilities {
            max_native_int_width: 32,
            ..vyre_test_support::backend_capabilities::all_granted()
        },
        1024,
    )
    .with_cooperative_launch(true)
    .with_compute_units(64)
    .with_subgroup_size(32)
    .with_occupancy(32, 4096)
    .with_launch_costs(200_000, 2_000)
}

/// Derive complete external facts from the graph itself: every symbolic
/// dimension bound, every constant value identified.
fn external_facts(graph: &ProgramGraph, expected_launch_batch: u32) -> ExternalFacts {
    let symbolic_bindings = graph
        .values()
        .iter()
        .flat_map(|value| &value.contract.shape)
        .filter_map(|dim| match dim {
            ShapeDim::Known(_) => None,
            ShapeDim::Symbol(symbol) => Some((symbol.clone(), SYMBOLIC_EXTENT)),
        })
        .collect();
    let constant_identities = graph
        .values()
        .iter()
        .filter(|value| value.contract.lifetime == ValueLifetime::Constant)
        .map(|value| (value.id, Digest([9; 32])))
        .collect();
    let mut facts = ExternalFacts::new(Digest([7; 32]), symbolic_bindings);
    facts.constant_identities = constant_identities;
    facts.with_expected_launch_batch(expected_launch_batch)
}

/// Compile one registered operation through the single neutral compiler entry
/// point. No caller-side geometry or route argument exists to pass.
fn compile_registered(
    graph: &ProgramGraph,
    expected_launch_batch: u32,
) -> Result<Artifact, String> {
    let request = CompileRequest::new(
        graph.clone(),
        external_facts(graph, expected_launch_batch),
        target_facts(),
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 4 << 20),
    )
    .validate()
    .map_err(|error| error.to_string())?;
    compile(&request).map_err(|error| error.to_string())
}

/// Count the linked library registrations, which is also what pulls every
/// inventory submission into this binary.
///
/// A test binary that names no `vyre-libs` item links no submission at all, so
/// the registry would read empty and every closure below would pass over an
/// empty roster.
fn linked_registrations() -> usize {
    vyre_libs::operation_catalog::all_entries().count()
}

/// Every registered operation that builds a program, paired with its graph.
///
/// A graph that fails to build is a failure, not a skip: a silently shrinking
/// roster is how a closure over the registry stops covering the registry.
fn registered_graphs() -> Vec<(&'static str, ProgramGraph)> {
    assert!(
        linked_registrations() > 0,
        "Fix: no library registration reached the link, so the registry roster is empty"
    );
    let mut graphs = Vec::new();
    let mut invalid = Vec::new();
    for entry in OperationRegistry::global().iter() {
        let Some(program) = entry.program() else {
            continue;
        };
        match ProgramGraph::from_program(entry.id, program) {
            Ok(graph) => graphs.push((entry.id, graph)),
            Err(error) => invalid.push(format!("{}: {error}", entry.id)),
        }
    }
    assert!(
        invalid.is_empty(),
        "Fix: every registered program must form a valid whole-program graph:\n{}",
        invalid.join("\n")
    );
    graphs
}

/// Name the route class without a catch-all arm, so a new variant fails to
/// compile until this closure records a decision for it.
fn mode_class(mode: &ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Static => "static",
        ExecutionMode::Persistent { saved_ns } => {
            assert!(
                *saved_ns > 0,
                "Fix: a persistent selection must record the launch overhead it saves"
            );
            "persistent"
        }
    }
}

/// Name the topology class without a catch-all arm.
fn topology_class(topology: &ExecutionTopology) -> &'static str {
    match topology {
        ExecutionTopology::Sequential => "sequential",
        ExecutionTopology::ConcurrentQueue { queues } => {
            assert!(*queues > 0, "Fix: a concurrent queue topology needs queues");
            "concurrent-queue"
        }
        ExecutionTopology::ResidentPartition { partitions, .. } => {
            assert!(
                *partitions > 0,
                "Fix: a resident partition topology needs partitions"
            );
            "resident-partition"
        }
    }
}

#[test]
fn every_registered_operation_receives_compiler_owned_positive_geometry() {
    let graphs = registered_graphs();
    assert!(
        !graphs.is_empty(),
        "Fix: the registry is empty in a binary that links vyre-libs, so inventory submissions are not reaching the link"
    );

    let mut failures = Vec::new();
    let mut distinct_workgroups = BTreeSet::new();
    let mut classes = BTreeSet::new();
    let mut compiled = 0usize;
    for (id, graph) in &graphs {
        let artifact = match compile_registered(graph, 1) {
            Ok(artifact) => artifact,
            Err(error) => {
                failures.push(format!("{id}: compilation failed: {error}"));
                continue;
            }
        };
        let plan = artifact.selected_plan();
        if let Err(error) = plan.validate() {
            failures.push(format!("{id}: selected plan is incomplete: {error}"));
            continue;
        }
        if artifact.geometry().is_empty() {
            failures.push(format!("{id}: artifact records no launch geometry"));
            continue;
        }
        for record in artifact.geometry() {
            if record.workgroup_size.iter().any(|extent| *extent == 0) {
                failures.push(format!(
                    "{id}: node {:?} admits zero geometry {:?}",
                    record.node, record.workgroup_size
                ));
            }
            distinct_workgroups.insert(record.workgroup_size);
        }
        classes.insert((mode_class(&plan.execution), topology_class(&plan.topology)));
        compiled += 1;
    }

    assert!(
        failures.is_empty(),
        "Fix: schedule search must admit complete positive geometry for every registered operation:\n{}",
        failures.join("\n")
    );
    assert_eq!(compiled, graphs.len());
    assert!(
        distinct_workgroups.len() > 1,
        "Fix: every operation admitted the same workgroup size {distinct_workgroups:?}, so geometry is a default rather than a derived decision"
    );
    assert!(
        !classes.is_empty(),
        "Fix: no route class was recorded for the registry"
    );
}

#[test]
fn one_submission_never_selects_a_persistent_route() {
    let mut persistent = Vec::new();
    for (id, graph) in &registered_graphs() {
        let Ok(artifact) = compile_registered(graph, 1) else {
            continue;
        };
        if matches!(
            artifact.selected_plan().execution,
            ExecutionMode::Persistent { .. }
        ) {
            persistent.push(*id);
        }
    }
    assert!(
        persistent.is_empty(),
        "Fix: a single submission cannot amortize persistent setup, so these selections are unearned: {}",
        persistent.join(", ")
    );
}

#[test]
fn launch_facts_alone_move_an_operation_onto_the_persistent_route() {
    let graphs = registered_graphs();
    let mut moved = Vec::new();
    let mut regressed = Vec::new();
    for (id, graph) in &graphs {
        let (Ok(single), Ok(batched)) = (
            compile_registered(graph, 1),
            compile_registered(graph, 4096),
        ) else {
            continue;
        };
        let before = mode_class(&single.selected_plan().execution);
        let after = mode_class(&batched.selected_plan().execution);
        match (before, after) {
            ("static", "persistent") => moved.push(*id),
            ("persistent", "static") => regressed.push(*id),
            _ => {}
        }
    }
    assert!(
        regressed.is_empty(),
        "Fix: a larger launch batch cannot make persistence less profitable: {}",
        regressed.join(", ")
    );
    assert!(
        !moved.is_empty(),
        "Fix: no registered operation reaches the persistent route from launch facts alone, so the route is not a compiler decision over {} operations",
        graphs.len()
    );
}

/// A registered fixture is built at a portable size, so the roster carries no
/// program that rendezvous across workgroups. The cooperative side is proved
/// from a library builder at a size whose semantics require grid-wide
/// ordering: the requirement is derived from the program, not declared here,
/// so a builder that stops needing the rendezvous stops proving this side and
/// turns the case red.
fn grid_ordered_graph() -> ProgramGraph {
    let program = vyre_libs::reduce::multi_block_prefix_scan::multi_block_prefix_scan_sum_u32(
        "input", "output", 1024,
    );
    let constraints = vyre_foundation::GeometryRequirements::from_program(&program)
        .expect("a multi-block scan must state compatible constraints");
    assert!(
        constraints.requires_cooperative_launch,
        "Fix: a multi-block prefix scan orders writes across workgroups, so its constraints must require a cooperative launch"
    );
    ProgramGraph::from_program("reduce.multi_block_prefix_scan.grid", program)
        .expect("a multi-block scan must build one graph")
}

#[test]
fn both_cooperative_and_non_cooperative_operations_cross_the_same_seam() {
    assert!(
        linked_registrations() > 0,
        "Fix: no library registration reached the link, so the registry roster is empty"
    );
    let mut cooperative = BTreeMap::new();
    let mut plain = BTreeMap::new();
    let mut conflicts = Vec::new();
    for entry in OperationRegistry::global().iter() {
        let Some(program) = entry.program() else {
            continue;
        };
        let Ok(graph) = ProgramGraph::from_program(entry.id, program) else {
            continue;
        };
        match entry.schedule_constraints() {
            Ok(constraints) if constraints.requires_cooperative_launch => {
                cooperative.insert(entry.id, graph);
            }
            Ok(_) => {
                plain.insert(entry.id, graph);
            }
            Err(error) => conflicts.push(format!("{}: {error}", entry.id)),
        }
    }
    assert!(
        conflicts.is_empty(),
        "Fix: every registration must expose one compatible schedule constraint decision:\n{}",
        conflicts.join("\n")
    );
    cooperative.insert("reduce.multi_block_prefix_scan.grid", grid_ordered_graph());
    assert!(
        !plain.is_empty(),
        "Fix: no registered operation runs without cooperative launch, so the sequential side of the seam is unproven"
    );

    let mut failures = Vec::new();
    for (id, graph) in cooperative.iter().chain(plain.iter()) {
        match compile_registered(graph, 1) {
            Ok(artifact) => {
                if artifact.geometry().is_empty() {
                    failures.push(format!("{id}: admitted no geometry"));
                }
            }
            Err(error) => failures.push(format!("{id}: {error}")),
        }
    }
    assert!(
        failures.is_empty(),
        "Fix: cooperative and non-cooperative operations must reach admitted geometry through the same compiler entry point:\n{}",
        failures.join("\n")
    );
}
