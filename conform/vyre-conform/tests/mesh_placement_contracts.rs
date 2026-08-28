//! Conformance of one coordinated topology across a device mesh.
//!
//! WHY: BACKLOG row 64 requires conformance to prove cross-device ordering,
//! numerical parity, deadlock freedom, and bounded progress for a placement that
//! spans devices. A placement is a schedule decision, so a defect here is a plan
//! that reorders combines across devices, waits forever, or computes something
//! other than what the program states while still validating as an artifact.
//!
//! What these cases do not prove: how fast a mesh runs, or that a specific
//! device carries a shard. Ranking is priced by the objective and placement is
//! proven against the mesh facts in `vyre-megakernel`.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::logical::LogicalExchangeKind;
use vyre_megakernel::mesh::{MeshTopologyPlan, TransferOrigin};
use vyre_reference::value::Value;

#[path = "../../../tests/support/artifact_fixtures.rs"]
mod artifact_fixtures;

use artifact_fixtures::{
    atomic_output_graph, chained_graph, compile_graph, compile_graph_on_mesh,
    compile_graph_on_mesh_for_memory, in_place_input_graph, two_device_mesh,
};

/// Every topology a mesh compile of the fixture graphs produces.
fn placements() -> Vec<(&'static str, MeshTopologyPlan)> {
    vec![
        (
            "spatial",
            compile_graph_on_mesh(in_place_input_graph(8), 0, two_device_mesh())
                .topology()
                .clone(),
        ),
        (
            "routed",
            compile_graph_on_mesh_for_memory(atomic_output_graph(8), 0, two_device_mesh())
                .topology()
                .clone(),
        ),
        (
            "pipeline",
            compile_graph_on_mesh_for_memory(chained_graph(), 0, two_device_mesh())
                .topology()
                .clone(),
        ),
    ]
}

/// WHY: a placement moves where work runs, never what it computes. An artifact
/// placed on a mesh that carried a different semantic identity would be ranked
/// against one program and submitted as another, and the reference oracle every
/// parity claim is anchored on would be judging the wrong one.
#[test]
fn a_mesh_placement_computes_what_the_reference_computes() {
    let graph = in_place_input_graph(8);
    let program = graph.nodes()[0].program.clone();
    let state = vec![0u8; 8 * 4];
    let outputs = vyre_reference::reference_eval(&program, &[Value::from(state.as_slice())])
        .expect("Fix: the reference oracle must execute the fixture program");
    let words = outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect::<Vec<_>>();
    assert_eq!(words, vec![1u32; 8]);

    let single = compile_graph(in_place_input_graph(8), 0);
    let placed = compile_graph_on_mesh(in_place_input_graph(8), 0, two_device_mesh());

    assert_eq!(
        single.provenance().semantic_graph,
        placed.provenance().semantic_graph,
        "Fix: cutting a program across devices must not change the semantics the \
         reference executes"
    );
    assert_ne!(
        single.digest(),
        placed.digest(),
        "Fix: the mesh a plan was placed on is part of what the artifact is"
    );
}

/// WHY: two collectives over the same devices in one stage have no stated order,
/// so two devices may enter them in opposite orders. Ordering across devices is
/// what the stage index states, and a topology that shared a stage between
/// intersecting collectives would state none.
#[test]
fn cross_device_transfers_state_one_order_per_stage() {
    for (name, topology) in placements() {
        let mut stages = topology
            .transfers
            .iter()
            .map(|transfer| transfer.stage)
            .collect::<Vec<_>>();
        let ascending = stages.windows(2).all(|pair| pair[0] <= pair[1]);
        assert!(
            ascending,
            "{name}: transfers are recorded out of stage order"
        );
        stages.dedup();
        assert_eq!(
            stages,
            (0..topology.stage_count()).collect::<Vec<_>>(),
            "{name}: stages are not contiguous from zero"
        );

        for stage in 0..topology.stage_count() {
            let mut participants = BTreeMap::<TransferOrigin, BTreeSet<_>>::new();
            for transfer in topology
                .transfers
                .iter()
                .filter(|transfer| transfer.stage == stage)
                .filter(|transfer| transfer.kind != LogicalExchangeKind::PointToPoint)
            {
                let entry = participants.entry(transfer.origin).or_default();
                entry.insert(transfer.from);
                entry.insert(transfer.to);
            }
            for (first, left) in &participants {
                for (second, right) in &participants {
                    assert!(
                        first == second || left.is_disjoint(right),
                        "{name}: {} and {} share a device in stage {stage}",
                        first.label(),
                        second.label()
                    );
                }
            }
        }
    }
}

/// WHY: a stage whose point-to-point transfers form a cycle is a mesh where
/// every device waits for the one behind it. The plan is the only thing a
/// submission reads, so the wait has to be excluded from the plan itself.
#[test]
fn no_stage_of_a_mesh_placement_waits_in_a_cycle() {
    for (name, topology) in placements() {
        for stage in 0..topology.stage_count() {
            let edges = topology
                .transfers
                .iter()
                .filter(|transfer| transfer.stage == stage)
                .filter(|transfer| transfer.kind == LogicalExchangeKind::PointToPoint)
                .map(|transfer| (transfer.from, transfer.to))
                .collect::<Vec<_>>();
            let mut reachable = edges.clone();
            // Transitive closure: a cycle is a device reachable from itself.
            loop {
                let mut grown = reachable.clone();
                for (from, mid) in &reachable {
                    for (start, to) in &reachable {
                        if start == mid && !grown.contains(&(*from, *to)) {
                            grown.push((*from, *to));
                        }
                    }
                }
                if grown.len() == reachable.len() {
                    break;
                }
                reachable = grown;
            }
            assert!(
                !reachable.iter().any(|(from, to)| from == to),
                "{name}: point-to-point transfers of stage {stage} wait in a cycle"
            );
        }
    }
}

/// WHY: progress is bounded only when the number of stages is finite, every
/// stage carries work, and every device a transfer names holds a shard. A stage
/// that carries nothing, or a transfer to a device with no work, is a submission
/// waiting on something no device will produce.
#[test]
fn a_mesh_placement_makes_bounded_progress() {
    for (name, topology) in placements() {
        topology
            .validate()
            .unwrap_or_else(|error| panic!("{name}: {}", error.diagnostic.message.as_ref()));
        let placed = topology.devices();
        assert!(
            !placed.is_empty(),
            "{name}: a placement that holds no device makes no progress"
        );
        for stage in 0..topology.stage_count() {
            assert!(
                topology
                    .transfers
                    .iter()
                    .any(|transfer| transfer.stage == stage),
                "{name}: stage {stage} carries no transfer"
            );
        }
        for transfer in &topology.transfers {
            assert!(
                placed.contains(&transfer.from) && placed.contains(&transfer.to),
                "{name}: a transfer names a device that holds no shard"
            );
            assert!(
                transfer.bytes > 0,
                "{name}: a transfer that moves no byte is recorded"
            );
        }
        assert!(
            topology.submission_devices().len() >= placed.len(),
            "{name}: a submission must bind every device the placement uses"
        );
    }
}
