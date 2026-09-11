//! Storage succession across a retained chain.
//!
//! WHY: a retained successor advances the storage of the value it replaces, so
//! the two hold the same bytes. Placing them in two regions severs the
//! succession, and the reader of the successor sees whatever the second
//! allocation happened to contain instead of what the predecessor published.
//! The class is every retained chain, not the one a whole-grid fence publishes:
//! a chain reaches the packer the same way whether a fission, a stream, or a
//! multi-submission graph produced it. These cases therefore drive the packer
//! directly with chains of every shape it admits, and the ranking peak is
//! checked against the assembled plan for each of them, because a figure the
//! ranking priced and the plan does not hold is a refused compile.
//!
//! What these do not catch: whether the graph names a chain correctly. A node
//! that publishes a successor without recording its predecessor arrives here as
//! two unrelated values and is placed as two, which is the right placement for
//! the facts it states.

use super::pack::ValueFact;
use super::{
    owned_by_artifact, peak, plan, AllocationPlan, DeviceSlot, PlacementLayout, RegionOwner,
    ValueLiveness,
};
use crate::identity::{ArtifactNodeId, ArtifactValueId, FusionGroupId};
use crate::mesh::{MeshTopologyPlan, PartitionKind, RegionPartition, ShardAssignment};
use crate::schema::ResourceLifetime;
use crate::{DeviceFacts, Digest};
use vyre_foundation::validate::BackendCapabilities;

/// One value of a test graph, in the shape both the ranking and the packer read.
#[derive(Clone, Copy)]
struct Value {
    id: u32,
    bytes: u64,
    lifetime: ResourceLifetime,
    producer: Option<u32>,
    consumer: Option<u32>,
    retained_predecessor: Option<u32>,
}

impl Value {
    const fn caller(id: u32, bytes: u64, consumer: u32) -> Self {
        Self {
            id,
            bytes,
            lifetime: ResourceLifetime::Invocation,
            producer: None,
            consumer: Some(consumer),
            retained_predecessor: None,
        }
    }

    const fn produced(id: u32, bytes: u64, producer: u32, lifetime: ResourceLifetime) -> Self {
        Self {
            id,
            bytes,
            lifetime,
            producer: Some(producer),
            consumer: None,
            retained_predecessor: None,
        }
    }

    const fn read_by(mut self, consumer: u32) -> Self {
        self.consumer = Some(consumer);
        self
    }

    const fn advancing(mut self, predecessor: u32) -> Self {
        self.retained_predecessor = Some(predecessor);
        self
    }
}

/// Stage of every node in the fixtures: one node per stage, in node order.
fn node_groups(nodes: u32) -> (Vec<FusionGroupId>, Vec<u32>) {
    (
        (0..nodes).map(FusionGroupId).collect(),
        (0..nodes).collect(),
    )
}

fn facts(values: &[Value], nodes: u32) -> Vec<ValueFact> {
    let (groups, stages) = node_groups(nodes);
    let final_stage = stages.iter().copied().max().unwrap_or(0);
    values
        .iter()
        .map(|value| {
            let consumers: Vec<ArtifactNodeId> =
                value.consumer.map(ArtifactNodeId).into_iter().collect();
            let (first_stage, last_stage) = super::span(
                value.producer.map(ArtifactNodeId),
                &consumers,
                matches!(
                    value.lifetime,
                    ResourceLifetime::Output | ResourceLifetime::Retained
                ),
                &groups,
                &stages,
                final_stage,
            );
            ValueFact {
                value: ArtifactValueId(value.id),
                producer: value.producer.map(ArtifactNodeId),
                bytes: value.bytes,
                element_bytes: 4,
                lifetime: value.lifetime,
                retained_predecessor: value.retained_predecessor.map(ArtifactValueId),
                first_stage,
                last_stage,
                produced: value.producer.is_some(),
                consumer_count: u32::from(value.consumer.is_some()),
                synchronized: false,
                in_place: false,
                layout: PlacementLayout {
                    element_bytes: 4,
                    storage_order: vec![0],
                    strides: vec![1],
                    contiguous: true,
                },
            }
        })
        .collect()
}

fn liveness(values: &[Value]) -> Vec<ValueLiveness> {
    values
        .iter()
        .map(|value| ValueLiveness {
            value: ArtifactValueId(value.id),
            bytes: value.bytes,
            producer: value.producer.map(ArtifactNodeId),
            consumers: value.consumer.map(ArtifactNodeId).into_iter().collect(),
            survives_to_end: matches!(
                value.lifetime,
                ResourceLifetime::Output | ResourceLifetime::Retained
            ),
            lifetime: value.lifetime,
            retained_predecessor: value.retained_predecessor.map(ArtifactValueId),
        })
        .collect()
}

fn topology(nodes: u32) -> MeshTopologyPlan {
    MeshTopologyPlan::single_device(
        Digest([0; 32]),
        DeviceSlot(0),
        (0..nodes)
            .map(|node| RegionPartition {
                node: ArtifactNodeId(node),
                kind: PartitionKind::Replicated,
                axis: None,
                region_points: 1,
                shards: vec![ShardAssignment {
                    shard: 0,
                    device: DeviceSlot(0),
                    coordinate: vec![0],
                    points: 1,
                }],
            })
            .collect(),
    )
}

fn planned(values: &[Value], nodes: u32) -> AllocationPlan {
    plan(
        &facts(values, nodes),
        DeviceFacts::new(BackendCapabilities::default(), 256),
        &topology(nodes),
    )
    .expect("the fixture plan is valid")
}

/// Region index holding `value`, over artifact-owned and caller regions alike.
fn region_of(plan: &AllocationPlan, value: u32) -> usize {
    plan.regions
        .iter()
        .position(|region| {
            region
                .placements
                .iter()
                .any(|placement| placement.value == ArtifactValueId(value))
        })
        .unwrap_or_else(|| panic!("value {value} is placed in no region"))
}

/// WHY: the succession itself. Every artifact-owned successor must hold the
/// bytes its predecessor holds, whatever the chain's length or the relative
/// size of its links, because the segment after a kernel cut reads the
/// successor and the segment before it wrote the predecessor.
#[test]
fn an_owned_retained_successor_holds_its_predecessors_bytes() {
    let cases: [(&str, Vec<Value>, u32, &[(u32, u32)]); 4] = [
        (
            "one successor",
            vec![
                Value::caller(0, 4096, 0),
                Value::produced(1, 512, 0, ResourceLifetime::Retained).read_by(1),
                Value::produced(2, 512, 1, ResourceLifetime::Retained).advancing(1),
            ],
            2,
            &[(1, 2)],
        ),
        (
            "three links",
            vec![
                Value::caller(0, 4096, 0),
                Value::produced(1, 512, 0, ResourceLifetime::Retained).read_by(1),
                Value::produced(2, 512, 1, ResourceLifetime::Retained)
                    .advancing(1)
                    .read_by(2),
                Value::produced(3, 512, 2, ResourceLifetime::Retained).advancing(2),
            ],
            3,
            &[(1, 2), (2, 3)],
        ),
        (
            "the successor is wider than the value it advances",
            vec![
                Value::caller(0, 4096, 0),
                Value::produced(1, 512, 0, ResourceLifetime::Retained).read_by(1),
                Value::produced(2, 8192, 1, ResourceLifetime::Retained).advancing(1),
            ],
            2,
            &[(1, 2)],
        ),
        (
            "a stream advances retained state",
            vec![
                Value::caller(0, 4096, 0),
                Value::produced(1, 256, 0, ResourceLifetime::Stream).read_by(1),
                Value::produced(2, 256, 1, ResourceLifetime::Stream).advancing(1),
            ],
            2,
            &[(1, 2)],
        ),
    ];

    for (name, values, nodes, chain) in cases {
        let plan = planned(&values, nodes);
        for (predecessor, successor) in chain {
            assert_eq!(
                region_of(&plan, *predecessor),
                region_of(&plan, *successor),
                "{name}: value {successor} advances value {predecessor} and holds other bytes"
            );
        }
        let widest = values
            .iter()
            .filter(|value| {
                chain
                    .iter()
                    .any(|(prior, next)| value.id == *prior || value.id == *next)
            })
            .map(|value| value.bytes)
            .max()
            .unwrap_or_default();
        let region = &plan.regions[region_of(&plan, chain[0].0)];
        assert_eq!(
            region.bytes, widest,
            "{name}: the shared region reserves {} bytes for a chain whose widest link holds {widest}",
            region.bytes
        );
    }
}

/// WHY: a chain is only one storage while the artifact owns both ends. A
/// caller-bound output is the caller's buffer, and folding it into the
/// artifact's region would bind the kernel to storage the caller never
/// supplied. An unproduced initial value is likewise the caller's.
#[test]
fn a_chain_that_leaves_artifact_storage_is_placed_apart() {
    let cases: [(&str, Vec<Value>, u32, u32, u32); 2] = [
        (
            "the successor is a caller output",
            vec![
                Value::caller(0, 4096, 0),
                Value::produced(1, 4, 0, ResourceLifetime::Retained).read_by(1),
                Value::produced(2, 4, 1, ResourceLifetime::Output).advancing(1),
            ],
            2,
            1,
            2,
        ),
        (
            "the predecessor is bound by the caller",
            vec![
                Value::caller(0, 4096, 0),
                Value::caller(1, 4, 0),
                Value::produced(2, 4, 0, ResourceLifetime::Retained).advancing(1),
            ],
            1,
            1,
            2,
        ),
    ];

    for (name, values, nodes, predecessor, successor) in cases {
        let plan = planned(&values, nodes);
        assert_ne!(
            region_of(&plan, predecessor),
            region_of(&plan, successor),
            "{name}: value {successor} took over storage the artifact does not own on both ends"
        );
    }
}

/// WHY: the compiler refuses an artifact whose assembled plan holds a different
/// byte total than the ranking that selected it priced. A chain charged twice
/// by one of them and once by the other is exactly that refusal, and it reaches
/// the caller as a failed compile of a correct program.
#[test]
fn the_ranked_peak_is_the_peak_the_assembled_plan_holds() {
    let cases: [(&str, Vec<Value>, u32); 5] = [
        (
            "one successor",
            vec![
                Value::caller(0, 4096, 0),
                Value::produced(1, 512, 0, ResourceLifetime::Retained).read_by(1),
                Value::produced(2, 512, 1, ResourceLifetime::Retained).advancing(1),
            ],
            2,
        ),
        (
            "three links",
            vec![
                Value::caller(0, 4096, 0),
                Value::produced(1, 512, 0, ResourceLifetime::Retained).read_by(1),
                Value::produced(2, 512, 1, ResourceLifetime::Retained)
                    .advancing(1)
                    .read_by(2),
                Value::produced(3, 512, 2, ResourceLifetime::Retained).advancing(2),
            ],
            3,
        ),
        (
            "the chain ends in a caller output",
            vec![
                Value::caller(0, 4096, 0),
                Value::produced(1, 4, 0, ResourceLifetime::Retained).read_by(1),
                Value::produced(2, 4, 1, ResourceLifetime::Output).advancing(1),
            ],
            2,
        ),
        (
            "the chain holds the widest storage in the graph",
            vec![
                Value::caller(0, 4, 0),
                Value::produced(1, 8192, 0, ResourceLifetime::Retained).read_by(1),
                Value::produced(2, 8192, 1, ResourceLifetime::Retained).advancing(1),
            ],
            2,
        ),
        (
            "no chain at all",
            vec![
                Value::caller(0, 4096, 0),
                Value::produced(1, 512, 0, ResourceLifetime::Invocation).read_by(1),
                Value::produced(2, 64, 1, ResourceLifetime::Output),
            ],
            2,
        ),
    ];

    for (name, values, nodes) in cases {
        let (groups, stages) = node_groups(nodes);
        let ranked = peak(&liveness(&values), &groups, &stages);
        let plan = planned(&values, nodes);
        assert_eq!(
            plan.aggregate_peak_bytes, ranked,
            "{name}: ranking priced {ranked} bytes and the assembled plan holds {}",
            plan.aggregate_peak_bytes
        );
    }
}

/// WHY: an artifact-owned region is one device buffer, and the runtime
/// allocates exactly the regions the plan records. A device that reserves fewer
/// bytes than its own placements hold is refused by the plan's own validation,
/// so the shared region must still cover the widest link of the chain.
#[test]
fn a_shared_region_reserves_at_least_every_value_it_holds() {
    let values = vec![
        Value::caller(0, 4096, 0),
        Value::produced(1, 512, 0, ResourceLifetime::Retained).read_by(1),
        Value::produced(2, 8192, 1, ResourceLifetime::Retained).advancing(1),
    ];
    let plan = planned(&values, 2);
    for region in plan.owned() {
        for placement in &region.placements {
            assert!(
                placement.byte_offset + placement.bytes <= region.bytes,
                "value {} runs past the {} bytes its region reserves",
                placement.value.0,
                region.bytes
            );
        }
    }
    let owned: u64 = plan
        .regions
        .iter()
        .filter(|region| region.owner == RegionOwner::Artifact)
        .map(|region| region.bytes)
        .sum();
    assert_eq!(
        owned, 8192,
        "the chain holds one region as wide as its widest link"
    );
}

/// WHY: the ownership rule decides whether the packer reserves bytes for a
/// value or states the caller's. Ranking and packing read it from one place, so
/// a lifetime added to the schema cannot be artifact storage to one of them and
/// caller storage to the other.
#[test]
fn ownership_follows_production_and_lifetime() {
    let cases = [
        (ResourceLifetime::Invocation, true),
        (ResourceLifetime::Retained, true),
        (ResourceLifetime::Stream, true),
        (ResourceLifetime::Constant, false),
        (ResourceLifetime::Output, false),
    ];
    for (lifetime, owned) in cases {
        assert_eq!(
            owned_by_artifact(true, lifetime),
            owned,
            "a produced {lifetime:?} value"
        );
        assert!(
            !owned_by_artifact(false, lifetime),
            "a {lifetime:?} value no node writes is bound by the caller"
        );
    }
}
