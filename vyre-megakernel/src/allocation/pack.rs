//! Placing every value one selected candidate executes.
//!
//! The stage span of every value is resolved once, by the record builder that
//! also reports it in the artifact, so the liveness this packer reuses storage
//! against and the liveness the artifact states are the same spans. Ranking and
//! assembly both call [`plan`], so the peak the objective ordered and the peak
//! the artifact records are one number.

use super::{
    AddressSpace, AliasClass, AllocationPlan, AllocationRegion, DevicePeak, DeviceSlot,
    PlacementLayout, PlacementPermits, RegionOwner, ValuePlacement, ALLOCATION_SCHEMA_VERSION,
    REGION_ALIGNMENT,
};
use crate::error::{overflow, CompileError};
use crate::identity::ArtifactValueId;
use crate::schema::ResourceLifetime;
use crate::DeviceFacts;

/// Everything the planner reads about one graph value.
///
/// The stage span is the one the resource records report, and the alias, effect
/// and layout facts are the ones the logical stage closed. The planner derives
/// no fact of its own, so a placement can be checked against the artifact's own
/// resource rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ValueFact {
    /// Canonical value identity.
    pub(crate) value: ArtifactValueId,
    /// Packed bytes the value occupies.
    pub(crate) bytes: u64,
    /// Packed bytes of one element.
    pub(crate) element_bytes: u32,
    /// Semantic lifetime class.
    pub(crate) lifetime: ResourceLifetime,
    /// Prior retained value whose storage this value advances.
    pub(crate) retained_predecessor: Option<ArtifactValueId>,
    /// First dependency stage that reads or writes the value.
    pub(crate) first_stage: u32,
    /// Last dependency stage that reads or writes the value.
    pub(crate) last_stage: u32,
    /// Whether a node of the graph writes the value.
    pub(crate) produced: bool,
    /// Nodes that read the value.
    pub(crate) consumer_count: u32,
    /// Whether an ordering, collective or atomic effect touches the value.
    pub(crate) synchronized: bool,
    /// Whether one entry reads and writes the value through a single read-write
    /// binding.
    pub(crate) in_place: bool,
    /// Index mapping of the stored value.
    pub(crate) layout: PlacementLayout,
}

impl ValueFact {
    /// Whether the artifact produces the storage and the runtime allocates it.
    ///
    /// A caller binds every other value, so the plan states its bytes and layout
    /// and reserves nothing for it.
    fn owned_by_artifact(&self) -> bool {
        self.produced
            && matches!(
                self.lifetime,
                ResourceLifetime::Invocation | ResourceLifetime::Retained
            )
    }
}

/// One region under construction, before offsets are assigned.
#[derive(Debug)]
struct Slot {
    bytes: u64,
    last_stage: u32,
    placements: Vec<ValuePlacement>,
}

/// Build the one allocation and layout plan for `values`.
///
/// # Errors
///
/// Returns [`CompileError`] when a byte total overflows the addressable range or
/// the assembled plan does not satisfy [`AllocationPlan::validate`].
pub(crate) fn plan(
    values: &[ValueFact],
    device: DeviceFacts,
) -> Result<AllocationPlan, CompileError> {
    let slot = DeviceSlot(0);
    let classes = alias_classes(values);
    let cache_capacity = device.cache_capacity_bytes();

    let mut order: Vec<usize> = (0..values.len())
        .filter(|index| values[*index].bytes > 0)
        .collect();
    order.sort_by_key(|index| (values[*index].first_stage, values[*index].value));

    let mut slots: Vec<Slot> = Vec::new();
    let mut caller: Vec<AllocationRegion> = Vec::new();
    for index in order {
        let fact = &values[index];
        let mut placement = ValuePlacement {
            value: fact.value,
            byte_offset: 0,
            bytes: fact.bytes,
            lifetime: fact.lifetime,
            alias_class: classes[index],
            first_stage: fact.first_stage,
            last_stage: fact.last_stage,
            synchronized: fact.synchronized,
            layout: fact.layout.clone(),
            permits: PlacementPermits {
                reuses: None,
                in_place: fact.in_place,
                rematerialize: fact.consumer_count <= 1
                    && !fact.synchronized
                    && fact.lifetime == ResourceLifetime::Invocation,
                spill: cache_capacity > 0 && fact.bytes > cache_capacity,
                prefetch: !fact.produced && fact.consumer_count > 0,
            },
        };
        if !fact.owned_by_artifact() {
            caller.push(AllocationRegion {
                device: slot,
                address_space: if fact.lifetime == ResourceLifetime::Constant {
                    AddressSpace::Constant
                } else {
                    AddressSpace::Device
                },
                owner: RegionOwner::Caller,
                offset: 0,
                bytes: fact.bytes,
                alignment: REGION_ALIGNMENT,
                padding_bytes: 0,
                placements: vec![placement],
            });
            continue;
        }
        let reused = slots.iter().position(|held| {
            held.last_stage < fact.first_stage
                && held.bytes >= fact.bytes
                && held
                    .placements
                    .iter()
                    .all(|prior| prior.alias_class != classes[index])
        });
        if let Some(existing) = reused {
            placement.permits.reuses = slots[existing].placements.last().map(|prior| prior.value);
            push(&mut slots[existing], placement);
            continue;
        }
        slots.push(Slot {
            bytes: fact.bytes,
            last_stage: fact.last_stage,
            placements: vec![placement],
        });
    }

    let mut regions = Vec::with_capacity(slots.len() + caller.len());
    let mut next = 0u64;
    for held in slots {
        let offset = align(next)?;
        let end = offset.checked_add(held.bytes).ok_or_else(|| {
            overflow(
                "planner.allocation.regions",
                "region end exceeds the addressable range",
            )
        })?;
        let padding = align(end)?.saturating_sub(end);
        regions.push(AllocationRegion {
            device: slot,
            address_space: AddressSpace::Device,
            owner: RegionOwner::Artifact,
            offset,
            bytes: held.bytes,
            alignment: REGION_ALIGNMENT,
            padding_bytes: padding,
            placements: held.placements,
        });
        next = end;
    }
    regions.extend(caller);
    regions.sort_by_key(|region| {
        (
            region.device,
            region.address_space,
            region.owner,
            region.offset,
            region.placements.first().map_or(0, |first| first.value.0),
        )
    });

    let mut plan = AllocationPlan {
        schema_version: ALLOCATION_SCHEMA_VERSION,
        regions,
        device_peaks: Vec::new(),
        aggregate_peak_bytes: 0,
    };
    let peak = plan.live_peak(slot, None)?;
    plan.device_peaks = vec![DevicePeak {
        device: slot,
        peak_bytes: peak,
        allocated_bytes: plan.owned_bytes(),
    }];
    plan.aggregate_peak_bytes = peak;
    plan.validate()?;
    Ok(plan)
}

fn push(slot: &mut Slot, placement: ValuePlacement) {
    slot.bytes = slot.bytes.max(placement.bytes);
    slot.last_stage = slot.last_stage.max(placement.last_stage);
    slot.placements.push(placement);
}

fn align(offset: u64) -> Result<u64, CompileError> {
    offset
        .checked_next_multiple_of(REGION_ALIGNMENT)
        .ok_or_else(|| {
            overflow(
                "planner.allocation.offset",
                "region offset exceeds the addressable range",
            )
        })
}

/// One class per retained chain, identified by the first value in the chain.
///
/// A retained successor advances the storage of the value it replaces, so the two
/// are one logical storage and never share a region with an unrelated value while
/// either is live. Every other value is its own class.
fn alias_classes(values: &[ValueFact]) -> Vec<AliasClass> {
    let mut root: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for value in values {
        let predecessor = value.retained_predecessor.map(|prior| prior.0);
        let class = predecessor
            .and_then(|prior| root.get(&prior).copied())
            .or(predecessor)
            .unwrap_or(value.value.0);
        root.insert(value.value.0, class);
    }
    values
        .iter()
        .map(|value| AliasClass(root.get(&value.value.0).copied().unwrap_or(value.value.0)))
        .collect()
}
