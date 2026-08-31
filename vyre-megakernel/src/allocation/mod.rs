//! One allocation and layout plan for every value a selected schedule executes.
//!
//! Before this existed, four modules answered "what bytes does this program
//! need" and none of them answered it the same way: a Program-level foundation
//! buffer enumeration, the artifact resource envelope's stage-liveness peak, the
//! cost model's workgroup scratch plus fusion-crossing bytes, and a workspace
//! packer that bump-allocated one offset per produced value and never reused a
//! dead range. The figure the objective ranked and the figure the artifact
//! recorded were different numbers about the same program.
//!
//! Schedule selection owns this plan. It maps every graph value to a physical
//! region using the stage spans the candidate's grouping resolves, the alias and
//! effect facts the logical stage closed, and the alignment the target states.
//! Lowering verifies each access against it, the runtime allocates it exactly,
//! and nothing downstream re-derives a byte count.

mod derive;
mod liveness;
mod pack;

use serde::{Deserialize, Serialize};

use crate::error::{failure, overflow, CompileError, CompilerFailureKind};
use crate::identity::ArtifactValueId;
use crate::schema::ResourceLifetime;

pub(crate) use derive::value_facts;
pub(crate) use liveness::{peak, span, ValueLiveness};
pub(crate) use pack::plan;

/// Schema version of the allocation plan carried inside one artifact.
///
/// Version 1 is the first plan that states offsets, alignment, address space,
/// device placement, alias classes and permitted storage operations. A stored
/// plan of another version is refused by version rather than reinterpreted.
pub const ALLOCATION_SCHEMA_VERSION: u16 = 1;

/// Byte alignment every region offset is a multiple of.
///
/// A backend binds a region at its recorded offset, and a binding offset below
/// the strictest alignment any current API requires would make an offset legal
/// in the artifact and unbindable on the device.
pub const REGION_ALIGNMENT: u64 = 256;

/// Memory space a region is addressed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddressSpace {
    /// Read-write device storage.
    Device,
    /// Immutable storage a caller fills once and no entry writes.
    Constant,
}

/// One authenticated device a plan places regions on.
///
/// A target that authenticates one device places every region on slot zero. The
/// slot is recorded rather than assumed so a plan states which device holds each
/// region instead of leaving it to whoever submits the artifact.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct DeviceSlot(pub u16);

/// Set of values that may name the same storage.
///
/// Two values in one class are one logical storage advanced across submissions,
/// so the plan gives them one region and states the in-place permit. Two values
/// in different classes may share a region only when their stage spans are
/// disjoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AliasClass(pub u32);

/// Who allocates the storage backing a region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionOwner {
    /// The artifact produces the value for itself and the runtime allocates it.
    Artifact,
    /// The caller binds the buffer; the artifact never allocates it.
    Caller,
}

/// How logical indices of one value map onto its storage.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlacementLayout {
    /// Packed bytes of one element.
    pub element_bytes: u32,
    /// Logical axes in increasing physical storage order.
    pub storage_order: Vec<u32>,
    /// Element strides in logical-axis order.
    pub strides: Vec<u64>,
    /// Whether every axis is densely row-major.
    pub contiguous: bool,
}

/// Storage operations the plan permits for one placement.
///
/// A permit is a decision the schedule already made, not a hint. Nothing
/// downstream performs one the plan did not state, and the runtime never
/// discovers one for itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlacementPermits {
    /// Value whose dead storage this placement takes over.
    pub reuses: Option<ArtifactValueId>,
    /// Whether one entry reads and writes this storage through a single
    /// read-write binding, so the value is advanced in place.
    pub in_place: bool,
    /// Whether recomputing the value is permitted instead of holding it.
    pub rematerialize: bool,
    /// Whether the value is larger than the device cache serves across its live
    /// range, so a spill to memory between stages is permitted.
    pub spill: bool,
    /// Whether the value may be moved toward the device before its first
    /// consuming stage.
    pub prefetch: bool,
}

/// One value's exact placement inside a region.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValuePlacement {
    /// Canonical value identity.
    pub value: ArtifactValueId,
    /// Byte offset of the value inside its region.
    pub byte_offset: u64,
    /// Packed bytes the value occupies.
    pub bytes: u64,
    /// Semantic lifetime class.
    pub lifetime: ResourceLifetime,
    /// Storage the value may be named by.
    pub alias_class: AliasClass,
    /// First dependency stage that reads or writes the value.
    pub first_stage: u32,
    /// Last dependency stage that reads or writes the value.
    pub last_stage: u32,
    /// Whether an ordering, collective or atomic effect touches the value.
    pub synchronized: bool,
    /// Index mapping of the stored value.
    pub layout: PlacementLayout,
    /// Storage operations the plan permits here.
    pub permits: PlacementPermits,
}

impl ValuePlacement {
    /// Exclusive byte end of the placement inside its region.
    ///
    /// # Errors
    ///
    /// Returns an overflow rejection when the end exceeds the addressable range.
    pub fn end(&self) -> Result<u64, CompileError> {
        self.byte_offset.checked_add(self.bytes).ok_or_else(|| {
            overflow(
                format!("artifact.allocation.values[{}]", self.value.0),
                "placement end exceeds the addressable range",
            )
        })
    }
}

/// One physical region of one device, holding every value placed in it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationRegion {
    /// Device holding the region.
    pub device: DeviceSlot,
    /// Space the region is addressed in.
    pub address_space: AddressSpace,
    /// Who allocates the backing storage.
    pub owner: RegionOwner,
    /// Byte offset of the region inside its device allocation.
    pub offset: u64,
    /// Bytes the region reserves.
    pub bytes: u64,
    /// Alignment the offset satisfies.
    pub alignment: u64,
    /// Bytes reserved past the last placement to reach the next aligned offset.
    pub padding_bytes: u64,
    /// Values placed in the region, ascending by first stage.
    pub placements: Vec<ValuePlacement>,
}

impl AllocationRegion {
    /// Last stage any placement in the region is live at.
    #[must_use]
    pub fn last_stage(&self) -> u32 {
        self.placements
            .iter()
            .map(|placement| placement.last_stage)
            .max()
            .unwrap_or(0)
    }

    /// Exclusive byte end of the region inside its device allocation.
    ///
    /// # Errors
    ///
    /// Returns an overflow rejection when the end exceeds the addressable range.
    pub fn end(&self) -> Result<u64, CompileError> {
        self.offset.checked_add(self.bytes).ok_or_else(|| {
            overflow(
                "artifact.allocation.regions",
                "region end exceeds the addressable range",
            )
        })
    }
}

/// Bytes one device holds under this plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DevicePeak {
    /// Device the figures belong to.
    pub device: DeviceSlot,
    /// Largest byte total simultaneously live in any one stage, over every
    /// placement the device holds.
    ///
    /// This is what the device holds at its busiest stage, so it is the figure
    /// candidate ranking prices and a hard memory bound is checked against.
    pub peak_bytes: u64,
    /// Bytes the artifact-owned regions of this device reserve.
    ///
    /// The runtime allocates exactly this. It is never below the liveness peak of
    /// the artifact-owned placements; the difference is what this packer left
    /// unused.
    pub allocated_bytes: u64,
}

/// Every physical allocation and layout decision one selected schedule made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationPlan {
    /// Schema version of the plan.
    pub schema_version: u16,
    /// Regions ascending by device, address space, owner and offset.
    pub regions: Vec<AllocationRegion>,
    /// One entry per device the plan places a region on, ascending by device.
    pub device_peaks: Vec<DevicePeak>,
    /// Sum of every device peak.
    pub aggregate_peak_bytes: u64,
}

impl Default for AllocationPlan {
    fn default() -> Self {
        Self::empty()
    }
}

impl AllocationPlan {
    /// A plan that places nothing.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            schema_version: ALLOCATION_SCHEMA_VERSION,
            regions: Vec::new(),
            device_peaks: Vec::new(),
            aggregate_peak_bytes: 0,
        }
    }

    /// Region and placement holding `value`, when the plan places it.
    #[must_use]
    pub fn placement(
        &self,
        value: ArtifactValueId,
    ) -> Option<(&AllocationRegion, &ValuePlacement)> {
        self.regions.iter().find_map(|region| {
            region
                .placements
                .iter()
                .find(|placement| placement.value == value)
                .map(|placement| (region, placement))
        })
    }

    /// Regions the runtime allocates.
    pub fn owned(&self) -> impl Iterator<Item = &AllocationRegion> {
        self.regions
            .iter()
            .filter(|region| region.owner == RegionOwner::Artifact)
    }

    /// Bytes the runtime allocates across every artifact-owned region.
    #[must_use]
    pub fn owned_bytes(&self) -> u64 {
        self.owned()
            .fold(0u64, |total, region| total.saturating_add(region.bytes))
    }

    /// Reject a plan no runtime could allocate and bind exactly.
    ///
    /// Every rule here is a statement the plan makes about itself, so a tampered
    /// field is refused by the plan rather than by whichever consumer noticed
    /// first.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] when the schema version is not the one this
    /// compiler writes, regions are unordered or overlap, a placement runs past
    /// its region, two placements of different alias classes hold the same bytes
    /// while both live, a permit names a value the plan does not place, a
    /// caller-owned region takes over storage it does not own, a reuse permit
    /// names a value that is not placed beside it or is still live, a device
    /// peak has no region or a region has no device peak, a recorded figure is
    /// not the one the placements hold, or the aggregate peak is not the sum of
    /// the device peaks.
    pub fn validate(&self) -> Result<(), CompileError> {
        if self.schema_version != ALLOCATION_SCHEMA_VERSION {
            return Err(failure(
                CompilerFailureKind::VersionSkew,
                "artifact.allocation.schema_version",
                format!(
                    "allocation plan states schema {} and this compiler writes {ALLOCATION_SCHEMA_VERSION}",
                    self.schema_version
                ),
                "recompile the source graph with this compiler",
            ));
        }
        self.validate_regions()?;
        self.validate_permits()?;
        self.validate_peaks()
    }

    fn validate_regions(&self) -> Result<(), CompileError> {
        let mut previous: Option<(DeviceSlot, AddressSpace, RegionOwner, u64)> = None;
        for (index, region) in self.regions.iter().enumerate() {
            let path = format!("artifact.allocation.regions[{index}]");
            if region.placements.is_empty() {
                return Err(invalid(
                    format!("{path}.placements"),
                    "region places no value",
                    "record one region per placed value set",
                ));
            }
            if region.bytes == 0 {
                return Err(invalid(
                    format!("{path}.bytes"),
                    "region reserves no byte",
                    "record the packed byte count the placements require",
                ));
            }
            if region.alignment == 0 || !region.alignment.is_power_of_two() {
                return Err(invalid(
                    format!("{path}.alignment"),
                    "region alignment is not a power of two",
                    "record the alignment the target states",
                ));
            }
            if region.offset % region.alignment != 0 {
                return Err(invalid(
                    format!("{path}.offset"),
                    "region offset is not a multiple of its alignment",
                    "align every region offset to its recorded alignment",
                ));
            }
            if region.owner == RegionOwner::Caller {
                if region.offset != 0 {
                    return Err(invalid(
                        format!("{path}.offset"),
                        "caller-bound storage states an offset inside another allocation",
                        "record caller-bound values at offset zero of their own buffer",
                    ));
                }
                if region.placements.len() != 1 {
                    return Err(invalid(
                        format!("{path}.placements"),
                        "caller-bound storage holds more than one value",
                        "record one caller-bound value per buffer the caller binds",
                    ));
                }
            }
            if region.address_space == AddressSpace::Constant
                && region.owner == RegionOwner::Artifact
            {
                return Err(invalid(
                    format!("{path}.owner"),
                    "the artifact allocates storage in the constant space",
                    "bind constant values the caller supplies, never allocate them",
                ));
            }
            let ordering = (region.device, region.address_space, region.owner);
            if let Some((device, space, owner, end)) = previous {
                if ordering < (device, space, owner) {
                    return Err(invalid(
                        format!("{path}.device"),
                        "regions are not ordered by device, address space and owner",
                        "record regions ascending by device, address space, owner and offset",
                    ));
                }
                if ordering == (device, space, owner)
                    && region.owner == RegionOwner::Artifact
                    && region.offset < end
                {
                    return Err(invalid(
                        format!("{path}.offset"),
                        "region overlaps the region before it",
                        "assign one disjoint offset per region",
                    ));
                }
            }
            previous = Some((
                region.device,
                region.address_space,
                region.owner,
                region.end()?,
            ));
            self.validate_placements(region, &path)?;
        }
        Ok(())
    }

    fn validate_placements(
        &self,
        region: &AllocationRegion,
        path: &str,
    ) -> Result<(), CompileError> {
        for (index, placement) in region.placements.iter().enumerate() {
            let placement_path = format!("{path}.placements[{index}]");
            if placement.bytes == 0 {
                return Err(invalid(
                    format!("{placement_path}.bytes"),
                    "placement reserves no byte",
                    "record the packed byte count of the value",
                ));
            }
            if (region.address_space == AddressSpace::Constant)
                != (placement.lifetime == ResourceLifetime::Constant)
            {
                return Err(invalid(
                    format!("{placement_path}.lifetime"),
                    "placement lifetime disagrees with the space that addresses it",
                    "address constant values in the constant space and every other value in the device space",
                ));
            }
            if placement.last_stage < placement.first_stage {
                return Err(invalid(
                    format!("{placement_path}.last_stage"),
                    "placement dies before it is produced",
                    "record the stage span the selected schedule resolved",
                ));
            }
            if placement.end()? > region.bytes {
                return Err(invalid(
                    format!("{placement_path}.bytes"),
                    "placement runs past its region",
                    "reserve the bytes the placements require",
                ));
            }
            if index > 0 && placement.first_stage < region.placements[index - 1].first_stage {
                return Err(invalid(
                    format!("{placement_path}.first_stage"),
                    "placements are not ordered by first stage",
                    "record placements ascending by first stage",
                ));
            }
            for other in region.placements.iter().skip(index + 1) {
                let same_bytes =
                    placement.byte_offset < other.end()? && other.byte_offset < placement.end()?;
                let both_live = placement.first_stage <= other.last_stage
                    && other.first_stage <= placement.last_stage;
                if same_bytes && both_live && placement.alias_class != other.alias_class {
                    return Err(invalid(
                        format!("{placement_path}.byte_offset"),
                        format!(
                            "value {} and value {} hold the same bytes while both are live",
                            placement.value.0, other.value.0
                        ),
                        "place values of different alias classes in disjoint bytes or disjoint stages",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_permits(&self) -> Result<(), CompileError> {
        for (index, region) in self.regions.iter().enumerate() {
            for (position, placement) in region.placements.iter().enumerate() {
                let path =
                    format!("artifact.allocation.regions[{index}].placements[{position}].permits");
                let Some(named) = placement.permits.reuses else {
                    continue;
                };
                if region.owner == RegionOwner::Caller {
                    return Err(invalid(
                        format!("{path}.reuses"),
                        "caller-bound storage takes over storage the artifact does not own",
                        "state a reuse permit on an artifact-owned region",
                    ));
                }
                if named == placement.value {
                    return Err(invalid(
                        format!("{path}.reuses"),
                        "reuse permit names the value holding it",
                        "name the value whose dead storage this placement takes over",
                    ));
                }
                let Some(prior) = region.placements.iter().find(|held| held.value == named) else {
                    return Err(invalid(
                        format!("{path}.reuses"),
                        format!(
                            "reuse permit names value {} which is not placed in this region",
                            named.0
                        ),
                        "take over storage only inside the region that holds it",
                    ));
                };
                if prior.last_stage >= placement.first_stage {
                    return Err(invalid(
                        format!("{path}.reuses"),
                        format!(
                            "value {} takes over storage value {} is still live in",
                            placement.value.0, named.0
                        ),
                        "take over storage only after its prior value's last stage",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_peaks(&self) -> Result<(), CompileError> {
        let mut aggregate = 0u64;
        let mut previous: Option<DeviceSlot> = None;
        for (index, peak) in self.device_peaks.iter().enumerate() {
            let path = format!("artifact.allocation.device_peaks[{index}]");
            if previous.is_some_and(|last| peak.device <= last) {
                return Err(invalid(
                    format!("{path}.device"),
                    "device peaks are not ascending and unique",
                    "record one ascending peak entry per placed device",
                ));
            }
            previous = Some(peak.device);
            let allocated = self
                .regions
                .iter()
                .filter(|region| {
                    region.device == peak.device && region.owner == RegionOwner::Artifact
                })
                .try_fold(0u64, |total, region| {
                    total.checked_add(region.bytes).ok_or_else(|| {
                        overflow(format!("{path}.allocated_bytes"), "region sum exceeds u64")
                    })
                })?;
            if allocated != peak.allocated_bytes {
                return Err(invalid(
                    format!("{path}.allocated_bytes"),
                    format!(
                        "device peak records {} allocated bytes and its regions reserve {allocated}",
                        peak.allocated_bytes
                    ),
                    "record the bytes the artifact-owned regions reserve",
                ));
            }
            let owned_peak = self.live_peak(peak.device, Some(RegionOwner::Artifact))?;
            if peak.allocated_bytes < owned_peak {
                return Err(invalid(
                    format!("{path}.allocated_bytes"),
                    format!(
                        "device allocates {} bytes and its own placements need {owned_peak}",
                        peak.allocated_bytes
                    ),
                    "reserve at least the peak the artifact-owned placements require",
                ));
            }
            let live = self.live_peak(peak.device, None)?;
            if live != peak.peak_bytes {
                return Err(invalid(
                    format!("{path}.peak_bytes"),
                    format!(
                        "device peak records {} live bytes and its placements hold {live}",
                        peak.peak_bytes
                    ),
                    "record the exact liveness peak of the placed values",
                ));
            }
            aggregate = aggregate.checked_add(peak.peak_bytes).ok_or_else(|| {
                overflow(
                    "artifact.allocation.aggregate_peak_bytes",
                    "device peak sum exceeds u64",
                )
            })?;
        }
        for (index, region) in self.regions.iter().enumerate() {
            if !self
                .device_peaks
                .iter()
                .any(|peak| peak.device == region.device)
            {
                return Err(invalid(
                    format!("artifact.allocation.regions[{index}].device"),
                    format!(
                        "device {} holds a region and no peak entry",
                        region.device.0
                    ),
                    "record one peak entry per device the plan places a region on",
                ));
            }
        }
        if aggregate != self.aggregate_peak_bytes {
            return Err(invalid(
                "artifact.allocation.aggregate_peak_bytes",
                format!(
                    "plan records {} aggregate bytes and its devices hold {aggregate}",
                    self.aggregate_peak_bytes
                ),
                "record the sum of the device peaks",
            ));
        }
        Ok(())
    }

    /// Largest byte total live in any one stage on `device`, over the placements
    /// of `owner` or over every placement when no owner is stated.
    fn live_peak(
        &self,
        device: DeviceSlot,
        owner: Option<RegionOwner>,
    ) -> Result<u64, CompileError> {
        let placements: Vec<&ValuePlacement> = self
            .regions
            .iter()
            .filter(|region| {
                region.device == device && owner.is_none_or(|owner| region.owner == owner)
            })
            .flat_map(|region| region.placements.iter())
            .collect();
        let final_stage = placements
            .iter()
            .map(|placement| placement.last_stage)
            .max()
            .unwrap_or(0);
        let mut peak = 0u64;
        for stage in 0..=final_stage {
            let live = placements
                .iter()
                .filter(|placement| placement.first_stage <= stage && stage <= placement.last_stage)
                .try_fold(0u64, |total, placement| {
                    total.checked_add(placement.bytes).ok_or_else(|| {
                        overflow(
                            "artifact.allocation.device_peaks",
                            "live placement sum exceeds u64",
                        )
                    })
                })?;
            peak = peak.max(live);
        }
        Ok(peak)
    }
}

fn invalid(
    path: impl Into<String>,
    message: impl Into<String>,
    fix: impl Into<String>,
) -> CompileError {
    failure(
        CompilerFailureKind::InvalidAllocationPlan,
        path,
        message,
        fix,
    )
}
