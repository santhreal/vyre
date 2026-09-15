//! Authenticated facts about the device mesh a schedule may be placed on.
//!
//! A single device is one mesh of one device, so nothing here is a multi-device
//! special case. The facts are generic: axes with extents, devices at
//! coordinates with a memory capacity and a failure domain, links with a
//! bandwidth and a latency, and the exchange kinds the mesh can carry. No
//! vendor, interconnect product, or driver name appears.
//!
//! Every field is authenticated. The digest covers the canonical bytes of the
//! whole record, so a caller cannot present one mesh to selection and another to
//! submission, and an artifact compiled for one mesh never admits on another.

use serde::{Deserialize, Serialize};
use vyre_foundation::logical::LogicalExchangeKind;

use crate::allocation::DeviceSlot;
use crate::error::{failure, serialization_failure, CompileError, CompilerFailureKind};
use crate::identity::{domain_digest, Digest};

/// Schema version of the mesh facts a compile is selected against.
pub const MESH_FACTS_VERSION: u16 = 1;

/// Identity domain the mesh authentication digest is bound to.
const MESH_DOMAIN: &[u8] = b"vyre-mesh-facts-v1\0";

/// One named axis of the device mesh.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshAxis {
    /// Neutral axis name reported by the target.
    pub name: String,
    /// Number of coordinates along this axis.
    pub extent: u32,
}

/// One device of the mesh, at one coordinate.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshDevice {
    /// Slot the allocation plan and the artifact payloads address.
    pub slot: DeviceSlot,
    /// Coordinate of this device, one entry per mesh axis.
    pub coordinate: Vec<u32>,
    /// Bytes this device can hold, or zero when the target reports none.
    pub memory_capacity_bytes: u64,
    /// Devices sharing a failure domain fail together.
    pub failure_domain: u32,
}

/// One directed link between two devices of the mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshLink {
    /// Device the bytes leave.
    pub from: DeviceSlot,
    /// Device the bytes arrive at.
    pub to: DeviceSlot,
    /// Sustained bandwidth of this link.
    pub bandwidth_bytes_per_ns: u64,
    /// Fixed cost of one transfer over this link.
    pub latency_ns: u64,
}

/// Exchange kinds the mesh can carry.
///
/// One field per [`LogicalExchangeKind`], read through an exhaustive match, so a
/// new exchange kind stops this crate compiling until the mesh states whether it
/// carries one.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(deny_unknown_fields)]
pub struct CollectiveSupport {
    /// Every participant contributes and receives the combined value.
    pub all_reduce: bool,
    /// Every participant receives every shard.
    pub all_gather: bool,
    /// Every participant receives the combined value of one shard.
    pub reduce_scatter: bool,
    /// One participant sends a value to every other.
    pub broadcast: bool,
    /// One participant sends a value to one other.
    pub point_to_point: bool,
}

impl CollectiveSupport {
    /// A mesh that carries no exchange.
    pub const NONE: Self = Self {
        all_reduce: false,
        all_gather: false,
        reduce_scatter: false,
        broadcast: false,
        point_to_point: false,
    };

    /// A mesh that carries every exchange kind.
    pub const ALL: Self = Self {
        all_reduce: true,
        all_gather: true,
        reduce_scatter: true,
        broadcast: true,
        point_to_point: true,
    };

    /// Whether the mesh carries one exchange kind.
    #[must_use]
    pub const fn carries(&self, kind: LogicalExchangeKind) -> bool {
        match kind {
            LogicalExchangeKind::AllReduce => self.all_reduce,
            LogicalExchangeKind::AllGather => self.all_gather,
            LogicalExchangeKind::ReduceScatter => self.reduce_scatter,
            LogicalExchangeKind::Broadcast => self.broadcast,
            LogicalExchangeKind::PointToPoint => self.point_to_point,
        }
    }
}

#[derive(Serialize)]
struct MeshBody<'a> {
    version: u16,
    axes: &'a [MeshAxis],
    devices: &'a [MeshDevice],
    links: &'a [MeshLink],
    collectives: CollectiveSupport,
}

/// Authenticated description of the mesh a compile may place work on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshFacts {
    version: u16,
    axes: Vec<MeshAxis>,
    devices: Vec<MeshDevice>,
    links: Vec<MeshLink>,
    collectives: CollectiveSupport,
    authentication: Digest,
}

impl MeshFacts {
    /// Validate and authenticate one mesh description.
    ///
    /// # Errors
    ///
    /// Returns when an axis has no extent, a device coordinate does not match the
    /// axes, two devices share a slot or a coordinate, a link names a device the
    /// mesh does not contain, a link connects a device to itself, a link reports
    /// no bandwidth, or two links share endpoints.
    pub fn new(
        axes: Vec<MeshAxis>,
        mut devices: Vec<MeshDevice>,
        mut links: Vec<MeshLink>,
        collectives: CollectiveSupport,
    ) -> Result<Self, CompileError> {
        if axes.is_empty() {
            return Err(invalid(
                "mesh.axes",
                "a mesh states no axis",
                "report at least one mesh axis, using one axis of extent one for a single device",
            ));
        }
        for (index, axis) in axes.iter().enumerate() {
            if axis.name.is_empty() {
                return Err(invalid(
                    format!("mesh.axes[{index}].name"),
                    "a mesh axis has no name",
                    "report the neutral name the target uses for this axis",
                ));
            }
            if axis.extent == 0 {
                return Err(invalid(
                    format!("mesh.axes[{index}].extent"),
                    "a mesh axis has extent zero",
                    "report a positive extent for every mesh axis",
                ));
            }
        }
        if devices.is_empty() {
            return Err(invalid(
                "mesh.devices",
                "a mesh states no device",
                "report every device the target authenticates",
            ));
        }
        devices.sort_by(|left, right| left.slot.cmp(&right.slot));
        for index in 1..devices.len() {
            if devices[index].slot == devices[index - 1].slot {
                return Err(invalid(
                    format!("mesh.devices[{index}].slot"),
                    format!("two mesh devices share slot {}", devices[index].slot.0),
                    "report each device once under its own slot",
                ));
            }
        }
        let mut coordinates = Vec::with_capacity(devices.len());
        for (index, device) in devices.iter().enumerate() {
            if device.coordinate.len() != axes.len() {
                return Err(invalid(
                    format!("mesh.devices[{index}].coordinate"),
                    format!(
                        "device coordinate has {} axes while the mesh states {}",
                        device.coordinate.len(),
                        axes.len()
                    ),
                    "report one coordinate entry for every mesh axis",
                ));
            }
            for (axis, position) in device.coordinate.iter().enumerate() {
                if *position >= axes[axis].extent {
                    return Err(invalid(
                        format!("mesh.devices[{index}].coordinate[{axis}]"),
                        format!(
                            "coordinate {position} is outside axis `{}` of extent {}",
                            axes[axis].name, axes[axis].extent
                        ),
                        "report a coordinate inside the extent of every mesh axis",
                    ));
                }
            }
            coordinates.push(device.coordinate.clone());
        }
        let unique = coordinates
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        if unique.len() != coordinates.len() {
            return Err(invalid(
                "mesh.devices",
                "two mesh devices share one coordinate",
                "report one device per mesh coordinate",
            ));
        }
        links.sort();
        for (index, link) in links.iter().enumerate() {
            let path = format!("mesh.links[{index}]");
            if link.from == link.to {
                return Err(invalid(
                    format!("{path}.to"),
                    format!("a mesh link connects device {} to itself", link.from.0),
                    "report links between distinct devices",
                ));
            }
            for (field, slot) in [("from", link.from), ("to", link.to)] {
                if !devices.iter().any(|device| device.slot == slot) {
                    return Err(invalid(
                        format!("{path}.{field}"),
                        format!("a mesh link names device {} which the mesh omits", slot.0),
                        "report every device the mesh links before its links",
                    ));
                }
            }
            if link.bandwidth_bytes_per_ns == 0 {
                return Err(invalid(
                    format!("{path}.bandwidth_bytes_per_ns"),
                    "a mesh link reports no bandwidth",
                    "report the sustained bandwidth of every link, or omit the link",
                ));
            }
            if index > 0 && links[index - 1].from == link.from && links[index - 1].to == link.to {
                return Err(invalid(
                    format!("{path}.to"),
                    format!(
                        "two mesh links connect device {} to device {}",
                        link.from.0, link.to.0
                    ),
                    "report one link per ordered device pair",
                ));
            }
        }
        let authentication =
            authenticate(MESH_FACTS_VERSION, &axes, &devices, &links, collectives)?;
        Ok(Self {
            version: MESH_FACTS_VERSION,
            axes,
            devices,
            links,
            collectives,
            authentication,
        })
    }

    /// The mesh of one device, holding `memory_capacity_bytes`.
    ///
    /// A target that authenticates one device is not a special case: it is a
    /// one-axis mesh of extent one with no links, so every placement rule below
    /// applies to it unchanged.
    ///
    /// # Errors
    ///
    /// Returns when the one-device mesh cannot be authenticated.
    pub fn single_device(memory_capacity_bytes: u64) -> Result<Self, CompileError> {
        Self::new(
            vec![MeshAxis {
                name: "device".to_owned(),
                extent: 1,
            }],
            vec![MeshDevice {
                slot: DeviceSlot(0),
                coordinate: vec![0],
                memory_capacity_bytes,
                failure_domain: 0,
            }],
            Vec::new(),
            CollectiveSupport::NONE,
        )
    }

    /// Mesh facts schema version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Mesh axes in report order.
    #[must_use]
    pub fn axes(&self) -> &[MeshAxis] {
        &self.axes
    }

    /// Mesh devices in slot order.
    #[must_use]
    pub fn devices(&self) -> &[MeshDevice] {
        &self.devices
    }

    /// Mesh links in endpoint order.
    #[must_use]
    pub fn links(&self) -> &[MeshLink] {
        &self.links
    }

    /// Exchange kinds this mesh carries.
    #[must_use]
    pub const fn collectives(&self) -> CollectiveSupport {
        self.collectives
    }

    /// Identity covering every mesh fact.
    #[must_use]
    pub const fn authentication(&self) -> Digest {
        self.authentication
    }

    /// The device occupying one slot.
    #[must_use]
    pub fn device(&self, slot: DeviceSlot) -> Option<&MeshDevice> {
        self.devices.iter().find(|device| device.slot == slot)
    }

    /// The link connecting two devices, with its index.
    #[must_use]
    pub fn link(&self, from: DeviceSlot, to: DeviceSlot) -> Option<(usize, &MeshLink)> {
        self.links
            .iter()
            .enumerate()
            .find(|(_, link)| link.from == from && link.to == to)
    }

    /// Reject facts whose identity does not cover their content.
    ///
    /// A mesh arrives from a driver as data, and a spoofed record that keeps a
    /// digest from a mesh with more devices or faster links would be selected
    /// against and then submitted somewhere else.
    ///
    /// # Errors
    ///
    /// Returns when the version is not the current schema or the recomputed
    /// identity differs from the recorded one.
    pub fn authenticate(&self) -> Result<(), CompileError> {
        if self.version != MESH_FACTS_VERSION {
            return Err(failure(
                CompilerFailureKind::VersionSkew,
                "mesh.version",
                format!(
                    "mesh facts schema {} is unsupported; expected {MESH_FACTS_VERSION}",
                    self.version
                ),
                "report mesh facts under the schema this compiler states",
            ));
        }
        let expected = authenticate(
            self.version,
            &self.axes,
            &self.devices,
            &self.links,
            self.collectives,
        )?;
        if expected != self.authentication {
            return Err(invalid(
                "mesh.authentication",
                "mesh identity does not cover the mesh facts it accompanies",
                "report mesh facts through MeshFacts::new so their identity covers them",
            ));
        }
        Ok(())
    }
}

fn authenticate(
    version: u16,
    axes: &[MeshAxis],
    devices: &[MeshDevice],
    links: &[MeshLink],
    collectives: CollectiveSupport,
) -> Result<Digest, CompileError> {
    let body = serde_json::to_vec(&MeshBody {
        version,
        axes,
        devices,
        links,
        collectives,
    })
    .map_err(serialization_failure)?;
    Ok(domain_digest(MESH_DOMAIN, &body))
}

fn invalid(
    path: impl Into<String>,
    message: impl Into<String>,
    fix: impl Into<String>,
) -> CompileError {
    failure(CompilerFailureKind::InvalidMeshFacts, path, message, fix)
}
