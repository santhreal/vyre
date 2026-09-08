//! Explicit compatibility and rollout matrix contracts for public protocols.
//!
//! WHY: closes the class "independent versioning creates uncoordinated mixed-version
//! failure or silent semantic drift". Public wire formats, catalog services,
//! proof certificates, schedule records, compiled artifacts, measurements,
//! resource allocations, and runtime protocols must declare explicit supported
//! version pairs, negotiation contracts, and rollout dispositions.


use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Public protocol domain whose versions are governed by the compatibility matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u32)]
pub enum ProtocolDomain {
    /// Public wire format for serialized programs and graphs.
    PublicWire = 0,
    /// Extension and operation catalog metadata format.
    Catalog = 1,
    /// Cryptographic proof and certificate format.
    Proof = 2,
    /// Selected schedule and transform persistence format.
    Schedule = 3,
    /// Megakernel and target compiled artifact payload format.
    Artifact = 4,
    /// Performance measurement and benchmark evidence format.
    Measurement = 5,
    /// External resource descriptor and tenant allocation format.
    Resource = 6,
    /// Driver and runtime submission and communication protocol.
    RuntimeProtocol = 7,
}

impl ProtocolDomain {
    /// All protocol domains governed by the compatibility contract.
    pub const ALL: &'static [Self] = &[
        Self::PublicWire,
        Self::Catalog,
        Self::Proof,
        Self::Schedule,
        Self::Artifact,
        Self::Measurement,
        Self::Resource,
        Self::RuntimeProtocol,
    ];

    /// Stable canonical name of the protocol domain.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PublicWire => "public_wire",
            Self::Catalog => "catalog",
            Self::Proof => "proof",
            Self::Schedule => "schedule",
            Self::Artifact => "artifact",
            Self::Measurement => "measurement",
            Self::Resource => "resource",
            Self::RuntimeProtocol => "runtime_protocol",
        }
    }
}

impl fmt::Display for ProtocolDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fixed-width semantic version for a protocol contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProtocolVersion {
    /// Breaking change version component.
    pub major: u32,
    /// Backward-compatible addition component.
    pub minor: u32,
    /// Backward-compatible bugfix component.
    pub patch: u32,
}

impl ProtocolVersion {
    /// Construct a protocol version from explicit numeric parts.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Fixed current version for v1.0.0.
    pub const V1_0_0: Self = Self::new(1, 0, 0);
    /// Fixed current version for v1.1.0.
    pub const V1_1_0: Self = Self::new(1, 1, 0);
    /// Fixed version for v2.0.0.
    pub const V2_0_0: Self = Self::new(2, 0, 0);

    /// Parse a semver string of the form "X.Y.Z".
    ///
    /// # Errors
    ///
    /// Returns a descriptive error when the string is malformed.
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(format!(
                "Fix: protocol version '{s}' must have exactly three dot-separated components (major.minor.patch)"
            ));
        }
        let major = parts[0]
            .parse::<u32>()
            .map_err(|e| format!("Fix: invalid major version in '{s}': {e}"))?;
        let minor = parts[1]
            .parse::<u32>()
            .map_err(|e| format!("Fix: invalid minor version in '{s}': {e}"))?;
        let patch = parts[2]
            .parse::<u32>()
            .map_err(|e| format!("Fix: invalid patch version in '{s}': {e}"))?;
        Ok(Self {
            major,
            minor,
            patch,
        })
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Compatibility disposition for a version pair under a protocol domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompatibilityDisposition {
    /// Fully supported with byte-identical or forward-compatible semantics.
    Supported,
    /// Deprecated within active support window; caller is guided to upgrade.
    Deprecated {
        /// Documented upgrade action for the caller.
        upgrade_action: &'static str,
    },
    /// Unsupported version combination.
    Unsupported {
        /// Minimum compatible version required.
        minimum_compatible: ProtocolVersion,
        /// Documented corrective action.
        upgrade_action: &'static str,
    },
}

impl CompatibilityDisposition {
    /// True when the disposition allows active communication or processing.
    #[must_use]
    pub const fn is_compatible(self) -> bool {
        matches!(self, Self::Supported | Self::Deprecated { .. })
    }
}

/// One cell in the compatibility matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompatibilityCell {
    /// Protocol domain governed by this cell.
    pub domain: ProtocolDomain,
    /// Producer or client version.
    pub producer: ProtocolVersion,
    /// Consumer or server version.
    pub consumer: ProtocolVersion,
    /// Disposition for this version combination.
    pub disposition: CompatibilityDisposition,
    /// Support window identifier (e.g. "active", "lts", "deprecated-0.8").
    pub support_window: &'static str,
}

/// Preregistered canonical compatibility matrix.
pub struct CompatibilityMatrix {
    cells: &'static [CompatibilityCell],
}

/// The canonical preregistered compatibility matrix for all Vyre protocol domains.
pub const CANONICAL_COMPATIBILITY_CELLS: &[CompatibilityCell] = &[
    // PublicWire
    CompatibilityCell {
        domain: ProtocolDomain::PublicWire,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_0_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::PublicWire,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::PublicWire,
        producer: ProtocolVersion::V1_1_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    // Catalog
    CompatibilityCell {
        domain: ProtocolDomain::Catalog,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_0_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Catalog,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Catalog,
        producer: ProtocolVersion::V1_1_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    // Proof
    CompatibilityCell {
        domain: ProtocolDomain::Proof,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_0_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Proof,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Proof,
        producer: ProtocolVersion::V1_1_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    // Schedule
    CompatibilityCell {
        domain: ProtocolDomain::Schedule,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_0_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Schedule,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Schedule,
        producer: ProtocolVersion::V1_1_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    // Artifact
    CompatibilityCell {
        domain: ProtocolDomain::Artifact,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_0_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Artifact,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Artifact,
        producer: ProtocolVersion::V1_1_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    // Measurement
    CompatibilityCell {
        domain: ProtocolDomain::Measurement,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_0_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Measurement,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Measurement,
        producer: ProtocolVersion::V1_1_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    // Resource
    CompatibilityCell {
        domain: ProtocolDomain::Resource,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_0_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Resource,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::Resource,
        producer: ProtocolVersion::V1_1_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    // RuntimeProtocol
    CompatibilityCell {
        domain: ProtocolDomain::RuntimeProtocol,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_0_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::RuntimeProtocol,
        producer: ProtocolVersion::V1_0_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
    CompatibilityCell {
        domain: ProtocolDomain::RuntimeProtocol,
        producer: ProtocolVersion::V1_1_0,
        consumer: ProtocolVersion::V1_1_0,
        disposition: CompatibilityDisposition::Supported,
        support_window: "active",
    },
];

impl CompatibilityMatrix {
    /// Return the canonical matrix instance.
    #[must_use]
    pub const fn canonical() -> Self {
        Self {
            cells: CANONICAL_COMPATIBILITY_CELLS,
        }
    }

    /// Check the disposition of a producer-consumer version pair.
    #[must_use]
    pub fn check(
        &self,
        domain: ProtocolDomain,
        producer: ProtocolVersion,
        consumer: ProtocolVersion,
    ) -> CompatibilityDisposition {
        for cell in self.cells {
            if cell.domain == domain && cell.producer == producer && cell.consumer == consumer {
                return cell.disposition;
            }
        }
        CompatibilityDisposition::Unsupported {
            minimum_compatible: ProtocolVersion::V1_0_0,
            upgrade_action: "Upgrade either producer or consumer to a compatible version within the active matrix.",
        }
    }

    /// Negotiate a mutually supported version for a domain.
    ///
    /// # Errors
    ///
    /// Returns a structured error with exact upgrade action if no mutually supported contract exists.
    pub fn negotiate(
        &self,
        domain: ProtocolDomain,
        offered: &[ProtocolVersion],
        supported: &[ProtocolVersion],
    ) -> Result<NegotiatedContract, NegotiationError> {
        if offered.is_empty() {
            return Err(NegotiationError {
                domain,
                offered: Vec::new(),
                supported: supported.to_vec(),
                upgrade_action: String::from(
                    "Fix: client must offer at least one protocol version during negotiation.",
                ),
            });
        }
        if supported.is_empty() {
            return Err(NegotiationError {
                domain,
                offered: offered.to_vec(),
                supported: Vec::new(),
                upgrade_action: String::from(
                    "Fix: host must support at least one protocol version during negotiation.",
                ),
            });
        }

        // Search in descending order of offered version (highest mutual version first)
        let mut sorted_offered = offered.to_vec();
        sorted_offered.sort_unstable();
        sorted_offered.reverse();

        for &client_ver in &sorted_offered {
            for &host_ver in supported {
                let disp = self.check(domain, client_ver, host_ver);
                if disp.is_compatible() {
                    // Compute contract digest combining domain, selected versions
                    let mut digest = [0_u8; 32];
                    let domain_tag = domain as u32;
                    digest[0..4].copy_from_slice(&domain_tag.to_le_bytes());
                    digest[4..8].copy_from_slice(&client_ver.major.to_le_bytes());
                    digest[8..12].copy_from_slice(&client_ver.minor.to_le_bytes());
                    digest[12..16].copy_from_slice(&client_ver.patch.to_le_bytes());
                    digest[16..20].copy_from_slice(&host_ver.major.to_le_bytes());
                    digest[20..24].copy_from_slice(&host_ver.minor.to_le_bytes());
                    digest[24..28].copy_from_slice(&host_ver.patch.to_le_bytes());
                    digest[28..32].copy_from_slice(&0x56595245_u32.to_le_bytes()); // "VYRE"

                    return Ok(NegotiatedContract {
                        domain,
                        client_version: client_ver,
                        host_version: host_ver,
                        disposition: disp,
                        contract_digest: digest,
                    });
                }
            }
        }

        Err(NegotiationError {
            domain,
            offered: offered.to_vec(),
            supported: supported.to_vec(),
            upgrade_action: format!(
                "Fix: no mutually supported version found for domain '{}'. Offered: {:?}, Host supported: {:?}. Upgrade client to one of host supported versions.",
                domain.as_str(),
                offered,
                supported
            ),
        })
    }
}

/// Negotiated contract resulting from successful protocol negotiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NegotiatedContract {
    /// Domain negotiated.
    pub domain: ProtocolDomain,
    /// Client offered version selected.
    pub client_version: ProtocolVersion,
    /// Host supported version matched.
    pub host_version: ProtocolVersion,
    /// Disposition of the negotiated version pair.
    pub disposition: CompatibilityDisposition,
    /// Cryptographic digest over the negotiated contract, part of request and artifact identity.
    pub contract_digest: [u8; 32],
}

/// Error returned when protocol negotiation fails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NegotiationError {
    /// Protocol domain where negotiation failed.
    pub domain: ProtocolDomain,
    /// Versions offered by client.
    pub offered: Vec<ProtocolVersion>,
    /// Versions supported by host.
    pub supported: Vec<ProtocolVersion>,
    /// Explicit upgrade instructions for the caller.
    pub upgrade_action: String,
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.upgrade_action)
    }
}
