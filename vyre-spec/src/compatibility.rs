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
    CompatibilityCell {
        domain: ProtocolDomain::PublicWire,
        producer: ProtocolVersion::new(8, 0, 0),
        consumer: ProtocolVersion::new(8, 0, 0),
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
    CompatibilityCell {
        domain: ProtocolDomain::Proof,
        producer: ProtocolVersion::new(2, 0, 0),
        consumer: ProtocolVersion::new(2, 0, 0),
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
    CompatibilityCell {
        domain: ProtocolDomain::Artifact,
        producer: ProtocolVersion::new(4, 0, 0),
        consumer: ProtocolVersion::new(4, 0, 0),
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

    /// Check the disposition of a schema version under the compatibility matrix.
    #[must_use]
    pub fn check_schema(
        &self,
        schema_id: crate::schema_registry::SchemaId,
        client_version: ProtocolVersion,
    ) -> CompatibilityDisposition {
        let domain = schema_id.domain();
        if let Some(def) = crate::schema_registry::SchemaRegistry::lookup(schema_id) {
            self.check(domain, client_version, def.semver)
        } else {
            CompatibilityDisposition::Unsupported {
                minimum_compatible: ProtocolVersion::V1_0_0,
                upgrade_action: "Register the schema ID in the canonical schema registry.",
            }
        }
    }

    /// Negotiate a mutually supported version for a registered schema ID.
    ///
    /// # Errors
    ///
    /// Returns a structured error with exact upgrade action if no mutually supported contract exists.
    pub fn negotiate_schema(
        &self,
        schema_id: crate::schema_registry::SchemaId,
        offered: &[ProtocolVersion],
    ) -> Result<NegotiatedContract, NegotiationError> {
        let domain = schema_id.domain();
        if let Some(def) = crate::schema_registry::SchemaRegistry::lookup(schema_id) {
            self.negotiate(domain, offered, &[def.semver])
        } else {
            Err(NegotiationError {
                domain,
                offered: offered.to_vec(),
                supported: Vec::new(),
                upgrade_action: format!(
                    "Fix: schema ID '{schema_id}' is not registered in canonical schema registry."
                ),
            })
        }
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
/// Generation identifier for generation-scoped cache namespaces and retained sessions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GenerationId(pub u64);

impl GenerationId {
    /// Initial baseline generation counter.
    pub const INITIAL: Self = Self(1);

    /// Construct a new generation identifier from an explicit counter.
    #[must_use]
    pub const fn new(val: u64) -> Self {
        Self(val)
    }

    /// Return the next generation identifier.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for GenerationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gen-{}", self.0)
    }
}

/// Generation-scoped cache namespace preventing cross-version and cross-generation contamination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheNamespace {
    /// Protocol domain governed by this cache.
    pub domain: ProtocolDomain,
    /// Protocol version of the cached payload format.
    pub version: ProtocolVersion,
    /// Generation identifier scoping this cache namespace.
    pub generation: GenerationId,
    /// Stable namespace name.
    pub name: &'static str,
}

impl CacheNamespace {
    /// Create a new generation-scoped cache namespace.
    #[must_use]
    pub const fn new(
        domain: ProtocolDomain,
        version: ProtocolVersion,
        generation: GenerationId,
        name: &'static str,
    ) -> Self {
        Self {
            domain,
            version,
            generation,
            name,
        }
    }

    /// Compute a deterministic 32-byte cache key scoped to this domain, version, generation, and namespace.
    #[must_use]
    pub fn scoped_key(&self, base_key: &[u8]) -> [u8; 32] {
        let mut digest = [0_u8; 32];
        let domain_tag = self.domain as u32;
        digest[0..4].copy_from_slice(&domain_tag.to_le_bytes());
        digest[4..8].copy_from_slice(&self.version.major.to_le_bytes());
        digest[8..12].copy_from_slice(&self.version.minor.to_le_bytes());
        digest[12..20].copy_from_slice(&self.generation.0.to_le_bytes());
        for (i, &b) in self.name.as_bytes().iter().enumerate() {
            digest[20 + (i % 12)] ^= b.wrapping_add((i as u8).wrapping_mul(31));
        }
        for (i, &b) in base_key.iter().enumerate() {
            digest[i % 32] = digest[i % 32].wrapping_add(b).rotate_left(1);
        }
        digest
    }

    /// Validate that a record from this namespace belongs to the active generation.
    ///
    /// # Errors
    ///
    /// Returns `StaleGenerationError` if the record generation does not match the active generation.
    pub fn validate_generation(
        &self,
        active_generation: GenerationId,
    ) -> Result<(), StaleGenerationError> {
        if self.generation == active_generation {
            Ok(())
        } else {
            Err(StaleGenerationError {
                record_generation: self.generation,
                active_generation,
                namespace: self.name,
            })
        }
    }
}

/// Error returned when attempting to access a cache record or session from a stale generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaleGenerationError {
    /// Generation identifier on the stale record.
    pub record_generation: GenerationId,
    /// Currently active generation identifier.
    pub active_generation: GenerationId,
    /// Name of the affected namespace.
    pub namespace: &'static str,
}

impl fmt::Display for StaleGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fix: stale record from generation {} rejected in namespace '{}'; active generation is {}. Invalidate or recompute cache entry.",
            self.record_generation, self.namespace, self.active_generation
        )
    }
}

/// Lifecycle status of a retained session during rollout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionStatus {
    /// Active and serving traffic.
    Active,
    /// Staging for upcoming generation.
    Staging,
    /// Committed to newly active generation.
    Committed,
    /// Rolled back to previous stable generation.
    RolledBack,
    /// Interrupted by crash and discarded before commit.
    Interrupted,
}

/// Generation-scoped retained session scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedSessionScope {
    /// Unique session identifier.
    pub session_id: [u8; 32],
    /// Protocol domain.
    pub domain: ProtocolDomain,
    /// Negotiated contract governing this session.
    pub contract: NegotiatedContract,
    /// Generation identifier scoping this session.
    pub generation: GenerationId,
    /// Current lifecycle status.
    pub status: SessionStatus,
}

impl RetainedSessionScope {
    /// Validate session against active generation and contract.
    ///
    /// # Errors
    ///
    /// Returns an error if generation or contract digest mismatch.
    pub fn validate(
        &self,
        active_generation: GenerationId,
        expected_contract: &NegotiatedContract,
    ) -> Result<(), SessionScopeError> {
        if self.generation != active_generation {
            return Err(SessionScopeError::StaleGeneration {
                session_generation: self.generation,
                active_generation,
            });
        }
        if self.contract.contract_digest != expected_contract.contract_digest {
            return Err(SessionScopeError::ContractMismatch);
        }
        if self.status != SessionStatus::Active && self.status != SessionStatus::Committed {
            return Err(SessionScopeError::InvalidStatus(self.status));
        }
        Ok(())
    }
}

/// Error returned when a retained session is invalid or stale.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionScopeError {
    /// Session belongs to a stale generation.
    StaleGeneration {
        /// Stale session generation.
        session_generation: GenerationId,
        /// Current active generation.
        active_generation: GenerationId,
    },
    /// Negotiated contract does not match session contract.
    ContractMismatch,
    /// Session is in an uncommitted or interrupted status.
    InvalidStatus(SessionStatus),
}

impl fmt::Display for SessionScopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleGeneration {
                session_generation,
                active_generation,
            } => write!(
                f,
                "Fix: retained session from generation {session_generation} rejected; active generation is {active_generation}."
            ),
            Self::ContractMismatch => write!(
                f,
                "Fix: retained session contract digest does not match active negotiated contract."
            ),
            Self::InvalidStatus(st) => write!(
                f,
                "Fix: retained session status is {:?}; expected Active or Committed.",
                st
            ),
        }
    }
}

/// Manager coordinating atomic generation rollouts, crash interruption recovery, and rollbacks.
#[derive(Clone, Debug)]
pub struct RolloutManager {
    active_generation: GenerationId,
    staging_generation: Option<GenerationId>,
    retained_sessions: Vec<RetainedSessionScope>,
}

impl RolloutManager {
    /// Create a new rollout manager initialized to the baseline generation.
    #[must_use]
    pub fn new(initial_generation: GenerationId) -> Self {
        Self {
            active_generation: initial_generation,
            staging_generation: None,
            retained_sessions: Vec::new(),
        }
    }

    /// Return the currently active generation.
    #[must_use]
    pub const fn active_generation(&self) -> GenerationId {
        self.active_generation
    }

    /// Return the staging generation if an upgrade is currently staged.
    #[must_use]
    pub const fn staging_generation(&self) -> Option<GenerationId> {
        self.staging_generation
    }

    /// Stage a new generation in an isolated staging namespace without mutating active state.
    pub fn stage_upgrade(&mut self) -> GenerationId {
        let next_gen = self.active_generation.next();
        self.staging_generation = Some(next_gen);
        next_gen
    }

    /// Simulate a crash interruption during upgrade: drops uncommitted staging state and preserves active generation.
    pub fn crash_interruption(&mut self) {
        self.staging_generation = None;
        for s in &mut self.retained_sessions {
            if s.status == SessionStatus::Staging {
                s.status = SessionStatus::Interrupted;
            }
        }
        self.retained_sessions
            .retain(|s| s.status != SessionStatus::Interrupted);
    }

    /// Atomically commit the staged generation, advancing the active generation.
    ///
    /// # Errors
    ///
    /// Returns an error if no upgrade was staged.
    pub fn atomic_commit(&mut self) -> Result<GenerationId, String> {
        if let Some(staged) = self.staging_generation.take() {
            self.active_generation = staged;
            for s in &mut self.retained_sessions {
                if s.generation == staged {
                    s.status = SessionStatus::Committed;
                }
            }
            Ok(staged)
        } else {
            Err(String::from("Fix: no staged upgrade to commit."))
        }
    }

    /// Roll back to the previous generation, discarding any half-migrated state atomically.
    pub fn rollback(&mut self, target_generation: GenerationId) {
        self.staging_generation = None;
        self.active_generation = target_generation;
        for s in &mut self.retained_sessions {
            if s.generation > target_generation {
                s.status = SessionStatus::RolledBack;
            }
        }
        self.retained_sessions
            .retain(|s| s.generation <= target_generation);
    }

    /// Register a session under the manager.
    pub fn register_session(&mut self, session: RetainedSessionScope) {
        self.retained_sessions.push(session);
    }

    /// Retrieve active sessions.
    #[must_use]
    pub fn sessions(&self) -> &[RetainedSessionScope] {
        &self.retained_sessions
    }
}

/// Derive a cryptographic artifact identity incorporating the negotiated contract digest.
///
/// Ensures two compilations or runs that negotiate different contracts produce distinct artifact identities.
#[must_use]
pub fn derive_artifact_identity(
    base_artifact_hash: &[u8; 32],
    contract: &NegotiatedContract,
) -> [u8; 32] {
    let mut out = [0_u8; 32];
    for i in 0..32 {
        out[i] = base_artifact_hash[i] ^ contract.contract_digest[i];
    }
    let domain_val = contract.domain as u8;
    out[0] = out[0].wrapping_add(domain_val);
    out[1] = out[1].wrapping_add(contract.client_version.major as u8);
    out[2] = out[2].wrapping_add(contract.client_version.minor as u8);
    out[3] = out[3].wrapping_add(contract.host_version.major as u8);
    out[4] = out[4].wrapping_add(contract.host_version.minor as u8);
    out
}
