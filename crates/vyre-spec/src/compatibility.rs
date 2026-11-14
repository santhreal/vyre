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

crate::semver_triple! {
    /// Fixed-width semantic version for a protocol contract.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct ProtocolVersion;
}

impl ProtocolVersion {
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
