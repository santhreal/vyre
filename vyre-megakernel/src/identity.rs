//! Stable content and record identities every canonical record refers to.

use serde::{Deserialize, Serialize};

/// Stable 256-bit content identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// Return the digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Canonical node identity inside an artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactNodeId(pub u32);

/// Canonical value identity inside an artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactValueId(pub u32);

/// Canonical fusion-group identity inside an artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FusionGroupId(pub u32);

/// Dependency endpoint with an explicit identity domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyEndpoint {
    /// An executable graph node.
    Node(ArtifactNodeId),
    /// A typed graph value materialized at a boundary.
    Value(ArtifactValueId),
}

/// Semantic reason that one artifact record depends on another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyKind {
    /// A produced value is consumed by another node.
    Data,
    /// A retained value is replaced by a type-preserving successor.
    Retained,
    /// A value must exist beyond its producing fusion group.
    Materialization,
}

/// One canonical typed dependency edge.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Edge source.
    pub from: DependencyEndpoint,
    /// Edge destination.
    pub to: DependencyEndpoint,
    /// Stable semantic edge kind.
    pub kind: DependencyKind,
    /// Connected value for data, retained, and materialization edges.
    pub value: Option<ArtifactValueId>,
}

pub(crate) fn domain_digest(domain: &[u8], bytes: &[u8]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Digest(*hasher.finalize().as_bytes())
}
