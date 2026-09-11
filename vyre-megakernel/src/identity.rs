//! Stable content and record identities every canonical record refers to.

use std::fmt;

use serde::de::{Unexpected, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Stable 256-bit content identity.
///
/// Serialized as 64 lowercase hex characters rather than as a byte array, so a
/// canonical record's byte length is a function of its structure and its stated
/// numbers alone. An artifact-byte ceiling is checked against the canonical
/// bytes of the artifact that states it, and a length that moved with hash
/// content made that check depend on the hash rather than on the program.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// Return the digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// The canonical 64-character lowercase hex form.
    #[must_use]
    pub fn to_hex(self) -> String {
        let mut hex = String::with_capacity(64);
        for byte in self.0 {
            hex.push(nibble(byte >> 4));
            hex.push(nibble(byte & 0x0f));
        }
        hex
    }
}

const fn nibble(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        _ => (b'a' + value - 10) as char,
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Digest({})", self.to_hex())
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(HexVisitor)
    }
}

struct HexVisitor;

impl Visitor<'_> for HexVisitor {
    type Value = Digest;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("64 lowercase hex characters")
    }

    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Digest, E> {
        let text = value.as_bytes();
        if text.len() != 64 {
            return Err(E::invalid_length(text.len(), &self));
        }
        let mut bytes = [0_u8; 32];
        for (byte, pair) in bytes.iter_mut().zip(text.chunks_exact(2)) {
            let high =
                value_of(pair[0]).ok_or_else(|| E::invalid_value(Unexpected::Str(value), &self))?;
            let low =
                value_of(pair[1]).ok_or_else(|| E::invalid_value(Unexpected::Str(value), &self))?;
            *byte = (high << 4) | low;
        }
        Ok(Digest(bytes))
    }
}

const fn value_of(character: u8) -> Option<u8> {
    match character {
        b'0'..=b'9' => Some(character - b'0'),
        b'a'..=b'f' => Some(character - b'a' + 10),
        _ => None,
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
