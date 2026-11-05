//! Stable typed identifiers for compiler substrate data.
//!
//! Strongly-typed identifiers replace raw integer and string indices across all
//! five compiler levels, enabling cache key derivation and dependency tracking.

use core::fmt;
use serde::{Deserialize, Serialize};

/// Canonical identifier for an interned string or symbol.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct InternedStringId(pub u32);

impl fmt::Display for InternedStringId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "str#{}", self.0)
    }
}

/// Canonical identifier for an interned data or tensor type.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct InternedTypeId(pub u32);

impl fmt::Display for InternedTypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type#{}", self.0)
    }
}

/// Canonical identifier for an interned constant value or literal blob.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct InternedConstId(pub u32);

impl fmt::Display for InternedConstId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "const#{}", self.0)
    }
}

/// Canonical identifier for an interned tensor or buffer layout.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct InternedLayoutId(pub u32);

impl fmt::Display for InternedLayoutId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "layout#{}", self.0)
    }
}

/// Canonical identifier for a slot in the hash-consed node arena.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct InternedNodeId(pub u32);

impl fmt::Display for InternedNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node#{}", self.0)
    }
}

/// Canonical identifier for a slot in the hash-consed logical region arena.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct InternedRegionId(pub u32);

impl fmt::Display for InternedRegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "region#{}", self.0)
    }
}

/// Canonical identifier for a schedule node in selected schedule IR.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct ScheduleNodeId(pub u32);

impl fmt::Display for ScheduleNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sched#{}", self.0)
    }
}

/// Canonical identifier for a lowered physical kernel descriptor.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct PhysicalKernelId(pub u32);

impl fmt::Display for PhysicalKernelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pkernel#{}", self.0)
    }
}

/// Canonical identifier for a compiled target artifact.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct ArtifactId(pub u32);

impl fmt::Display for ArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "art#{}", self.0)
    }
}

/// Monotonic revision sequence for compiler query tracking and invalidation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct Revision(pub u64);

impl Revision {
    /// Initial zero revision.
    #[must_use]
    pub const fn zero() -> Self {
        Self(0)
    }

    /// Advance to next sequential revision.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Numeric revision value.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for Revision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}
