//! Coarse semantic tier, identifier namespace, and operation effects/tolerances.

use crate::ir::{BufferAccess, Program};

/// Coarse semantic tier used by catalog and conformance consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OperationTier {
    /// Foundation IR or built-in operation.
    Foundation,
    /// Category C hardware intrinsic owned by `vyre-primitives`.
    Intrinsic,
    /// Category A library composition over typed IR, owned by `vyre-libs`.
    Library,
    /// External extension operation.
    External,
    /// Identifier does not match an accepted semantic namespace.
    Unknown,
}

impl OperationTier {
    /// Every tier, in declaration order.
    ///
    /// A consumer that has to enumerate the taxonomy reads this instead of
    /// restating the variants, so adding one reaches every reader. The enum is
    /// `non_exhaustive`, so only this crate can match it exhaustively:
    /// `the_roster_carries_every_tier` below does, and a new variant stops
    /// compiling there until it is listed here.
    pub const ALL: &'static [Self] = &[
        Self::Foundation,
        Self::Intrinsic,
        Self::Library,
        Self::External,
        Self::Unknown,
    ];

    /// Stable operation-matrix spelling.
    #[must_use]
    pub const fn matrix_value(self) -> &'static str {
        match self {
            Self::Foundation => "foundation_ir",
            Self::Intrinsic => "intrinsic",
            Self::Library => "libs",
            Self::External => "external",
            Self::Unknown => "unknown",
        }
    }
}

/// Which crate minted one operation identity.
///
/// The namespace is frozen when the identity is published. It records the crate
/// that minted the id, not the crate the definition lives in today: eighteen
/// composition domains moved to `vyre-libs` keeping their `vyre-primitives::`
/// ids, so 130 identities name a crate that no longer holds their code. Nothing
/// derives a tier or a placement fact from this prefix. Tier is declared by the
/// registration and cross-checked against the tree by `crate-structure` and by
/// the operation schema. Host-side runtime capabilities are reached through the
/// driver and runtime capability surfaces, so `core.`, `io.` and `mem.` name no
/// namespace and are refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IdNamespace<'a> {
    /// Minted by the named workspace crate.
    Workspace(&'a str),
    /// Minted by a consumer outside the workspace.
    External(&'a str),
    /// Names no crate.
    Unknown,
}

/// Read the minting namespace of one operation identity.
#[must_use]
pub fn operation_id_namespace(id: &str) -> IdNamespace<'_> {
    let Some((namespace, rest)) = id.split_once("::") else {
        return IdNamespace::Unknown;
    };
    if namespace.is_empty() || rest.is_empty() {
        IdNamespace::Unknown
    } else if namespace.starts_with("vyre-") {
        IdNamespace::Workspace(namespace)
    } else {
        IdNamespace::External(namespace)
    }
}

/// Semantic memory and synchronization effects derived from an operation program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OperationEffects {
    /// The operation reads caller-visible storage.
    pub reads: bool,
    /// The operation writes caller-visible storage.
    pub writes: bool,
    /// The operation contains atomic memory effects.
    pub atomics: bool,
    /// The operation requires intra- or inter-workgroup synchronization.
    pub synchronizes: bool,
}

impl OperationEffects {
    /// Conservative strongest applicable effects (all effects active).
    /// Pure computation with no side-effects or external storage access.
    pub const NONE: Self = Self {
        reads: false,
        writes: false,
        atomics: false,
        synchronizes: false,
    };

    /// Standard read-write storage access without atomics or barriers.
    pub const READ_WRITE: Self = Self {
        reads: true,
        writes: true,
        atomics: false,
        synchronizes: false,
    };

    /// Read-only storage access.
    pub const READ_ONLY: Self = Self {
        reads: true,
        writes: false,
        atomics: false,
        synchronizes: false,
    };

    /// Barrier synchronization effect.
    pub const SYNCHRONIZES: Self = Self {
        reads: false,
        writes: false,
        atomics: false,
        synchronizes: true,
    };

    /// Read-write storage access with synchronization barrier.
    pub const READ_WRITE_SYNCHRONIZES: Self = Self {
        reads: true,
        writes: true,
        atomics: false,
        synchronizes: true,
    };

    /// Conservative strongest applicable effects (all effects active).
    pub const ALL: Self = Self {
        reads: true,
        writes: true,
        atomics: true,
        synchronizes: true,
    };

    /// Merge another effect set into this one (field-wise OR).
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            reads: self.reads || other.reads,
            writes: self.writes || other.writes,
            atomics: self.atomics || other.atomics,
            synchronizes: self.synchronizes || other.synchronizes,
        }
    }

    /// Derive neutral effects from the canonical program declaration and statistics.
    #[must_use]
    pub fn from_program(program: &Program) -> Self {
        let mut effects = Self::default();
        for buffer in program.buffers() {
            match buffer.access() {
                BufferAccess::ReadOnly => effects.reads = true,
                BufferAccess::ReadWrite => {
                    effects.reads = true;
                    effects.writes = true;
                }
                BufferAccess::WriteOnly => effects.writes = true,
                _ => {
                    effects.reads = true;
                    effects.writes = true;
                }
            }
        }
        let stats = program.stats();
        effects.atomics = stats.atomic_op_count > 0;
        effects.synchronizes = stats.has_node_barrier() || stats.distributed_collectives();
        effects
    }
}
