//! Closed ownership, mutability, and linear discipline types.

/// Ownership and mutability discipline for semantic values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum OwnershipMutability {
    /// Immutable value; may be freely shared and duplicated.
    Immutable,
    /// Unique exclusive ownership; may be mutated in place.
    ExclusiveOwned,
    /// Shared borrow; read access guaranteed without race.
    SharedBorrowed,
    /// Linear single-use value; must be consumed exactly once.
    LinearConsumed,
}

impl OwnershipMutability {
    /// Whether this ownership mode permits in-place mutation.
    #[must_use]
    pub const fn is_mutable(&self) -> bool {
        matches!(self, Self::ExclusiveOwned)
    }

    /// Whether this ownership mode enforces linear consumption.
    #[must_use]
    pub const fn is_linear(&self) -> bool {
        matches!(self, Self::LinearConsumed)
    }

    /// Whether this value may be aliased by other references.
    #[must_use]
    pub const fn can_alias(&self) -> bool {
        match self {
            Self::Immutable | Self::SharedBorrowed => true,
            Self::ExclusiveOwned | Self::LinearConsumed => false,
        }
    }
}
