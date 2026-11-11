//! Closed resource capabilities for buffers and memory objects.

/// Resource access capabilities granted to a value or buffer binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum ResourceCapability {
    /// Read-only access permitted.
    ReadOnly,
    /// Write-only access permitted (destination buffer).
    WriteOnly,
    /// Read and write access permitted.
    ReadWrite,
    /// Atomic read-modify-write operations permitted.
    Atomic,
    /// Append-only / streaming write permitted.
    AppendOnly,
    /// Exclusive write ownership with no concurrent observers.
    Exclusive,
}

impl ResourceCapability {
    /// Whether reading from this resource is permitted.
    #[must_use]
    pub const fn can_read(&self) -> bool {
        match self {
            Self::ReadOnly | Self::ReadWrite | Self::Atomic | Self::Exclusive => true,
            Self::WriteOnly | Self::AppendOnly => false,
        }
    }

    /// Whether writing to this resource is permitted.
    #[must_use]
    pub const fn can_write(&self) -> bool {
        match self {
            Self::WriteOnly
            | Self::ReadWrite
            | Self::Atomic
            | Self::AppendOnly
            | Self::Exclusive => true,
            Self::ReadOnly => false,
        }
    }

    /// Whether atomic operations are permitted.
    #[must_use]
    pub const fn can_atomic(&self) -> bool {
        matches!(self, Self::Atomic | Self::Exclusive)
    }

    /// Combine two capabilities to the least permissive capability satisfying both constraints.
    #[must_use]
    pub const fn join(&self, other: Self) -> Self {
        match (*self, other) {
            (Self::Exclusive, _) | (_, Self::Exclusive) => Self::Exclusive,
            (Self::Atomic, _) | (_, Self::Atomic) => Self::Atomic,
            (Self::ReadWrite, _) | (_, Self::ReadWrite) => Self::ReadWrite,
            (Self::ReadOnly, Self::WriteOnly) | (Self::WriteOnly, Self::ReadOnly) => {
                Self::ReadWrite
            }
            (Self::ReadOnly, Self::ReadOnly) => Self::ReadOnly,
            (Self::WriteOnly, Self::WriteOnly) => Self::WriteOnly,
            (Self::AppendOnly, Self::AppendOnly) => Self::AppendOnly,
            (Self::AppendOnly, Self::ReadOnly) | (Self::ReadOnly, Self::AppendOnly) => {
                Self::ReadWrite
            }
            (Self::AppendOnly, Self::WriteOnly) | (Self::WriteOnly, Self::AppendOnly) => {
                Self::WriteOnly
            }
        }
    }
}
