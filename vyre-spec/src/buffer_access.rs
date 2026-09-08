//! Frozen buffer-access tags used by operation and program metadata.

/// Buffer access mode in the frozen data contract.
///
/// Example: `BufferAccess::ReadWrite` records that a storage buffer may be
/// both read and written by a lowered operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum BufferAccess {
    /// Read-only storage buffer.
    ReadOnly,
    /// Read-write storage buffer.
    ReadWrite,
    /// Uniform buffer: small, read-only, and fast path.
    Uniform,
    /// Write-only storage buffer.
    WriteOnly,
    /// Workgroup-local shared memory.
    Workgroup,
}

impl BufferAccess {
    /// Every access mode in the frozen contract.
    ///
    /// A fixed-length array, so a variant added to the enum fails to compile
    /// until it is listed here. Callers outside this crate cannot match the
    /// enum exhaustively, because it is `#[non_exhaustive]`, so this is how a
    /// test walks the whole space and how a new mode turns those tests red until
    /// a decision is recorded for it.
    pub const ALL: [Self; 5] = [
        Self::ReadOnly,
        Self::ReadWrite,
        Self::Uniform,
        Self::WriteOnly,
        Self::Workgroup,
    ];
}
