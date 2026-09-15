//! Closed sparsity format definitions.

/// Closed sparsity format representation.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum Sparsity {
    /// Fully dense storage with no sparse compression.
    Dense,
    /// Compressed Sparse Row format.
    Csr,
    /// Compressed Sparse Column format.
    Csc,
    /// Coordinate list format.
    Coo,
    /// Block Compressed Sparse Row format.
    Bsr {
        /// Number of rows per dense block.
        block_rows: u32,
        /// Number of columns per dense block.
        block_cols: u32,
    },
    /// ELLPACK format.
    Ellpack,
    /// Structured 2:4 sparse format (2 non-zeros per 4 elements).
    Structured2to4,
    /// Ragged nested dimension with explicit segment offsets metadata.
    Ragged {
        /// Symbolic name of the buffer containing segment offsets.
        segment_offsets_symbol: String,
    },
}

impl Sparsity {
    /// Whether this represents uncompressed dense storage.
    #[must_use]
    pub const fn is_dense(&self) -> bool {
        matches!(self, Self::Dense)
    }

    /// Canonical name of the sparsity format.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Csr => "csr",
            Self::Csc => "csc",
            Self::Coo => "coo",
            Self::Bsr { .. } => "bsr",
            Self::Ellpack => "ellpack",
            Self::Structured2to4 => "structured_2to4",
            Self::Ragged { .. } => "ragged",
        }
    }
}
