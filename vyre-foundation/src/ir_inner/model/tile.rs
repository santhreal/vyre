//! Tile values in the IR.
//!
//! A tile is a first-class value with an element type, static extents, a layout,
//! and a residency. It is produced, consumed, and passed between operations
//! without requiring a backing buffer.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ir_inner::model::op_signature::DataType;

/// Typed error for Tile size and layout calculations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TileError {
    /// Arithmetic overflow in size or layout calculation.
    #[error("Tile layout computation overflow in field `{field}`")]
    Overflow {
        /// Name of the field where overflow occurred.
        field: &'static str,
    },
    /// Element data type has unknown or variable byte size.
    #[error("Tile element type `{element:?}` has unknown or variable element byte size")]
    UnknownElementSize {
        /// Element data type.
        element: DataType,
    },
}

/// Residency names where the tile data lives in the hardware hierarchy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Residency {
    /// Private to one invocation (registers).
    Register,
    /// Distributed fragment across the invocations of one subgroup (matrix fragments).
    Subgroup,
    /// Shared memory within a workgroup.
    Workgroup,
    /// Global memory buffer view.
    Global,
}

/// Layout describes how logical tile indices map to storage indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layout {
    /// Standard row-major storage.
    RowMajor,
    /// Standard column-major storage.
    ColumnMajor,
    /// Bank-conflict-free swizzled layout with permutation and swizzle period.
    Swizzled {
        /// Dimension permutation mapping.
        permutation: Vec<u32>,
        /// Swizzle period.
        period: u32,
    },
}

impl Layout {
    /// Compute linear storage index from multi-dimensional logical coordinates with checked arithmetic.
    pub fn checked_linear_index(&self, coords: &[u32], extents: &[u32]) -> Result<u64, TileError> {
        match self {
            Self::RowMajor => {
                let mut index = 0u64;
                let mut stride = 1u64;
                for (&c, &e) in coords.iter().rev().zip(extents.iter().rev()) {
                    let term = (c as u64).checked_mul(stride).ok_or(TileError::Overflow {
                        field: "coords * stride",
                    })?;
                    index = index.checked_add(term).ok_or(TileError::Overflow {
                        field: "linear_index",
                    })?;
                    stride = stride
                        .checked_mul(e as u64)
                        .ok_or(TileError::Overflow { field: "stride" })?;
                }
                Ok(index)
            }
            Self::ColumnMajor => {
                let mut index = 0u64;
                let mut stride = 1u64;
                for (&c, &e) in coords.iter().zip(extents.iter()) {
                    let term = (c as u64).checked_mul(stride).ok_or(TileError::Overflow {
                        field: "coords * stride",
                    })?;
                    index = index.checked_add(term).ok_or(TileError::Overflow {
                        field: "linear_index",
                    })?;
                    stride = stride
                        .checked_mul(e as u64)
                        .ok_or(TileError::Overflow { field: "stride" })?;
                }
                Ok(index)
            }
            Self::Swizzled {
                permutation,
                period,
            } => {
                let mut permuted_coords = coords.to_vec();
                if !permutation.is_empty() {
                    for (dst, &src) in permutation.iter().enumerate() {
                        if (src as usize) < coords.len() && dst < permuted_coords.len() {
                            permuted_coords[dst] = coords[src as usize];
                        }
                    }
                }
                if *period > 0 && permuted_coords.len() >= 2 {
                    let row = permuted_coords[0];
                    permuted_coords[1] ^= (row / period) % period;
                }
                let mut index = 0u64;
                let mut stride = 1u64;
                for (&c, &e) in permuted_coords.iter().rev().zip(extents.iter().rev()) {
                    let term = (c as u64).checked_mul(stride).ok_or(TileError::Overflow {
                        field: "coords * stride",
                    })?;
                    index = index.checked_add(term).ok_or(TileError::Overflow {
                        field: "linear_index",
                    })?;
                    stride = stride
                        .checked_mul(e as u64)
                        .ok_or(TileError::Overflow { field: "stride" })?;
                }
                Ok(index)
            }
        }
    }

    /// Compute linear storage index from multi-dimensional logical coordinates.
    #[must_use]
    pub fn linear_index(&self, coords: &[u32], extents: &[u32]) -> usize {
        self.checked_linear_index(coords, extents).unwrap_or(0) as usize
    }
}

/// A multidimensional tile value in the IR.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tile {
    /// Element data type.
    pub element: DataType,
    /// Static dimensions / extents.
    pub extents: Vec<u32>,
    /// Storage layout mapping.
    pub layout: Layout,
    /// Hardware residency level.
    pub residency: Residency,
}

impl Tile {
    /// Construct a new Tile description.
    #[must_use]
    pub fn new(
        element: DataType,
        extents: impl Into<Vec<u32>>,
        layout: Layout,
        residency: Residency,
    ) -> Self {
        Self {
            element,
            extents: extents.into(),
            layout,
            residency,
        }
    }

    /// Total number of elements in the tile with checked arithmetic.
    pub fn checked_element_count(&self) -> Result<u64, TileError> {
        if self.extents.is_empty() {
            return Ok(0);
        }
        let mut count = 1u64;
        for &e in &self.extents {
            count = count
                .checked_mul(e as u64)
                .ok_or(TileError::Overflow { field: "extents" })?;
        }
        Ok(count)
    }

    /// Total number of elements in the tile.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.checked_element_count().unwrap_or(0) as usize
    }

    /// Total byte size required by the tile storage with checked arithmetic.
    pub fn checked_byte_size(&self) -> Result<u64, TileError> {
        let count = self.checked_element_count()?;
        let elem_bytes = match &self.element {
            DataType::U8
            | DataType::I8
            | DataType::F8E4M3
            | DataType::F8E5M2
            | DataType::I4
            | DataType::FP4
            | DataType::NF4 => 1u64,
            DataType::U16 | DataType::I16 | DataType::F16 | DataType::BF16 => 2u64,
            DataType::Bool
            | DataType::U32
            | DataType::I32
            | DataType::F32
            | DataType::Handle(_) => 4u64,
            DataType::U64 | DataType::I64 | DataType::F64 | DataType::Vec2U32 => 8u64,
            DataType::Vec4U32 => 16u64,
            DataType::Vec {
                element,
                count: lanes,
            } => {
                let elem_size = match element.size_bytes() {
                    Some(s) => s as u64,
                    None => {
                        return Err(TileError::UnknownElementSize {
                            element: *element.clone(),
                        })
                    }
                };
                elem_size
                    .checked_mul(*lanes as u64)
                    .ok_or(TileError::Overflow {
                        field: "vector element size",
                    })?
            }
            DataType::Array { element_size } => *element_size as u64,
            other => match other.size_bytes() {
                Some(s) => s as u64,
                None => {
                    return Err(TileError::UnknownElementSize {
                        element: other.clone(),
                    })
                }
            },
        };
        count
            .checked_mul(elem_bytes)
            .ok_or(TileError::Overflow { field: "byte_size" })
    }

    /// Total byte size required by the tile storage.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        self.checked_byte_size().unwrap_or(0)
    }
}
