//! Tile values in the IR.
//!
//! A tile is a first-class value with an element type, static extents, a layout,
//! and a residency. It is produced, consumed, and passed between operations
//! without requiring a backing buffer.

use serde::{Deserialize, Serialize};

use crate::ir_inner::model::op_signature::DataType;

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
    /// Compute linear storage index from multi-dimensional logical coordinates.
    #[must_use]
    pub fn linear_index(&self, coords: &[u32], extents: &[u32]) -> usize {
        match self {
            Self::RowMajor => {
                let mut index = 0usize;
                let mut stride = 1usize;
                for (&c, &e) in coords.iter().rev().zip(extents.iter().rev()) {
                    index += (c as usize) * stride;
                    stride *= e as usize;
                }
                index
            }
            Self::ColumnMajor => {
                let mut index = 0usize;
                let mut stride = 1usize;
                for (&c, &e) in coords.iter().zip(extents.iter()) {
                    index += (c as usize) * stride;
                    stride *= e as usize;
                }
                index
            }
            Self::Swizzled { permutation, period } => {
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
                let mut index = 0usize;
                let mut stride = 1usize;
                for (&c, &e) in permuted_coords.iter().rev().zip(extents.iter().rev()) {
                    index += (c as usize) * stride;
                    stride *= e as usize;
                }
                index
            }
        }
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
    pub fn new(element: DataType, extents: impl Into<Vec<u32>>, layout: Layout, residency: Residency) -> Self {
        Self {
            element,
            extents: extents.into(),
            layout,
            residency,
        }
    }

    /// Total number of elements in the tile.
    #[must_use]
    pub fn element_count(&self) -> usize {
        if self.extents.is_empty() {
            0
        } else {
            self.extents.iter().map(|&x| x as usize).product()
        }
    }

    /// Total byte size required by the tile storage.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let elem_bytes = match self.element {
            DataType::U8 | DataType::I8 | DataType::Bool => 1,
            DataType::U16 | DataType::I16 | DataType::F16 | DataType::BF16 => 2,
            DataType::U32 | DataType::I32 | DataType::F32 => 4,
            DataType::U64 | DataType::I64 | DataType::F64 | DataType::Vec2U32 => 8,
            DataType::Vec4U32 => 16,
            _ => 4,
        };
        (self.element_count() as u64) * elem_bytes
    }
}
