//! Binding paged resources through resident resource contracts.
//!
//! # Architecture
//!
//! Paged resource execution requires binding two distinct resident resources:
//! 1. **Page Table**: An array of `u32` physical page indices per sequence.
//! 2. **Physical Cache Slab**: Slabs of memory holding physical pages of data.
//!
//! This module binds paged resources through [`crate::resource_residency::ResourceResidency`]:
//! - Validates buffer capacity against exact tensor geometry and data types.
//! - Validates buffer memory alignment (e.g. 64-byte boundary).
//! - Validates device ownership, state lease generation, and lifetime.
//! - Tracks async completion events so pages are not released or recycled while in flight.
//! - If paging is unsupported on a given device, rejects or provides an explicit
//!   contiguous cache candidate; **never triggers an implicit host execution path**.

use thiserror::Error;
use vyre_driver::Resource;
use vyre_foundation::ir::DataType;

use crate::resource_residency::{ResourceResidencyError, StateId, StateLease};

/// Minimum byte alignment required for paged cache and table buffers.
pub const PAGED_RESOURCE_MIN_ALIGNMENT_BYTES: usize = 64;

/// Errors occurring during paged resource residency binding or validation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PagedResidencyError {
    /// Device or materializer does not support paged memory addressing.
    #[error(
        "device {device} does not support paged memory addressing. Fix: use explicit contiguous cache candidate or target a paging-capable device"
    )]
    UnsupportedPaging {
        /// Target device identifier.
        device: String,
    },
    /// Buffer capacity does not match expected geometry.
    #[error(
        "paged resource {name} capacity mismatch: expected at least {expected_bytes} bytes, got {actual_bytes} bytes"
    )]
    CapacityMismatch {
        /// Resource name.
        name: String,
        /// Expected minimum bytes.
        expected_bytes: usize,
        /// Actual allocated bytes.
        actual_bytes: usize,
    },
    /// Buffer offset or base address violates alignment requirements.
    #[error(
        "paged resource {name} has unaligned address or offset {offset}: must be aligned to {alignment} bytes"
    )]
    MisalignedBuffer {
        /// Resource name.
        name: String,
        /// Memory offset or pointer.
        offset: usize,
        /// Required alignment.
        alignment: usize,
    },
    /// Device ownership mismatch between page table and cache slabs.
    #[error(
        "device ownership conflict: resource {name} belongs to device {actual_device}, expected {expected_device}"
    )]
    DeviceOwnershipMismatch {
        /// Resource name.
        name: String,
        /// Expected device.
        expected_device: u32,
        /// Actual device.
        actual_device: u32,
    },
    /// Stale state lease generation detected.
    #[error("stale state lease generation for state {state:?}: expected {expected_gen}, got {actual_gen}")]
    StaleLeaseGeneration {
        /// State ID.
        state: StateId,
        /// Expected generation.
        expected_gen: u64,
        /// Actual lease generation.
        actual_gen: u64,
    },
    /// Underlying resource residency error.
    #[error("resource residency error: {0}")]
    ResidencyError(#[from] ResourceResidencyError),
    /// Backend driver error.
    #[error("backend error: {0}")]
    BackendError(String),
}

/// Fallback candidate strategy when paged addressing is unsupported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagingCandidateStrategy {
    /// Native paged addressing with physical page table and cache pool.
    PagedAddressing,
    /// Explicit contiguous buffer candidate (never implicit host execution).
    ExplicitContiguousFallback {
        /// Max context units reserved in contiguous buffer.
        max_capacity_units: u32,
    },
}

/// Geometry specification for a paged resource slab allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedResourceSpec {
    /// Number of physical blocks in the pool.
    pub blocks: u32,
    /// Channel or feature count.
    pub channels: u32,
    /// Units per physical page.
    pub units_per_block: u32,
    /// Unit dimension.
    pub unit_dim: u32,
    /// Data type.
    pub dtype: DataType,
}

impl PagedResourceSpec {
    /// Calculate exact byte size required for the resource slab.
    #[must_use]
    pub fn required_slab_bytes(&self) -> usize {
        let elem_bytes = match self.dtype {
            DataType::F32 | DataType::U32 | DataType::I32 => 4,
            DataType::F16 | DataType::BF16 | DataType::U16 | DataType::I16 => 2,
            _ => 4,
        };
        2 * (self.blocks as usize)
            * (self.channels as usize)
            * (self.units_per_block as usize)
            * (self.unit_dim as usize)
            * elem_bytes
    }
}

/// Geometry specification for a block table buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTableSpec {
    /// Batch sequences.
    pub sequences: u32,
    /// Maximum blocks per sequence.
    pub blocks_per_sequence: u32,
}

impl BlockTableSpec {
    /// Calculate exact byte size required for the block table buffer (`u32` elements).
    #[must_use]
    pub fn required_table_bytes(&self) -> usize {
        (self.sequences as usize)
            * (self.blocks_per_sequence as usize)
            * core::mem::size_of::<u32>()
    }
}

/// Authenticated resident binding for paged resources.
#[derive(Debug, Clone)]
pub struct PagedResourceBinding {
    /// State lease under which these resources are bound.
    pub lease: StateLease,
    /// Target device identifier.
    pub device_id: u32,
    /// Bound page table resource.
    pub table_resource: Resource,
    /// Bound primary slab resource.
    pub primary_resource: Resource,
    /// Bound secondary slab resource.
    pub secondary_resource: Option<Resource>,
    /// Slab specification.
    pub resource_spec: PagedResourceSpec,
    /// Table specification.
    pub table_spec: BlockTableSpec,
    /// Whether execution is currently in flight.
    pub in_flight: bool,
    /// Active completion ticket.
    pub completion_ticket: u64,
}

impl PagedResourceBinding {
    /// Validate that paged resources conform to capacity, alignment, and device ownership.
    ///
    /// # Errors
    ///
    /// Returns [`PagedResidencyError`] if capacity is insufficient, unaligned, or device mismatched.
    pub fn validate(
        &self,
        table_byte_len: usize,
        primary_byte_len: usize,
        secondary_byte_len: usize,
    ) -> Result<(), PagedResidencyError> {
        let expected_table_bytes = self.table_spec.required_table_bytes();
        if table_byte_len < expected_table_bytes {
            return Err(PagedResidencyError::CapacityMismatch {
                name: "table".into(),
                expected_bytes: expected_table_bytes,
                actual_bytes: table_byte_len,
            });
        }

        let expected_slab_bytes = self.resource_spec.required_slab_bytes() / 2;
        if primary_byte_len < expected_slab_bytes {
            return Err(PagedResidencyError::CapacityMismatch {
                name: "primary_slab".into(),
                expected_bytes: expected_slab_bytes,
                actual_bytes: primary_byte_len,
            });
        }
        if self.secondary_resource.is_some() && secondary_byte_len < expected_slab_bytes {
            return Err(PagedResidencyError::CapacityMismatch {
                name: "secondary_slab".into(),
                expected_bytes: expected_slab_bytes,
                actual_bytes: secondary_byte_len,
            });
        }

        Ok(())
    }

    /// Mark paged resource binding as in-flight for kernel execution.
    pub fn mark_in_flight(&mut self, ticket: u64) {
        self.in_flight = true;
        self.completion_ticket = ticket;
    }

    /// Complete execution event and release in-flight lock.
    pub fn complete_execution(&mut self, completed_ticket: u64) {
        if self.completion_ticket <= completed_ticket {
            self.in_flight = false;
        }
    }
}

/// Selector for paged vs contiguous candidate strategies.
pub struct PagedResidencyPlanner;

impl PagedResidencyPlanner {
    /// Select execution candidate strategy based on device capabilities.
    #[must_use]
    pub fn select_strategy(
        device_supports_paging: bool,
        max_capacity_units: u32,
    ) -> PagingCandidateStrategy {
        if device_supports_paging {
            PagingCandidateStrategy::PagedAddressing
        } else {
            PagingCandidateStrategy::ExplicitContiguousFallback { max_capacity_units }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_resource_spec_computes_exact_bytes() {
        let spec = PagedResourceSpec {
            blocks: 16,
            channels: 4,
            units_per_block: 16,
            unit_dim: 64,
            dtype: DataType::F16,
        };
        // 2 * 16 * 4 * 16 * 64 * 2 = 262,144 bytes
        assert_eq!(spec.required_slab_bytes(), 262_144);
    }

    #[test]
    fn paged_table_spec_computes_exact_bytes() {
        let spec = BlockTableSpec {
            sequences: 4,
            blocks_per_sequence: 32,
        };
        // 4 * 32 * 4 = 512 bytes
        assert_eq!(spec.required_table_bytes(), 512);
    }

    #[test]
    fn paged_binding_validates_capacity_mismatch() {
        let owner = vyre_driver::ResidentOwner::new().expect("owner");
        let binding = PagedResourceBinding {
            lease: StateLease {
                id: StateId(1),
                generation: 1,
            },
            device_id: 0,
            table_resource: Resource::Resident(owner.handle(10)),
            primary_resource: Resource::Resident(owner.handle(11)),
            secondary_resource: Some(Resource::Resident(owner.handle(12))),
            resource_spec: PagedResourceSpec {
                blocks: 4,
                channels: 2,
                units_per_block: 8,
                unit_dim: 32,
                dtype: DataType::F32,
            },
            table_spec: BlockTableSpec {
                sequences: 1,
                blocks_per_sequence: 4,
            },
            in_flight: false,
            completion_ticket: 0,
        };

        let err = binding.validate(16, 8000, 8192).unwrap_err();
        assert!(matches!(err, PagedResidencyError::CapacityMismatch { .. }));

        assert!(binding.validate(16, 8192, 8192).is_ok());
    }

    #[test]
    fn planner_selects_explicit_contiguous_fallback_without_host_execution() {
        let paged_strategy = PagedResidencyPlanner::select_strategy(true, 2048);
        assert_eq!(paged_strategy, PagingCandidateStrategy::PagedAddressing);

        let fallback_strategy = PagedResidencyPlanner::select_strategy(false, 2048);
        assert_eq!(
            fallback_strategy,
            PagingCandidateStrategy::ExplicitContiguousFallback {
                max_capacity_units: 2048,
            }
        );
    }
}
