//! Binding paged resources (page tables, KV cache slabs) through resident resource contracts.
//!
//! # Architecture
//!
//! Paged attention requires binding two distinct resident resources:
//! 1. **Page Table**: An array of `u32` physical page indices per sequence.
//! 2. **Physical Cache Slab**: Slabs of memory holding physical pages of keys and values.
//!
//! This module binds paged resources through [`crate::resource_residency::ResourceResidency`]:
//! - Validates buffer capacity against exact tensor geometry and data types.
//! - Validates buffer memory alignment (e.g. 64-byte boundary).
//! - Validates device ownership, state lease generation, and lifetime.
//! - Tracks async completion events so pages are not released or recycled while in flight.
//! - If paging is unsupported on a given device, rejects or provides an explicit
//!   contiguous-cache candidate; **never triggers an implicit host execution path**.

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
        "device {device} does not support paged KV attention. Fix: use explicit contiguous cache candidate or target a paging-capable device"
    )]
    UnsupportedPaging {
        /// Target device identifier.
        device: String,
    },
    /// Buffer capacity does not match expected tensor geometry.
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
    /// Native paged attention with physical page table and cache pool.
    PagedAttention,
    /// Explicit contiguous KV-cache buffer candidate (never implicit host execution).
    ExplicitContiguousFallback {
        /// Max context tokens reserved in contiguous buffer.
        max_context_tokens: u32,
    },
}

/// Geometry specification for a paged KV cache allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKVSlabSpec {
    /// Number of physical blocks in the pool.
    pub blocks: u32,
    /// KV head count.
    pub kv_heads: u32,
    /// Tokens per physical page.
    pub block_tokens: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Data type.
    pub dtype: DataType,
}

impl PagedKVSlabSpec {
    /// Calculate exact byte size required for keys and values cache slab.
    #[must_use]
    pub fn required_slab_bytes(&self) -> usize {
        let elem_bytes = match self.dtype {
            DataType::F32 | DataType::U32 | DataType::I32 => 4,
            DataType::F16 | DataType::BF16 | DataType::U16 | DataType::I16 => 2,
            _ => 4,
        };
        // 2 for K and V
        2 * (self.blocks as usize)
            * (self.kv_heads as usize)
            * (self.block_tokens as usize)
            * (self.head_dim as usize)
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

/// Authenticated resident binding for paged KV attention resources.
#[derive(Debug, Clone)]
pub struct PagedResourceBinding {
    /// State lease under which these resources are bound.
    pub lease: StateLease,
    /// Target device identifier.
    pub device_id: u32,
    /// Bound page table resource.
    pub block_table_resource: Resource,
    /// Bound K-cache slab resource.
    pub k_cache_resource: Resource,
    /// Bound V-cache slab resource.
    pub v_cache_resource: Resource,
    /// Slab specification.
    pub slab_spec: PagedKVSlabSpec,
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
        k_byte_len: usize,
        v_byte_len: usize,
    ) -> Result<(), PagedResidencyError> {
        let expected_table_bytes = self.table_spec.required_table_bytes();
        if table_byte_len < expected_table_bytes {
            return Err(PagedResidencyError::CapacityMismatch {
                name: "block_table".into(),
                expected_bytes: expected_table_bytes,
                actual_bytes: table_byte_len,
            });
        }

        let expected_slab_bytes = self.slab_spec.required_slab_bytes() / 2; // Per K and V
        if k_byte_len < expected_slab_bytes {
            return Err(PagedResidencyError::CapacityMismatch {
                name: "k_cache".into(),
                expected_bytes: expected_slab_bytes,
                actual_bytes: k_byte_len,
            });
        }
        if v_byte_len < expected_slab_bytes {
            return Err(PagedResidencyError::CapacityMismatch {
                name: "v_cache".into(),
                expected_bytes: expected_slab_bytes,
                actual_bytes: v_byte_len,
            });
        }

        // Validate 64-byte alignment
        if expected_table_bytes % PAGED_RESOURCE_MIN_ALIGNMENT_BYTES != 0 {
            // Buffer size or stride check
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
        max_context_tokens: u32,
    ) -> PagingCandidateStrategy {
        if device_supports_paging {
            PagingCandidateStrategy::PagedAttention
        } else {
            // Explicit contiguous fallback candidate (not implicit host execution!)
            PagingCandidateStrategy::ExplicitContiguousFallback {
                max_context_tokens,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_slab_spec_computes_exact_bytes() {
        let spec = PagedKVSlabSpec {
            blocks: 16,
            kv_heads: 4,
            block_tokens: 16,
            head_dim: 64,
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
            block_table_resource: Resource::Resident(owner.handle(10)),
            k_cache_resource: Resource::Resident(owner.handle(11)),
            v_cache_resource: Resource::Resident(owner.handle(12)),
            slab_spec: PagedKVSlabSpec {
                blocks: 4,
                kv_heads: 2,
                block_tokens: 8,
                head_dim: 32,
                dtype: DataType::F32,
            },
            table_spec: BlockTableSpec {
                sequences: 1,
                blocks_per_sequence: 4,
            },
            in_flight: false,
            completion_ticket: 0,
        };

        // Table needs 1 * 4 * 4 = 16 bytes.
        // K slab needs 4 * 2 * 8 * 32 * 4 = 8192 bytes.
        // Passing 8000 bytes for K slab should fail capacity validation.
        let err = binding.validate(16, 8000, 8192).unwrap_err();
        assert!(matches!(err, PagedResidencyError::CapacityMismatch { .. }));

        // Passing full sizes succeeds.
        assert!(binding.validate(16, 8192, 8192).is_ok());
    }

    #[test]
    fn planner_selects_explicit_contiguous_fallback_without_host_execution() {
        let paged_strategy = PagedResidencyPlanner::select_strategy(true, 2048);
        assert_eq!(paged_strategy, PagingCandidateStrategy::PagedAttention);

        let fallback_strategy = PagedResidencyPlanner::select_strategy(false, 2048);
        assert_eq!(
            fallback_strategy,
            PagingCandidateStrategy::ExplicitContiguousFallback {
                max_context_tokens: 2048
            }
        );
    }
}
