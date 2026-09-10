//! Typed resource transfers, residency composition, transfer lifetime, and integrity contracts.
//!
//! Bridges authenticated resource byte ranges and container identities with
//! artifact residency. Selects direct storage, registered host memory, or staged upload
//! paths from concrete device, filesystem, and driver capabilities.
//!
//! Unsupported direct storage never masquerades as zero-copy.
//! No resource becomes dispatch-visible before validation and transfer completion.

use std::collections::{BTreeMap, BTreeSet};

/// Concrete hardware and driver transfer capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceTransferCapabilities {
    /// True when device and kernel driver support NVMe-to-VRAM GPUDirect storage.
    pub supports_direct_storage: bool,
    /// True when pinned / registered host memory is supported.
    pub supports_pinned_host_memory: bool,
    /// Required byte alignment for direct storage DMA (typically 4096 bytes).
    pub required_direct_alignment_bytes: u64,
    /// Maximum concurrent in-flight transfer queue depth.
    pub max_transfer_queue_depth: usize,
}

impl Default for DeviceTransferCapabilities {
    fn default() -> Self {
        Self {
            supports_direct_storage: false,
            supports_pinned_host_memory: true,
            required_direct_alignment_bytes: 4096,
            max_transfer_queue_depth: 32,
        }
    }
}

/// Selected physical transfer path for a resource byte range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceTransferPath {
    /// Direct DMA from storage / NVMe to device memory (zero host bounce).
    DirectStorage,
    /// Registered / pinned host memory staging (accelerated DMA).
    RegisteredHostMemory,
    /// Standard staged host-buffer copy.
    StagedUpload,
}

/// Decision output for path selection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathSelectionDecision {
    /// Selected transfer path.
    pub selected_path: ResourceTransferPath,
    /// Whether the transfer is genuinely zero-copy without host staging.
    pub is_zero_copy: bool,
    /// Human-readable rationale for diagnostic tracing.
    pub reason: &'static str,
}

/// Select transfer path from device capabilities, file alignment, and filesystem support.
///
/// WHY: Unsupported direct transfer never masquerades as zero-copy. If direct storage
/// is unavailable or alignment requirements are not met, fallback path is chosen and
/// `is_zero_copy` is explicitly false.
#[must_use]
pub fn select_transfer_path(
    caps: &DeviceTransferCapabilities,
    file_offset: u64,
    fs_supports_direct: bool,
) -> PathSelectionDecision {
    if caps.supports_direct_storage && fs_supports_direct {
        let is_aligned = (file_offset % caps.required_direct_alignment_bytes) == 0;
        if is_aligned {
            return PathSelectionDecision {
                selected_path: ResourceTransferPath::DirectStorage,
                is_zero_copy: true,
                reason: "direct storage supported by device, filesystem, and 4KB alignment",
            };
        } else {
            return PathSelectionDecision {
                selected_path: ResourceTransferPath::RegisteredHostMemory,
                is_zero_copy: false,
                reason: "file offset is not 4KB aligned; falling back to pinned host staging",
            };
        }
    }

    if caps.supports_pinned_host_memory {
        PathSelectionDecision {
            selected_path: ResourceTransferPath::RegisteredHostMemory,
            is_zero_copy: false,
            reason: "direct storage unavailable; using registered host memory staging",
        }
    } else {
        PathSelectionDecision {
            selected_path: ResourceTransferPath::StagedUpload,
            is_zero_copy: false,
            reason: "using standard staged buffer upload",
        }
    }
}

/// Authenticated transfer descriptor binding a resource range to device residency.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceTransferDescriptor {
    /// Resource identifier.
    pub resource_name: String,
    /// Immutable file identity (BLAKE3 header/manifest digest).
    pub file_identity: [u8; 32],
    /// Byte offset within container file.
    pub offset: u64,
    /// Byte length of payload.
    pub length: u64,
    /// Content digest (BLAKE3 of resource payload).
    pub content_digest: [u8; 32],
    /// Destination allocated device resource name.
    pub destination_resource: String,
    /// Target device generation at transfer initiation.
    pub device_generation: u64,
    /// Selected transfer path.
    pub path: ResourceTransferPath,
}

/// Lifecycle state for one in-flight or completed transfer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceTransferState {
    /// Transfer descriptor registered; validation pending.
    Initiated,
    /// File header and digest pre-validation in progress.
    Validating,
    /// Byte transfer currently in progress.
    InProgress {
        /// Number of bytes successfully transferred so far.
        bytes_transferred: u64,
    },
    /// Transfer successfully finished and validated on device.
    Completed {
        /// Hardware or runtime completion event token.
        completion_event: u64,
    },
    /// Transfer failed with a specific integrity or hardware error.
    Failed {
        /// Failure details.
        error: ResourceTransferError,
    },
    /// Transfer explicitly cancelled by caller.
    Cancelled,
}

/// Integrity and transfer failure error conditions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, thiserror::Error)]
pub enum ResourceTransferError {
    /// Read returned fewer bytes than requested.
    #[error(
        "short read on resource `{resource_name}`: expected {expected} bytes, read {actual} bytes"
    )]
    ShortRead {
        /// Target resource name.
        resource_name: String,
        /// Expected byte length.
        expected: u64,
        /// Actual bytes read.
        actual: u64,
    },
    /// File identity was replaced or modified during transfer.
    #[error("file identity replaced during transfer of `{resource_name}`")]
    FileReplaced {
        /// Target resource name.
        resource_name: String,
    },
    /// File was truncated before expected end offset.
    #[error(
        "container file truncated: length {file_len} is less than required end {required_end}"
    )]
    Truncation {
        /// Observed container file length in bytes.
        file_len: u64,
        /// Required byte end offset.
        required_end: u64,
    },
    /// Alignment does not satisfy DMA requirements.
    #[error("alignment failure: offset {offset} not aligned to {required_alignment} bytes")]
    AlignmentFailure {
        /// Target byte offset.
        offset: u64,
        /// Required byte alignment.
        required_alignment: u64,
    },
    /// Device was reset or lost during transfer.
    #[error("device loss detected (device {device_id}, expected gen {expected_gen}, actual gen {actual_gen})")]
    DeviceLoss {
        /// Physical device index.
        device_id: u32,
        /// Expected hardware generation.
        expected_gen: u64,
        /// Actual observed hardware generation.
        actual_gen: u64,
    },
    /// Payload digest does not match authenticated descriptor.
    #[error("content digest mismatch for resource `{resource_name}`")]
    DigestMismatch {
        /// Target resource name.
        resource_name: String,
    },
    /// In-flight transfer queue depth exceeded maximum capacity.
    #[error(
        "transfer queue backpressure: {queue_depth} in-flight transfers exceed max {max_depth}"
    )]
    QueueBackpressure {
        /// Current in-flight transfer count.
        queue_depth: usize,
        /// Configured queue depth capacity.
        max_depth: usize,
    },
    /// Partial transfer failed and destination resource was safely reverted.
    #[error("partial transfer failed and cleaned up ({bytes_reverted} bytes reverted)")]
    PartialUploadCleanedUp {
        /// Number of allocated bytes reverted during cleanup.
        bytes_reverted: u64,
    },
}

/// Lifecycle engine governing resource transfers and dispatch visibility.
#[derive(Debug)]
pub struct ResourceTransferLifecycleEngine {
    caps: DeviceTransferCapabilities,
    device_generation: u64,
    transfers: BTreeMap<String, (ResourceTransferDescriptor, ResourceTransferState)>,
    completed_resources: BTreeSet<String>,
}

impl ResourceTransferLifecycleEngine {
    /// Create a new transfer engine for a given device generation.
    #[must_use]
    pub fn new(caps: DeviceTransferCapabilities, device_generation: u64) -> Self {
        Self {
            caps,
            device_generation,
            transfers: BTreeMap::new(),
            completed_resources: BTreeSet::new(),
        }
    }

    /// Register and initiate a new resource transfer.
    pub fn initiate_transfer(
        &mut self,
        descriptor: ResourceTransferDescriptor,
    ) -> Result<(), ResourceTransferError> {
        let in_flight_count = self
            .transfers
            .values()
            .filter(|(_, state)| {
                matches!(
                    state,
                    ResourceTransferState::Initiated
                        | ResourceTransferState::Validating
                        | ResourceTransferState::InProgress { .. }
                )
            })
            .count();

        if in_flight_count >= self.caps.max_transfer_queue_depth {
            return Err(ResourceTransferError::QueueBackpressure {
                queue_depth: in_flight_count,
                max_depth: self.caps.max_transfer_queue_depth,
            });
        }

        if descriptor.device_generation != self.device_generation {
            return Err(ResourceTransferError::DeviceLoss {
                device_id: 0,
                expected_gen: descriptor.device_generation,
                actual_gen: self.device_generation,
            });
        }

        let name = descriptor.resource_name.clone();
        self.transfers
            .insert(name, (descriptor, ResourceTransferState::Initiated));
        Ok(())
    }

    /// Update progress for an in-flight transfer.
    pub fn progress_transfer(
        &mut self,
        resource_name: &str,
        bytes: u64,
    ) -> Result<(), ResourceTransferError> {
        let (desc, state) = self.transfers.get_mut(resource_name).ok_or_else(|| {
            ResourceTransferError::ShortRead {
                resource_name: resource_name.to_string(),
                expected: 0,
                actual: 0,
            }
        })?;

        if desc.device_generation != self.device_generation {
            let err = ResourceTransferError::DeviceLoss {
                device_id: 0,
                expected_gen: desc.device_generation,
                actual_gen: self.device_generation,
            };
            *state = ResourceTransferState::Failed { error: err.clone() };
            return Err(err);
        }

        *state = ResourceTransferState::InProgress {
            bytes_transferred: bytes,
        };
        Ok(())
    }

    /// Complete and verify a transfer.
    pub fn complete_transfer(
        &mut self,
        resource_name: &str,
        completion_event: u64,
        actual_digest: [u8; 32],
    ) -> Result<(), ResourceTransferError> {
        let (desc, state) = self.transfers.get_mut(resource_name).ok_or_else(|| {
            ResourceTransferError::ShortRead {
                resource_name: resource_name.to_string(),
                expected: 0,
                actual: 0,
            }
        })?;

        if desc.device_generation != self.device_generation {
            let err = ResourceTransferError::DeviceLoss {
                device_id: 0,
                expected_gen: desc.device_generation,
                actual_gen: self.device_generation,
            };
            *state = ResourceTransferState::Failed { error: err.clone() };
            return Err(err);
        }

        if actual_digest != desc.content_digest {
            let err = ResourceTransferError::DigestMismatch {
                resource_name: resource_name.to_string(),
            };
            *state = ResourceTransferState::Failed { error: err.clone() };
            return Err(err);
        }

        *state = ResourceTransferState::Completed { completion_event };
        self.completed_resources.insert(resource_name.to_string());
        Ok(())
    }

    /// Record transfer failure and perform cleanup.
    pub fn fail_transfer(&mut self, resource_name: &str, error: ResourceTransferError) {
        if let Some((_, state)) = self.transfers.get_mut(resource_name) {
            *state = ResourceTransferState::Failed { error };
        }
        self.completed_resources.remove(resource_name);
    }

    /// Explicitly cancel an in-flight transfer.
    pub fn cancel_transfer(&mut self, resource_name: &str) {
        if let Some((_, state)) = self.transfers.get_mut(resource_name) {
            *state = ResourceTransferState::Cancelled;
        }
        self.completed_resources.remove(resource_name);
    }

    /// Returns whether the resource has completed validation and transfer and is dispatch-visible.
    #[must_use]
    pub fn is_dispatch_visible(&self, resource_name: &str) -> bool {
        self.completed_resources.contains(resource_name)
    }

    /// Handle device reset or loss event, invalidating all current transfers and prior state.
    pub fn handle_device_loss(&mut self, new_generation: u64) {
        self.device_generation = new_generation;
        self.completed_resources.clear();
        for (_, (_, state)) in self.transfers.iter_mut() {
            *state = ResourceTransferState::Failed {
                error: ResourceTransferError::DeviceLoss {
                    device_id: 0,
                    expected_gen: 0,
                    actual_gen: new_generation,
                },
            };
        }
    }
}
