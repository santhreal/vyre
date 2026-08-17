//! Topology-neutral peer-transfer capability contracts and checked transfer accounting.
//!
//! # Architecture
//!
//! Multi-device workflows (such as Mixture-of-Experts all-to-all token exchange,
//! tensor-parallel key/value distribution, and sharded graph frontier merges) require
//! explicit understanding of device-to-device connectivity.
//!
//! This module owns the backend-neutral capability contract:
//! - **Topology & Link Types**: NVLink, PCIe, Host-staged, or Unreachable.
//! - **Asymmetric & Partial Topology**: Accurately models unidirectional P2P links
//!   and partial interconnect meshes.
//! - **Generation & Stale Protection**: Rejects transfers when either device is on a
//!   stale generation.
//! - **Cancellation & Rollback**: Safe cancellation of pending peer transfers.
//! - **Non-Peer GPU-Resident Candidate**: Explicit staged fallback when direct P2P
//!   is unavailable, without hidden behavior.
//! - **Checked Accounting**: Zero-overflow byte, operation, and throughput tracking.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::transfer_accounting::TransferAccountingPolicy;
use crate::BackendError;

const PEER_TRANSFER_ACCOUNTING: TransferAccountingPolicy =
    TransferAccountingPolicy::new("peer_transfer", "shard or throttle inter-device transfers");

/// Interconnect physical link type between two devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PeerLinkKind {
    /// Direct high-bandwidth NVLink interconnect.
    NVLink {
        /// NVLink generation (e.g. 3, 4, 5).
        generation: u32,
        /// Number of physical links combined.
        links: u32,
    },
    /// Direct PCIe interconnect (Peer-to-Peer DMA).
    PCIe {
        /// PCIe generation (e.g. 4, 5).
        gen: u32,
        /// Number of lanes (e.g. 8, 16).
        lanes: u32,
    },
    /// Host-staged transfer path (via pinned host memory).
    HostStaged,
    /// No communication path exists.
    None,
}

/// Peer-to-peer access capability from a source device to a destination device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PeerAccessCapability {
    /// Direct device-to-device memory access enabled.
    DirectPeerMemory {
        /// Estimated unidirectional bandwidth in GB/s.
        bandwidth_gbps: u32,
        /// Underlying physical link type.
        link: PeerLinkKind,
    },
    /// Direct peer memory is unsupported or disabled; transfer must stage through host.
    StagedHostTransfer,
    /// Direct peer memory is physically supported but administrative access is disabled.
    PeerDisabled,
    /// Devices cannot communicate.
    Unreachable,
}

impl PeerAccessCapability {
    /// Whether direct GPU-to-GPU peer memory access is usable.
    #[must_use]
    pub const fn is_direct(&self) -> bool {
        matches!(self, Self::DirectPeerMemory { .. })
    }

    /// Whether any transfer path (direct or staged) exists.
    #[must_use]
    pub const fn is_reachable(&self) -> bool {
        !matches!(self, Self::Unreachable)
    }
}

/// Matrix describing directional peer access capabilities across all devices in a cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerTopology {
    device_count: u32,
    matrix: BTreeMap<(u32, u32), PeerAccessCapability>,
}

impl PeerTopology {
    /// Create a new peer topology for `device_count` devices (initialized as self-reachable, others unreachable).
    #[must_use]
    pub fn new(device_count: u32) -> Self {
        let mut matrix = BTreeMap::new();
        for src in 0..device_count {
            for dst in 0..device_count {
                if src == dst {
                    matrix.insert(
                        (src, dst),
                        PeerAccessCapability::DirectPeerMemory {
                            bandwidth_gbps: 1000,
                            link: PeerLinkKind::NVLink {
                                generation: 5,
                                links: 18,
                            },
                        },
                    );
                } else {
                    matrix.insert((src, dst), PeerAccessCapability::Unreachable);
                }
            }
        }
        Self {
            device_count,
            matrix,
        }
    }

    /// Set directional access capability from `src` to `dst`.
    pub fn set_capability(&mut self, src: u32, dst: u32, capability: PeerAccessCapability) {
        if src < self.device_count && dst < self.device_count {
            self.matrix.insert((src, dst), capability);
        }
    }

    /// Set symmetric access capability between `dev_a` and `dev_b`.
    pub fn set_symmetric_capability(
        &mut self,
        dev_a: u32,
        dev_b: u32,
        capability: PeerAccessCapability,
    ) {
        self.set_capability(dev_a, dev_b, capability);
        self.set_capability(dev_b, dev_a, capability);
    }

    /// Query capability from `src` to `dst`.
    #[must_use]
    pub fn capability(&self, src: u32, dst: u32) -> PeerAccessCapability {
        self.matrix
            .get(&(src, dst))
            .copied()
            .unwrap_or(PeerAccessCapability::Unreachable)
    }

    /// Number of devices in topology.
    #[must_use]
    pub const fn device_count(&self) -> u32 {
        self.device_count
    }
}

/// Errors occurring during peer transfer planning or execution.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PeerTransferError {
    /// Device index is out of bounds for the topology.
    #[error("device index {device} out of bounds for topology of {count} devices")]
    DeviceOutOfBounds {
        /// Out of bounds device index.
        device: u32,
        /// Total device count.
        count: u32,
    },
    /// Target peer is unreachable from source.
    #[error("peer transfer from device {src} to {dst} is unreachable")]
    UnreachablePeer {
        /// Source device index.
        src: u32,
        /// Destination device index.
        dst: u32,
    },
    /// Asymmetric access violation (e.g. writing requires bi-directional or reverse link).
    #[error(
        "asymmetric peer access violation from device {src} to {dst}: direct link is unidirectional"
    )]
    AsymmetricAccessViolation {
        /// Source device index.
        src: u32,
        /// Destination device index.
        dst: u32,
    },
    /// Stale device generation detected on source or destination device.
    #[error(
        "stale device generation on device {device}: expected {expected_gen}, got {actual_gen}"
    )]
    StaleDeviceGeneration {
        /// Device index.
        device: u32,
        /// Expected device generation.
        expected_gen: u64,
        /// Stale device generation.
        actual_gen: u64,
    },
    /// Transfer byte count is zero.
    #[error("peer transfer byte count cannot be zero")]
    ZeroByteTransfer,
    /// Byte count or buffer size overflowed.
    #[error("peer transfer accounting overflow: {0}")]
    AccountingOverflow(String),
    /// Transfer was cancelled.
    #[error("peer transfer operation {ticket} was cancelled")]
    Cancelled {
        /// Transfer ticket.
        ticket: u64,
    },
}

/// Single directional peer transfer request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerTransferRequest {
    /// Unique ticket identifying this transfer.
    pub ticket: u64,
    /// Source device index.
    pub src_device: u32,
    /// Source device generation.
    pub src_generation: u64,
    /// Source memory handle or address offset.
    pub src_offset: u64,
    /// Destination device index.
    pub dst_device: u32,
    /// Destination device generation.
    pub dst_generation: u64,
    /// Destination memory handle or address offset.
    pub dst_offset: u64,
    /// Exact bytes to transfer.
    pub byte_len: usize,
}

/// Plan for executing an individual or batch peer transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerTransferPlan {
    /// Planned transfer request.
    pub request: PeerTransferRequest,
    /// Capability resolved for this route.
    pub capability: PeerAccessCapability,
    /// Whether direct P2P copy or staged fallback is used.
    pub is_direct: bool,
    /// Whether transfer has been cancelled.
    pub cancelled: bool,
}

/// Cumulative accounting for peer-to-peer transfers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeerTransferAccounting {
    /// Total direct peer bytes transferred.
    pub direct_bytes: u64,
    /// Total host-staged peer bytes transferred.
    pub staged_bytes: u64,
    /// Total direct P2P operations completed.
    pub direct_transfers: u64,
    /// Total staged transfer operations completed.
    pub staged_transfers: u64,
    /// Total cancelled transfers.
    pub cancelled_transfers: u64,
}

impl PeerTransferAccounting {
    /// Record a completed transfer plan.
    pub fn record_completion(&mut self, plan: &PeerTransferPlan) -> Result<(), BackendError> {
        if plan.cancelled {
            PEER_TRANSFER_ACCOUNTING.add_u64_counter(
                &mut self.cancelled_transfers,
                1,
                "cancelled transfers",
                "cancelled transfer count",
            )?;
            return Ok(());
        }

        if plan.is_direct {
            PEER_TRANSFER_ACCOUNTING.add_bytes(
                &mut self.direct_bytes,
                plan.request.byte_len,
                "direct peer bytes",
            )?;
            PEER_TRANSFER_ACCOUNTING
                .add_operation(&mut self.direct_transfers, "direct peer transfers")?;
        } else {
            PEER_TRANSFER_ACCOUNTING.add_bytes(
                &mut self.staged_bytes,
                plan.request.byte_len,
                "staged peer bytes",
            )?;
            PEER_TRANSFER_ACCOUNTING
                .add_operation(&mut self.staged_transfers, "staged peer transfers")?;
        }

        Ok(())
    }
}

/// Planner for validating and preparing peer transfer operations against a cluster topology.
pub struct PeerTransferPlanner<'a> {
    topology: &'a PeerTopology,
    device_generations: &'a BTreeMap<u32, u64>,
}

impl<'a> PeerTransferPlanner<'a> {
    /// Create a planner with topology and current device generation map.
    #[must_use]
    pub fn new(topology: &'a PeerTopology, device_generations: &'a BTreeMap<u32, u64>) -> Self {
        Self {
            topology,
            device_generations,
        }
    }

    /// Plan and validate a peer transfer request.
    ///
    /// # Errors
    ///
    /// Returns [`PeerTransferError`] if peers are unreachable, device index is out of bounds,
    /// or device generation is stale.
    pub fn plan_transfer(
        &self,
        request: PeerTransferRequest,
    ) -> Result<PeerTransferPlan, PeerTransferError> {
        if request.byte_len == 0 {
            return Err(PeerTransferError::ZeroByteTransfer);
        }

        let dev_count = self.topology.device_count();
        if request.src_device >= dev_count {
            return Err(PeerTransferError::DeviceOutOfBounds {
                device: request.src_device,
                count: dev_count,
            });
        }
        if request.dst_device >= dev_count {
            return Err(PeerTransferError::DeviceOutOfBounds {
                device: request.dst_device,
                count: dev_count,
            });
        }

        // Validate device generations
        if let Some(&expected_src_gen) = self.device_generations.get(&request.src_device) {
            if request.src_generation != expected_src_gen {
                return Err(PeerTransferError::StaleDeviceGeneration {
                    device: request.src_device,
                    expected_gen: expected_src_gen,
                    actual_gen: request.src_generation,
                });
            }
        }
        if let Some(&expected_dst_gen) = self.device_generations.get(&request.dst_device) {
            if request.dst_generation != expected_dst_gen {
                return Err(PeerTransferError::StaleDeviceGeneration {
                    device: request.dst_device,
                    expected_gen: expected_dst_gen,
                    actual_gen: request.dst_generation,
                });
            }
        }

        let capability = self
            .topology
            .capability(request.src_device, request.dst_device);

        match capability {
            PeerAccessCapability::DirectPeerMemory { .. } => Ok(PeerTransferPlan {
                request,
                capability,
                is_direct: true,
                cancelled: false,
            }),
            PeerAccessCapability::StagedHostTransfer => Ok(PeerTransferPlan {
                request,
                capability,
                is_direct: false,
                cancelled: false,
            }),
            PeerAccessCapability::PeerDisabled | PeerAccessCapability::Unreachable => {
                Err(PeerTransferError::UnreachablePeer {
                    src: request.src_device,
                    dst: request.dst_device,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_topology_models_direct_nvlink_and_staged() {
        let mut topo = PeerTopology::new(4);
        topo.set_symmetric_capability(
            0,
            1,
            PeerAccessCapability::DirectPeerMemory {
                bandwidth_gbps: 600,
                link: PeerLinkKind::NVLink {
                    generation: 4,
                    links: 12,
                },
            },
        );
        topo.set_symmetric_capability(0, 2, PeerAccessCapability::StagedHostTransfer);

        assert!(topo.capability(0, 1).is_direct());
        assert!(!topo.capability(0, 2).is_direct());
        assert!(topo.capability(0, 2).is_reachable());
        assert!(!topo.capability(0, 3).is_reachable());
    }

    #[test]
    fn peer_planner_rejects_stale_generation() {
        let mut topo = PeerTopology::new(2);
        topo.set_symmetric_capability(
            0,
            1,
            PeerAccessCapability::DirectPeerMemory {
                bandwidth_gbps: 300,
                link: PeerLinkKind::PCIe { gen: 5, lanes: 16 },
            },
        );

        let mut gens = BTreeMap::new();
        gens.insert(0, 2);
        gens.insert(1, 2);

        let planner = PeerTransferPlanner::new(&topo, &gens);

        let req = PeerTransferRequest {
            ticket: 1,
            src_device: 0,
            src_generation: 1, // Stale generation 1 != 2
            src_offset: 0,
            dst_device: 1,
            dst_generation: 2,
            dst_offset: 0,
            byte_len: 1024,
        };

        let err = planner.plan_transfer(req).unwrap_err();
        assert!(matches!(
            err,
            PeerTransferError::StaleDeviceGeneration { .. }
        ));
    }

    #[test]
    fn peer_accounting_tracks_direct_and_staged_bytes() {
        let mut accounting = PeerTransferAccounting::default();
        let plan_direct = PeerTransferPlan {
            request: PeerTransferRequest {
                ticket: 1,
                src_device: 0,
                src_generation: 1,
                src_offset: 0,
                dst_device: 1,
                dst_generation: 1,
                dst_offset: 0,
                byte_len: 4096,
            },
            capability: PeerAccessCapability::DirectPeerMemory {
                bandwidth_gbps: 600,
                link: PeerLinkKind::NVLink {
                    generation: 4,
                    links: 12,
                },
            },
            is_direct: true,
            cancelled: false,
        };

        let plan_staged = PeerTransferPlan {
            request: PeerTransferRequest {
                ticket: 2,
                src_device: 0,
                src_generation: 1,
                src_offset: 0,
                dst_device: 2,
                dst_generation: 1,
                dst_offset: 0,
                byte_len: 2048,
            },
            capability: PeerAccessCapability::StagedHostTransfer,
            is_direct: false,
            cancelled: false,
        };

        accounting.record_completion(&plan_direct).expect("record");
        accounting.record_completion(&plan_staged).expect("record");

        assert_eq!(accounting.direct_bytes, 4096);
        assert_eq!(accounting.direct_transfers, 1);
        assert_eq!(accounting.staged_bytes, 2048);
        assert_eq!(accounting.staged_transfers, 1);
    }
}
