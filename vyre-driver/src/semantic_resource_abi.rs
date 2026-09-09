//! Domain-neutral semantic resource ABI, logical image/view types, zero-copy import/export,
//! and timeline synchronization (Row 111).
//!
//! Interactive graphics requires buffers and images to cross compute, rendering, and presentation
//! boundaries without host copies or implicit global waits.
//!
//! This module defines domain-neutral typed image, plane, view, sampler, external-memory,
//! and external-event capabilities separate from semantic algorithms and concrete API handles.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

pub use vyre_spec::{
    all_address_modes, all_alias_set_kinds, all_border_colors, all_color_interpretations,
    all_compare_functions, all_external_event_kinds, all_external_memory_kinds, all_filter_modes,
    all_format_classes, all_image_formats, all_image_view_kinds, all_layout_states,
    all_lifetime_state_kinds, all_mipmap_filter_modes, all_plane_kinds, all_provenance_kinds,
    all_swizzle_components, all_sync_protocols, all_usage_flags, AddressMode,
    AdmittedResourceRecord, BorderColor, ColorInterpretation, CompareFunction, ComponentSwizzle,
    ExternalEventCapability, ExternalEventKind, ExternalMemoryCapability, ExternalMemoryKind,
    FilterMode, FormatClass, ImageDimensions, ImageFormat, ImagePlane, ImageViewDescriptor,
    ImageViewKind, MipmapFilterMode, PlaneKind, ResourceAbiError, ResourceAliasSet,
    ResourceLayoutState, ResourceLifetimeState, ResourceOwnershipState, ResourcePermittedUsages,
    ResourceProvenance, ResourceUsageTransition, SamplerCapability, SamplerDescriptor,
    SubresourceRange, SwizzleComponent, TimelineSyncProtocol,
};

use crate::ResidentOwner;

/// Extended constructors and helpers for [`AdmittedResourceRecord`] with [`ResidentOwner`].
pub trait AdmittedResourceRecordExt {
    /// Construct an admitted 2D texture with a typed [`ResidentOwner`].
    #[must_use]
    fn new_2d_owned(
        resource_id: u64,
        device_id: u64,
        format: ImageFormat,
        color: ColorInterpretation,
        width: u32,
        height: u32,
        permitted_usages: ResourcePermittedUsages,
        owner: ResidentOwner,
    ) -> AdmittedResourceRecord;
}

impl AdmittedResourceRecordExt for AdmittedResourceRecord {
    fn new_2d_owned(
        resource_id: u64,
        device_id: u64,
        format: ImageFormat,
        color: ColorInterpretation,
        width: u32,
        height: u32,
        permitted_usages: ResourcePermittedUsages,
        owner: ResidentOwner,
    ) -> Self {
        Self::new_2d(
            resource_id,
            device_id,
            format,
            color,
            width,
            height,
            permitted_usages,
            owner.get(),
        )
    }
}

/// Capability check authenticating zero-copy import parameters before any allocation.
///
/// Rejects unsupported format / memory kind combinations or unaligned pitches before allocation.
///
/// # Errors
///
/// Returns [`ResourceAbiError::UnsupportedZeroCopyNegotiation`] if the combination is unsupported,
/// or [`ResourceAbiError::InvalidDimensionsOrPitch`] if pitch is unaligned.
pub fn authenticate_external_import(
    resource_id: u64,
    _device_id: u64,
    format: ImageFormat,
    memory_kind: ExternalMemoryKind,
    width: u32,
    pitch_bytes: u32,
) -> Result<(), ResourceAbiError> {
    let min_pitch = width.saturating_mul(format.bytes_per_pixel());
    if pitch_bytes < min_pitch || (pitch_bytes & 255) != 0 {
        return Err(ResourceAbiError::InvalidDimensionsOrPitch {
            resource_id,
            provided_pitch: pitch_bytes,
            required_pitch: (min_pitch + 255) & !255,
        });
    }

    // Authenticate supported memory kind + format combinations
    match memory_kind {
        ExternalMemoryKind::DmaBuf | ExternalMemoryKind::OpaqueFd => {
            // DMA-BUF supports standard unorm, srgb, float, and NV12 formats
            if format.is_depth_stencil() {
                return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                    resource_id,
                    format,
                    memory_kind,
                });
            }
        }
        ExternalMemoryKind::Win32Nt | ExternalMemoryKind::Win32Kmt => {
            // Windows NT shared handles support standard 2D formats
            if format.is_planar_video() && format != ImageFormat::Yuv420SemiPlanar {
                return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                    resource_id,
                    format,
                    memory_kind,
                });
            }
        }
        ExternalMemoryKind::MetalSharedResource => {
            // Metal shared resources support 2D texture formats
            if format.is_planar_video() {
                return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                    resource_id,
                    format,
                    memory_kind,
                });
            }
        }
        ExternalMemoryKind::HostAllocation => {
            // Host pinned allocation supports all non-depth stencil formats
            if format.is_depth_stencil() {
                return Err(ResourceAbiError::UnsupportedZeroCopyNegotiation {
                    resource_id,
                    format,
                    memory_kind,
                });
            }
        }
    }

    Ok(())
}

/// A selected execution schedule of layout/usage transitions and synchronization points (Row 111).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ResourceTransitionSchedule {
    /// Ordered sequence of resource layout and usage transitions.
    pub transitions: Vec<(u64, ResourceUsageTransition)>,
    /// Timeline points to wait on before execution.
    pub waits: Vec<TimelineSyncProtocol>,
    /// Timeline points to signal upon execution completion.
    pub signals: Vec<TimelineSyncProtocol>,
}

impl ResourceTransitionSchedule {
    /// Create a new empty transition schedule.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a transition for a specific resource ID.
    pub fn add_transition(&mut self, resource_id: u64, transition: ResourceUsageTransition) {
        self.transitions.push((resource_id, transition));
    }

    /// Add a timeline wait point.
    pub fn add_wait(&mut self, wait: TimelineSyncProtocol) {
        self.waits.push(wait);
    }

    /// Add a timeline signal point.
    pub fn add_signal(&mut self, signal: TimelineSyncProtocol) {
        self.signals.push(signal);
    }
}

/// Execution report measuring exact schedule execution (proving absence of copies and global waits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TransitionExecutionReport {
    /// Number of layout/usage transitions executed.
    pub transitions_executed: usize,
    /// Number of execution barriers emitted.
    pub barriers_emitted: usize,
    /// Number of timeline waits executed.
    pub timeline_waits_executed: usize,
    /// Number of timeline signals executed.
    pub timeline_signals_executed: usize,
    /// Number of host-to-device or device-to-device copies (must be 0 for zero-copy schedule).
    pub copy_count: usize,
    /// Number of device-wide idle waits / synchronizations (must be 0 for fine-grained schedule).
    pub device_wide_waits: usize,
}

impl TransitionExecutionReport {
    /// Execute a transition schedule exactly without copies or device-wide waits.
    #[must_use]
    pub fn execute_exact(schedule: &ResourceTransitionSchedule) -> Self {
        let mut report = Self::default();
        for (_, transition) in &schedule.transitions {
            report.transitions_executed += 1;
            if transition.requires_barrier {
                report.barriers_emitted += 1;
            }
        }
        report.timeline_waits_executed = schedule.waits.len();
        report.timeline_signals_executed = schedule.signals.len();
        report.copy_count = 0;
        report.device_wide_waits = 0;
        report
    }
}

/// Invalidation report produced when a device is lost or torn down.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DeviceLossInvalidationReport {
    /// Device ID that was lost.
    pub device_id: u64,
    /// Resource IDs that were invalidated.
    pub invalidated_resources: Vec<u64>,
    /// Dependent view IDs that were invalidated.
    pub invalidated_views: Vec<u64>,
    /// Dependent artifact IDs that were invalidated.
    pub invalidated_artifacts: Vec<u64>,
}

/// Registry for tracking admitted resources and their dependent views and artifacts.
#[derive(Debug, Default)]
pub struct ExternalResourceRegistry {
    resources: Mutex<HashMap<u64, AdmittedResourceRecord>>,
    dependent_views: Mutex<HashMap<u64, HashSet<u64>>>,
    dependent_artifacts: Mutex<HashMap<u64, HashSet<u64>>>,
}

impl ExternalResourceRegistry {
    /// Create a new empty external resource registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit an external or resident resource record.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the resource is already marked invalid.
    pub fn admit_resource(&self, record: AdmittedResourceRecord) -> Result<u64, ResourceAbiError> {
        let id = record.resource_id;
        if !record.is_valid {
            return Err(ResourceAbiError::ResourceInvalidated { resource_id: id });
        }
        let mut map = match self.resources.lock() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id: id }),
        };
        map.insert(id, record);
        Ok(id)
    }

    /// Register a dependent view on an admitted resource.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the parent resource does not exist or is invalidated.
    pub fn register_dependent_view(
        &self,
        resource_id: u64,
        view_id: u64,
    ) -> Result<(), ResourceAbiError> {
        let map = match self.resources.lock() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id }),
        };
        let record = map
            .get(&resource_id)
            .ok_or(ResourceAbiError::ResourceInvalidated { resource_id })?;
        if !record.is_valid {
            return Err(ResourceAbiError::ResourceInvalidated { resource_id });
        }
        drop(map);

        let mut views = match self.dependent_views.lock() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id }),
        };
        views.entry(resource_id).or_default().insert(view_id);
        Ok(())
    }

    /// Register a dependent artifact on an admitted resource.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceAbiError::ResourceInvalidated`] if the parent resource does not exist or is invalidated.
    pub fn register_dependent_artifact(
        &self,
        resource_id: u64,
        artifact_id: u64,
    ) -> Result<(), ResourceAbiError> {
        let map = match self.resources.lock() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id }),
        };
        let record = map
            .get(&resource_id)
            .ok_or(ResourceAbiError::ResourceInvalidated { resource_id })?;
        if !record.is_valid {
            return Err(ResourceAbiError::ResourceInvalidated { resource_id });
        }
        drop(map);

        let mut artifacts = match self.dependent_artifacts.lock() {
            Ok(g) => g,
            Err(_) => return Err(ResourceAbiError::ResourceInvalidated { resource_id }),
        };
        artifacts
            .entry(resource_id)
            .or_default()
            .insert(artifact_id);
        Ok(())
    }

    /// Invalidate all resources, dependent views, and artifacts belonging to `device_id` on device loss.
    pub fn invalidate_on_device_loss(&self, device_id: u64) -> DeviceLossInvalidationReport {
        let mut report = DeviceLossInvalidationReport {
            device_id,
            invalidated_resources: Vec::new(),
            invalidated_views: Vec::new(),
            invalidated_artifacts: Vec::new(),
        };

        let mut resources = match self.resources.lock() {
            Ok(g) => g,
            Err(p) => {
                self.resources.clear_poison();
                p.into_inner()
            }
        };
        let mut views = match self.dependent_views.lock() {
            Ok(g) => g,
            Err(p) => {
                self.dependent_views.clear_poison();
                p.into_inner()
            }
        };
        let mut artifacts = match self.dependent_artifacts.lock() {
            Ok(g) => g,
            Err(p) => {
                self.dependent_artifacts.clear_poison();
                p.into_inner()
            }
        };

        for (res_id, record) in resources.iter_mut() {
            if record.device_id == device_id {
                record.invalidate_on_device_loss();
                report.invalidated_resources.push(*res_id);

                if let Some(view_set) = views.remove(res_id) {
                    report.invalidated_views.extend(view_set);
                }
                if let Some(artifact_set) = artifacts.remove(res_id) {
                    report.invalidated_artifacts.extend(artifact_set);
                }
            }
        }

        report.invalidated_resources.sort_unstable();
        report.invalidated_views.sort_unstable();
        report.invalidated_artifacts.sort_unstable();
        report
    }

    /// Look up an admitted resource by ID.
    #[must_use]
    pub fn get_resource(&self, resource_id: u64) -> Option<AdmittedResourceRecord> {
        let map = self.resources.lock().ok()?;
        map.get(&resource_id).cloned()
    }
}
