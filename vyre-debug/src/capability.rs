//! Classification and boundaries for neutral vs target-specific debug capabilities.
//!
//! Neutral inspection (descriptor dumps, diffs, source assignments, loop carriers,
//! dangling references, and artifact reports) is available without concrete emitters.
//! Naga/WGSL tracing is an explicit target inspection capability. Binary emitters
//! are inspected via structured byte/payload metadata rather than forced text disassembly.

use serde::{Deserialize, Serialize};

/// Debug capability category.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DebugCapabilityKind {
    /// Pure neutral IR / descriptor analysis (requires no emitter or GPU runtime).
    NeutralDescriptor,
    /// Pure neutral artifact container diagnostics.
    NeutralArtifact,
    /// Target-specific Naga / WGSL inspection.
    TargetNagaWgsl,
    /// Target-specific binary payload inspection (PTX, SPIR-V, Metal) via structured metadata.
    TargetBinaryPayload {
        /// Target payload format identifier.
        format: String,
    },
}

/// Metadata description for one debug capability.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DebugCapabilityInfo {
    /// Capability name.
    pub name: &'static str,
    /// Category classification.
    pub kind: DebugCapabilityKind,
    /// Whether this capability requires concrete target emitter linkage.
    pub requires_target_emitter: bool,
    /// Whether this capability operates on binary payloads without text disassembly.
    pub is_binary_safe: bool,
}

/// Available debug capabilities in `vyre-debug`.
pub const DEBUG_CAPABILITIES: &[DebugCapabilityInfo] = &[
    DebugCapabilityInfo {
        name: "descriptor_dump",
        kind: DebugCapabilityKind::NeutralDescriptor,
        requires_target_emitter: false,
        is_binary_safe: true,
    },
    DebugCapabilityInfo {
        name: "descriptor_diff",
        kind: DebugCapabilityKind::NeutralDescriptor,
        requires_target_emitter: false,
        is_binary_safe: true,
    },
    DebugCapabilityInfo {
        name: "carrier_summary",
        kind: DebugCapabilityKind::NeutralDescriptor,
        requires_target_emitter: false,
        is_binary_safe: true,
    },
    DebugCapabilityInfo {
        name: "find_dangling_refs",
        kind: DebugCapabilityKind::NeutralDescriptor,
        requires_target_emitter: false,
        is_binary_safe: true,
    },
    DebugCapabilityInfo {
        name: "source_assignments",
        kind: DebugCapabilityKind::NeutralDescriptor,
        requires_target_emitter: false,
        is_binary_safe: true,
    },
    DebugCapabilityInfo {
        name: "artifact_report",
        kind: DebugCapabilityKind::NeutralArtifact,
        requires_target_emitter: false,
        is_binary_safe: true,
    },
    DebugCapabilityInfo {
        name: "dump_wgsl",
        kind: DebugCapabilityKind::TargetNagaWgsl,
        requires_target_emitter: true,
        is_binary_safe: false,
    },
    DebugCapabilityInfo {
        name: "dump_naga_module",
        kind: DebugCapabilityKind::TargetNagaWgsl,
        requires_target_emitter: true,
        is_binary_safe: false,
    },
    DebugCapabilityInfo {
        name: "failure_trace",
        kind: DebugCapabilityKind::TargetNagaWgsl,
        requires_target_emitter: true,
        is_binary_safe: false,
    },
];

/// Returns all capabilities that are completely neutral and require no emitter.
#[must_use]
pub fn neutral_debug_capabilities() -> Vec<&'static DebugCapabilityInfo> {
    DEBUG_CAPABILITIES
        .iter()
        .filter(|c| !c.requires_target_emitter)
        .collect()
}
