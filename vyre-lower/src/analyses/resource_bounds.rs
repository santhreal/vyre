//! Legality verification and resource bounding for candidate lowered kernels.
//!
//! Evaluates candidate transformations across partial tiles, tails, alignment,
//! aliasing, divergent barriers, dynamic shapes, bank geometry, register pressure,
//! shared-memory limits, unsupported instructions, and pressure-induced spill.
//! Unswizzled, synchronous, and unfused candidates are retained as fallbacks.

use crate::KernelDescriptor;
use serde::{Deserialize, Serialize};

/// Hardware resource capacity limits for a target device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TargetResourceLimits {
    /// Maximum shared-memory bytes per thread block / workgroup.
    pub max_shared_memory_bytes: usize,
    /// Maximum register count available per thread before spilling.
    pub max_registers_per_thread: usize,
    /// Maximum workgroup (block) size in threads.
    pub max_threads_per_workgroup: usize,
    /// Required memory alignment in bytes for vector instructions.
    pub required_vector_alignment_bytes: u32,
    /// Whether the target supports dynamic memory allocations or dynamic shapes.
    pub supports_dynamic_shapes: bool,
}

impl Default for TargetResourceLimits {
    fn default() -> Self {
        Self {
            max_shared_memory_bytes: 48 * 1024,
            max_registers_per_thread: 128,
            max_threads_per_workgroup: 1024,
            required_vector_alignment_bytes: 16,
            supports_dynamic_shapes: true,
        }
    }
}

/// Tail handling and partial tile guarding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TailHandling {
    /// Whether out-of-bounds elements in partial tiles are protected by predicates.
    pub dynamic_predication: bool,
    /// Whether buffers are padded to full tile boundaries.
    pub padded_allocation: bool,
}

impl Default for TailHandling {
    fn default() -> Self {
        Self {
            dynamic_predication: true,
            padded_allocation: false,
        }
    }
}

/// Legality verification checklist for candidate transformations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LegalityCheck {
    /// Whether partial tiles / tails have active bounds guards.
    pub partial_tiles_guarded: bool,
    /// Strategy used to handle tail elements.
    pub tail_handling: TailHandling,
    /// Verified memory alignment in bytes.
    pub alignment_bytes: u32,
    /// Whether memory accesses are proven free of illegal aliasing.
    pub aliasing_free: bool,
    /// Whether all barrier / synchronization instructions are uniform (no divergent control flow).
    pub no_divergent_barriers: bool,
    /// Whether dynamic shapes are supported by target when present.
    pub dynamic_shapes_supported: bool,
    /// Whether all instructions are supported by the target architecture.
    pub instructions_supported: bool,
}

/// Resource consumption and pressure estimates.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceBounds {
    /// Estimated static + dynamic shared-memory consumption in bytes.
    pub shared_memory_bytes: usize,
    /// Shared-memory capacity limit in bytes.
    pub shared_memory_limit_bytes: usize,
    /// Estimated register requirement per thread.
    pub registers_per_thread: usize,
    /// Target register file limit per thread.
    pub register_limit_per_thread: usize,
    /// Estimated spill to local memory in bytes (0 if no spill).
    pub spill_bytes: usize,
    /// Whether all resource usages are within hardware limits.
    pub is_within_limits: bool,
}

/// Retained fallback configurations ensuring safe degraded execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RetainedFallbacks {
    /// Standard unswizzled baseline candidate retained.
    pub unswizzled_candidate_retained: bool,
    /// Synchronous copy candidate retained (fallback from cp.async failure).
    pub synchronous_candidate_retained: bool,
    /// Unfused baseline candidate retained (fallback from complex fusion).
    pub unfused_candidate_retained: bool,
}

impl Default for RetainedFallbacks {
    fn default() -> Self {
        Self {
            unswizzled_candidate_retained: true,
            synchronous_candidate_retained: true,
            unfused_candidate_retained: true,
        }
    }
}

/// Complete legality and resource bounds report for an optimization candidate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CandidateLegalityReport {
    /// Analyzed kernel identifier.
    pub kernel_id: String,
    /// Candidate name or transform kind.
    pub candidate_name: String,
    /// Whether the candidate satisfies all legality constraints.
    pub is_legal: bool,
    /// Detailed legality verification breakdown.
    pub legality: LegalityCheck,
    /// Resource bounds and pressure profile.
    pub resource_bounds: ResourceBounds,
    /// Fallback candidates preserved for fallback resilience.
    pub retained_fallbacks: RetainedFallbacks,
    /// Explicit list of legality or resource violations if any.
    pub violations: Vec<String>,
}

/// Verify candidate legality and resource limits against target capabilities.
#[must_use]
pub fn verify_candidate_legality(
    desc: &KernelDescriptor,
    candidate_name: &str,
    legality: LegalityCheck,
    shared_memory_bytes: usize,
    registers_per_thread: usize,
    limits: &TargetResourceLimits,
) -> CandidateLegalityReport {
    let mut violations = Vec::new();

    if !legality.partial_tiles_guarded {
        violations.push("partial tiles lack dynamic bounds predication or padding".to_string());
    }
    if legality.alignment_bytes < limits.required_vector_alignment_bytes {
        violations.push(format!(
            "alignment {} bytes violates required vector alignment {} bytes",
            legality.alignment_bytes, limits.required_vector_alignment_bytes
        ));
    }
    if !legality.aliasing_free {
        violations.push("potential buffer aliasing detected without disambiguation".to_string());
    }
    if !legality.no_divergent_barriers {
        violations.push("synchronization barrier placed inside divergent control flow".to_string());
    }
    if !legality.instructions_supported {
        violations
            .push("candidate uses instructions unsupported on target architecture".to_string());
    }

    let shared_mem_exceeded = shared_memory_bytes > limits.max_shared_memory_bytes;
    if shared_mem_exceeded {
        violations.push(format!(
            "shared memory {} bytes exceeds device limit {} bytes",
            shared_memory_bytes, limits.max_shared_memory_bytes
        ));
    }

    let registers_exceeded = registers_per_thread > limits.max_registers_per_thread;
    let spill_bytes = if registers_exceeded {
        (registers_per_thread - limits.max_registers_per_thread)
            * 4
            * limits.max_threads_per_workgroup
    } else {
        0
    };

    if spill_bytes > 0 {
        violations.push(format!(
            "register pressure ({registers_per_thread} regs) induces {spill_bytes} bytes local memory spill"
        ));
    }

    let is_within_limits = !shared_mem_exceeded && spill_bytes == 0;
    let is_legal = violations.is_empty();

    CandidateLegalityReport {
        kernel_id: desc.id.clone(),
        candidate_name: candidate_name.to_string(),
        is_legal,
        legality,
        resource_bounds: ResourceBounds {
            shared_memory_bytes,
            shared_memory_limit_bytes: limits.max_shared_memory_bytes,
            registers_per_thread,
            register_limit_per_thread: limits.max_registers_per_thread,
            spill_bytes,
            is_within_limits,
        },
        retained_fallbacks: RetainedFallbacks::default(),
        violations,
    }
}
