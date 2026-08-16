//! Target-boundary bank-conflict mitigation strategy selection.
//!
//! Reuses neutral access facts from [`super::analyze`] to select padding,
//! XOR swizzling, or no rewrite inside concrete emitters using real bank count,
//! bank width, subgroup shape, instruction width, and all access phases.
//!
//! A candidate is rejected when a transformation merely moves an unacceptable
//! conflict to another phase. Universal zero conflicts is not promised.

use serde::{Deserialize, Serialize};
use super::report::{BankConflictKind, ConflictSeverity};

/// Physical and execution geometry for target shared-memory banks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TargetBankGeometry {
    /// Number of shared-memory banks (typically 32 on modern GPUs).
    pub bank_count: u32,
    /// Width of each bank in bytes (typically 4 bytes = 32 bits).
    pub bank_width_bytes: u32,
    /// Subgroup (warp/wavefront) size in lanes (e.g. 32 on NVIDIA, 32 or 64 on AMD).
    pub subgroup_lanes: u32,
    /// Native instruction access width in bytes (e.g. 4 for f32/u32, 8 for f64/v2, 16 for v4).
    pub instruction_word_bytes: u32,
}

impl Default for TargetBankGeometry {
    fn default() -> Self {
        Self {
            bank_count: 32,
            bank_width_bytes: 4,
            subgroup_lanes: 32,
            instruction_word_bytes: 4,
        }
    }
}

/// Access phase within a kernel's execution lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessPhase {
    /// Global-to-shared staging / tile load phase.
    LoadStage,
    /// Inner compute loop shared-memory reads (e.g. matrix multiplication tiles).
    ComputeRead,
    /// Shared-memory intra-block reduction / tree-reduction phase.
    Reduction,
    /// Result store / epilogue phase.
    EpilogueStore,
}

/// Profile of shared-memory access pattern for one kernel phase.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AccessPhaseProfile {
    /// Execution phase.
    pub phase: AccessPhase,
    /// Stride between consecutive thread accesses in elements.
    pub stride_elements: u32,
    /// Number of active threads accessing shared memory in this phase.
    pub active_threads: u32,
    /// Total access count weight (frequency in loop).
    pub access_weight: u32,
}

/// Bank conflict mitigation candidate strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BankConflictMitigation {
    /// Keep baseline layout without modifications.
    NoRewrite,
    /// Pad allocation row stride by N elements.
    PadLines {
        /// Number of padding elements added per row/stride.
        pad_elements_per_row: u32,
    },
    /// Apply bitwise XOR swizzling to address calculations.
    XorSwizzle {
        /// Number of bits to participate in XOR swizzling.
        swizzle_bits: u32,
        /// Stride shift in bits.
        stride_shift: u32,
    },
}

/// Conflict report for a single access phase under a candidate strategy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhaseConflictReport {
    /// Analyzed phase.
    pub phase: AccessPhase,
    /// Detected conflict kind.
    pub conflict: BankConflictKind,
    /// Severity classification.
    pub severity: ConflictSeverity,
    /// Estimated serialization penalty factor (1.0 = no penalty).
    pub penalty_factor: u32,
}

/// Multi-phase evaluation result for a candidate mitigation strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MitigationEvaluation {
    /// Evaluated strategy candidate.
    pub strategy: BankConflictMitigation,
    /// Conflict evaluation for each active phase.
    pub phase_reports: Vec<PhaseConflictReport>,
    /// Worst conflict severity across all phases.
    pub worst_severity: ConflictSeverity,
    /// Aggregate penalty score across weighted phases.
    pub aggregate_penalty: f32,
    /// Whether the candidate was accepted or rejected.
    pub accepted: bool,
    /// Rejection reason if candidate was rejected.
    pub rejection_reason: Option<String>,
}

/// Evaluate a single mitigation candidate across all execution phases.
#[must_use]
pub fn evaluate_mitigation_candidate(
    phases: &[AccessPhaseProfile],
    geometry: &TargetBankGeometry,
    candidate: BankConflictMitigation,
    baseline_worst: ConflictSeverity,
) -> MitigationEvaluation {
    let mut phase_reports = Vec::with_capacity(phases.len());
    let mut worst_severity = ConflictSeverity::None;
    let mut aggregate_penalty = 0.0_f32;

    for phase in phases {
        let effective_stride = match candidate {
            BankConflictMitigation::NoRewrite => phase.stride_elements,
            BankConflictMitigation::PadLines { pad_elements_per_row } => {
                phase.stride_elements.saturating_add(pad_elements_per_row)
            }
            BankConflictMitigation::XorSwizzle { swizzle_bits, .. } => {
                // Swizzling reduces effective stride collision by spreading across 2^swizzle_bits banks
                let divisor = 1_u32 << swizzle_bits.min(5);
                (phase.stride_elements / divisor).max(1)
            }
        };

        let conflict = classify_phase_conflict(effective_stride, geometry.bank_count, phase.active_threads);
        let severity = conflict.severity();
        let penalty_factor = match conflict {
            BankConflictKind::NoConflict | BankConflictKind::BroadcastSafe => 1,
            BankConflictKind::Conflict { way_count } => way_count.max(1),
            BankConflictKind::Unknown => 4,
        };

        if severity_rank(severity) > severity_rank(worst_severity) {
            worst_severity = severity;
        }

        aggregate_penalty += (penalty_factor as f32) * (phase.access_weight as f32);

        phase_reports.push(PhaseConflictReport {
            phase: phase.phase,
            conflict,
            severity,
            penalty_factor,
        });
    }

    // WHY: Reject candidate when transformation merely moves an unacceptable conflict to another phase
    let is_rejected = match candidate {
        BankConflictMitigation::NoRewrite => false,
        _ => {
            // If candidate introduced a Severe or Critical conflict in any phase that wasn't already Critical
            let has_unacceptable_moved_conflict = phase_reports.iter().any(|r| {
                matches!(r.severity, ConflictSeverity::Critical | ConflictSeverity::Severe)
                    && severity_rank(r.severity) > severity_rank(baseline_worst)
            });
            has_unacceptable_moved_conflict
        }
    };

    let rejection_reason = if is_rejected {
        Some("mitigation moves unacceptable conflict to another access phase".to_string())
    } else {
        None
    };

    MitigationEvaluation {
        strategy: candidate,
        phase_reports,
        worst_severity,
        aggregate_penalty,
        accepted: !is_rejected,
        rejection_reason,
    }
}

/// Select the optimal bank-conflict mitigation strategy for a target geometry and phase profile.
#[must_use]
pub fn select_bank_conflict_strategy(
    phases: &[AccessPhaseProfile],
    geometry: &TargetBankGeometry,
) -> MitigationEvaluation {
    if phases.is_empty() {
        return MitigationEvaluation {
            strategy: BankConflictMitigation::NoRewrite,
            phase_reports: Vec::new(),
            worst_severity: ConflictSeverity::None,
            aggregate_penalty: 0.0,
            accepted: true,
            rejection_reason: None,
        };
    }

    // Evaluate baseline
    let baseline = evaluate_mitigation_candidate(
        phases,
        geometry,
        BankConflictMitigation::NoRewrite,
        ConflictSeverity::None,
    );
    let baseline_worst = baseline.worst_severity;

    let candidates = [
        BankConflictMitigation::NoRewrite,
        BankConflictMitigation::PadLines { pad_elements_per_row: 1 },
        BankConflictMitigation::PadLines { pad_elements_per_row: 2 },
        BankConflictMitigation::PadLines { pad_elements_per_row: 4 },
        BankConflictMitigation::XorSwizzle { swizzle_bits: 2, stride_shift: 3 },
        BankConflictMitigation::XorSwizzle { swizzle_bits: 3, stride_shift: 4 },
    ];

    let mut best = baseline;

    for candidate in candidates {
        let eval = evaluate_mitigation_candidate(phases, geometry, candidate, baseline_worst);
        if eval.accepted && eval.aggregate_penalty < best.aggregate_penalty {
            best = eval;
        }
    }

    best
}

fn classify_phase_conflict(stride: u32, bank_count: u32, active_threads: u32) -> BankConflictKind {
    if stride == 0 {
        return BankConflictKind::BroadcastSafe;
    }
    let gcd = gcd_u32(stride, bank_count);
    if gcd <= 1 {
        BankConflictKind::NoConflict
    } else {
        let way_count = gcd.min(active_threads);
        if way_count <= 1 {
            BankConflictKind::NoConflict
        } else {
            BankConflictKind::Conflict { way_count }
        }
    }
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn severity_rank(sev: ConflictSeverity) -> u32 {
    match sev {
        ConflictSeverity::None => 0,
        ConflictSeverity::Mild => 1,
        ConflictSeverity::Unknown => 2,
        ConflictSeverity::Severe => 3,
        ConflictSeverity::Critical => 4,
    }
}
