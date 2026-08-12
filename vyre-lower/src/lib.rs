#![allow(
    clippy::doc_lazy_continuation,
    clippy::double_must_use,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::match_like_matches_macro,
    clippy::redundant_closure
)]
//! Substrate-neutral verified lowering for Vyre.
//!
//! `lower_verified` runs the canonical semantic `Program` optimizer once,
//! expands representation-only constructs, builds a neutral
//! `KernelDescriptor`, and verifies the result. Raw descriptor fixtures use
//! `verify_descriptor`, whose bounded canonicalization only orders pure
//! same-body dependencies needed by emitters.
//!
//! ```text
//! vyre-foundation Program
//!         ↓ semantic optimize once
//! verified KernelDescriptor
//!         ↓
//! concrete emitter strategy
//!         ↓
//! backend artifact
//! ```
//!
//! `KernelDescriptor` carries binding layout, dispatch shape, and a lowered
//! kernel body. Descriptor analyses are read-only. Semantic rewrites belong in
//! `vyre-foundation`; target strategy belongs in concrete emitters and drivers.

pub mod analyses;
pub mod audit;
mod canonicalize;
pub mod descriptor;
pub mod emit_adversarial_corpus;
pub mod error;
mod lower;
pub(crate) mod op_properties;
pub(crate) mod operand_semantics;
mod pre_emit;
pub mod target;
pub mod verify;

pub use audit::{
    audit, audit_with_histogram, PerfAuditReport, Recommendation, RecommendationCategory,
};

/// Verify a raw descriptor, apply bounded representation canonicalization, and
/// verify the emitter-ready result.
///
/// This boundary does not perform semantic optimization. Production `Program`
/// callers use [`lower_verified`]; descriptor fixtures and tooling use this
/// function before invoking a pure emitter.
pub fn verify_descriptor(desc: &KernelDescriptor) -> Result<KernelDescriptor, VerifyFailure> {
    if let Err(errors) = verify::verify(desc) {
        return Err(VerifyFailure::Input(errors));
    }
    let canonical = canonicalize::canonicalize_for_emit(desc);
    if let Err(errors) = verify::verify(&canonical) {
        return Err(VerifyFailure::Output(errors));
    }
    Ok(canonical)
}

/// Which descriptor verification step failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyFailure {
    /// The descriptor was invalid before canonicalization.
    Input(Vec<verify::VerifyError>),
    /// Bounded representation canonicalization produced an invalid descriptor.
    Output(Vec<verify::VerifyError>),
}

impl VerifyFailure {
    /// Return the verifier errors carried by either failure stage.
    pub fn errors(&self) -> &[verify::VerifyError] {
        match self {
            VerifyFailure::Input(e) | VerifyFailure::Output(e) => e,
        }
    }
}

/// Run every read-only descriptor analysis and verification in one call.
#[must_use]
pub fn full_report(desc: &KernelDescriptor) -> FullReport {
    let verify = verify::verify(desc);
    let fix_text = build_full_report_fix_text(&verify);
    FullReport {
        descriptor_id: desc.id.clone(),
        summary: desc.summary(),
        histogram: analyses::op_histogram::analyze(desc),
        perf: audit::audit(desc),
        verify,
        fix_text,
    }
}

/// Read-only descriptor analysis bundle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FullReport {
    #[serde(default)]
    /// Stable descriptor identifier.
    pub descriptor_id: String,
    /// Descriptor summary.
    pub summary: String,
    /// Input operation histogram.
    pub histogram: analyses::op_histogram::OpHistogram,
    /// Input performance audit.
    pub perf: PerfAuditReport,
    /// Descriptor verification result.
    pub verify: verify::VerifyResult,
    #[serde(default)]
    /// Actionable verification failure guidance.
    pub fix_text: String,
}

impl FullReport {
    /// Return the stable verification status label.
    #[must_use]
    pub fn verify_status(&self) -> &'static str {
        verification_status(&self.verify)
    }

    /// One-line headline drawn from the underlying parts. Useful for
    /// log lines.
    pub fn format_short(&self) -> String {
        format!(
            "{} | id {} | {} | {} | verify {}",
            self.summary,
            self.descriptor_id,
            self.histogram.format_short(),
            self.perf.format_short(),
            self.verify_status(),
        )
    }

    /// Multi-line human-readable view, suitable for `--verbose` CLI
    /// output. Each section has a header and is indented for readability.
    pub fn format_long(&self) -> String {
        let mut out = String::new();
        use std::fmt::Write as _;
        let _ = writeln!(out, "Kernel:");
        let _ = writeln!(out, "  descriptor id: {}", self.descriptor_id);
        let _ = writeln!(out, "  summary: {}", self.summary);
        let _ = writeln!(out, "Histogram:");
        let _ = writeln!(out, "  {}", self.histogram.format_short());
        if let Some((cat, n)) = self.histogram.dominant() {
            let _ = writeln!(out, "  dominant: {cat} ({n})");
        }
        let _ = writeln!(out, "Perf audit:");
        let _ = writeln!(out, "  {}", self.perf.format_short());
        for r in &self.perf.recommendations {
            let _ = writeln!(
                out,
                "  - [p{}] {:?}: {} (≤{:.2}× speedup)",
                r.priority, r.category, r.message, r.estimated_speedup_upper_bound
            );
        }
        let _ = writeln!(out, "Verify:");
        write_verify_section(&mut out, &self.verify);
        if !self.fix_text.is_empty() {
            let _ = writeln!(out, "Fix:");
            let _ = writeln!(out, "  {}", self.fix_text);
        }
        out
    }
}

fn verification_status(result: &verify::VerifyResult) -> &'static str {
    if result.is_ok() {
        "OK"
    } else {
        "FAIL"
    }
}

fn write_verify_section(out: &mut String, result: &verify::VerifyResult) {
    use std::fmt::Write as _;
    match result {
        Ok(()) => {
            let _ = writeln!(out, "  OK");
        }
        Err(errs) => {
            let _ = writeln!(out, "  FAIL ({} errors)", errs.len());
            for e in errs {
                let _ = writeln!(out, "    {:?}", e);
            }
        }
    }
}

fn build_full_report_fix_text(verification: &verify::VerifyResult) -> String {
    let mut messages = Vec::new();
    push_verify_fix_text(verification, &mut messages);
    messages.join(" ")
}

fn push_verify_fix_text(result: &verify::VerifyResult, messages: &mut Vec<String>) {
    if let Err(errs) = result {
        if errs.is_empty() {
            messages.push(format!(
                "Fix: descriptor verification returned an empty error list; treat this as a verifier contract bug and preserve the descriptor for triage."
            ));
        } else {
            messages.push(format!(
                "Fix: descriptor verification failed with {} error(s); repair the descriptor before emission. First error: {:?}",
                errs.len(),
                errs[0]
            ));
        }
    }
}

impl std::fmt::Display for FullReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.format_short())
    }
}
pub use verify::{verify, VerifyError, VerifyErrorKind, VerifyResult};

pub use descriptor::{
    scan_construct_intent_mapping, BindingLayout, BindingSlot, BindingVisibility, DescriptorIntent,
    DescriptorIntentError, DescriptorIntentEvidence, DescriptorIntentKind, DescriptorIntentSet,
    DescriptorIntentStrategy, Dispatch, IntentAnnotatedDescriptor, KernelBody, KernelDescriptor,
    KernelOp, KernelOpKind, LiteralValue, MatrixMmaElement, MatrixMmaLayout, MatrixMmaShape,
    MemoryClass, OpaqueExprData, OpaqueNodeData, ScanConstructIntentClass,
    ScanConstructIntentMapping, DESCRIPTOR_INTENT_SCHEMA_VERSION, SCAN_CONSTRUCT_INTENT_MAPPINGS,
    TRAP_SIDECAR_NAME, TRAP_SIDECAR_WORDS,
};
pub use error::LowerError;
pub use pre_emit::{lower_verified, LowerVerifiedError, VerifiedLowering};
pub use target::{
    required_subgroup_capabilities, validate_workgroup_size, EmissionTargetCapabilities,
    SubgroupCapabilities, WorkgroupLimitViolation, WorkgroupLimits,
};
/// Re-exported so consumers matching/constructing `KernelOpKind::SubgroupReduce`
/// can name the reduction operator without depending on `vyre-foundation`.
pub use vyre_foundation::ir::SubgroupReduceOp;

#[cfg(test)]
mod verify_descriptor_tests {
    use super::*;

    #[test]
    fn valid_input_returns_descriptor_directly() {
        let desc = KernelDescriptor {
            id: "k".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                }],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(7)],
            },
        };
        let out = verify_descriptor(&desc).unwrap();
        assert_eq!(out, desc);
    }

    #[test]
    fn invalid_input_returns_input_failure() {
        // Descriptor with zero workgroup_size dim  -  caught by verify.
        let desc = KernelDescriptor {
            id: "bad".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(0, 1, 1),
            body: KernelBody {
                ops: vec![],
                child_bodies: vec![],
                literals: vec![],
            },
        };
        let r = verify_descriptor(&desc);
        assert!(matches!(r, Err(VerifyFailure::Input(_))));
    }

    #[test]
    fn full_report_runs_read_only_analyses() {
        let desc = KernelDescriptor {
            id: "fr".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(0),
                    },
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(1),
                    },
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(7)],
            },
        };
        let report = full_report(&desc);
        assert_eq!(report.descriptor_id, "fr");
        assert!(report.summary.contains("fr:"));
        assert_eq!(report.histogram.literal, 2);
        assert_eq!(report.perf.kernel_id, "fr");
        assert!(report.verify.is_ok());
        assert_eq!(report.verify_status(), "OK");
        assert!(report.fix_text.is_empty());
        let rendered = format!("{report}");
        assert!(rendered.contains("fr:"));
        assert!(rendered.contains("id fr"));
        assert!(rendered.contains("OK"));
    }

    #[test]
    fn full_report_serializes_to_json() {
        let desc = KernelDescriptor {
            id: "fr".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                }],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(7)],
            },
        };
        let report = full_report(&desc);
        assert_eq!(report.descriptor_id, "fr");
        let json = serde_json::to_string(&report).expect("Fix: serialize");
        assert!(json.contains("\"descriptor_id\""));
        assert!(json.contains("\"summary\""));
        assert!(json.contains("\"histogram\""));
        assert!(json.contains("\"perf\""));
        assert!(json.contains("\"verify\""));
        assert!(json.contains("\"fix_text\""));

        // Round-trip back through Deserialize.
        let _back: FullReport = serde_json::from_str(&json).expect("Fix: round-trip");
    }

    #[test]
    fn full_report_format_long_includes_all_sections() {
        let desc = KernelDescriptor {
            id: "fr".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                }],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(7)],
            },
        };
        let r = full_report(&desc);
        let long = r.format_long();
        assert!(long.contains("Kernel:"));
        assert!(long.contains("descriptor id: fr"));
        assert!(long.contains("Histogram:"));
        assert!(long.contains("Perf audit:"));
        assert!(long.contains("Verify:"));
        assert!(long.contains("OK"));
    }

    #[test]
    fn full_report_records_verify_fix_text_for_bad_descriptor() {
        let desc = KernelDescriptor {
            id: "bad".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(0, 1, 1),
            body: KernelBody {
                ops: vec![],
                child_bodies: vec![],
                literals: vec![],
            },
        };
        let report = full_report(&desc);
        assert_eq!(report.descriptor_id, "bad");
        assert_eq!(report.verify_status(), "FAIL");
        assert!(
            report
                .fix_text
                .contains("Fix: descriptor verification failed"),
            "Fix: invalid descriptor reports must carry operator-actionable verifier repair text."
        );
        let long = report.format_long();
        assert!(long.contains("Verify:"));
        assert!(long.contains("Fix:"));
    }

    #[test]
    fn errors_accessor_yields_underlying() {
        let desc = KernelDescriptor {
            id: "bad".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(0, 1, 1),
            body: KernelBody {
                ops: vec![],
                child_bodies: vec![],
                literals: vec![],
            },
        };
        let f = verify_descriptor(&desc).unwrap_err();
        assert_ne!(f.errors().len(), 0);
    }
}
