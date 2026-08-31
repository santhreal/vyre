//! Substrate-neutral verified lowering for Vyre.
//!
//! `lower_scheduled` applies a validated selected phase to schedule-free
//! `Program` IR and delegates to `lower_physical`, which expands and optimizes
//! already physical IR before building verified physical kernel IR. The result
//! is the only type accepted by megakernel target compilation. Raw descriptor
//! fixtures use `verify_descriptor`, whose bounded canonicalization only orders
//! pure same-body dependencies needed by emitters.
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
/// Byte-stability harness for emitted backend artifacts. Test-only, like
/// `descriptor_builder`: enable `test-fixtures` to reach it.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod artifact_golden;
pub(crate) mod audit;
pub(crate) mod canonicalize;
pub(crate) mod descriptor;
/// Fixture builders for kernel descriptors. Every consumer is a test, so this
/// is not part of the shipped surface: enable `test-fixtures` to reach it.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod descriptor_builder;
pub mod emit_adversarial_corpus;
pub(crate) mod equivalence;
pub(crate) mod error;
mod level_stage;
pub(crate) mod lower;
pub(crate) mod op_facts;
pub mod operand_class;
pub mod pattern_audit;
/// Backend-neutral `Program` corpus shared by byte-stability goldens.
/// Test-only, like `descriptor_builder`: enable `test-fixtures` to reach it.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod program_stability_corpus;
pub(crate) mod result_id_remap;
pub mod rewrites;
pub(crate) mod target;
mod verified_lowering;
pub(crate) mod verify;

pub use audit::{
    audit, audit_with_histogram, PerfAuditReport, Recommendation, RecommendationCategory,
};

/// Verify a raw descriptor, apply bounded representation canonicalization, and
/// verify the emitter-ready result.
///
/// This boundary does not perform semantic optimization. Production semantic
/// `Program` callers use [`lower_scheduled`]; descriptor fixtures and tooling
/// use this function before invoking a pure emitter.
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
///
/// `facts` carries the device capacities the target reported, passed through
/// to [`audit`]. A caller with none passes [`analyses::AnalysisFacts::none`]
/// and the report carries no section that would need one.
#[must_use]
pub fn full_report(desc: &KernelDescriptor, facts: &analyses::AnalysisFacts) -> FullReport {
    let verify = verify::verify(desc);
    let fix_text = build_full_report_fix_text(&verify);
    FullReport {
        descriptor_id: desc.id.clone(),
        summary: desc.summary(),
        histogram: analyses::op_histogram::analyze(desc),
        perf: audit::audit(desc, facts),
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
            messages.push("Fix: descriptor verification returned an empty error list; treat this as a verifier contract bug and preserve the descriptor for triage.".to_string());
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
pub use verify::format_verify_errors;
pub use verify::{verify, VerifyError, VerifyErrorKind, VerifyResult};

pub use descriptor::{
    descriptor_trap_tags, scan_construct_intent_mapping, AsyncTransaction, AsyncTransactionError,
    AsyncWaitSpec, BarrierPhase, BindingLayout, BindingSlot, BindingVisibility, DescriptorIntent,
    DescriptorIntentError, DescriptorIntentEvidence, DescriptorIntentKind, DescriptorIntentSet,
    DescriptorIntentStrategy, DescriptorTrapTag, Dispatch, FragmentOperand, FragmentValue,
    IntentAnnotatedDescriptor, KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue,
    MatrixMmaElement, MatrixMmaLayout, MatrixMmaSpec, MatrixSpecError, MatrixTileShape,
    MemoryClass, MemoryProxyFence, OpaqueExprData, OpaqueNodeData, PhysicalSchedule,
    ScanConstructIntentClass, ScanConstructIntentMapping, StageSlot, StorageLayout,
    StorageLayoutError, StorageLifetime, StorageRegion, TensorAccessMap, TransactionScope,
    DESCRIPTOR_INTENT_SCHEMA_VERSION, PHYSICAL_SCHEDULE_VERSION, SCAN_CONSTRUCT_INTENT_MAPPINGS,
    STORAGE_LAYOUT_VERSION, TRAP_SIDECAR_NAME, TRAP_SIDECAR_WORDS,
};
pub use descriptor::{KernelOpsIter, Name};
pub use equivalence::{check_effects, BindingEffects, EffectSignature, EquivalenceError};
pub use error::LowerError;
pub use level_stage::registered_level_stage;
/// Re-exported so a caller building a `KernelDescriptor` by hand through
/// `descriptor_builder` can place a Shared or Scratch binding in the range
/// `verify` accepts. A shared slot below this value is rejected with
/// `VerifyErrorKind::WorkgroupBindingInHostRange`.
pub use lower::{lower, WORKGROUP_SLOT_BASE};
pub use op_facts::{facts_for, OpFacts};
pub use target::{
    required_subgroup_capabilities, validate_workgroup_size, EmissionTargetCapabilities,
    SubgroupCapabilities, WorkgroupLimitViolation, WorkgroupLimits,
};
pub use verified_lowering::{
    lower_physical, lower_scheduled, PhysicalKernel, PhysicalLowering, PhysicalLoweringError,
};
/// Re-exported so consumers matching/constructing `KernelOpKind::SubgroupReduce`
/// can name the reduction operator without depending on `vyre-foundation`.
pub use vyre_foundation::ir::SubgroupReduceOp;
