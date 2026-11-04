//! Contract for `EmitError::retargeted_diagnostic`.
//!
//! WHY: `vyre-emit-metal` and `vyre-emit-spirv` reach naga through this crate
//! and report a naga refusal under their own target. Each one rebuilt the same
//! projection by hand, and a rebuilt copy is free to drop the code, the fix, or
//! the causes that make the diagnostic actionable. This holds the projection to
//! replacing the target and appending one note, for every variant, so a variant
//! added here cannot reach a downstream emitter unprojected.

use vyre_emit_naga::EmitError;
use vyre_lower::{KernelOp, KernelOpKind, WorkgroupLimitViolation};

/// One value per `EmitError` variant.
///
/// The match below carries no catch-all arm, so a variant added to the enum
/// fails to compile here until someone adds a case for it.
fn every_variant() -> Vec<EmitError> {
    let variants = vec![
        EmitError::UnsupportedOp(KernelOp {
            kind: KernelOpKind::Literal,
            operands: Vec::new(),
            result: None,
        }),
        EmitError::UnsupportedCapability("subgroup_ballot"),
        EmitError::UnsupportedWorkgroup(WorkgroupLimitViolation::DimensionExceeded {
            axis: 0,
            actual: 4096,
            limit: 1024,
        }),
        EmitError::NagaConstructionFailed("handle arena exhausted".to_string()),
        EmitError::InvalidBinding {
            slot: 7,
            reason: "slot exceeds the target namespace".to_string(),
        },
        EmitError::InvalidDescriptor("entry point is not compute".to_string()),
    ];
    for variant in &variants {
        match variant {
            EmitError::UnsupportedOp(_)
            | EmitError::UnsupportedCapability(_)
            | EmitError::UnsupportedWorkgroup(_)
            | EmitError::NagaConstructionFailed(_)
            | EmitError::InvalidBinding { .. }
            | EmitError::InvalidDescriptor { .. } => {}
        }
    }
    variants
}

/// WHY: the downstream emitter names the artifact it was producing, and an
/// operator reads that from the target field. A projection that left the naga
/// target in place reported a Metal build failure as a naga failure.
#[test]
fn every_variant_is_reported_under_the_downstream_target_with_its_stage_note() {
    for variant in every_variant() {
        let projected = variant.retargeted_diagnostic("metal", "during Metal emission");
        assert_eq!(
            projected.target.as_deref(),
            Some("metal"),
            "{variant:?} kept a target other than the downstream one"
        );
        assert_eq!(
            projected.notes.last().map(AsRef::as_ref),
            Some("during Metal emission"),
            "{variant:?} did not record the stage note last"
        );
    }
}

/// WHY: the code, the fix and the cause chain are what make a diagnostic
/// actionable, and a hand-rebuilt projection is free to lose them. Comparing
/// the whole diagnostic rather than a field list keeps a field added to
/// `Diagnostic` inside the contract instead of outside it.
#[test]
fn projection_changes_only_the_target_and_appends_one_note() {
    for variant in every_variant() {
        let direct = variant.diagnostic();
        let projected = variant.retargeted_diagnostic("spirv", "during SPIR-V emission");

        let mut expected = direct.clone();
        expected.target = Some("spirv".to_string());
        expected.notes.push("during SPIR-V emission".into());
        assert_eq!(
            projected, expected,
            "{variant:?} changed something other than its target and its notes"
        );
    }
}

/// WHY: two emitters project the same naga failure, and a projection that
/// mutated shared state would let the first caller decide what the second one
/// reports.
#[test]
fn projecting_twice_produces_two_independent_diagnostics() {
    let variant = EmitError::InvalidDescriptor("entry point is not compute".to_string());
    let metal = variant.retargeted_diagnostic("metal", "during Metal emission");
    let spirv = variant.retargeted_diagnostic("spirv", "during SPIR-V emission");

    assert_eq!(metal.target.as_deref(), Some("metal"));
    assert_eq!(spirv.target.as_deref(), Some("spirv"));
    assert_eq!(metal.notes.len(), spirv.notes.len());
    assert_ne!(metal.notes.last(), spirv.notes.last());
}
