//! WHY: owner-local reference failures must retain specific corrective action instead of
//! erasing it behind a generic cross-domain error. These checks do not classify individual
//! interpreter operations; they protect formatting at the public error boundary.

use vyre_reference::ReferenceError;

#[test]
fn preserves_specific_recovery_guidance_once() {
    let error =
        ReferenceError::new("unsupported subgroup width. Fix: use a power-of-two subgroup width");
    let rendered = error.to_string();

    assert_eq!(
        rendered,
        "vyre reference interpreter: unsupported subgroup width. Fix: use a power-of-two subgroup width"
    );
    assert_eq!(rendered.matches("Fix:").count(), 1);
}

#[test]
fn supplies_recovery_guidance_when_caller_has_none() {
    let rendered = ReferenceError::new("input buffer is too short").to_string();

    assert_eq!(
        rendered,
        "vyre reference interpreter: input buffer is too short. Fix: validate the Program and input buffer set before invoking the reference backend."
    );
}
