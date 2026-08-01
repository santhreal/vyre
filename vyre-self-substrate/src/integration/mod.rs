//! Integration-only substrate modules.
//!
//! These modules consume the platform substrate to answer readiness,
//! coverage, evidence, and release-process questions. They are deliberately
//! separated from the platform substrate modules because they may describe
//! downstream integration surfaces while the platform itself remains
//! consumer-neutral.

pub(crate) fn first_missing_text_evidence<'a>(
    text: &str,
    required: &'a [(&'static str, &str)],
) -> Option<&'static str> {
    required
        .iter()
        .find_map(|(evidence, needle)| (!text.contains(needle)).then_some(*evidence))
}
pub(crate) fn require_text_evidence<E>(
    text: &str,
    required: &[(&'static str, &str)],
    missing: impl FnOnce(&'static str) -> E,
) -> Result<(), E> {
    match first_missing_text_evidence(text, required) {
        Some(evidence) => Err(missing(evidence)),
        None => Ok(()),
    }
}

pub mod coverage;
pub mod evidence;
pub mod quality;
pub mod release;
