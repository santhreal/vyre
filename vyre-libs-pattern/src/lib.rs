//! Substring matching, DFA, NFA, regex scanning pipelines, and bracket matching.

#[cfg(feature = "nfa")]
pub mod nfa;
#[cfg(feature = "pattern")]
pub mod pattern;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
