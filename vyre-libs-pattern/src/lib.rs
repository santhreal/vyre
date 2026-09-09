//! Substring matching, DFA, NFA, regex scanning pipelines, and bracket matching.

pub mod pattern;
pub mod nfa;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
