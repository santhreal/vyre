//! Substring matching, DFA, NFA, regex scanning pipelines, and bracket matching.

#[cfg(feature = "nfa")]
pub mod nfa;
#[cfg(feature = "pattern")]
pub mod pattern;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
