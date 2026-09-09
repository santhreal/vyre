//! Neural network activations, linear, normalization, attention, MoE, and LLM inference.

#[cfg(feature = "llm")]
pub mod llm;
pub mod nn;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
