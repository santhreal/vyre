//! Neural network activations, linear, normalization, attention, MoE, and LLM inference.

#[cfg(feature = "llm")]
pub mod llm;
pub mod nn;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
