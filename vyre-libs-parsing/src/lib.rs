//! Lexer drivers, LR(1) table walkers, and language-specific AST construction kernels.

pub mod parsing;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
