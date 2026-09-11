//! Lexer drivers, LR(1) table walkers, and language-specific AST construction kernels.

pub mod parsing;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
