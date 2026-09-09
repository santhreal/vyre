//! Base64, hex, DEFLATE, and encodex data decoding and decompression compositions.

pub mod decode;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
