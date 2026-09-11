//! Base64, hex, DEFLATE, and encodex data decoding and decompression compositions.

#[cfg(feature = "decode")]
pub mod decode;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
