//! Cat-A hash compositions.
//!
//! Owns hash compositions that build on more than one intrinsic. Single-kernel
//! checksums and hashes are registered once, in `vyre_primitives::hash`, and are
//! called from here rather than re-registered.

pub mod blake3_compress;

pub use blake3_compress::blake3_compress;
