//! What a built `Program` declares about itself.
//!
//! One module reads a Program's structure back out for tooling; the other
//! rewrites which of its buffers the host still reads back once two passes are
//! fused. Both answer questions about a finished Program rather than building
//! one, so neither belongs to the dialect that produced it.

pub(crate) mod descriptor;

#[cfg(any(feature = "reduce", feature = "text"))]
pub(crate) mod outputs;
