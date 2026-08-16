//! What a built `Program` declares about itself.
//!
//! One module reads a Program's structure back out for tooling, one rewrites
//! which of its buffers the host still reads back once two passes are fused,
//! and one records which composition selected it. All three answer questions
//! about a finished Program rather than building one, so none of them belongs
//! to the dialect that produced it.

#[cfg(any(feature = "math-scan", feature = "llm"))]
pub(crate) mod attribution;
pub(crate) mod descriptor;

// `text` is not named: it enables `reduce`, and `reduce` is what the module's
// callers there compile under.
#[cfg(any(feature = "reduce", feature = "llm"))]
pub(crate) mod outputs;
