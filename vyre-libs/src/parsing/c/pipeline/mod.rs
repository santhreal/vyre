//! Reusable C source pipeline stages and source-cache support.

/// Content-keyed source pipeline cache.
pub mod source_cache;
/// Named GPU stages for embedders (`c11_lexer`, preprocess, …).
pub mod stages;
