//! Stable typed identifiers for causal spans, traces, and work units.

use core::fmt;
use serde::{Deserialize, Serialize};

/// Stable typed 64-bit identifier for a causal span.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize,
)]
#[repr(transparent)]
pub struct CausalSpanId(pub u64);

impl CausalSpanId {
    /// Create a new span id from raw integer.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Return the raw integer id.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for CausalSpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "span:{:016x}", self.0)
    }
}

/// Stable 128-bit identifier for a causal trace execution.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize,
)]
pub struct TraceId(pub u128);

impl TraceId {
    /// Create a new trace id from a 128-bit integer.
    pub const fn new(id: u128) -> Self {
        Self(id)
    }

    /// Return the raw 128-bit integer.
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trace:{:032x}", self.0)
    }
}

/// Typed source span reference connecting IR/Region to high-level origin.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub struct SourceSpanRef {
    /// Source file path or logical module name.
    pub file_or_module: String,
    /// Starting line number (1-based).
    pub line_start: u32,
    /// Starting column number (1-based).
    pub col_start: u32,
    /// Ending line number.
    pub line_end: u32,
    /// Ending column number.
    pub col_end: u32,
}

impl fmt::Display for SourceSpanRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}-{}:{}",
            self.file_or_module, self.line_start, self.col_start, self.line_end, self.col_end
        )
    }
}
