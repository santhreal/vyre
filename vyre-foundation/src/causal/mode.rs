//! Tracing activation mode and sampling policy.

use serde::{Deserialize, Serialize};

/// Mode of causal trace recording.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalTraceMode {
    /// Disabled. Guarantees zero heap allocation and minimal overhead.
    Off,
    /// Sampled mode: captures 1 out of every `sample_every` requests.
    Sampled {
        /// Sampling interval (e.g. 100 captures 1% of requests).
        sample_every: u32,
    },
    /// Full recording mode: captures every event across all compiler and runtime stages.
    Full,
}

impl Default for CausalTraceMode {
    fn default() -> Self {
        Self::Off
    }
}

impl CausalTraceMode {
    /// Returns true if tracing is completely disabled.
    #[inline(always)]
    pub const fn is_off(self) -> bool {
        matches!(self, Self::Off)
    }

    /// Returns true if tracing is active for a given monotonic sequence counter.
    #[inline]
    pub fn should_trace(self, sequence: u64) -> bool {
        match self {
            Self::Off => false,
            Self::Sampled { sample_every } => {
                if sample_every == 0 {
                    false
                } else {
                    sequence % (sample_every as u64) == 0
                }
            }
            Self::Full => true,
        }
    }
}
