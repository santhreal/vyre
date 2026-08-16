//! Why a subset construction could not complete.

use std::error::Error;
use std::fmt;

/// Why a subset construction couldn't complete.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum NfaToDfaError {
    /// Subset construction would create more than the caller-supplied
    /// `max_dfa_states` DFA states. State explosion has hit; the
    /// caller's options are raise the cap, shard the pattern set, or
    /// fall back to the NFA scan kernel.
    StateExplosion {
        /// Number of DFA states discovered before the cap was hit.
        produced: usize,
        /// Cap the caller passed.
        cap: usize,
    },
    /// One of the input bit-tables had a length inconsistent with the
    /// declared `num_states`.
    ShapeMismatch {
        /// Which table failed the length cross-check.
        reason: &'static str,
    },
}

impl fmt::Display for NfaToDfaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StateExplosion { produced, cap } => write!(
                formatter,
                "NFA→DFA subset construction exceeded the {cap}-state cap after producing {produced} DFA states. Fix: raise the cap, shard the pattern set, or dispatch via the NFA scan kernel."
            ),
            Self::ShapeMismatch { reason } => {
                write!(formatter, "NFA bit-table shape mismatch: {reason}.")
            }
        }
    }
}

impl Error for NfaToDfaError {}
