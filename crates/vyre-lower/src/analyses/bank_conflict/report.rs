//! Bank-conflict classes and their severity.

use serde::{Deserialize, Serialize};

/// Shared-memory bank access classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BankConflictKind {
    /// Threads in the subgroup access addresses that map to distinct
    /// banks (or the same bank with broadcast semantics on a read).
    /// Full single-cycle throughput.
    NoConflict,
    /// Threads access addresses that map to the same bank but for
    /// reads where hardware broadcast is supported, the same
    /// 32-bit word. Single cycle.
    BroadcastSafe,
    /// All N threads in a subgroup hit the same bank with N distinct
    /// addresses. Worst case  -  N-way serialization.
    Conflict {
        /// Number of accesses serialized through one bank.
        way_count: u32,
    },
    /// Index pattern not classifiable. Default conservative answer
    /// when phase-1 analysis can't prove safety.
    Unknown,
}

/// Severity assigned to a shared-memory bank access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConflictSeverity {
    /// `NoConflict` or `BroadcastSafe`  -  performance is fine.
    None,
    /// `Conflict { way_count: 2..=4 }`  -  typically a 2-4x slowdown
    /// on the affected accesses.
    Mild,
    /// `Conflict { way_count: 5..=15 }`  -  5-15x slowdown.
    Severe,
    /// `Conflict { way_count: 16+ }`  -  full subgroup serialization.
    Critical,
    /// Pattern unknown  -  caller should treat as suspect until phase-2
    /// upgrades the analysis.
    Unknown,
}

impl BankConflictKind {
    /// Classify the performance severity of this access pattern.
    #[must_use]
    pub fn severity(&self) -> ConflictSeverity {
        match self {
            Self::NoConflict | Self::BroadcastSafe => ConflictSeverity::None,
            Self::Conflict { way_count } => match *way_count {
                0..=1 => ConflictSeverity::None,
                2..=4 => ConflictSeverity::Mild,
                5..=15 => ConflictSeverity::Severe,
                _ => ConflictSeverity::Critical,
            },
            Self::Unknown => ConflictSeverity::Unknown,
        }
    }
}

// Inline: covers the crate-private severity mapping.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_conflict_severity_is_none() {
        assert_eq!(
            BankConflictKind::NoConflict.severity(),
            ConflictSeverity::None
        );
    }

    #[test]
    fn broadcast_safe_severity_is_none() {
        assert_eq!(
            BankConflictKind::BroadcastSafe.severity(),
            ConflictSeverity::None
        );
    }

    #[test]
    fn way_2_conflict_is_mild() {
        assert_eq!(
            BankConflictKind::Conflict { way_count: 2 }.severity(),
            ConflictSeverity::Mild
        );
    }

    #[test]
    fn way_4_conflict_is_mild() {
        assert_eq!(
            BankConflictKind::Conflict { way_count: 4 }.severity(),
            ConflictSeverity::Mild
        );
    }

    #[test]
    fn way_8_conflict_is_severe() {
        assert_eq!(
            BankConflictKind::Conflict { way_count: 8 }.severity(),
            ConflictSeverity::Severe
        );
    }

    #[test]
    fn way_15_conflict_is_severe() {
        assert_eq!(
            BankConflictKind::Conflict { way_count: 15 }.severity(),
            ConflictSeverity::Severe
        );
    }

    #[test]
    fn way_16_conflict_is_critical() {
        assert_eq!(
            BankConflictKind::Conflict { way_count: 16 }.severity(),
            ConflictSeverity::Critical
        );
    }

    #[test]
    fn way_32_conflict_is_critical() {
        assert_eq!(
            BankConflictKind::Conflict { way_count: 32 }.severity(),
            ConflictSeverity::Critical
        );
    }

    #[test]
    fn unknown_severity_is_unknown() {
        assert_eq!(
            BankConflictKind::Unknown.severity(),
            ConflictSeverity::Unknown
        );
    }
}
