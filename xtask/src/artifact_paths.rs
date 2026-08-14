//! Paths of the release evidence artifacts more than one subcommand names.
//!
//! A writer and its readers must agree on the path, so the literal lives here
//! once instead of being retyped at each end.

/// Leaderboard the frontier benchmark writes and the release gate reads back.
pub const FRONTIER_LEADERBOARD_ARTIFACT: &str =
    "release/evidence/benchmarks/frontier-leaderboard.json";

/// Duplicate registered operations the lego audit records.
pub const REGISTERED_OP_DUPLICATES_ARTIFACT: &str =
    "release/evidence/dedup/registered-op-duplicates.json";
/// Duplicate source families the lego audit records.
pub const LEGO_AUDIT_DUPLICATES_ARTIFACT: &str =
    "release/evidence/dedup/lego-audit-duplicates.json";
