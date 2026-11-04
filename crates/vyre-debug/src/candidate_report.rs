//! Search certificate, candidate funnel, and prune reason diagnostics.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use vyre::compiler::SearchCertificate;

/// Formatted candidate inspection report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateReport {
    /// Number of derived candidate productions.
    pub total_derived: usize,
    /// Number of pruned candidate productions.
    pub total_pruned: usize,
    /// Grammar version used during candidate derivation.
    pub grammar_version: u32,
    /// Expansion depth reached.
    pub depth: u32,
    /// Whether budget was exhausted before grammar was exhausted.
    pub budget_exhausted: bool,
    /// Pruned candidate families grouped by reason code and category.
    pub prune_summary: BTreeMap<String, usize>,
    /// Detailed eliminated candidate descriptions.
    pub eliminated_families: Vec<EliminatedFamilyReport>,
}

/// Report on one pruned production family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EliminatedFamilyReport {
    /// Stable prune reason code.
    pub code: String,
    /// Diagnostic message explaining why the family was eliminated.
    pub reason: String,
    /// Production name or grammar identifier.
    pub production: String,
    /// Number of candidates eliminated in this family.
    pub count: u32,
}

impl CandidateReport {
    /// Project candidate search diagnostics from a SearchCertificate.
    #[must_use]
    pub fn from_certificate(cert: &SearchCertificate) -> Self {
        let mut prune_summary = BTreeMap::new();
        let mut eliminated_families = Vec::new();

        let total_derived = cert.derived.iter().map(|d| d.derived as usize).sum();
        let total_pruned = cert.pruned.iter().map(|p| p.count as usize).sum();

        for pruned in &cert.pruned {
            let code = pruned.reason.code().to_string();
            *prune_summary.entry(code.clone()).or_insert(0) += pruned.count as usize;
            eliminated_families.push(EliminatedFamilyReport {
                code,
                reason: pruned.reason.explanation().to_string(),
                production: format!("{:?}", pruned.production),
                count: pruned.count,
            });
        }

        Self {
            total_derived,
            total_pruned,
            grammar_version: cert.grammar_version,
            depth: cert.depth,
            budget_exhausted: cert.budget_exhausted,
            prune_summary,
            eliminated_families,
        }
    }
}

/// Difference between two search certificates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchCertificateDiff {
    /// Candidate count delta.
    pub candidate_count_delta: i64,
    /// Pruned reasons added or removed.
    pub prune_count_deltas: BTreeMap<String, i64>,
    /// Whether both certificates produced identical candidate decisions.
    pub is_identical: bool,
}

/// Structurally diff two search certificates.
#[must_use]
pub fn diff_search_certificates(
    before: &SearchCertificate,
    after: &SearchCertificate,
) -> SearchCertificateDiff {
    let rep1 = CandidateReport::from_certificate(before);
    let rep2 = CandidateReport::from_certificate(after);

    let candidate_count_delta = (rep2.total_derived as i64) - (rep1.total_derived as i64);
    let mut prune_count_deltas = BTreeMap::new();

    let mut all_codes = BTreeMap::new();
    for code in rep1.prune_summary.keys() {
        all_codes.insert(code.clone(), ());
    }
    for code in rep2.prune_summary.keys() {
        all_codes.insert(code.clone(), ());
    }

    for code in all_codes.keys() {
        let count1 = rep1.prune_summary.get(code).copied().unwrap_or(0) as i64;
        let count2 = rep2.prune_summary.get(code).copied().unwrap_or(0) as i64;
        let diff = count2 - count1;
        if diff != 0 {
            prune_count_deltas.insert(code.clone(), diff);
        }
    }

    let is_identical = candidate_count_delta == 0 && prune_count_deltas.is_empty();

    SearchCertificateDiff {
        candidate_count_delta,
        prune_count_deltas,
        is_identical,
    }
}
