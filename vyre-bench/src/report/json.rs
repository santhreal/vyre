use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::api::case::{Correctness, PerformanceContract, PerformanceEvaluation};
use crate::api::metric::MetricStats;
use crate::probes::environment::EnvironmentData;

#[derive(Debug, Serialize, Deserialize)]
pub struct ReportSchema {
    pub schema: String,
    pub run_id: String,
    pub suite: String,
    #[serde(default)]
    pub selected_backend: Option<String>,
    pub git: BTreeMap<String, String>,
    #[serde(default)]
    pub source_fingerprint: String,
    #[serde(default)]
    pub source_tree_fingerprint: String,
    pub environment: EnvironmentData,
    pub features: Vec<String>,
    pub cases: Vec<CaseReport>,
    pub summary: ReportSummary,
    #[serde(default)]
    pub blockers: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CaseReport {
    pub id: String,
    #[serde(default)]
    pub workload_fingerprint: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub owner_crate: String,
    #[serde(default)]
    pub workload_class: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub backend_id: Option<String>,
    #[serde(default)]
    pub needs_gpu: bool,
    #[serde(default)]
    pub min_vram_bytes: Option<u64>,
    #[serde(default)]
    pub min_input_bytes: Option<u64>,
    #[serde(default)]
    pub required_features: Vec<String>,
    pub status: String,
    pub wall_ns: Option<f64>,
    pub correctness: Correctness,
    pub contract: Option<PerformanceContract>,
    pub performance: Option<PerformanceEvaluation>,
    pub metrics: BTreeMap<String, MetricStats>,
    #[serde(default)]
    pub optimization_passes_applied: Vec<String>,
    pub artifacts: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_cases: usize,
    pub passed: usize,
    pub failed: usize,
    pub total_time_ns: u64,
    pub cache_hit_rate: Option<f64>,
}

impl CaseReport {
    pub fn passes_summary_evidence(&self) -> bool {
        self.status == "pass"
            && !matches!(self.correctness, Correctness::Invalid { .. })
            && !self
                .performance
                .as_ref()
                .is_some_and(|performance| !performance.contract_passed)
    }

    pub fn evidence_blockers(&self) -> Vec<String> {
        let mut blockers = Vec::new();
        if self.status != "pass" {
            blockers.push(format!("case `{}` status `{}`", self.id, self.status));
        }
        if let Correctness::Invalid { reason } = &self.correctness {
            blockers.push(format!("case `{}` correctness invalid: {reason}", self.id));
        }
        if let Some(performance) = &self.performance {
            if !performance.contract_passed {
                if performance.violations.is_empty() {
                    blockers.push(format!(
                        "case `{}` failed its performance contract without a violation reason",
                        self.id
                    ));
                } else {
                    for violation in &performance.violations {
                        blockers.push(format!(
                            "case `{}` performance contract failed: {violation}",
                            self.id
                        ));
                    }
                }
            }
        }
        blockers
    }
}

impl ReportSchema {
    pub fn evidence_summary_counts(&self) -> (usize, usize) {
        let passed = self
            .cases
            .iter()
            .filter(|case| case.passes_summary_evidence())
            .count();
        (passed, self.cases.len().saturating_sub(passed))
    }

    pub fn validate_summary_evidence(&self) -> Result<(), String> {
        if self.summary.total_cases != self.cases.len() {
            return Err(format!(
                "summary.total_cases={} does not match {} case report(s). Fix: regenerate the benchmark report from case evidence.",
                self.summary.total_cases,
                self.cases.len()
            ));
        }
        let (passed, failed) = self.evidence_summary_counts();
        if self.summary.passed != passed || self.summary.failed != failed {
            return Err(format!(
                "summary pass/fail ({}/{}) contradicts case evidence ({}/{}). Fix: regenerate the benchmark report from case status, correctness, and performance contracts.",
                self.summary.passed, self.summary.failed, passed, failed
            ));
        }
        Ok(())
    }

    pub fn validate_blocker_evidence(&self) -> Result<(), String> {
        let derived = self.derived_blockers();
        if self.blockers != derived {
            return Err(format!(
                "top-level blockers {:?} contradict case-derived blockers {:?}. Fix: regenerate the benchmark report from case status, correctness, and performance contracts.",
                self.blockers, derived
            ));
        }
        Ok(())
    }

    pub fn derived_blockers(&self) -> Vec<String> {
        self.cases
            .iter()
            .flat_map(CaseReport::evidence_blockers)
            .collect()
    }
}

pub fn generate_json_report(report: &ReportSchema) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case_report(
        status: &str,
        correctness: Correctness,
        performance: Option<PerformanceEvaluation>,
    ) -> CaseReport {
        CaseReport {
            id: "release.condition_eval.1m".to_string(),
            workload_fingerprint: "bench-case:release.condition_eval.1m".to_string(),
            name: "release condition eval".to_string(),
            owner_crate: "vyre-bench".to_string(),
            workload_class: "Release".to_string(),
            tags: Vec::new(),
            backend_id: Some("cuda".to_string()),
            needs_gpu: true,
            min_vram_bytes: None,
            min_input_bytes: None,
            required_features: Vec::new(),
            status: status.to_string(),
            wall_ns: Some(1.0),
            correctness,
            contract: None,
            performance,
            metrics: BTreeMap::new(),
            optimization_passes_applied: Vec::new(),
            artifacts: Vec::new(),
        }
    }

    fn performance(contract_passed: bool) -> PerformanceEvaluation {
        PerformanceEvaluation {
            speedup_x: Some(100.0),
            contract_passed,
            violations: if contract_passed {
                Vec::new()
            } else {
                vec!["speedup below release floor".to_string()]
            },
        }
    }

    #[test]
    fn summary_pass_requires_pass_status_valid_correctness_and_contract() {
        assert!(
            case_report("pass", Correctness::Exact, Some(performance(true)))
                .passes_summary_evidence(),
            "Fix: valid pass evidence should still count as a passed benchmark case."
        );

        for rejected in [
            case_report("failed", Correctness::Exact, Some(performance(true))),
            case_report(
                "pass",
                Correctness::Invalid {
                    reason: "CUDA/WGPU output mismatch at row 17".to_string(),
                },
                Some(performance(true)),
            ),
            case_report("pass", Correctness::Exact, Some(performance(false))),
            case_report("unstable", Correctness::Exact, Some(performance(true))),
            case_report(
                "thermal_unstable",
                Correctness::Exact,
                Some(performance(true)),
            ),
        ] {
            assert!(
                !rejected.passes_summary_evidence(),
                "Fix: summary.passed must not count failed, invalid, contract-failed, or unstable case evidence: {rejected:?}"
            );
        }
    }
}
