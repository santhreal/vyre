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
    #[serde(default)]
    pub backend_profile: Option<ReportBackendProfile>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReportBackendProfile {
    pub backend: String,
    pub timing_quality: String,
    pub supports_device_timestamps: bool,
    pub supports_hardware_counters: bool,
    pub supports_subgroup_ops: bool,
    pub supports_indirect_dispatch: bool,
    pub max_workgroup_size: [u32; 3],
    pub max_invocations_per_workgroup: u32,
    pub max_shared_memory_bytes: u32,
    pub max_storage_buffer_binding_size: u64,
    pub subgroup_size: u32,
    pub compute_units: u32,
    pub mem_bw_gbps: u32,
}

impl ReportBackendProfile {
    #[must_use]
    pub fn from_device_profile(profile: vyre_driver::DeviceProfile) -> Self {
        Self {
            backend: profile.backend.to_string(),
            timing_quality: profile.timing_quality.as_str().to_string(),
            supports_device_timestamps: profile.supports_device_timestamps,
            supports_hardware_counters: profile.supports_hardware_counters,
            supports_subgroup_ops: profile.supports_subgroup_ops,
            supports_indirect_dispatch: profile.supports_indirect_dispatch,
            max_workgroup_size: profile.max_workgroup_size,
            max_invocations_per_workgroup: profile.max_invocations_per_workgroup,
            max_shared_memory_bytes: profile.max_shared_memory_bytes,
            max_storage_buffer_binding_size: profile.max_storage_buffer_binding_size,
            subgroup_size: profile.subgroup_size,
            compute_units: profile.compute_units,
            mem_bw_gbps: profile.mem_bw_gbps,
        }
    }

    #[must_use]
    pub fn has_valid_timing_quality(&self) -> bool {
        matches!(
            self.timing_quality.as_str(),
            "host_only" | "host_enqueue_wait" | "device_timestamps" | "hardware_counters"
        )
    }
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

    pub fn validate_backend_profile_evidence(
        &self,
        expected_backend: Option<&str>,
    ) -> Result<(), String> {
        let expected_backend = expected_backend.or(self.selected_backend.as_deref());
        if let Some(expected_backend) = expected_backend {
            let profile = self.backend_profile.as_ref().ok_or_else(|| {
                format!(
                    "report for backend `{expected_backend}` lacks backend_profile. Fix: regenerate the benchmark report with a current vyre-bench binary so backend profile and timing-quality evidence are recorded."
                )
            })?;
            if profile.backend != expected_backend {
                return Err(format!(
                    "backend_profile.backend `{}` contradicts expected backend `{expected_backend}`. Fix: regenerate the report from the selected backend instead of editing JSON by hand.",
                    profile.backend
                ));
            }
        }
        if let Some(selected_backend) = self.selected_backend.as_deref() {
            if let Some(profile) = self.backend_profile.as_ref() {
                if profile.backend != selected_backend {
                    return Err(format!(
                        "backend_profile.backend `{}` contradicts selected_backend `{selected_backend}`. Fix: regenerate the benchmark report from one backend acquisition path.",
                        profile.backend
                    ));
                }
            }
        }
        if let Some(profile) = self.backend_profile.as_ref() {
            if !profile.has_valid_timing_quality() {
                return Err(format!(
                    "backend_profile.timing_quality `{}` is not a stable timing-quality value. Fix: use DeviceTimingQuality::as_str() when generating reports.",
                    profile.timing_quality
                ));
            }
            if profile.max_workgroup_size.contains(&0) {
                return Err(format!(
                    "backend_profile.max_workgroup_size {:?} contains zero. Fix: report conservative nonzero dispatch limits for benchmark evidence.",
                    profile.max_workgroup_size
                ));
            }
            if profile.max_invocations_per_workgroup == 0 {
                return Err(
                    "backend_profile.max_invocations_per_workgroup is zero. Fix: report a conservative nonzero invocation limit for benchmark evidence."
                        .to_string(),
                );
            }
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
    fn backend_profile_projects_timing_quality_for_reports() {
        let mut profile = vyre_driver::DeviceProfile::conservative("metal");
        profile.timing_quality = vyre_driver::DeviceTimingQuality::HostEnqueueWait;
        profile.supports_device_timestamps = false;
        profile.supports_hardware_counters = false;
        profile.supports_subgroup_ops = true;
        profile.max_workgroup_size = [1024, 1, 1];
        profile.max_invocations_per_workgroup = 1024;
        profile.max_storage_buffer_binding_size = 1 << 30;

        let report_profile = ReportBackendProfile::from_device_profile(profile);

        assert_eq!(report_profile.backend, "metal");
        assert_eq!(report_profile.timing_quality, "host_enqueue_wait");
        assert!(!report_profile.supports_device_timestamps);
        assert!(!report_profile.supports_hardware_counters);
        assert!(report_profile.supports_subgroup_ops);
        assert_eq!(report_profile.max_workgroup_size, [1024, 1, 1]);
        assert_eq!(report_profile.max_invocations_per_workgroup, 1024);
        assert_eq!(report_profile.max_storage_buffer_binding_size, 1 << 30);
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
