//! Benchmark report fragments shared by the evidence tests.
//!
//! Every reader of a benchmark report is fed a schema-2 case carrying a
//! CPU-SOTA baseline contract. That contract is twelve lines of JSON saying the
//! same thing every time, the run summary another six, and the measured case
//! around it another fifteen, so the fixtures were the largest block of copied
//! source in this crate: adding a field to any of them meant editing every
//! copy, and a copy that was missed read as a deliberate variation. Build them
//! here and let each test spell out only the values it is actually about.
//!
//! Two contract shapes exist because two layers parse them. The suite
//! inspection readers require the named, crate-attributed baseline; the
//! semantic readers and the release gate checks parse only the class, the
//! backends and the demanded speedup.

use std::fs;
use std::path::Path;

use serde::Serialize;
use serde_json::{json, Value};
use tempfile::TempDir;

/// A single-case run summary in the long form the artifact readers require.
pub(crate) fn case_summary(passed: u64, failed: u64) -> Value {
    json!({
        "total_cases": 1,
        "passed": passed,
        "failed": failed,
        "total_time_ns": 0,
        "cache_hit_rate": null
    })
}

/// The CPU-SOTA baseline contract a release benchmark case carries.
pub(crate) fn cpu_sota_contract(primitive: &str, backend_ids: &[&str]) -> Value {
    json!({
        "primitive": primitive,
        "baselines": [
            {
                "name": "CPU-SOTA",
                "crate_name": "vyre-runtime",
                "class": "CpuSota",
                "min_speedup_x": 100.0,
                "backend_ids": backend_ids
            }
        ]
    })
}

/// The CPU-SOTA baseline contract in the short form the semantic readers and the
/// release gate checks parse.
pub(crate) fn cpu_sota_baseline(backend_ids: &[&str], min_speedup_x: f64) -> Value {
    json!({
        "baselines": [
            {
                "class": "CpuSota",
                "backend_ids": backend_ids,
                "min_speedup_x": min_speedup_x
            }
        ]
    })
}

/// The short-form CPU-SOTA contract attributed to the primitive it measures.
///
/// Readers that report a failure by name need the attribution; readers that only
/// decide pass or fail never look at it, which is why [`cpu_sota_baseline`] omits
/// it rather than carrying a placeholder.
pub(crate) fn cpu_sota_baseline_for(
    primitive: &str,
    backend_ids: &[&str],
    min_speedup_x: f64,
) -> Value {
    let mut contract = cpu_sota_baseline(backend_ids, min_speedup_x);
    contract["primitive"] = json!(primitive);
    contract
}

/// One measured CPU-SOTA case: which backend ran it, the status the runner
/// recorded, and the wall timings a reader derives the achieved speedup from.
///
/// The claimed `performance.speedup_x` is fixed at a generous 200x on purpose.
/// Every reader under test must decide from the measured
/// `baseline_wall_ns / wall_ns` ratio, so a case that passes while the claim
/// alone would carry it proves the reader ignored the claim.
pub(crate) fn cpu_sota_case(
    id: &str,
    backend_id: &str,
    status: &str,
    contract_backend_ids: &[&str],
    wall_p50: u64,
    baseline_wall_p50: u64,
) -> Value {
    json!({
        "id": id,
        "backend_id": backend_id,
        "status": status,
        "contract": cpu_sota_baseline(contract_backend_ids, 100.0),
        "metrics": {
            "wall_ns": {"p50": wall_p50},
            "baseline_wall_ns": {"p50": baseline_wall_p50}
        },
        "performance": {"contract_passed": true, "speedup_x": 200.0}
    })
}

/// One benchmark case: the identity triple every reader keys on, plus whatever
/// measured fields the reader under test grades.
pub(crate) fn benchmark_case(
    id: &str,
    backend_id: &str,
    status: &str,
    measured: impl IntoIterator<Item = (&'static str, Value)>,
) -> Value {
    let mut case = json!({
        "id": id,
        "backend_id": backend_id,
        "status": status
    });
    let fields = case
        .as_object_mut()
        .expect("Fix: benchmark_case builds a JSON object.");
    for (key, value) in measured {
        fields.insert(key.to_string(), value);
    }
    case
}

/// The correctness reason every hidden-failure fixture reports.
pub(crate) const HIDDEN_INVALID_REASON: &str = "CUDA/WGPU output mismatch at row 17";

/// A case the runner recorded as `pass` whose correctness evidence says the
/// output was wrong, plus whatever measured fields the reader under test needs
/// before it will look at correctness at all.
///
/// Four readers reject this shape and each carried its own copy of it, keyed on
/// a reason string spelled out four times. A reader that started reporting the
/// reason differently would have been corrected in one copy, and the other
/// three would have gone on asserting the old text against a fixture that no
/// longer produced it.
pub(crate) fn hidden_invalid_case(
    id: &str,
    backend_id: &str,
    measured: impl IntoIterator<Item = (&'static str, Value)>,
) -> Value {
    let correctness = json!({"Invalid": {"reason": HIDDEN_INVALID_REASON}});
    benchmark_case(
        id,
        backend_id,
        "pass",
        std::iter::once(("correctness", correctness)).chain(measured),
    )
}

/// [`hidden_invalid_case`] carrying the contract, timings and winning
/// performance claim a reader demands before it will read correctness at all.
///
/// The claim is a generous 200x in every copy on purpose: a reader that accepts
/// the case has taken the claim over the correctness evidence beside it.
pub(crate) fn hidden_invalid_measured_case(
    id: &str,
    backend_id: &str,
    contract: Value,
    metrics: Value,
) -> Value {
    hidden_invalid_case(
        id,
        backend_id,
        [
            ("contract", contract),
            ("metrics", metrics),
            (
                "performance",
                json!({"contract_passed": true, "speedup_x": 200.0}),
            ),
        ],
    )
}

/// Wall and baseline wall timings as `[p50, p95, p99]`, each declaring the thirty
/// samples the release gate demands before it will read a percentile.
pub(crate) fn percentile_metrics(wall_ns: [u64; 3], baseline_wall_ns: [u64; 3]) -> Value {
    json!({
        "wall_ns": {
            "samples": 30,
            "p50": wall_ns[0],
            "p95": wall_ns[1],
            "p99": wall_ns[2]
        },
        "baseline_wall_ns": {
            "samples": 30,
            "p50": baseline_wall_ns[0],
            "p95": baseline_wall_ns[1],
            "p99": baseline_wall_ns[2]
        }
    })
}

/// [`percentile_metrics`] plus the kernel launch count a GPU case must report.
///
/// The release gate refuses a GPU measurement that never launched a kernel, so
/// every artifact fixture for a dispatching backend carries this metric and the
/// count is the only part of it a fixture varies.
pub(crate) fn launched_percentile_metrics(
    wall_ns: [u64; 3],
    baseline_wall_ns: [u64; 3],
    kernel_launches_p50: u64,
) -> Value {
    let mut metrics = percentile_metrics(wall_ns, baseline_wall_ns);
    metrics["kernel_launches"] = json!({"samples": 30, "p50": kernel_launches_p50});
    metrics
}

/// [`launched_percentile_metrics`] plus the PTX source cache counters a CUDA case
/// reports, as `[entries, hits, misses]`.
///
/// The suite readers demand all three before they will call a CUDA measurement
/// release-grade, so a fixture that omits one is testing the missing-metric path
/// rather than the path it names.
pub(crate) fn cuda_cached_metrics(
    wall_ns: [u64; 3],
    baseline_wall_ns: [u64; 3],
    kernel_launches_p50: u64,
    ptx_source_cache: [u64; 3],
) -> Value {
    let mut metrics = launched_percentile_metrics(wall_ns, baseline_wall_ns, kernel_launches_p50);
    metrics["cuda_ptx_source_cache_entries"] = json!({"samples": 30, "p50": ptx_source_cache[0]});
    metrics["cuda_ptx_source_cache_hits"] = json!({"samples": 30, "p50": ptx_source_cache[1]});
    metrics["cuda_ptx_source_cache_misses"] = json!({"samples": 30, "p50": ptx_source_cache[2]});
    metrics
}

/// A backend suite descriptor over the artifacts of one benchmark family.
///
/// `artifacts` is derived from the same paths the statuses name. A fixture that
/// spells the list twice can disagree with itself by a typo, and a reader under
/// test would then be judged against a shape no suite writer emits; a fixture
/// that is about that disagreement assigns to `suite["artifacts"]`.
pub(crate) fn backend_suite(
    family_id: &str,
    requested_case_id: &str,
    artifact_paths: &[&str],
) -> Value {
    json!({
        "artifact_statuses": artifact_paths
            .iter()
            .map(|path| json!({
                "path": path,
                "family_id": family_id,
                "requested_case_id": requested_case_id
            }))
            .collect::<Vec<_>>(),
        "artifacts": artifact_paths
    })
}

/// The environment block an artifact carries when the GPU memory evidence is the
/// only part of it a reader looks at: the release axis proves a `max_vram_mib`
/// claim from `gpu_devices[0].memory_total_mib` and reads nothing else.
pub(crate) fn gpu_memory_environment(memory_total_mib: u64) -> Value {
    json!({"gpu_devices": [{"memory_total_mib": memory_total_mib}]})
}

/// The environment block the suite inspection readers parse, keyed
/// `host_cpu_model`.
///
/// Every attribution string is a parameter because the provenance readers reject
/// blank attribution, and that rejection is what several fixtures are about: a
/// caller passing whitespace is stating the negative case in one line instead of
/// restating the whole block to change four strings.
pub(crate) fn host_environment(
    host_cpu_model: &str,
    gpu_name: &str,
    nvidia_driver_version: &str,
    nvidia_cuda_version: &str,
) -> Value {
    json!({
        "host_cpu_model": host_cpu_model,
        "gpu_devices": [
            {
                "name": gpu_name,
                "memory_total_mib": 24576,
                "compute_capability_major": 8,
                "compute_capability_minor": 9
            }
        ],
        "nvidia_driver_version": nvidia_driver_version,
        "nvidia_cuda_version": nvidia_cuda_version
    })
}

/// A temporary workspace root holding release benchmark evidence, plus the two
/// source fingerprints an artifact written into it must carry to read as
/// current.
///
/// The provenance readers resolve a workspace root by walking up from the
/// evidence path to a `Cargo.toml`, and then recompute the fingerprints of that
/// root to decide whether the evidence is stale. A fixture for any of them has
/// to lay down the manifest, the `release/evidence/benchmarks` directory and the
/// current fingerprints before it writes a single report, in that order, because
/// the tree fingerprint is taken over the tree as it stands. Eleven fixtures
/// carried that opening and each one named it differently, so a reader that
/// changed where it looks for the root would have been corrected in one of them.
pub(crate) struct EvidenceWorkspace {
    root: TempDir,
    source_fingerprint: String,
    source_tree_fingerprint: String,
}

impl EvidenceWorkspace {
    pub(crate) fn new() -> Self {
        let root = TempDir::new().expect("Fix: create temp workspace for evidence fixture.");
        fs::write(root.path().join("Cargo.toml"), "[workspace]\n")
            .expect("Fix: write temp workspace manifest.");
        fs::create_dir_all(root.path().join("release/evidence/benchmarks"))
            .expect("Fix: create temp benchmark evidence directory.");
        let git = vyre_bench::probes::capture_git_info_at(root.path());
        let source_fingerprint = vyre_bench::probes::source_fingerprint(&git);
        let source_tree_fingerprint = vyre_bench::probes::source_tree_fingerprint_at(root.path());
        Self {
            root,
            source_fingerprint,
            source_tree_fingerprint,
        }
    }

    /// The workspace root a checker is pointed at.
    pub(crate) fn path(&self) -> &Path {
        self.root.path()
    }

    /// The git fingerprint of this root, as evidence measured on it would carry.
    pub(crate) fn source_fingerprint(&self) -> &str {
        &self.source_fingerprint
    }

    /// The source tree fingerprint of this root, as evidence measured on it
    /// would carry.
    pub(crate) fn source_tree_fingerprint(&self) -> &str {
        &self.source_tree_fingerprint
    }

    /// Write one report under `release/evidence/benchmarks` and return the
    /// workspace-relative path the axes and suite arrays name it by.
    pub(crate) fn write_report(&self, file_name: &str, report: &Value) -> String {
        let relative = format!("release/evidence/benchmarks/{file_name}");
        fs::write(
            self.root.path().join(&relative),
            serde_json::to_string_pretty(report).expect("Fix: serialize evidence report fixture."),
        )
        .expect("Fix: write evidence report fixture.");
        relative
    }

    /// Write one current CUDA artifact carrying exactly `cases`, and return the
    /// path the axes and suite arrays name it by.
    ///
    /// A fixture about one defect in one case still has to satisfy every check
    /// that runs before the one it is about: the artifact must claim CUDA, name
    /// the source it was measured on, and agree with its own summary. Those four
    /// keys are the same in every such fixture, so only the cases are a caller's
    /// business, and the summary is derived from them rather than restated.
    pub(crate) fn write_cuda_release_artifact(&self, file_name: &str, cases: Value) -> String {
        let total_cases = cases
            .as_array()
            .expect("Fix: pass a cases array to write_cuda_release_artifact.")
            .len();
        self.write_report(
            file_name,
            &json!({
                "selected_backend": "cuda",
                "source_fingerprint": self.source_fingerprint(),
                "source_tree_fingerprint": self.source_tree_fingerprint(),
                "summary": {"total_cases": total_cases, "passed": total_cases, "failed": 0},
                "cases": cases
            }),
        )
    }

    /// The CUDA release suite that lists exactly `artifacts` and claims CUDA.
    ///
    /// The suite is cross-checked against the axes for artifacts either side
    /// omits, so a fixture about anything else has to make the two agree.
    pub(crate) fn cuda_release_suite<S: Serialize>(artifacts: &[S]) -> Value {
        json!({
            "backend": "cuda",
            "artifacts": artifacts
        })
    }

    /// The release axes that cite exactly `artifacts` and no scalar of their own.
    pub(crate) fn cuda_release_axes<S: Serialize>(artifacts: &[S]) -> Value {
        json!({
            "source_artifacts": artifacts
        })
    }

    /// The twelve current CUDA workload artifacts a release axis needs cited
    /// before it will compute a scalar at all, returned in the order the axes and
    /// suite arrays list them.
    ///
    /// Each one proves every axis: a warm wall time, a cold build time, a scan
    /// throughput, the device memory the memory axis reads, and the ULP drift the
    /// correctness axis takes its maximum over. `case_prefix` keeps the case ids
    /// of one fixture distinct from another's, and `max_observed_ulp` is the only
    /// measurement a caller normally varies.
    pub(crate) fn cuda_release_axis_artifacts(
        &self,
        case_prefix: &str,
        max_observed_ulp: u64,
    ) -> Vec<String> {
        (1..=12)
            .map(|index| {
                self.write_report(
                    &format!("workload-{index:02}.json"),
                    &json!({
                        "selected_backend": "cuda",
                        "source_fingerprint": self.source_fingerprint(),
                        "source_tree_fingerprint": self.source_tree_fingerprint(),
                        "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                        "environment": gpu_memory_environment(24576),
                        "cases": [
                            {
                                "id": format!("{case_prefix}.{index}"),
                                "backend_id": "cuda",
                                "status": "pass",
                                "metrics": {
                                    "wall_ns": {"p50": 17_000},
                                    "cold_compile_ns": {"p50": 2_000_000},
                                    "wall_gb_s_x1000": {"p50": 4_000}
                                },
                                "correctness": {
                                    "Toleranced": {"max_observed_ulp": max_observed_ulp}
                                }
                            }
                        ]
                    }),
                )
            })
            .collect()
    }
}

/// The release-axis issues a single written CUDA source artifact draws.
///
/// A fixture about one artifact still has to satisfy the axes-against-suite
/// cross-check, so every such test wrote the same workspace, the same
/// one-element axes and the same one-element suite around the case list it
/// actually cared about.
pub(crate) fn cuda_release_axis_issues(file_name: &str, cases: Value) -> Vec<String> {
    let workspace = EvidenceWorkspace::new();
    let artifact = workspace.write_cuda_release_artifact(file_name, cases);
    let axes = EvidenceWorkspace::cuda_release_axes(&[&artifact]);
    let cuda_suite = EvidenceWorkspace::cuda_release_suite(&[&artifact]);
    crate::bench::benchmark_evidence_semantics::cuda_release_axes_source_artifact_issues(
        workspace.path(),
        &axes,
        &cuda_suite,
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{case_summary, cpu_sota_baseline, cpu_sota_case, cpu_sota_contract};

    /// WHY: the fragments stand in for hand-written JSON in a dozen fixtures,
    /// so a drift in either shape would silently change what those fixtures
    /// mean. Pin the exact keys and values the readers parse.
    #[test]
    fn case_summary_carries_every_field_a_reader_parses() {
        let summary = case_summary(1, 0);
        assert_eq!(summary["total_cases"], 1);
        assert_eq!(summary["passed"], 1);
        assert_eq!(summary["failed"], 0);
        assert_eq!(summary["total_time_ns"], 0);
        assert!(
            summary["cache_hit_rate"].is_null(),
            "Fix: cache_hit_rate must stay present and null; readers distinguish absent from null."
        );
        assert_eq!(
            summary.as_object().map(serde_json::Map::len),
            Some(5),
            "Fix: an extra summary field changes every fixture built on this helper."
        );
    }

    /// WHY: `backend_ids` is what decides whether a baseline applies to the
    /// case under inspection, and the class/name pair is what marks it as the
    /// CPU-SOTA baseline. A fixture that got either wrong would exercise the
    /// wrong branch while still looking like a CPU-SOTA case.
    #[test]
    fn cpu_sota_contract_names_the_baseline_and_its_backends() {
        let contract = cpu_sota_contract("release condition eval", &["cuda", "wgpu"]);
        assert_eq!(contract["primitive"], "release condition eval");
        let baselines = contract["baselines"]
            .as_array()
            .expect("Fix: contract must carry a baselines array.");
        assert_eq!(baselines.len(), 1);
        assert_eq!(baselines[0]["name"], "CPU-SOTA");
        assert_eq!(baselines[0]["class"], "CpuSota");
        assert_eq!(baselines[0]["crate_name"], "vyre-runtime");
        assert_eq!(baselines[0]["min_speedup_x"], 100.0);
        assert_eq!(
            baselines[0]["backend_ids"],
            json!(["cuda", "wgpu"]),
            "Fix: backend_ids decides which case the baseline applies to."
        );
    }

    /// WHY: the short contract is what the semantic readers and the release gate
    /// checks match on. `backend_ids` decides whether a baseline applies to the
    /// case at all, and `min_speedup_x` is the threshold under test, so a
    /// fixture that got either wrong would exercise the wrong branch.
    #[test]
    fn the_short_baseline_carries_only_what_the_readers_match_on() {
        let contract = cpu_sota_baseline(&["cuda"], 1.01);
        let baselines = contract["baselines"]
            .as_array()
            .expect("Fix: contract must carry a baselines array.");
        assert_eq!(baselines.len(), 1);
        assert_eq!(baselines[0]["class"], "CpuSota");
        assert_eq!(baselines[0]["backend_ids"], json!(["cuda"]));
        assert_eq!(baselines[0]["min_speedup_x"], 1.01);
        assert_eq!(
            baselines[0].as_object().map(serde_json::Map::len),
            Some(3),
            "Fix: the short baseline must stay short; the named form is cpu_sota_contract."
        );
    }

    /// WHY: every reader under test must derive the achieved speedup from the
    /// measured wall timings. The generous claimed speedup_x is what makes a
    /// claim-trusting reader visible, so it has to be present and unchanged, and
    /// the timings have to be exactly what the caller asked for.
    #[test]
    fn a_case_reports_measured_timings_alongside_a_generous_claim() {
        let case = cpu_sota_case(
            "release.condition_eval.1m",
            "cuda",
            "pass",
            &["cuda"],
            10,
            2000,
        );
        assert_eq!(case["id"], "release.condition_eval.1m");
        assert_eq!(case["backend_id"], "cuda");
        assert_eq!(case["status"], "pass");
        assert_eq!(case["metrics"]["wall_ns"]["p50"], 10);
        assert_eq!(case["metrics"]["baseline_wall_ns"]["p50"], 2000);
        assert_eq!(case["performance"]["contract_passed"], true);
        assert_eq!(case["performance"]["speedup_x"], 200.0);
        assert_eq!(
            case["contract"],
            cpu_sota_baseline(&["cuda"], 100.0),
            "Fix: a case carries the short baseline the gate checks parse."
        );
    }
}
