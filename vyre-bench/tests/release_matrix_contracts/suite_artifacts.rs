use super::*;

/// Lowest workload-family inventory either release suite may claim.
///
/// `release/evidence/docs/benchmark-doc-proof.md` binds both suites to at least
/// this many families. Without the floor the family-closure assertions below are
/// satisfied by an empty `release-workload-matrix.json`, so truncating the
/// matrix would narrow the release claim instead of turning this gate red.
const RELEASE_WORKLOAD_FAMILY_FLOOR: usize = 12;

/// Release-class recorded device memory floor, in MiB. This matches the
/// producer and release-gate minimum rather than a particular device's capacity.
const RELEASE_GPU_MEMORY_FLOOR_MIB: usize = 16 * 1024;

/// Lowest compute capability the release backends are probed against.
///
/// The backend matrix records CUDA release support at sm_80 and above, so a
/// recorded run below it did not exercise the release path.
const RELEASE_COMPUTE_CAPABILITY_FLOOR: (usize, usize) = (8, 0);

/// One recorded accelerator identity.
///
/// A suite status row and the workload artifact it cites each write this from
/// the same probe of a single run. Comparing the two recordings is the only
/// artifact-internal way to separate one measured sweep from rows glued
/// together out of several, and it asks nothing of the machine reading the
/// evidence.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RecordedDevice {
    model: String,
    memory_total_mib: usize,
    compute_capability: (usize, usize),
    driver_version: String,
    cuda_version: String,
}

impl RecordedDevice {
    fn from_status(status: &Value) -> Self {
        Self {
            model: json_str(status, "gpu_model").to_owned(),
            memory_total_mib: json_usize(status, "gpu_memory_total_mib"),
            compute_capability: (
                json_usize(status, "gpu_compute_capability_major"),
                json_usize(status, "gpu_compute_capability_minor"),
            ),
            driver_version: json_str(status, "nvidia_driver_version").to_owned(),
            cuda_version: json_str(status, "nvidia_cuda_version").to_owned(),
        }
    }

    fn from_environment_inventory(environment: &Value) -> BTreeSet<Self> {
        let devices = environment["gpu_devices"]
            .as_array()
            .expect("Fix: benchmark environment must record probed gpu_devices.");
        devices
            .iter()
            .map(|device| Self {
                model: json_str(device, "name").to_owned(),
                memory_total_mib: json_usize(device, "memory_total_mib"),
                compute_capability: (
                    json_usize(device, "compute_capability_major"),
                    json_usize(device, "compute_capability_minor"),
                ),
                driver_version: json_str(environment, "nvidia_driver_version").to_owned(),
                cuda_version: json_str(environment, "nvidia_cuda_version").to_owned(),
            })
            .collect()
    }
}

fn recorded_device_json(
    name: &str,
    memory_total_mib: usize,
    compute_capability_major: usize,
    compute_capability_minor: usize,
) -> Value {
    serde_json::json!({
        "name": name,
        "memory_total_mib": memory_total_mib,
        "compute_capability_major": compute_capability_major,
        "compute_capability_minor": compute_capability_minor,
    })
}

/// WHY: a release sweep selects one qualifying device from a complete probe.
/// Requiring the probe itself to contain one row rejects valid multi-GPU hosts.
#[test]
fn recorded_device_matches_any_member_of_environment_inventory() {
    let environment = serde_json::json!({
        "nvidia_driver_version": "580.173.02",
        "nvidia_cuda_version": "13.0",
        "gpu_devices": [
            recorded_device_json("sub-floor-device", 8192, 6, 1),
            recorded_device_json("qualifying-device", 24564, 8, 9),
        ]
    });
    let selected = RecordedDevice {
        model: "qualifying-device".to_string(),
        memory_total_mib: 24564,
        compute_capability: (8, 9),
        driver_version: "580.173.02".to_string(),
        cuda_version: "13.0".to_string(),
    };

    assert!(
        RecordedDevice::from_environment_inventory(&environment).contains(&selected),
        "Fix: release-suite provenance must find the selected device anywhere in the complete inventory."
    );
}

/// Single-run provenance closure over a release suite artifact.
///
/// A suite status row records the selected device and measured checkout; the
/// workload artifact records that device within its complete probe inventory
/// and repeats the checkout identity. Comparing those independent records is
/// the only artifact-internal way to separate one measured sweep from rows
/// glued together out of several.
///
/// Trusting the suite's own `blockers` array instead cannot see a violation.
/// The WGPU fallback suite carried an empty `blockers` array while two of its
/// sixteen rows cited artifacts recorded against a different source tree on a
/// different driver version, and its gate stayed green.
#[derive(Default)]
struct SuiteProvenance {
    devices: BTreeSet<RecordedDevice>,
    source_fingerprints: BTreeSet<String>,
    source_tree_fingerprints: BTreeSet<String>,
    paths: BTreeSet<String>,
}

impl SuiteProvenance {
    fn record(&mut self, suite: &str, status: &Value, artifact: &Value) {
        let path = json_str(status, "path");
        assert!(
            self.paths.insert(path.to_owned()),
            "Fix: {suite} lists `{path}` in more than one status row."
        );

        let recorded = RecordedDevice::from_status(status);
        assert!(
            !recorded.model.trim().is_empty(),
            "Fix: {suite} status `{path}` must record the probed device model."
        );
        assert!(
            recorded.memory_total_mib >= RELEASE_GPU_MEMORY_FLOOR_MIB,
            "Fix: {suite} status `{path}` records {} MiB of device memory, below the release floor of {RELEASE_GPU_MEMORY_FLOOR_MIB} MiB.",
            recorded.memory_total_mib
        );
        assert!(
            recorded.compute_capability >= RELEASE_COMPUTE_CAPABILITY_FLOOR,
            "Fix: {suite} status `{path}` records compute capability {:?}, below the release floor {:?}.",
            recorded.compute_capability,
            RELEASE_COMPUTE_CAPABILITY_FLOOR
        );
        assert_probed_version(
            suite,
            path,
            "nvidia_driver_version",
            &recorded.driver_version,
        );
        assert_probed_version(suite, path, "nvidia_cuda_version", &recorded.cuda_version);
        assert!(
            RecordedDevice::from_environment_inventory(&artifact["environment"])
                .contains(&recorded),
            "Fix: {suite} status `{path}` names a device absent from the artifact's complete GPU inventory."
        );
        self.devices.insert(recorded);

        for field in ["source_fingerprint", "source_tree_fingerprint"] {
            let recorded_by_status = json_str(status, field);
            assert!(
                !recorded_by_status.trim().is_empty(),
                "Fix: {suite} status `{path}` must record `{field}`."
            );
            assert_eq!(
                recorded_by_status,
                json_str(artifact, field),
                "Fix: {suite} status `{path}` `{field}` disagrees with the artifact it cites."
            );
        }
        self.source_fingerprints
            .insert(json_str(status, "source_fingerprint").to_owned());
        self.source_tree_fingerprints
            .insert(json_str(status, "source_tree_fingerprint").to_owned());
    }

    fn assert_single_run(&self, suite: &str, rows: usize) {
        assert_eq!(
            self.paths.len(),
            rows,
            "Fix: {suite} must cite a distinct workload artifact in every status row."
        );
        assert_eq!(
            self.devices.len(),
            1,
            "Fix: {suite} records {} distinct devices; one release sweep runs on one device: {:?}",
            self.devices.len(),
            self.devices
        );
        assert_eq!(
            self.source_fingerprints.len(),
            1,
            "Fix: {suite} records {} distinct source fingerprints; one release sweep measures one checkout: {:?}",
            self.source_fingerprints.len(),
            self.source_fingerprints
        );
        assert_eq!(
            self.source_tree_fingerprints.len(),
            1,
            "Fix: {suite} records {} distinct source trees; one release sweep measures one tree: {:?}",
            self.source_tree_fingerprints.len(),
            self.source_tree_fingerprints
        );
    }
}

/// A probed version field is dotted decimal.
///
/// A run that could not reach the driver writes a word there, so this separates
/// a measured field from a placeholder without matching a vendor spelling.
fn assert_probed_version(suite: &str, path: &str, field: &str, value: &str) {
    let dotted_decimal = value.contains('.')
        && value.split('.').all(|component| {
            !component.is_empty() && component.bytes().all(|byte| byte.is_ascii_digit())
        });
    assert!(
        dotted_decimal,
        "Fix: {suite} status `{path}` `{field}` must be a probed dotted-decimal version, not `{value}`."
    );
}

/// CUDA release evidence must use the current digest-bound suite schema and
/// prove every release workload on real NVIDIA hardware.
#[test]
fn cuda_release_suite_artifact_proves_real_gpu_macro_workloads() {
    let workspace = workspace_root();
    let suite_path = workspace.join("release/evidence/benchmarks/cuda-release-suite.json");
    let suite = read_json(&suite_path);
    let matrix =
        read_json(&workspace.join("release/evidence/benchmarks/release-workload-matrix.json"));
    let matrix_families = matrix["families"]
        .as_array()
        .expect("Fix: release workload matrix must list families.")
        .iter()
        .map(|family| json_str(family, "id").to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    let matrix_family_speedups = matrix["families"]
        .as_array()
        .expect("Fix: release workload matrix must list families.")
        .iter()
        .map(|family| {
            (
                json_str(family, "id").to_owned(),
                family["max_cpu_sota_min_speedup_x"].as_f64().unwrap_or(0.0),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let matrix_family_required = matrix["families"]
        .as_array()
        .expect("Fix: release workload matrix must list families.")
        .iter()
        .map(|family| {
            (
                json_str(family, "id").to_owned(),
                family["required"].as_bool().unwrap_or(true),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let matrix_family_baseline_required = matrix["families"]
        .as_array()
        .expect("Fix: release workload matrix must list families.")
        .iter()
        .map(|family| {
            (
                json_str(family, "id").to_owned(),
                family["requires_cpu_sota_baseline"]
                    .as_bool()
                    .unwrap_or(true),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(
        suite["schema_version"], 3,
        "Fix: CUDA release benchmark suite evidence must use digest-bound schema v3."
    );
    assert_eq!(
        suite["backend"], "cuda",
        "Fix: CUDA release benchmark suite must be CUDA-bound evidence."
    );
    assert_eq!(
        json_usize(&suite, "family_count"),
        matrix_families.len(),
        "Fix: CUDA release benchmark suite must cover every release workload matrix family."
    );
    assert!(
        matrix_families.len() >= RELEASE_WORKLOAD_FAMILY_FLOOR,
        "Fix: release-workload-matrix declares {} families, below the release floor of {RELEASE_WORKLOAD_FAMILY_FLOOR}.",
        matrix_families.len()
    );

    let artifacts = suite["artifacts"]
        .as_array()
        .expect("Fix: CUDA release suite must list artifacts.");
    let statuses = suite["artifact_statuses"]
        .as_array()
        .expect("Fix: CUDA release suite must list artifact_statuses.");
    assert_eq!(
        artifacts.len(),
        statuses.len(),
        "Fix: CUDA release suite artifacts and statuses must have one row per workload."
    );

    let mut covered_families = std::collections::BTreeSet::new();
    let mut provenance = SuiteProvenance::default();
    for status in statuses {
        let path = json_str(status, "path");
        let family_id = json_str(status, "family_id");
        let family_matrix_speedup = *matrix_family_speedups.get(family_id).unwrap_or_else(|| {
            panic!("Fix: CUDA suite family `{family_id}` is absent from release-workload-matrix.")
        });
        let family_is_required = *matrix_family_required.get(family_id).unwrap_or_else(|| {
            panic!("Fix: CUDA suite family `{family_id}` is absent from release-workload-matrix.")
        });
        let family_requires_baseline = *matrix_family_baseline_required
            .get(family_id)
            .unwrap_or_else(|| {
                panic!(
                    "Fix: CUDA suite family `{family_id}` is absent from release-workload-matrix."
                )
            });
        assert!(
            artifacts
                .iter()
                .any(|artifact| artifact.as_str() == Some(path)),
            "Fix: CUDA release suite status references `{path}` but artifacts[] does not."
        );
        assert_eq!(
            status["exists"], true,
            "Fix: CUDA workload artifact `{path}` must exist."
        );
        assert!(
            json_usize(status, "bytes") > 16_000,
            "Fix: CUDA workload artifact `{path}` is too small to be real benchmark evidence."
        );
        assert!(
            status["read_error"].is_null(),
            "Fix: CUDA workload artifact `{path}` must be readable."
        );
        assert_eq!(
            json_str(status, "selected_backend"),
            "cuda",
            "Fix: CUDA workload artifact `{path}` status must be CUDA-selected."
        );
        assert!(
            json_usize(status, "min_wall_samples") >= 30
                && json_usize(status, "min_baseline_wall_samples") >= 30,
            "Fix: CUDA workload artifact `{path}` must record at least 30 GPU and baseline timing samples."
        );
        assert!(
            json_usize(status, "case_count") >= 1 && json_usize(status, "failed_count") == 0,
            "Fix: CUDA workload artifact `{path}` must contain at least one passing benchmark case."
        );
        let requires_cpu_sota_100x = status["cpu_sota_100x_required"].as_bool().expect(
            "Fix: CUDA suite status must state whether the 100x CPU-SOTA contract is required.",
        );
        if requires_cpu_sota_100x {
            assert!(
                json_usize(status, "cpu_sota_100x_contract_cases") >= 1
                    && json_usize(status, "cpu_sota_100x_passing_cases")
                        == json_usize(status, "cpu_sota_100x_contract_cases"),
                "Fix: CUDA workload artifact `{path}` must pass every required CPU-SOTA 100x contract case."
            );
        } else if family_requires_baseline {
            assert!(
                family_matrix_speedup >= 10.0,
                "Fix: CUDA workload artifact `{path}` must map to a matrix CPU-SOTA contract of at least 10x."
            );
        } else {
            assert_eq!(
                family_matrix_speedup, 0.0,
                "Fix: CUDA workload artifact `{path}` without a CPU-SOTA baseline must have null/zero matrix speedup."
            );
        }
        assert!(
            status["blockers"].as_array().is_some_and(Vec::is_empty),
            "Fix: CUDA workload artifact `{path}` must not carry blockers."
        );

        let artifact = read_json(&workspace.join(path));
        provenance.record("CUDA release suite", status, &artifact);
        assert_eq!(
            artifact["schema"], "vyre-bench.result.v1",
            "Fix: `{path}` must be a vyre-bench result artifact."
        );
        assert_eq!(
            artifact["suite"], "release",
            "Fix: `{path}` must be release-suite evidence."
        );
        assert_eq!(
            artifact["selected_backend"], "cuda",
            "Fix: `{path}` must be CUDA evidence."
        );
        assert_eq!(
            artifact["environment"]["has_gpu"], true,
            "Fix: `{path}` must record a live GPU environment."
        );
        assert!(
            artifact["environment"]["features"]
                .as_array()
                .expect("Fix: benchmark environment features must be an array.")
                .iter()
                .any(|feature| feature.as_str() == Some("backend.usable.cuda")),
            "Fix: `{path}` must prove CUDA was usable, not merely linked."
        );
        let cases = artifact["cases"]
            .as_array()
            .expect("Fix: benchmark artifact cases must be an array.");
        assert!(
            !cases.is_empty(),
            "Fix: `{path}` must include benchmark cases."
        );
        for case in cases {
            assert_eq!(
                case["status"], "pass",
                "Fix: `{path}` has a non-passing benchmark case."
            );
            assert_eq!(
                case["backend_id"], "cuda",
                "Fix: `{path}` contains a non-CUDA case."
            );
            assert_eq!(
                case["needs_gpu"], true,
                "Fix: `{path}` release cases must require GPU execution."
            );
            if family_is_required && family_requires_baseline {
                assert_eq!(
                    case["workload_class"], "Macro",
                    "Fix: `{path}` must prove macro workloads, not primitive-only microbenchmarks."
                );
                assert!(
                    case["min_input_bytes"].as_u64().unwrap_or(0) >= 512 * 1024,
                    "Fix: `{path}` release cases must use at least 512 KiB input."
                );
                assert!(
                    case["performance"]["contract_passed"]
                        .as_bool()
                        .unwrap_or(false),
                    "Fix: `{path}` benchmark case failed its performance contract."
                );
                let min_cuda_cpu_sota_speedup = cuda_cpu_sota_min_speedup(case);
                assert!(
                    min_cuda_cpu_sota_speedup >= family_matrix_speedup,
                    "Fix: `{path}` case contract must be at least as strong as release-workload-matrix family `{family_id}`."
                );
                assert!(
                    case["performance"]["speedup_x"].as_f64().unwrap_or(0.0)
                        >= min_cuda_cpu_sota_speedup,
                    "Fix: `{path}` benchmark case must prove its CUDA CPU-SOTA speedup contract."
                );
                if requires_cpu_sota_100x {
                    assert!(
                        min_cuda_cpu_sota_speedup >= 100.0,
                        "Fix: `{path}` is marked 100x-required but its CUDA CPU-SOTA contract is weaker."
                    );
                } else {
                    assert!(
                        min_cuda_cpu_sota_speedup >= family_matrix_speedup,
                        "Fix: `{path}` non-required release contract is weaker than release-workload-matrix family `{family_id}`."
                    );
                }
                assert!(
                    case["performance"]["speedup_x"].as_f64().unwrap_or(0.0) >= 25.0,
                    "Fix: `{path}` benchmark case must prove at least the non-100x release floor."
                );
            } else if family_is_required {
                if let Some(performance) = case.get("performance").filter(|p| !p.is_null()) {
                    assert!(
                        performance["contract_passed"].as_bool().unwrap_or(false),
                        "Fix: `{path}` benchmark case failed its performance contract."
                    );
                    let min_cuda_cpu_sota_speedup = cuda_cpu_sota_min_speedup(case);
                    assert!(
                        performance["speedup_x"].as_f64().unwrap_or(0.0)
                            >= min_cuda_cpu_sota_speedup,
                        "Fix: `{path}` benchmark case must prove its CUDA speedup contract."
                    );
                }
            } else if let Some(performance) = case.get("performance").filter(|p| !p.is_null()) {
                assert!(
                    performance["contract_passed"].as_bool().unwrap_or(false),
                    "Fix: `{path}` benchmark case failed its performance contract."
                );
            }
            assert!(
                case["metrics"]["wall_ns"]["samples"].as_u64().unwrap_or(0) >= 30,
                "Fix: `{path}` benchmark case must contain at least 30 wall-clock samples."
            );
        }
        covered_families.insert(json_str(status, "family_id").to_owned());
    }
    provenance.assert_single_run("CUDA release suite", statuses.len());

    assert_eq!(
        covered_families, matrix_families,
        "Fix: CUDA release suite family coverage must match release-workload-matrix exactly."
    );
}

/// WGPU release evidence must use the current digest-bound suite schema and
/// preserve its own exact historical workload-family inventory.
#[test]
fn wgpu_fallback_suite_covers_executable_release_workload_families() {
    let workspace = workspace_root();
    let suite = read_json(&workspace.join("release/evidence/benchmarks/wgpu-fallback-suite.json"));
    assert_eq!(
        suite["schema_version"], 3,
        "Fix: WGPU fallback suite evidence must use digest-bound schema v3."
    );
    assert_eq!(
        json_usize(&suite, "family_count"),
        suite["artifact_statuses"]
            .as_array()
            .expect("Fix: WGPU fallback suite must list artifact_statuses.")
            .len(),
        "Fix: WGPU fallback suite family_count must equal its authenticated status inventory."
    );
    assert!(
        json_usize(&suite, "family_count") >= RELEASE_WORKLOAD_FAMILY_FLOOR,
        "Fix: WGPU fallback suite records {} families, below the release floor of {RELEASE_WORKLOAD_FAMILY_FLOOR}.",
        json_usize(&suite, "family_count")
    );

    let artifacts = suite["artifacts"]
        .as_array()
        .expect("Fix: WGPU fallback suite must list artifacts.");
    let statuses = suite["artifact_statuses"]
        .as_array()
        .expect("Fix: WGPU fallback suite must list artifact_statuses.");
    assert_eq!(
        artifacts.len(),
        statuses.len(),
        "Fix: WGPU fallback suite artifacts and statuses must have one row per workload."
    );

    let mut covered_families = BTreeSet::new();
    let mut provenance = SuiteProvenance::default();
    for status in statuses {
        let path = json_str(status, "path");
        assert!(
            artifacts
                .iter()
                .any(|artifact| artifact.as_str() == Some(path)),
            "Fix: WGPU fallback suite status references `{path}` but artifacts[] does not."
        );
        assert_eq!(
            status["exists"], true,
            "Fix: WGPU workload artifact `{path}` must exist."
        );
        assert!(
            status["blockers"].as_array().is_some(),
            "Fix: WGPU workload artifact `{path}` status must carry an explicit blockers array."
        );
        let artifact = read_json(&workspace.join(path));
        provenance.record("WGPU fallback suite", status, &artifact);
        assert_eq!(
            artifact["schema"], "vyre-bench.result.v1",
            "Fix: `{path}` must be a vyre-bench result artifact."
        );
        assert_eq!(
            artifact["suite"], "release",
            "Fix: `{path}` must be release-suite evidence."
        );
        assert_eq!(
            artifact["selected_backend"], "wgpu",
            "Fix: `{path}` must be WGPU evidence."
        );
        assert!(
            artifact["environment"]["features"]
                .as_array()
                .expect("Fix: benchmark environment features must be an array.")
                .iter()
                .any(|feature| feature.as_str() == Some("backend.usable.wgpu")),
            "Fix: `{path}` must prove WGPU was usable, not merely linked."
        );
        covered_families.insert(json_str(status, "family_id").to_owned());
    }
    provenance.assert_single_run("WGPU fallback suite", statuses.len());

    assert_eq!(
        covered_families.len(),
        json_usize(&suite, "family_count"),
        "Fix: WGPU fallback suite status rows must name each recorded family exactly once."
    );
}

fn cuda_cpu_sota_min_speedup(case: &Value) -> f64 {
    case["contract"]["baselines"]
        .as_array()
        .expect("Fix: benchmark case contract baselines must be an array.")
        .iter()
        .filter(|baseline| {
            baseline["class"].as_str() == Some("CpuSota")
                && baseline["backend_ids"]
                    .as_array()
                    .expect("Fix: CPU-SOTA baseline backend_ids must be an array.")
                    .iter()
                    .any(|backend| backend.as_str() == Some("cuda"))
        })
        .filter_map(|baseline| baseline["min_speedup_x"].as_f64())
        .fold(0.0, f64::max)
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("Fix: `{}` must be readable: {error}", path.display())),
    )
    .unwrap_or_else(|error| panic!("Fix: `{}` must be valid JSON: {error}", path.display()))
}

fn json_str<'a>(json: &'a Value, key: &str) -> &'a str {
    json[key]
        .as_str()
        .unwrap_or_else(|| panic!("Fix: JSON field `{key}` must be a string."))
}

fn json_usize(json: &Value, key: &str) -> usize {
    json[key]
        .as_u64()
        .unwrap_or_else(|| panic!("Fix: JSON field `{key}` must be an unsigned integer."))
        .try_into()
        .unwrap_or_else(|_| panic!("Fix: JSON field `{key}` must fit usize."))
}
