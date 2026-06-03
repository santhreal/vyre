//! Release-gate contract tests for conform wiring.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

const RUNTIME_DIALECT_CONTRACT_OPS: &[&str] = &[
    "core.indirect_dispatch",
    "io.dma_from_nvme",
    "io.write_back_to_nvme",
    "mem.unmap",
    "mem.zerocopy_map",
];

fn repo_file(path: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("Fix: vyre-conform-runner must stay under the repository conform directory.");
    std::fs::read_to_string(root.join(path)).unwrap_or_else(|error| {
        panic!("Fix: expected release-gate file `{path}` to exist: {error}")
    })
}

fn repo_json(path: &str) -> Value {
    serde_json::from_str(&repo_file(path)).unwrap_or_else(|error| {
        panic!("Fix: release artifact `{path}` must be valid JSON: {error}")
    })
}

fn floor(script: &str, crate_name: &str) -> u32 {
    let needle = format!("FLOOR[\"{crate_name}\"]=");
    let line = script
        .lines()
        .find(|line| line.trim_start().starts_with(&needle))
        .unwrap_or_else(|| {
            panic!("Fix: `{crate_name}` must have an explicit test-coverage floor.")
        });
    let value = line[needle.len()..]
        .split(|ch: char| !ch.is_ascii_digit())
        .next()
        .expect("Fix: coverage floor must start with a decimal percentage.");
    value.parse().unwrap_or_else(|error| {
        panic!("Fix: coverage floor for `{crate_name}` must parse: {error}")
    })
}

#[test]
fn release_conformance_artifacts_prove_three_backend_catalog_completeness() {
    let gate = repo_json("release/evidence/conformance/release-gate-log.json");
    assert_eq!(
        gate["schema_version"], 2,
        "Fix: release gate log schema must be v2."
    );
    assert_json_string_array_contains_exactly(
        &gate["requested_backends"],
        &["cuda", "wgpu", "cpu-ref"],
        "requested_backends",
    );
    assert_json_string_array_contains_exactly(
        &gate["required_artifacts"],
        &[
            "cuda-conformance.json",
            "wgpu-conformance.json",
            "reference-conformance.json",
        ],
        "required_artifacts",
    );
    assert!(
        gate["blockers"].as_array().is_some_and(Vec::is_empty),
        "Fix: release conformance gate must have zero blockers."
    );
    for status in gate["artifact_statuses"]
        .as_array()
        .expect("Fix: release gate log must contain artifact_statuses")
    {
        let path = status["path"]
            .as_str()
            .expect("Fix: conformance artifact status must name a path");
        assert_eq!(
            status["exists"], true,
            "Fix: required conformance artifact `{path}` must exist."
        );
        assert!(
            status["bytes"].as_u64().unwrap_or(0) > 1000,
            "Fix: required conformance artifact `{path}` is too small to be a real certificate."
        );
        assert!(
            status["read_error"].is_null(),
            "Fix: required conformance artifact `{path}` must be readable."
        );
    }

    let matrix = repo_json("release/evidence/conformance/conformance-matrix.json");
    let matrix_summary = ConformanceSummary::from_json(&matrix, "conformance-matrix.json");
    assert!(
        matrix_summary.distinct_op_count >= 400,
        "Fix: release conformance matrix must cover the full catalog-scale op surface."
    );
    assert_eq!(
        matrix_summary.catalog_required_op_count, matrix_summary.catalog_covered_op_count,
        "Fix: every required catalog op must be covered by release conformance rows."
    );
    assert!(
        matrix_summary.missing_catalog_ops.is_empty(),
        "Fix: release conformance matrix has missing catalog ops: {:?}",
        matrix_summary.missing_catalog_ops
    );
    assert_eq!(
        matrix_summary.release_backend_row_count,
        matrix_summary.catalog_required_op_count * 3,
        "Fix: release conformance matrix must contain exactly one reference, CUDA, and WGPU row for every required catalog op."
    );
    let expected_rows = release_backend_rows(&matrix, "conformance-matrix.json");
    assert_eq!(
        expected_rows.len(),
        matrix_summary.release_backend_row_count,
        "Fix: release conformance row count field must match release_backend_rows length."
    );
    assert_complete_backend_rows(&expected_rows, matrix_summary.catalog_required_op_count);

    for (backend_id, artifact) in [
        ("cuda", "cuda-conformance.json"),
        ("wgpu", "wgpu-conformance.json"),
        ("cpu-ref", "reference-conformance.json"),
    ] {
        let artifact_path = format!("release/evidence/conformance/{artifact}");
        let json = repo_json(&artifact_path);
        let summary = ConformanceSummary::from_json(&json, &artifact_path);
        assert_eq!(
            json["backend_id"], backend_id,
            "Fix: `{artifact}` must declare backend_id `{backend_id}`."
        );
        let command = json["command"]
            .as_str()
            .expect("Fix: conformance artifact must record the command that generated it");
        assert!(
            command.contains("cargo_full")
                && command.contains("vyre-conform-runner")
                && command.contains("dispatch --backend")
                && command.contains(backend_id),
            "Fix: `{artifact}` must record a reproducible cargo_full dispatch command for `{backend_id}`, got `{command}`."
        );
        assert_eq!(
            summary.distinct_op_count, matrix_summary.distinct_op_count,
            "Fix: `{artifact}` distinct_op_count must agree with conformance-matrix.json."
        );
        assert_eq!(
            summary.catalog_required_op_count, matrix_summary.catalog_required_op_count,
            "Fix: `{artifact}` catalog_required_op_count must agree with conformance-matrix.json."
        );
        assert_eq!(
            summary.catalog_covered_op_count, matrix_summary.catalog_covered_op_count,
            "Fix: `{artifact}` catalog_covered_op_count must agree with conformance-matrix.json."
        );
        assert!(
            summary.missing_catalog_ops.is_empty(),
            "Fix: `{artifact}` reports missing catalog ops: {:?}",
            summary.missing_catalog_ops
        );
        assert!(
            json["stdout_diagnostics"]
                .as_array()
                .is_some_and(Vec::is_empty),
            "Fix: `{artifact}` must not carry ignored stdout diagnostics."
        );
        assert_conformance_artifact_has_no_failures(&json, artifact);
        assert_runtime_dialect_rows(&json, backend_id, artifact);
        let rows = release_backend_rows(&json, &artifact_path);
        assert_eq!(
            rows, expected_rows,
            "Fix: `{artifact}` release backend rows must match conformance-matrix.json exactly."
        );
    }
}

fn assert_conformance_artifact_has_no_failures(json: &Value, label: &str) {
    let total_pairs = json_usize(json, "total_pairs", label)
        .unwrap_or_else(|| panic!("Fix: `{label}` must define total_pairs."));
    let passed_pairs = json_usize(json, "passed_pairs", label)
        .unwrap_or_else(|| panic!("Fix: `{label}` must define passed_pairs."));
    let failed_pairs = json_usize(json, "failed_pairs", label)
        .unwrap_or_else(|| panic!("Fix: `{label}` must define failed_pairs."));
    assert_eq!(
        failed_pairs, 0,
        "Fix: `{label}` must not ship a release conformance artifact with failing pairs."
    );
    assert_eq!(
        passed_pairs, total_pairs,
        "Fix: `{label}` passed_pairs must equal total_pairs."
    );
    assert!(
        json["blockers"].as_array().is_some_and(Vec::is_empty),
        "Fix: `{label}` must not ship release conformance blockers."
    );
    let pairs = json["pairs"]
        .as_array()
        .unwrap_or_else(|| panic!("Fix: `{label}` must include conformance pairs."));
    assert_eq!(
        pairs.len(),
        total_pairs,
        "Fix: `{label}` total_pairs must match the pairs array length."
    );
    for pair in pairs {
        assert_eq!(
            pair["passed"], true,
            "Fix: `{label}` pair ({:?}, {:?}) must pass before release evidence is accepted.",
            pair["backend_id"], pair["op_id"]
        );
    }
}

fn assert_runtime_dialect_rows(json: &Value, backend_id: &str, label: &str) {
    let rows = release_backend_rows(json, label);
    let matrix_backend_id = match backend_id {
        "cpu-ref" => "reference",
        other => other,
    };
    let expected_status = match matrix_backend_id {
        "reference" => "not_applicable",
        "cuda" | "wgpu" => "experimental",
        other => panic!("Fix: unknown release backend `{other}` in `{label}`."),
    };
    for op in RUNTIME_DIALECT_CONTRACT_OPS {
        let row = format!("{op}:{matrix_backend_id}:{expected_status}");
        assert!(
            rows.contains(&row),
            "Fix: `{label}` must include runtime dialect release row `{row}`."
        );
    }
}

#[test]
fn concrete_driver_coverage_floors_are_nonzero_release_gates() {
    let script = repo_file("scripts/check_test_coverage_per_crate.sh");

    for crate_name in concrete_driver_crates() {
        assert!(
            floor(&script, &crate_name) > 0,
            "Fix: concrete driver `{crate_name}` must not be exempt from per-crate test coverage."
        );
    }
}

struct ConformanceSummary {
    distinct_op_count: usize,
    catalog_required_op_count: usize,
    catalog_covered_op_count: usize,
    release_backend_row_count: usize,
    missing_catalog_ops: Vec<String>,
}

impl ConformanceSummary {
    fn from_json(json: &Value, label: &str) -> Self {
        assert_eq!(
            json["schema_version"], 2,
            "Fix: `{label}` must use conformance evidence schema v2."
        );
        let total_pairs = json_usize(json, "total_pairs", label).unwrap_or_else(|| {
            json_usize(json, "op_count", label)
                .unwrap_or_else(|| panic!("Fix: `{label}` must define total_pairs or op_count."))
        });
        let distinct_op_count = json_usize(json, "distinct_op_count", label)
            .unwrap_or_else(|| panic!("Fix: `{label}` must define distinct_op_count."));
        assert_eq!(
            total_pairs, distinct_op_count,
            "Fix: `{label}` must not contain duplicate op-pair certificates."
        );
        Self {
            distinct_op_count,
            catalog_required_op_count: json_usize(json, "catalog_required_op_count", label)
                .unwrap_or_else(|| panic!("Fix: `{label}` must define catalog_required_op_count.")),
            catalog_covered_op_count: json_usize(json, "catalog_covered_op_count", label)
                .unwrap_or_else(|| panic!("Fix: `{label}` must define catalog_covered_op_count.")),
            release_backend_row_count: json_usize(json, "release_backend_row_count", label)
                .unwrap_or_else(|| panic!("Fix: `{label}` must define release_backend_row_count.")),
            missing_catalog_ops: json["missing_catalog_ops"]
                .as_array()
                .unwrap_or_else(|| panic!("Fix: `{label}` must define missing_catalog_ops."))
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .unwrap_or_else(|| {
                            panic!("Fix: `{label}` missing_catalog_ops entries must be strings.")
                        })
                        .to_string()
                })
                .collect(),
        }
    }
}

fn json_usize(json: &Value, key: &str, label: &str) -> Option<usize> {
    json[key].as_u64().map(|value| {
        usize::try_from(value).unwrap_or_else(|error| {
            panic!("Fix: `{label}` field `{key}` cannot fit usize: {error}")
        })
    })
}

fn release_backend_rows(json: &Value, label: &str) -> BTreeSet<String> {
    json["release_backend_rows"]
        .as_array()
        .unwrap_or_else(|| panic!("Fix: `{label}` must define release_backend_rows."))
        .iter()
        .map(|value| {
            value
                .as_str()
                .unwrap_or_else(|| {
                    panic!("Fix: `{label}` release_backend_rows entries must be strings.")
                })
                .to_string()
        })
        .collect()
}

fn assert_complete_backend_rows(rows: &BTreeSet<String>, expected_ops_per_backend: usize) {
    let mut per_backend: BTreeMap<&str, usize> = BTreeMap::new();
    let mut ops = BTreeSet::new();
    let mut runtime_status_rows = BTreeSet::new();
    for row in rows {
        let (op, backend, status) = parse_release_backend_row(row);
        assert!(
            matches!(backend, "reference" | "cuda" | "wgpu"),
            "Fix: release conformance row `{row}` has unexpected backend `{backend}`."
        );
        if RUNTIME_DIALECT_CONTRACT_OPS.contains(&op) {
            let expected = if backend == "reference" {
                "not_applicable"
            } else {
                "experimental"
            };
            assert_eq!(
                status, expected,
                "Fix: runtime dialect contract row `{row}` must use status `{expected}` until a concrete backend lowering is promoted."
            );
            runtime_status_rows.insert(row.clone());
        } else {
            assert_eq!(
                status, "supported",
                "Fix: non-runtime release conformance row `{row}` must be supported."
            );
        }
        *per_backend.entry(backend).or_default() += 1;
        ops.insert(op.to_string());
    }
    assert_eq!(
        runtime_status_rows.len(),
        RUNTIME_DIALECT_CONTRACT_OPS.len() * 3,
        "Fix: runtime dialect exceptions must be explicit and limited to the Category C runtime contract ops."
    );
    for backend in ["reference", "cuda", "wgpu"] {
        assert_eq!(
            per_backend.get(backend).copied().unwrap_or(0),
            expected_ops_per_backend,
            "Fix: release conformance must contain one `{backend}` row for every required catalog op."
        );
    }
    assert_eq!(
        ops.len(),
        expected_ops_per_backend,
        "Fix: release conformance row set must contain exactly the required catalog op set."
    );
}

fn parse_release_backend_row(row: &str) -> (&str, &str, &str) {
    let (prefix, status) = row
        .rsplit_once(':')
        .unwrap_or_else(|| panic!("Fix: release backend row `{row}` must include a status."));
    let (op, backend) = prefix
        .rsplit_once(':')
        .unwrap_or_else(|| panic!("Fix: release backend row `{row}` must include a backend."));
    (op, backend, status)
}

fn assert_json_string_array_contains_exactly(json: &Value, expected: &[&str], field: &str) {
    let actual = json
        .as_array()
        .unwrap_or_else(|| panic!("Fix: `{field}` must be an array."))
        .iter()
        .map(|value| {
            value
                .as_str()
                .unwrap_or_else(|| panic!("Fix: `{field}` entries must be strings."))
                .to_string()
        })
        .collect::<BTreeSet<_>>();
    let expected = expected
        .iter()
        .map(|value| (*value).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected, "Fix: `{field}` has the wrong entries.");
}

#[test]
fn cuda_parity_gate_documents_int4_gpu_parity_coverage() {
    let script = repo_file("scripts/check_cuda_parity_perf_gate.sh");
    assert!(
        script.contains("check_cuda_parity_perf_gate.sh")
            && script.contains("*gpu_parity*")
            && script.contains("int4_quantized_gpu_parity"),
        "Fix: CUDA parity gate must auto-discover INT4 gpu_parity integration tests."
    );

    let evidence: serde_json::Value =
        serde_json::from_str(&repo_file("release/evidence/tests/cuda-release-gate.json"))
            .expect("Fix: CUDA release gate evidence must be valid JSON.");
    let int4_ops = evidence["int4_conformance_ops"]
        .as_array()
        .expect("Fix: cuda-release-gate.json must list int4_conformance_ops.");
    assert_eq!(
        int4_ops.len(),
        6,
        "Fix: INT4 release gate must enumerate all six harness-backed quant.int4 ops."
    );
    assert!(
        evidence["gpu_parity_integration_tests"]
            .as_array()
            .is_some_and(|tests| tests.iter().any(|test| test == "int4_quantized_gpu_parity")),
        "Fix: cuda-release-gate.json must name int4_quantized_gpu_parity as a gpu_parity integration test."
    );
}

#[test]
fn nightly_ci_runs_backend_gates_and_real_conform_subcommands() {
    let script = repo_file("scripts/nightly_ci.sh");
    assert!(
        script.contains("source scripts/lib/cargo_runner.sh") && script.contains("vyre_select_cargo_runner"),
        "Fix: nightly_ci.sh must fall back to cargo under CARGO_BUILD_JOBS-gated execution when cargo_full is absent."
    );

    for required in [
        "nvidia-smi",
        "scripts/check_test_coverage_per_crate.sh",
        "scripts/check_roadmap_status_split.sh",
        "scripts/check_ownership_boundaries.sh",
        "scripts/check_cuda_parity_perf_gate.sh",
        "dispatch --backend",
    ] {
        assert!(
            script.contains(required),
            "Fix: nightly_ci.sh must contain `{required}` so backend gates cannot be silently unchecked."
        );
    }
    assert!(
        !script.contains(" -- run --backend "),
        "Fix: nightly_ci.sh must call the implemented `dispatch` subcommand, not the stale `run` spelling."
    );
    assert!(
        !script.contains("\"\" test"),
        "Fix: nightly_ci.sh must invoke the selected cargo runner, not an empty command string."
    );
    for required_test in [
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-reference --test oracle_program_edges",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-reference --test quantized_buffer_contract",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-spec --test invariant_catalog_surface",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-spec --test data_type_layout_matrix",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-spec --test collective_op_contracts",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-macros --test adversarial",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-foundation --test wire_fuzz_infra_contracts",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-foundation --test autodiff_transform_contracts",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-foundation --test collective_ir_contracts",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-libs --test hash_single_source_contracts",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-bench --test release_matrix_contracts",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre --test wire_malformed_adversarial",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-self-substrate --test organization_contracts",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-self-substrate --test graph_single_source_contracts",
        "CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\" \"$CARGO_RUNNER\" test -q -p vyre-self-substrate --test platform_doc_consumer_boundary",
    ] {
        assert!(
            script.contains(required_test),
            "Fix: nightly_ci.sh must run focused release-blocker test `{required_test}`."
        );
    }
}

#[test]

fn release_shell_scripts_use_shared_cargo_runner_selection() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("Fix: vyre-conform-runner must stay under the repository conform directory.");
    let helper = repo_file("scripts/lib/cargo_runner.sh");
    assert!(
        helper.contains("vyre_select_cargo_runner")
            && helper.contains("[[ -x ./cargo_full ]]")
            && helper.contains("CARGO_RUNNER=\"cargo\"")
            && helper.contains("CARGO_BUILD_JOBS=\"${CARGO_BUILD_JOBS:-1}\""),
        "Fix: scripts/lib/cargo_runner.sh must centralize cargo_full/cargo fallback with single-job builds."
    );

    for script in shell_scripts_under(root.join("scripts")) {
        let display = script
            .strip_prefix(root)
            .unwrap_or(&script)
            .display()
            .to_string();
        let contents = std::fs::read_to_string(&script).unwrap_or_else(|error| {
            panic!("Fix: shell script `{display}` must be readable: {error}")
        });
        assert!(
            !contents.contains("VYRE_CARGO_RUNNER:-./cargo_full"),
            "Fix: `{display}` must use scripts/lib/cargo_runner.sh instead of hardcoding a brittle ./cargo_full default."
        );
    }
}

fn shell_scripts_under(root: PathBuf) -> Vec<PathBuf> {
    let mut scripts = Vec::new();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|error| {
            panic!(
                "Fix: script directory `{}` must be readable: {error}",
                dir.display()
            )
        }) {
            let path = entry
                .unwrap_or_else(|error| {
                    panic!("Fix: script directory entry must be readable: {error}")
                })
                .path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "sh") {
                scripts.push(path);
            }
        }
    }
    scripts.sort();
    scripts
}

#[test]
fn conformance_tests_use_wrapped_backend_acquisition() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("src");
    let tests_dir = manifest_dir.join("tests");
    let mut findings = Vec::new();
    scan_for_raw_backend_factory_calls(&src_dir, &src_dir, &mut findings);
    scan_for_raw_backend_factory_calls(&tests_dir, &tests_dir, &mut findings);

    assert!(
        findings.is_empty(),
        "Fix: conformance runner code must call BackendRegistration::acquire() so grid-sync split and backend wrappers are applied:\n{}",
        findings.join("\n")
    );
}

#[test]
fn dispatch_conformance_isolates_backend_instance_per_pair() {
    let source = repo_file("conform/vyre-conform-runner/src/main.rs");
    let dispatch_start = source
        .find("fn dispatch_pairs(")
        .expect("Fix: conformance runner must expose dispatch_pairs.");
    let dispatch_end = source[dispatch_start..]
        .find("fn acquire_backend(")
        .map(|offset| dispatch_start + offset)
        .expect("Fix: dispatch_pairs must remain before acquire_backend.");
    let dispatch = &source[dispatch_start..dispatch_end];
    let prepare_pos = dispatch
        .find("let prepared = match prepare_entry(entry)")
        .expect("Fix: dispatch_pairs must prepare each entry before backend comparison.");
    let acquire_pos = dispatch
        .find("let backend = match acquire_backend(&backend_id)")
        .expect("Fix: dispatch_pairs must acquire the selected backend.");
    let compare_pos = dispatch
        .find("compare_backend_against_reference(")
        .expect("Fix: dispatch_pairs must compare backend output against reference.");
    assert!(
        prepare_pos < acquire_pos && acquire_pos < compare_pos,
        "Fix: dispatch conformance must acquire a fresh backend per prepared pair so a poisoned CUDA instance cannot taint later release evidence."
    );
    assert!(
        !dispatch[..prepare_pos].contains("acquire_backend(&backend_id)"),
        "Fix: dispatch conformance must not share one backend instance across all selected pairs."
    );
}

#[test]
fn release_conformance_static_sizing_uses_packed_buffer_lengths() {
    for path in [
        "conform/vyre-conform-runner/src/main.rs",
        "conform/vyre-conform-runner/tests/__split/parity_matrix_chunk1.rs",
        "vyre-libs/src/primitive_catalog.rs",
    ] {
        let source = repo_file(path);
        assert!(
            source.contains(".static_byte_len()"),
            "Fix: `{path}` must use BufferDecl::static_byte_len() so sub-byte static buffers use packed lengths."
        );
        assert!(
            !source.contains("buffer.element().min_bytes()"),
            "Fix: `{path}` must not size static dispatch buffers with min_bytes(); I4/FP4/NF4 buffers require packed byte lengths."
        );
    }

    for path in [
        "vyre-reference/src/execution/hashmap/mod.rs",
        "vyre-reference/src/execution/hashmap/memory.rs",
    ] {
        let source = repo_file(path);
        assert!(
            source.contains(".static_byte_len()"),
            "Fix: `{path}` must use BufferDecl::static_byte_len() so reference allocation mirrors packed backend buffer lengths."
        );
    }

    for path in [
        "vyre-reference/src/execution/hashmap/sync.rs",
        "vyre-reference/src/oob.rs",
    ] {
        let source = repo_file(path);
        assert!(
            source.contains(".bit_width()"),
            "Fix: `{path}` must compute logical element counts from DataType::bit_width() so sub-byte buffers report packed logical lengths."
        );
    }
}

#[test]
fn parity_matrix_input_planner_tracks_dynamic_fixture_contract() {
    for path in [
        "conform/vyre-conform-runner/src/main.rs",
        "conform/vyre-conform-runner/tests/__split/parity_matrix_chunk1.rs",
    ] {
        let source = repo_file(path);
        assert!(
            source.contains("matching_fixture_bytes(")
                && source.contains("fixture_index")
                && source.contains("byte_len: Option<usize>"),
            "Fix: `{path}` must route backend witness inputs by logical fixture order with optional static byte lengths, not only raw Program::buffers indices."
        );
        assert!(
            source.contains("runtime-sized read-write buffer"),
            "Fix: `{path}` must reject omitted runtime-sized read-write buffers instead of silently zeroing an unknown byte length."
        );
        assert!(
            !source.contains("fixture_buffer_count"),
            "Fix: `{path}` must not infer read-write fixture presence from a raw fixture count; use per-buffer fixture matching."
        );
    }
}

#[test]
fn conformance_tests_do_not_compile_out_gpu_gates() {
    let tests_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut findings = Vec::new();
    scan_for_cfg_gated_gpu_tests(&tests_dir, &tests_dir, &mut findings);

    assert!(
        findings.is_empty(),
        "Fix: conformance GPU gates must fail loudly when GPU drivers are not linked; do not compile out tests/modules with cfg(feature = \"gpu\"):\n{}",
        findings.join("\n")
    );
}

fn scan_for_raw_backend_factory_calls(root: &Path, path: &Path, findings: &mut Vec<String>) {
    let entries = std::fs::read_dir(path).unwrap_or_else(|error| {
        panic!(
            "Fix: expected test directory `{}` to be readable: {error}",
            path.display()
        )
    });
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "Fix: expected directory entry under `{}` to be readable: {error}",
                path.display()
            )
        });
        let path = entry.path();
        let file_type = entry.file_type().unwrap_or_else(|error| {
            panic!(
                "Fix: expected `{}` metadata to be readable: {error}",
                path.display()
            )
        });
        if file_type.is_dir() {
            scan_for_raw_backend_factory_calls(root, &path, findings);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!("Fix: expected `{}` to be readable: {error}", path.display())
        });
        let compact = source
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<String>();
        let member_factory_call = [".", "factory", ")()"].concat();
        let registration_factory_call = ["(", "registration", ".", "factory", ")()"].concat();
        if compact.contains(&member_factory_call) || compact.contains(&registration_factory_call) {
            let relative = path.strip_prefix(root).unwrap_or(path.as_path());
            findings.push(relative.display().to_string());
        }
    }
}

fn scan_for_cfg_gated_gpu_tests(root: &Path, path: &Path, findings: &mut Vec<String>) {
    let entries = std::fs::read_dir(path).unwrap_or_else(|error| {
        panic!(
            "Fix: expected test directory `{}` to be readable: {error}",
            path.display()
        )
    });
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "Fix: expected directory entry under `{}` to be readable: {error}",
                path.display()
            )
        });
        let path = entry.path();
        let file_type = entry.file_type().unwrap_or_else(|error| {
            panic!(
                "Fix: expected `{}` metadata to be readable: {error}",
                path.display()
            )
        });
        if file_type.is_dir() {
            scan_for_cfg_gated_gpu_tests(root, &path, findings);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!("Fix: expected `{}` to be readable: {error}", path.display())
        });
        let mut lines = source.lines().enumerate().peekable();
        while let Some((index, line)) = lines.next() {
            if !compacted_eq(line, "#[cfg(feature=\"gpu\")]") {
                continue;
            }
            let mut next = "";
            while let Some((_, candidate)) = lines.peek() {
                let trimmed = candidate.trim();
                if trimmed.is_empty() || trimmed.starts_with("//") {
                    lines.next();
                    continue;
                }
                next = *candidate;
                break;
            }
            if compacted_eq(next, "#[test]") || compacted_starts_with(next, "mod") {
                let relative = path.strip_prefix(root).unwrap_or(path.as_path());
                findings.push(format!("{}:{}", relative.display(), index + 1));
            }
        }
    }
}

fn compacted_eq(input: &str, expected: &str) -> bool {
    let mut expected = expected.chars();
    for ch in input.chars().filter(|ch| !ch.is_whitespace()) {
        match expected.next() {
            Some(expected_ch) if expected_ch == ch => {}
            _ => return false,
        }
    }
    expected.next().is_none()
}

fn compacted_starts_with(input: &str, expected: &str) -> bool {
    let mut expected = expected.chars();
    for ch in input.chars().filter(|ch| !ch.is_whitespace()) {
        match expected.next() {
            Some(expected_ch) if expected_ch == ch => {}
            Some(_) => return false,
            None => return true,
        }
    }
    expected.next().is_none()
}

fn concrete_driver_crates() -> Vec<String> {
    let manifest = repo_file("Cargo.toml");
    // The workspace `members` section uses bare quoted strings like
    // `"vyre-driver-wgpu",`. The `[workspace.dependencies]` table
    // uses the same prefix in lines like
    // `vyre-driver-wgpu = { version = ... }`  -  those must NOT match
    // here, otherwise the whole dep line gets treated as a crate name.
    // Restrict to lines whose trimmed-of-quotes/commas form is a bare
    // crate name (no spaces, no `=`).
    manifest
        .lines()
        .filter_map(|line| {
            let member = line.trim().trim_matches(',').trim_matches('"');
            (member.starts_with("vyre-driver-") && !member.contains(' ') && !member.contains('='))
                .then(|| member.to_string())
        })
        .collect()
}
