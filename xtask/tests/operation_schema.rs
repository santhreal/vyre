//! Canonical live operation schema contract tests.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("Fix: xtask must remain directly under the workspace root")
        .to_path_buf()
}

fn schema_path() -> PathBuf {
    workspace_root().join("docs/generated/OP_SCHEMA.json")
}

fn read_schema() -> Value {
    serde_json::from_str(
        &fs::read_to_string(schema_path())
            .expect("Fix: generated operation schema must be readable"),
    )
    .expect("Fix: generated operation schema must be valid JSON")
}

fn run_xtask(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(arguments)
        .current_dir(workspace_root())
        .output()
        .expect("Fix: xtask operation schema command must launch")
}

fn assert_mutation_rejected<F>(expected_error: &str, mutate: F)
where
    F: FnOnce(&mut Value),
{
    let mut candidate = read_schema();
    mutate(&mut candidate);
    let temp = tempfile::tempdir().expect("Fix: schema mutation directory must be creatable");
    let path = temp.path().join("candidate.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(&candidate).expect("Fix: mutation must remain serializable"),
    )
    .expect("Fix: mutated schema must be writable");
    let output = run_xtask(&["operation-schema", "--validate", path.to_str().unwrap()]);
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "Fix: schema mutation was accepted"
    );
    assert!(
        error.contains(expected_error),
        "Fix: rejection must name `{expected_error}`; stderr={error}"
    );
}

/// The committed JSON must be the exact serialization of today's live registrations.
///
/// This prevents a source operation change from leaving release catalogs and counts on
/// the previous registry state.
#[test]
fn committed_schema_matches_live_registrations() {
    let output = run_xtask(&["operation-schema", "--check"]);
    assert!(
        output.status.success(),
        "Fix: regenerate the canonical operation schema: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Both Markdown catalogs must be exact projections of the committed JSON contract.
///
/// This prevents a correct schema from coexisting with stale public counts or rows in
/// the generated inventory and subsystem pages.
#[test]
fn schema_derived_markdown_views_are_current() {
    for arguments in [
        ["list-ops", "--check"].as_slice(),
        ["catalog", "--check"].as_slice(),
    ] {
        let output = run_xtask(arguments);
        assert!(
            output.status.success(),
            "Fix: regenerate `{}` from the canonical operation schema: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// Every live row must carry all nine documentation surfaces and aggregate counts must be exact.
///
/// This locks out the former ID-only inventory, which could look current while omitting
/// signatures, feature gates, oracles, backend truth, laws, and composition evidence.
#[test]
fn schema_rows_cover_every_required_operation_contract() {
    let schema = read_schema();
    let operations = schema["operations"]
        .as_array()
        .expect("Fix: operations must be an array");
    assert_eq!(
        schema["operation_count"].as_u64().unwrap() as usize,
        operations.len()
    );
    assert_eq!(
        schema["tier_counts"]
            .as_object()
            .unwrap()
            .values()
            .map(|count| count.as_u64().unwrap())
            .sum::<u64>() as usize,
        operations.len()
    );
    assert_eq!(
        schema["category_counts"]
            .as_object()
            .unwrap()
            .values()
            .map(|count| count.as_u64().unwrap())
            .sum::<u64>() as usize,
        operations.len()
    );
    for operation in operations {
        let id = operation["id"].as_str().expect("Fix: op id must be text");
        assert!(!id.is_empty(), "Fix: op id must not be empty");
        assert!(!operation["tier"].as_str().unwrap().is_empty());
        assert!(!operation["category"].as_str().unwrap().is_empty());
        let signature = operation["signature"].as_object().unwrap();
        match signature["kind"].as_str().unwrap() {
            "program_buffers" => assert!(!signature["buffers"].as_array().unwrap().is_empty()),
            "dialect_parameters" => assert!(
                !signature["inputs"].as_array().unwrap().is_empty()
                    || !signature["outputs"].as_array().unwrap().is_empty()
            ),
            kind => panic!("Fix: {id} has unknown signature kind `{kind}`"),
        }
        assert!(!operation["features"].as_array().unwrap().is_empty());
        let reference_supported =
            operation["backend_support"]["reference"]["status"] == "supported";
        assert_eq!(
            operation["oracle"]["reference_eval"].as_bool().unwrap(),
            reference_supported
        );
        for backend in ["reference", "cuda", "wgpu"] {
            assert!(
                !operation["backend_support"][backend]["status"]
                    .as_str()
                    .unwrap()
                    .is_empty(),
                "Fix: {id} needs an exact {backend} status"
            );
        }
        assert!(operation["laws"].is_array());
        assert!(operation["composition_chain"].is_array());
    }
}

/// Runtime dialect operations must share the same authority as Program-backed operations.
///
/// This prevents the five live `OpDef` registrations from disappearing behind a
/// 360-row `OpEntry` count while the backend matrix still requires 365 operations.
#[test]
fn runtime_dialect_contracts_are_typed_and_fail_closed_without_reference_fallback() {
    let schema = read_schema();
    let operations = schema["operations"].as_array().unwrap();
    let expected = [
        (
            "core.indirect_dispatch",
            &["GpuBufferHandle<[u32;3]>"][..],
            &[][..],
        ),
        (
            "io.dma_from_nvme",
            &["i32", "u64", "u64"][..],
            &["GpuBufferHandle"][..],
        ),
        (
            "io.write_back_to_nvme",
            &["GpuBufferHandle", "i32", "u64"][..],
            &[][..],
        ),
        ("mem.unmap", &["GpuBufferHandle"][..], &[][..]),
        ("mem.zerocopy_map", &["i32"][..], &["GpuBufferHandle"][..]),
    ];
    assert_eq!(schema["tier_counts"]["runtime"], expected.len());
    for (id, expected_inputs, expected_outputs) in expected {
        let operation = operations
            .iter()
            .find(|operation| operation["id"] == id)
            .unwrap_or_else(|| panic!("Fix: runtime operation `{id}` must be cataloged"));
        assert_eq!(operation["tier"], "runtime");
        assert_eq!(operation["signature"]["kind"], "dialect_parameters");
        let inputs = operation["signature"]["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|parameter| parameter["data_type"].as_str().unwrap())
            .collect::<Vec<_>>();
        let outputs = operation["signature"]["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|parameter| parameter["data_type"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(inputs, expected_inputs);
        assert_eq!(outputs, expected_outputs);
        assert_eq!(operation["features"], serde_json::json!(["default"]));
        assert_eq!(operation["oracle"]["reference_eval"], false);
        assert_eq!(
            operation["backend_support"]["reference"]["status"],
            "not_applicable"
        );
        assert_eq!(
            operation["backend_support"]["cuda"]["status"],
            "experimental"
        );
        assert_eq!(
            operation["backend_support"]["wgpu"]["status"],
            "experimental"
        );
        assert_eq!(operation["composition_chain"], serde_json::json!([]));
    }
}

/// Duplicate operation IDs must fail before a catalog can merge two unrelated contracts.
#[test]
fn duplicate_operation_id_fails_closed() {
    assert_mutation_rejected("empty or duplicated", |schema| {
        let operations = schema["operations"].as_array_mut().unwrap();
        operations[1]["id"] = operations[0]["id"].clone();
    });
}

/// An operation tier cannot drift from the canonical ID classifier.
#[test]
fn mismatched_tier_fails_closed() {
    assert_mutation_rejected("tier", |schema| {
        schema["operations"][0]["tier"] = Value::String("libs".to_string());
    });
}

/// A missing category must not be rendered as a plausible uncategorized catalog row.
#[test]
fn empty_category_fails_closed() {
    assert_mutation_rejected("invalid category", |schema| {
        schema["operations"][0]["category"] = Value::String(String::new());
    });
}

/// A built program without its exact buffer signature must not enter documentation.
#[test]
fn empty_signature_fails_closed() {
    assert_mutation_rejected("invalid operation signature", |schema| {
        let operation = schema["operations"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|operation| operation["signature"]["kind"] == "program_buffers")
            .expect("Fix: live schema must include a Program-backed operation");
        operation["signature"]["buffers"] = Value::Array(Vec::new());
    });
}

/// Feature documentation must name the canonical route that links the registration.
#[test]
fn changed_feature_route_fails_closed() {
    assert_mutation_rejected("feature route", |schema| {
        schema["operations"][0]["features"] = Value::Array(Vec::new());
    });
}

/// Every backend-supported reference oracle must remain explicit in the schema.
#[test]
fn disabled_reference_oracle_fails_closed() {
    assert_mutation_rejected("reference oracle", |schema| {
        let operation = schema["operations"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|operation| operation["oracle"]["reference_eval"] == true)
            .expect("Fix: live schema must include a reference-evaluated operation");
        operation["oracle"]["reference_eval"] = Value::Bool(false);
    });
}

/// Removing one release backend status must invalidate the whole schema.
#[test]
fn missing_backend_support_fails_closed() {
    assert_mutation_rejected("missing valid backend `cuda`", |schema| {
        schema["operations"][0]["backend_support"]
            .as_object_mut()
            .unwrap()
            .remove("cuda");
    });
}

/// Empty or duplicated law names must not create ambiguous algebraic claims.
#[test]
fn malformed_laws_fail_closed() {
    assert_mutation_rejected("laws must be", |schema| {
        schema["operations"][0]["laws"] = Value::Array(vec![
            Value::String(String::new()),
            Value::String(String::new()),
        ]);
    });
}

/// A composition step cannot claim registration when its operation ID is only an internal stage.
#[test]
fn false_registered_composition_step_fails_closed() {
    assert_mutation_rejected("inconsistent composition chain", |schema| {
        let operations = schema["operations"].as_array_mut().unwrap();
        let operation = operations
            .iter_mut()
            .find(|operation| {
                operation["composition_chain"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|step| step["registered"] == false)
            })
            .expect("Fix: live schema must retain an internal composition stage");
        let step = operation["composition_chain"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|step| step["registered"] == false)
            .unwrap();
        step["registered"] = Value::Bool(true);
    });
}

/// Aggregate counts must be recomputed from rows rather than copied into prose.
#[test]
fn stale_operation_count_fails_closed() {
    assert_mutation_rejected("operation_count", |schema| {
        let count = schema["operation_count"].as_u64().unwrap();
        schema["operation_count"] = Value::from(count + 1);
    });
}
