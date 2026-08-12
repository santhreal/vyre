//! Current architecture documentation coherence tests.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{json, Value};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives under the workspace root")
        .to_path_buf()
}

fn run_checker(root: &Path) -> Output {
    Command::new("python3")
        .arg(workspace_root().join("scripts/architecture_docs.py"))
        .arg(root)
        .arg("--check")
        .output()
        .expect("architecture docs checker must launch")
}

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

fn current_header(title: &str, body: &str) -> String {
    format!("# {title}\n\nLast verified: 2026-08-04\n\nVyre 0.7.9.\n\n{body}\n")
}

fn write_fixture(root: &Path) {
    for directory in ["docs/generated", "docs/rfcs", "release/evidence/backends"] {
        fs::create_dir_all(root.join(directory)).unwrap();
    }
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = [\"a\", \"vyre-megakernel\"]\n",
    )
    .unwrap();
    fs::write(
        root.join("release/release-train.toml"),
        "[versions]\nvyre = \"0.7.9\"\n",
    )
    .unwrap();
    fs::write(
        root.join("docs/CRATE_OWNERSHIP.toml"),
        "schema_version = 1\n\n[[crate]]\npackage = \"vyre-megakernel\"\npath = \"vyre-megakernel\"\nowner = \"megakernel-compiler\"\nlayer = \"compiler-boundary\"\nresponsibility = \"Compile validated ProgramGraph inputs into canonical static and persistent megakernel artifacts without owning admission, execution, or lifecycle policy.\"\nallowed_dependencies = [\"vyre-foundation\"]\n",
    )
    .unwrap();
    write_json(
        &root.join("docs/generated/OP_SCHEMA.json"),
        &json!({
            "schema_version": 2,
            "operation_count": 4,
            "tier_counts": {"intrinsic": 1, "primitive": 1, "libs": 1, "runtime": 1},
            "operations": [
                {"id": "i", "tier": "intrinsic"},
                {"id": "p", "tier": "primitive"},
                {"id": "l", "tier": "libs"},
                {"id": "r", "tier": "runtime"}
            ]
        }),
    );
    write_json(
        &root.join("release/evidence/backends/backend-matrix.json"),
        &json!({
            "preferred_backend_id": "cuda",
            "blockers": [],
            "backends": [
                {"id": "cuda"},
                {"id": "wgpu"}
            ]
        }),
    );

    fs::write(
        root.join("docs/ARCHITECTURE.md"),
        current_header(
            "Architecture",
            "Use generated/OP_SCHEMA.json. The semantic authority is vyre-foundation::operation_registry; derived catalogs do not own shadow operation identities. Evidence selects CUDA as the preferred backend. Cross-program composition target. vyre-megakernel emits Artifact envelopes. Persistent protocol lives in vyre-runtime/src/megakernel/. The older bytecode interpreter design is superseded.",
        ),
    )
    .unwrap();
    fs::write(
        root.join("docs/OPTIMIZATION_ARCHITECTURE.md"),
        current_header(
            "Optimization architecture",
            "Layer 1: semantic IR optimization. Layer 2: concrete lowering strategy. Current scheduling is in vyre-runtime/src/megakernel/. Artifact freeze is owned by vyre-megakernel. IR pre-dispatch fusion lives in vyre-foundation/src/optimizer/megakernel.",
        ),
    )
    .unwrap();
    fs::write(
        root.join("docs/RUNTIME_PIPELINE.md"),
        current_header(
            "Runtime pipeline",
            "Cache: vyre-runtime/src/pipeline_cache/. Runtime: vyre-runtime/src/megakernel/. Neutral artifacts come from vyre-megakernel and enter through artifact_admission. Failure does not silently rerun. Prose does not substitute for raw samples.",
        ),
    )
    .unwrap();
    fs::write(
        root.join("docs/megakernel-wiring.md"),
        current_header(
            "Megakernel wiring",
            "Execution starts from the same validated `Program` and does not consume a general VIR bytecode interpreter. Ownership: vyre-runtime/src/megakernel/. Artifact compiler vyre-megakernel emits Artifact. Wave policy lives in vyre-driver/src/megakernel_execution. IR fusion lives in vyre-foundation/src/optimizer/megakernel. Residue note: vyre-driver-wgpu/src/megakernel was removed.",
        ),
    )
    .unwrap();
    fs::write(
        root.join("docs/rfcs/0005-persistent-megakernel.md"),
        "# RFC\n\nLast verified: 2026-08-04\n\nStatus: **Superseded**\n\nHistorical motivation. Superseded design. Current resolution. Vyre does not support a general interpreter. `vyre-megakernel` is a current workspace member that compiles typed graphs to canonical artifacts.\n",
    )
    .unwrap();
    fs::write(
        root.join("docs/DOCS.toml"),
        "schema_version = 1\n\n[[page]]\npath = \"ARCHITECTURE.md\"\nstatus = \"current\"\n\n[[page]]\npath = \"OPTIMIZATION_ARCHITECTURE.md\"\nstatus = \"current\"\n\n[[page]]\npath = \"RUNTIME_PIPELINE.md\"\nstatus = \"current\"\n\n[[page]]\npath = \"megakernel-wiring.md\"\nstatus = \"current\"\n\n[[page]]\npath = \"rfcs/0005-persistent-megakernel.md\"\nstatus = \"superseded\"\n",
    )
    .unwrap();
}

/// Current architecture guides must agree with live workspace, operation, backend, and ownership authorities.
#[test]
fn workspace_architecture_documents_are_current() {
    let output = run_checker(&workspace_root());
    assert!(
        output.status.success(),
        "Fix: repair current architecture docs: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A complete fixture proves the checker accepts the current ownership split.
#[test]
fn coherent_architecture_fixture_passes() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path());
    assert!(
        run_checker(temp.path()).status.success(),
        "Fix: coherent architecture fixture must pass: {}",
        String::from_utf8_lossy(&run_checker(temp.path()).stderr)
    );
}

/// A retired release claim must not return to a current architecture guide.
#[test]
fn retired_architecture_version_fails_closed() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path());
    let path = temp.path().join("docs/RUNTIME_PIPELINE.md");
    fs::write(
        &path,
        format!("{}\nVyre 0.6.9.\n", fs::read_to_string(&path).unwrap()),
    )
    .unwrap();
    let output = run_checker(temp.path());
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("stale architecture pattern"));
}

/// WGPU cannot be presented as the primary production route when evidence selects CUDA.
#[test]
fn wgpu_primary_route_fails_closed() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path());
    let path = temp.path().join("docs/ARCHITECTURE.md");
    fs::write(
        &path,
        format!(
            "{}\nWGPU is the primary production path.\n",
            fs::read_to_string(&path).unwrap()
        ),
    )
    .unwrap();
    let output = run_checker(temp.path());
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("primary production path"));
}

/// Docs must not revive the pre-crate planned/absent claim for vyre-megakernel.
#[test]
fn planned_absent_megakernel_claim_fails_closed() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path());
    let path = temp.path().join("docs/ARCHITECTURE.md");
    fs::write(
        &path,
        format!(
            "{}\nThe planned `vyre-megakernel` crate is not a current workspace member.\n",
            fs::read_to_string(&path).unwrap()
        ),
    )
    .unwrap();
    let output = run_checker(temp.path());
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("stale architecture pattern")
            || stderr.contains("architecture still presents"),
        "Fix: planned/absent claim must fail closed; stderr={stderr}"
    );
}

/// Removing the semantic registry authority must fail the joined operation model.
#[test]
fn missing_semantic_operation_registry_fails_closed() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path());
    let path = temp.path().join("docs/ARCHITECTURE.md");
    let text = fs::read_to_string(&path)
        .unwrap()
        .replace("vyre-foundation::operation_registry", "runtime registry");
    fs::write(&path, text).unwrap();
    let output = run_checker(temp.path());
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("operation_registry"));
}

/// A preferred backend without an executable row must not support an architecture claim.
#[test]
fn missing_preferred_backend_probe_fails_closed() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path());
    let path = temp
        .path()
        .join("release/evidence/backends/backend-matrix.json");
    let mut matrix: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    matrix["backends"] = json!([{"id": "wgpu"}]);
    write_json(&path, &matrix);
    let output = run_checker(temp.path());
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no executable probe row"));
}

/// The bytecode-interpreter RFC must remain visibly superseded in both file and manifest.
#[test]
fn megakernel_rfc_cannot_return_to_current_status() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path());
    let path = temp.path().join("docs/DOCS.toml");
    fs::write(
        &path,
        fs::read_to_string(&path).unwrap().replace(
            "path = \"rfcs/0005-persistent-megakernel.md\"\nstatus = \"superseded\"",
            "path = \"rfcs/0005-persistent-megakernel.md\"\nstatus = \"current\"",
        ),
    )
    .unwrap();
    let output = run_checker(temp.path());
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("classify"));
}

/// A current architecture page must remain current in the documentation manifest.
#[test]
fn architecture_page_manifest_status_fails_closed() {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path());
    let path = temp.path().join("docs/DOCS.toml");
    fs::write(
        &path,
        fs::read_to_string(&path).unwrap().replace(
            "path = \"ARCHITECTURE.md\"\nstatus = \"current\"",
            "path = \"ARCHITECTURE.md\"\nstatus = \"archived\"",
        ),
    )
    .unwrap();
    let output = run_checker(temp.path());
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("ARCHITECTURE.md"));
}
