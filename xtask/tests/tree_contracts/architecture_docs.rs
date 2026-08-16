//! Current architecture documentation coherence tests.
//!
//! WHY: `docs/ARCHITECTURE.md` is prose about six live authorities, and prose
//! drifts from all six without a compile error. Each case here poisons exactly
//! one authority or one phrase in a fixture workspace and asserts the
//! `architecture-contract` gate reports it, plus one case over the real checkout
//! so the gate cannot pass fixtures while judging no published document.

use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::{json, Value};

use super::workspace_sources::workspace_root;
use xtask::gate::{Gate, GateCtx, Report};
use xtask::gates::architecture_contract::ArchitectureContract;

fn judge(root: &Path) -> Report {
    ArchitectureContract
        .run(&GateCtx::new(root.to_path_buf(), Vec::new()))
        .expect("Fix: the gate must be able to read the fixture tree")
}

/// Every finding rendered as one line, for substring assertions.
fn rendered(report: &Report) -> String {
    report
        .findings
        .iter()
        .map(|finding| finding.message.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

/// The schema version and tier vocabulary the contract actually enforces.
///
/// Both are read from the generated schema this repository ships rather than
/// restated here. A hardcoded fixture is how these tests went red unnoticed:
/// the schema version was bumped and two tiers were deleted, and the fixture
/// kept asserting the retired shape, so every case failed on the coherence
/// check before reaching the behaviour it names.
fn enforced_schema_shape() -> (u64, Vec<String>) {
    let path = workspace_root().join("docs/generated/OP_SCHEMA.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!("Fix: the generated operation schema must be readable at {path:?}: {err}")
    });
    let schema: Value = serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("Fix: {path:?} must be valid JSON: {err}"));
    let version = schema["schema_version"]
        .as_u64()
        .expect("Fix: the generated operation schema must carry a numeric schema_version");
    let tiers: Vec<String> = schema["tier_counts"]
        .as_object()
        .expect("Fix: the generated operation schema must carry a tier_counts table")
        .keys()
        .cloned()
        .collect();
    assert!(
        !tiers.is_empty(),
        "Fix: tier_counts must name at least one tier, or the fixture below covers no tier at all"
    );
    (version, tiers)
}

fn current_header(title: &str, body: &str) -> String {
    format!("# {title}\n\nLast verified: 2026-08-04\n\nVyre 0.7.9.\n\n{body}\n")
}

fn write_fixture(root: &Path) {
    for directory in ["docs/generated", "release/evidence/backends"] {
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
        "schema_version = 2\n\n[[crate]]\npackage = \"vyre-megakernel\"\npath = \"vyre-megakernel\"\nowner = \"megakernel-compiler\"\nlayer = \"compiler-boundary\"\nresponsibility = \"Compile validated ProgramGraph inputs into canonical static and persistent megakernel artifacts without owning admission, execution, or lifecycle policy.\"\n\n[[crate.dependency]]\npackage = \"vyre-foundation\"\npurpose = \"Use typed IR and graph contracts.\"\nfeatures = []\nconditions = [\"always\"]\nkinds = [\"normal\"]\noptional = false\ndefault_features = true\nboundary = \"public\"\nseam = \"foundation-ir\"\n",
    )
    .unwrap();
    let (schema_version, tiers) = enforced_schema_shape();
    let operations: Vec<Value> = tiers
        .iter()
        .map(|tier| json!({"id": tier, "tier": tier}))
        .collect();
    let tier_counts: serde_json::Map<String, Value> =
        tiers.iter().map(|tier| (tier.clone(), json!(1))).collect();
    write_json(
        &root.join("docs/generated/OP_SCHEMA.json"),
        &json!({
            "schema_version": schema_version,
            "operation_count": operations.len(),
            "tier_counts": tier_counts,
            "operations": operations
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
            "Use generated/OP_SCHEMA.json. The semantic authority is vyre-foundation::operation::OperationRegistry; derived catalogs do not own shadow operation identities. Evidence selects CUDA as the preferred backend. Cross-program composition target. vyre-megakernel emits Artifact envelopes. Persistent protocol lives in vyre-runtime/src/resident_work_queue/. The older bytecode interpreter design is superseded.",
        ),
    )
    .unwrap();
    fs::write(
        root.join("docs/DOCS.toml"),
        "schema_version = 1\n\n[[page]]\npath = \"ARCHITECTURE.md\"\nstatus = \"current\"\n",
    )
    .unwrap();
    // Member manifests and one optimization lane: the gate also asserts that
    // every lane names paths that exist and `-p` packages a manifest declares,
    // so a fixture without them is not a coherent workspace.
    for member in ["a", "vyre-megakernel"] {
        fs::create_dir_all(root.join(member).join("src")).unwrap();
        fs::write(
            root.join(member).join("Cargo.toml"),
            format!("[package]\nname = \"{member}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n"),
        )
        .unwrap();
        fs::write(root.join(member).join("src/lib.rs"), "").unwrap();
    }
    fs::create_dir_all(root.join("docs/optimization")).unwrap();
    fs::write(
        root.join("docs/optimization/OWNERSHIP.toml"),
        "[lane.megakernel]\npurpose = \"Artifact freeze\"\nlayer = \"compiler-boundary\"\nwrite = [\"vyre-megakernel/src/**\"]\nrequired_commands = [\"test -p vyre-megakernel\"]\n",
    )
    .unwrap();
    // The gate lists the tree through git, so a fixture that is not a checkout
    // reports as unreadable rather than as coherent.
    let status = Command::new("git")
        .args(["init", "-q"])
        .current_dir(root)
        .status()
        .expect("Fix: git must launch for the fixture tree listing");
    assert!(status.success());
}

fn fixture() -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    write_fixture(temp.path());
    temp
}

/// Current architecture guides must agree with live workspace, operation, backend, and ownership authorities.
#[test]
fn workspace_architecture_documents_are_current() {
    let report = judge(&workspace_root());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
}

/// A complete fixture proves the gate accepts the current ownership split.
#[test]
fn coherent_architecture_fixture_passes() {
    let temp = fixture();
    let report = judge(temp.path());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
}

/// A retired release claim must not return to a current architecture guide.
///
/// This wrote `Vyre 0.6.9.` into one of three pages the fixture created that
/// the document list does not name, so the checker never opened the file the
/// test poisoned and the assertion below could only have passed by accident.
/// The fixture now writes exactly what the contract reads.
#[test]
fn retired_architecture_version_fails_closed() {
    let temp = fixture();
    let path = temp.path().join("docs/ARCHITECTURE.md");
    fs::write(
        &path,
        format!("{}\nVyre 0.6.9.\n", fs::read_to_string(&path).unwrap()),
    )
    .unwrap();
    let report = judge(temp.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(text.contains("stale architecture phrase `version 0.6`"), "{text}");
}

/// WGPU cannot be presented as the primary production route when evidence selects CUDA.
#[test]
fn wgpu_primary_route_fails_closed() {
    let temp = fixture();
    let path = temp.path().join("docs/ARCHITECTURE.md");
    fs::write(
        &path,
        format!(
            "{}\nWGPU is the primary production path.\n",
            fs::read_to_string(&path).unwrap()
        ),
    )
    .unwrap();
    let report = judge(temp.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(text.contains("primary production path"), "{text}");
}

/// Docs must not revive the pre-crate planned/absent claim for vyre-megakernel.
#[test]
fn planned_absent_megakernel_claim_fails_closed() {
    let temp = fixture();
    let path = temp.path().join("docs/ARCHITECTURE.md");
    fs::write(
        &path,
        format!(
            "{}\nThe planned `vyre-megakernel` crate is not a current workspace member.\n",
            fs::read_to_string(&path).unwrap()
        ),
    )
    .unwrap();
    let report = judge(temp.path());
    let text = rendered(&report);
    assert!(
        text.contains("not a current workspace"),
        "Fix: the planned/absent claim must fail closed; findings={text}"
    );
}

/// Removing the semantic registry authority must fail the joined operation model.
#[test]
fn missing_semantic_operation_registry_fails_closed() {
    let temp = fixture();
    let path = temp.path().join("docs/ARCHITECTURE.md");
    let text = fs::read_to_string(&path).unwrap().replace(
        "vyre-foundation::operation::OperationRegistry",
        "runtime registry",
    );
    fs::write(&path, text).unwrap();
    let report = judge(temp.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(text.contains("operation::OperationRegistry"), "{text}");
}

/// A preferred backend without an executable row must not support an architecture claim.
#[test]
fn missing_preferred_backend_probe_fails_closed() {
    let temp = fixture();
    let path = temp
        .path()
        .join("release/evidence/backends/backend-matrix.json");
    let mut matrix: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    matrix["backends"] = json!([{"id": "wgpu"}]);
    write_json(&path, &matrix);
    let report = judge(temp.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(text.contains("no executable probe row"), "{text}");
}

/// A current architecture page must remain current in the documentation manifest.
#[test]
fn architecture_page_manifest_status_fails_closed() {
    let temp = fixture();
    let path = temp.path().join("docs/DOCS.toml");
    fs::write(
        &path,
        fs::read_to_string(&path).unwrap().replace(
            "path = \"ARCHITECTURE.md\"\nstatus = \"current\"",
            "path = \"ARCHITECTURE.md\"\nstatus = \"archived\"",
        ),
    )
    .unwrap();
    let report = judge(temp.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(text.contains("rather than current"), "{text}");
}

/// A lane whose write glob matches nothing hands an owner scope they cannot
/// enter, and nothing fails until they try.
#[test]
fn a_lane_naming_absent_scope_fails_closed() {
    let temp = fixture();
    let path = temp.path().join("docs/optimization/OWNERSHIP.toml");
    fs::write(
        &path,
        fs::read_to_string(&path)
            .unwrap()
            .replace("vyre-megakernel/src/**", "vyre-deleted/src/**"),
    )
    .unwrap();
    let report = judge(temp.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(text.contains("matches nothing in the tree"), "{text}");
}

/// A lane command naming a package no manifest declares is the same defect one
/// level further in, and it is the one that outlived a deleted crate three times.
#[test]
fn a_lane_command_naming_an_absent_package_fails_closed() {
    let temp = fixture();
    let path = temp.path().join("docs/optimization/OWNERSHIP.toml");
    fs::write(
        &path,
        fs::read_to_string(&path)
            .unwrap()
            .replace("test -p vyre-megakernel", "test -p vyre-scan"),
    )
    .unwrap();
    let report = judge(temp.path());
    let text = rendered(&report);
    assert_eq!(report.count(), 1, "{text}");
    assert!(
        text.contains("names package `vyre-scan`, which no workspace manifest declares"),
        "{text}"
    );
}
