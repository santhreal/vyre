//! Test: rewrite layer contract.
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

fn workspace_root() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root()
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("failed to read {}: {error}", path.display());
    })
}

#[derive(Debug, Deserialize)]
struct KernelFamilySurfaceManifest {
    schema_version: u32,
    contract: String,
    family: Vec<KernelFamilySurface>,
}

#[derive(Debug, Deserialize)]
struct KernelFamilySurface {
    family_id: String,
    owner_lane: String,
    root: String,
    public_reexport: String,
    schedule_config: String,
    evidence_writer: String,
    forbid_section_dividers: bool,
    forbidden_private_import_prefixes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct KernelFamilySchedule {
    schema_version: u32,
    contract: String,
    family_id: String,
    stage_order: Vec<String>,
    evidence_policy: String,
}

#[test]
fn descriptor_lowering_has_no_semantic_rewrite_module() {
    assert!(
        !workspace_root().join("vyre-lower/src/rewrites").exists(),
        "verified lowering must not contain a semantic descriptor rewrite layer"
    );
}

#[test]
fn kernel_family_surfaces_have_single_reexport_schedule_and_evidence_contracts() {
    let root = workspace_root();
    let manifest_path = root.join("vyre-lower/rules/kernel_family_surfaces.toml");
    let manifest: KernelFamilySurfaceManifest = toml::from_str(&read(&manifest_path))
        .expect("Fix: kernel family surface manifest must be valid TOML.");

    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.contract, "vyre-kernel-family-surfaces:v1");
    assert!(
        !manifest.family.is_empty(),
        "Fix: kernel family surface manifest must declare at least one family."
    );

    let mut public_reexports = std::collections::BTreeSet::new();
    let mut failures = Vec::new();
    for family in &manifest.family {
        if family.family_id.trim().is_empty() {
            failures.push("family_id is blank".to_string());
        }
        if family.owner_lane.trim().is_empty() {
            failures.push(format!("{} owner_lane is blank", family.family_id));
        }
        for (label, rel_path) in [
            ("root", &family.root),
            ("public_reexport", &family.public_reexport),
            ("schedule_config", &family.schedule_config),
            ("evidence_writer", &family.evidence_writer),
        ] {
            if !root.join(rel_path).exists() {
                failures.push(format!(
                    "{} {label} `{rel_path}` does not exist",
                    family.family_id
                ));
            }
        }
        if !public_reexports.insert(family.public_reexport.as_str()) {
            failures.push(format!(
                "duplicate public re-export point `{}`",
                family.public_reexport
            ));
        }
        if let Ok(schedule_text) = fs::read_to_string(root.join(&family.schedule_config)) {
            let schedule: KernelFamilySchedule =
                toml::from_str(&schedule_text).unwrap_or_else(|error| {
                    panic!(
                        "Fix: schedule config `{}` must be valid TOML: {error}",
                        family.schedule_config
                    )
                });
            if schedule.schema_version != 1 {
                failures.push(format!(
                    "{} schedule schema_version must be 1",
                    family.family_id
                ));
            }
            if schedule.contract != "vyre-kernel-family-schedule:v1" {
                failures.push(format!(
                    "{} schedule contract must be vyre-kernel-family-schedule:v1",
                    family.family_id
                ));
            }
            if schedule.family_id != family.family_id {
                failures.push(format!(
                    "{} schedule family_id `{}` does not match manifest",
                    family.family_id, schedule.family_id
                ));
            }
            if schedule.stage_order.len() < 3 {
                failures.push(format!(
                    "{} schedule must declare at least three stages",
                    family.family_id
                ));
            }
            if schedule.evidence_policy.trim().is_empty() {
                failures.push(format!(
                    "{} schedule evidence_policy is blank",
                    family.family_id
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Kernel family organization contract failed:\n{}",
        failures.join("\n")
    );
}
