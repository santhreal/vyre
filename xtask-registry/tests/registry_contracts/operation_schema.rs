//! Canonical live operation schema contract tests.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;
use vyre_foundation::operation::{operation_id_namespace, IdNamespace};

fn workspace_root() -> PathBuf {
    structure_gate::workspace_root()
}

/// Workspace member crate names, read from the root manifest at run time.
fn workspace_members() -> BTreeSet<String> {
    let manifest = fs::read_to_string(workspace_root().join("Cargo.toml"))
        .expect("Fix: workspace root manifest must be readable");
    let parsed: toml::Value =
        toml::from_str(&manifest).expect("Fix: workspace root manifest must be valid TOML");
    parsed["workspace"]["members"]
        .as_array()
        .expect("Fix: [workspace] must declare members")
        .iter()
        .filter_map(|member| member.as_str())
        .map(|path| path.rsplit('/').next().unwrap_or(path).to_string())
        .collect()
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
    Command::new(env!("CARGO_BIN_EXE_xtask-registry"))
        .args(arguments)
        .current_dir(workspace_root())
        .output()
        .expect("Fix: xtask operation schema command must launch")
}

/// Assert the validator rejects a mutated schema, and says why.
///
/// A delegated gate binary returns its report as JSON on stdout and exits 0; the
/// parent dispatcher treats a non-zero exit as a gate that could not run at all.
/// So the verdict is the report, not the status: asserting on the status made
/// every one of these mutations read as rejected while the validator was
/// accepting them, and the same assertion passed against a schema that already
/// disagreed with the live registry.
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
    let findings = reported_findings(&output);
    assert!(
        !findings.is_empty(),
        "Fix: schema mutation was accepted; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains(expected_error)),
        "Fix: rejection must name `{expected_error}`; findings={findings:?}"
    );
}

/// Every finding message the validator reported, as `message` plus its fix.
fn reported_findings(output: &Output) -> Vec<String> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|error| {
        panic!(
            "Fix: the validator must print one JSON report; {error}; stdout={stdout} stderr={}",
            String::from_utf8_lossy(&output.stderr)
        )
    });
    report
        .get("findings")
        .and_then(Value::as_array)
        .map(|findings| {
            findings
                .iter()
                .map(|finding| {
                    let message = finding.get("message").and_then(Value::as_str).unwrap_or("");
                    let fix = finding.get("fix").and_then(Value::as_str).unwrap_or("");
                    format!("{message} {fix}")
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The committed JSON must be the exact serialization of today's live registrations.
///
/// This prevents a source operation change from leaving release catalogs and counts on
/// the previous registry state.
#[test]
fn committed_schema_matches_live_registrations() {
    let output = run_xtask(&["operation-schema"]);
    assert!(
        output.status.success(),
        "Fix: regenerate the canonical operation schema: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Both generated inventories must be exact projections of the live registry.
///
/// This prevents a correct schema from coexisting with stale counts or rows in
/// `docs/generated/op-inventory.toml` and `docs/generated/catalog.toml`.
#[test]
fn schema_derived_markdown_views_are_current() {
    for arguments in [["list-ops"].as_slice(), ["catalog"].as_slice()] {
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
        let constraints = operation["schedule_constraints"].as_object().unwrap();
        assert!(!constraints["workgroup_width"].as_str().unwrap().is_empty());
        assert!(!constraints["subgroup_width"].as_str().unwrap().is_empty());
        assert!(!constraints["element_policy"].as_str().unwrap().is_empty());
        assert!(!constraints["uniformity"].as_str().unwrap().is_empty());
        assert!(constraints["min_shared_bytes"].is_u64());
        assert!(constraints["cooperative_launch"].is_boolean());
        assert!(
            constraints["memory_ordering"].is_null() || constraints["memory_ordering"].is_string()
        );
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
/// Every operation id names a crate that exists.
///
/// WHY: `vyre-driver` registered `core.indirect_dispatch`, `io.dma_from_nvme`,
/// `io.write_back_to_nvme`, `mem.zerocopy_map` and `mem.unmap` as operations
/// carrying a signature and no program. A host-side runtime capability has no
/// program to lower and no fixture to compare against, so those five ids were a
/// second identity for capabilities that already live on the backend capability
/// surface and in `vyre-runtime`. `operation_id_namespace` returns `Unknown` for
/// a dotted id and `OperationRegistry` refuses it, so re-adding one turns this
/// red at registry construction.
///
/// The member list is the live registry, not a literal, so a newly registered
/// violation is judged without editing this test.
///
/// What it does not catch: where the definition lives. The namespace is frozen
/// at mint time and 130 operations moved crate while keeping theirs, so
/// `crate-structure` owns placement by reading the file each registration lives
/// in.
#[test]
fn every_registered_operation_id_names_a_workspace_crate() {
    let members = workspace_members();
    let registry = vyre_registry_link::operation::live_operation_registry();

    assert!(
        registry.iter().len() > 100,
        "Fix: xtask must link every operation crate; saw only {} registrations, so this rule is judging nothing",
        registry.iter().len()
    );

    let mut findings = Vec::new();
    for operation in registry.iter() {
        let Some((crate_name, rest)) = operation.id.split_once("::") else {
            findings.push(format!(
                "`{}` carries no `crate::` namespace; an operation id names its owning crate, and a host capability is reached through the driver capability surface instead",
                operation.id
            ));
            continue;
        };
        if crate_name.is_empty() || rest.is_empty() {
            findings.push(format!("`{}` has an empty namespace segment", operation.id));
            continue;
        }
        if crate_name.starts_with("vyre-") && !members.contains(crate_name) {
            findings.push(format!(
                "`{}` claims workspace crate `{crate_name}`, which is not a workspace member",
                operation.id
            ));
        }
        if operation_id_namespace(operation.id) == IdNamespace::Unknown {
            findings.push(format!(
                "`{}` names no minting crate; an id is `<crate>::<path>`",
                operation.id
            ));
        }
    }

    assert!(
        findings.is_empty(),
        "{} operation id namespace violation(s):\n  - {}",
        findings.len(),
        findings.join("\n  - ")
    );
}

/// The catalog tier vocabulary carries no retired spelling.
#[test]
fn schema_tier_counts_use_only_live_tier_spellings() {
    let schema = read_schema();
    let tier_counts = schema["tier_counts"].as_object().unwrap();

    for tier in tier_counts.keys() {
        assert!(
            ["foundation_ir", "intrinsic", "libs", "external"].contains(&tier.as_str()),
            "Fix: `{tier}` is not a live OperationTier spelling"
        );
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

/// Every width decision is positive or explicitly agnostic.
#[test]
fn invalid_schedule_width_fails_closed() {
    for field in ["workgroup_width", "subgroup_width"] {
        assert_mutation_rejected("invalid", |schema| {
            schema["operations"][0]["schedule_constraints"][field] =
                Value::String("exactly:0".to_string());
        });
    }
}

/// Cooperative launch is inseparable from grid-scope ordering.
#[test]
fn cooperative_launch_without_grid_ordering_fails_closed() {
    assert_mutation_rejected("cooperative launch", |schema| {
        schema["operations"][0]["schedule_constraints"]["cooperative_launch"] = Value::Bool(true);
        schema["operations"][0]["schedule_constraints"]["memory_ordering"] =
            Value::String("wire:4".to_string());
    });
}

/// Unknown ordering tags cannot enter the generated operation contract.
#[test]
fn unknown_schedule_ordering_fails_closed() {
    assert_mutation_rejected("memory ordering", |schema| {
        schema["operations"][0]["schedule_constraints"]["memory_ordering"] =
            Value::String("wire:255".to_string());
    });
}

/// An operation tier cannot drift from the canonical ID classifier.
///
/// WHY the mutated value is derived: this test wrote the literal `"libs"` into
/// the first operation's tier, and the first operation is a `vyre-libs`
/// registration whose tier already reads `libs`. The mutation was a no-op, so
/// the candidate schema was identical to the live one and the run was accepted,
/// which the test then reported as a fail-open. It only ever proved anything
/// while the first operation happened to be an intrinsic. The replacement takes
/// a tier the schema itself uses for some other operation, so the mutation is a
/// real disagreement whatever order the registry is in.
#[test]
fn mismatched_tier_fails_closed() {
    assert_mutation_rejected("tier", |schema| {
        let operations = schema["operations"].as_array().unwrap();
        let current = operations[0]["tier"].as_str().unwrap().to_string();
        let other = operations
            .iter()
            .find_map(|operation| {
                let tier = operation["tier"].as_str()?;
                (tier != current).then(|| tier.to_string())
            })
            .expect("Fix: the schema must carry more than one tier for this mutation to disagree");
        schema["operations"][0]["tier"] = Value::String(other);
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

/// Target facets are a sorted identity join, not free-form catalog prose.
#[test]
fn malformed_target_facets_fail_closed() {
    assert_mutation_rejected("target facets must be", |schema| {
        let operation = schema["operations"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|operation| !operation["target_facets"].as_array().unwrap().is_empty())
            .expect("Fix: live schema must include a target-backed operation");
        operation["target_facets"] = serde_json::json!(["wgpu", "", "wgpu"]);
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

/// Every crate that submits a registration in source appears in the live
/// registry.
///
/// WHY: the registry is observed by linking the operation crates into this
/// binary. Drop one of those links and the registry still answers with hundreds
/// of ids from the crate that is still linked, every count agrees with itself,
/// and the catalog loses a whole tier without a single assertion failing. The
/// inventory used to be smoke-tested from another crate's test target by
/// grepping `list-ops` output for the string `vyre-primitives::`, which judged
/// one crate and only through the CLI.
///
/// The expected set is read out of the sources at run time, so a third
/// registering crate is covered the day it registers.
///
/// What it does not catch: a registration whose id reaches
/// `inventory::submit!` through a macro parameter defined in another file. The
/// source scan does not model those, so a crate that registers only that way is
/// not in the expected set.
#[test]
fn every_registering_crate_appears_in_the_live_registry() {
    let scanned = structure_gate::scan(&workspace_root());
    let registering: BTreeSet<&str> = scanned
        .registrations
        .iter()
        .map(|registration| registration.crate_name.as_str())
        .collect();
    assert!(
        !registering.is_empty(),
        "Fix: no crate in the workspace submits an operation registration the source scan can see, so this contract is judging nothing"
    );

    let registry = vyre_registry_link::operation::live_operation_registry();
    let linked: BTreeSet<&str> = registry
        .iter()
        .filter_map(|operation| operation.id.split_once("::"))
        .map(|(crate_name, _)| crate_name)
        .collect();

    let missing: Vec<&str> = registering.difference(&linked).copied().collect();
    assert!(
        missing.is_empty(),
        "Fix: {} crate(s) register operations in source but contribute none to the live registry, so xtask-registry does not link them: {}",
        missing.len(),
        missing.join(", ")
    );
}
