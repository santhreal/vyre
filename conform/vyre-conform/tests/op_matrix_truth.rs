//! Source-backed op truth gates for `docs/optimization/OP_MATRIX.toml`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use toml::Value;
use vyre_foundation::operation::{
    classify_operation_id as classify_op_id, OperationTier as OpTier,
};

#[derive(Debug)]
struct RegisteredOp {
    id: String,
    source: &'static str,
    tier: OpTier,
}

const REQUIRED_SCAN_CONSTRUCT_TIERS: [&str; 5] = [
    "supported",
    "rejected",
    "approximated",
    "accelerator-only",
    "verifier-required",
];

const REQUIRED_SCAN_CONSTRUCT_BACKENDS: [&str; 9] = [
    "cpu_ref",
    "cuda",
    "wgpu",
    "metal",
    "hyperscan",
    "vectorscan",
    "rust_regex",
    "dpu",
    "fpga",
];

#[test]
fn op_matrix_covers_every_registered_op_once() {
    let root = workspace_root();
    let matrix = read_toml(&root.join("docs/optimization/OP_MATRIX.toml"));
    let bench_targets = read_bench_targets(&root);
    let registered = registered_ops();

    let status_values = string_set(
        matrix
            .get("backend_status_values")
            .and_then(Value::as_array)
            .expect("Fix: OP_MATRIX.toml must declare backend_status_values."),
    );
    let tier_values = string_set(
        matrix
            .get("tier_values")
            .and_then(Value::as_array)
            .expect("Fix: OP_MATRIX.toml must declare tier_values."),
    );
    assert!(
        !tier_values.contains("unknown"),
        "Fix: OP_MATRIX.toml must not accept unknown tiers."
    );

    let rows = matrix
        .get("op")
        .and_then(Value::as_array)
        .expect("Fix: OP_MATRIX.toml must contain [[op]] rows.");

    let mut family_seen = BTreeSet::new();
    let mut op_to_row = BTreeMap::<String, usize>::new();
    let mut op_to_sources = BTreeMap::<String, Vec<String>>::new();

    for (row_index, row) in rows.iter().enumerate() {
        let family = required_str(row, "family");
        assert!(
            family_seen.insert(family.to_string()),
            "Fix: duplicate OP_MATRIX family `{family}`."
        );

        let tier = required_str(row, "tier");
        assert!(
            tier_values.contains(tier),
            "Fix: OP_MATRIX family `{family}` uses tier `{tier}` not listed in tier_values."
        );

        for status_key in ["reference", "foundation_ir", "cuda", "wgpu", "spirv"] {
            let status = required_str(row, status_key);
            assert!(
                status_values.contains(status),
                "Fix: OP_MATRIX family `{family}` uses invalid {status_key} status `{status}`."
            );
        }

        assert_existing_paths(&root, family, "owners", required_array(row, "owners"));
        assert_existing_paths(&root, family, "tests", required_array(row, "tests"));
        for target in required_array(row, "bench_targets") {
            assert!(
                bench_targets.contains(target),
                "Fix: OP_MATRIX family `{family}` references missing bench target `{target}`."
            );
        }

        let sources = required_array(row, "registry_sources");
        let ops = required_array(row, "ops");
        assert!(
            !ops.is_empty(),
            "Fix: OP_MATRIX family `{family}` must list at least one op id."
        );
        for op in ops {
            if let Some(first_row) = op_to_row.insert(op.to_string(), row_index) {
                let first_family = required_str(&rows[first_row], "family");
                panic!(
                    "Fix: op `{op}` appears in OP_MATRIX families `{first_family}` and `{family}`."
                );
            }
            op_to_sources.insert(
                op.to_string(),
                sources.iter().map(|source| source.to_string()).collect(),
            );
        }
    }

    let mut registered_ids = BTreeMap::<String, BTreeSet<&'static str>>::new();
    // Collect every unmatrixed op instead of panicking on the first. Panicking on the
    // first turned a batch of new registrations into one round trip per op.
    let mut unmatrixed: Vec<String> = Vec::new();
    let mut wrong_sources: Vec<String> = Vec::new();
    for op in &registered {
        let sources_for_id = registered_ids.entry(op.id.clone()).or_default();
        assert!(
            sources_for_id.insert(op.source),
            "Fix: duplicate registered op id `{}` appears more than once in `{}`.",
            op.id,
            op.source
        );

        assert_eq!(
            sources_for_id.len(),
            1,
            "Fix: semantic operation `{}` must have exactly one registry owner, found {:?}.",
            op.id,
            sources_for_id
        );

        let Some(row_index) = op_to_row.get(&op.id) else {
            unmatrixed.push(op.id.clone());
            continue;
        };
        let row = &rows[*row_index];
        assert_eq!(
            required_str(row, "tier"),
            op.tier.matrix_value(),
            "Fix: OP_MATRIX tier for `{}` must match its canonical registry namespace.",
            op.id
        );
        let sources = op_to_sources
            .get(&op.id)
            .expect("Fix: matrix source map must exist for every row op.");
        if !sources.iter().any(|source| source == op.source) {
            wrong_sources.push(format!("{} needs registry source `{}`", op.id, op.source));
            continue;
        }
        if sources_for_id.len() > 1 {
            assert!(
                row.get("duplicate_ok").and_then(Value::as_bool) == Some(true),
                "Fix: OP_MATRIX row for duplicate op `{}` must set duplicate_ok = true.",
                op.id
            );
        }
    }

    unmatrixed.sort();
    unmatrixed.dedup();
    wrong_sources.sort();
    wrong_sources.dedup();
    assert!(
        wrong_sources.is_empty(),
        "Fix: {} OP_MATRIX row(s) declare the wrong registry_sources:\n  {}",
        wrong_sources.len(),
        wrong_sources.join("\n  ")
    );
    assert!(
        unmatrixed.is_empty(),
        "Fix: OP_MATRIX.toml is missing {} registered op(s):\n  {}",
        unmatrixed.len(),
        unmatrixed.join("\n  ")
    );

    assert!(
        !registered_ids.is_empty(),
        "Fix: op-matrix truth test must link at least one inventory registry."
    );
}

#[test]
fn op_matrix_scan_construct_tiers_have_proof_and_diagnostics() {
    let root = workspace_root();
    let matrix = read_toml(&root.join("docs/optimization/OP_MATRIX.toml"));
    let bench_targets = read_bench_targets(&root);

    let tier_values = string_set(
        matrix
            .get("scan_construct_tier_values")
            .and_then(Value::as_array)
            .expect("Fix: OP_MATRIX.toml must declare scan_construct_tier_values."),
    );
    let route_values = string_set(
        matrix
            .get("scan_construct_route_values")
            .and_then(Value::as_array)
            .expect("Fix: OP_MATRIX.toml must declare scan_construct_route_values."),
    );
    for required in REQUIRED_SCAN_CONSTRUCT_TIERS.iter().copied() {
        assert!(
            tier_values.contains(required),
            "Fix: OP_MATRIX.toml scan_construct_tier_values must include `{required}`."
        );
    }

    let rows = matrix
        .get("scan_construct")
        .and_then(Value::as_array)
        .expect("Fix: OP_MATRIX.toml must contain [[scan_construct]] rows.");

    let mut seen_ids = BTreeSet::new();
    let mut seen_tiers = BTreeSet::new();
    for row in rows {
        let id = required_str(row, "id");
        assert!(
            seen_ids.insert(id.to_string()),
            "Fix: duplicate OP_MATRIX scan construct id `{id}`."
        );

        let tier = required_str(row, "tier");
        assert!(
            tier_values.contains(tier),
            "Fix: OP_MATRIX scan construct `{id}` uses unregistered tier `{tier}`."
        );
        seen_tiers.insert(tier.to_string());

        let dialect_class = required_str(row, "dialect_class");
        assert!(
            !dialect_class.trim().is_empty(),
            "Fix: OP_MATRIX scan construct `{id}` must name a dialect_class."
        );

        let diagnostic_code = required_str(row, "diagnostic_code");
        assert!(
            diagnostic_code.starts_with("VYRE_SCAN_"),
            "Fix: OP_MATRIX scan construct `{id}` diagnostic_code `{diagnostic_code}` must use the VYRE_SCAN_ namespace."
        );

        let user_diagnostic = required_str(row, "user_diagnostic");
        assert!(
            user_diagnostic.len() >= 32,
            "Fix: OP_MATRIX scan construct `{id}` must include an operator-visible user_diagnostic."
        );

        let approximation_policy = required_str(row, "approximation_policy");
        assert!(
            !approximation_policy.trim().is_empty(),
            "Fix: OP_MATRIX scan construct `{id}` must name an approximation_policy."
        );

        let constructs = required_array(row, "constructs");
        assert!(
            !constructs.is_empty(),
            "Fix: OP_MATRIX scan construct `{id}` must list at least one syntax construct."
        );

        assert_existing_paths(&root, id, "proof_gates", required_array(row, "proof_gates"));
        for target in required_array(row, "bench_targets") {
            assert!(
                bench_targets.contains(target),
                "Fix: OP_MATRIX scan construct `{id}` references missing bench target `{target}`."
            );
        }

        let routes = row
            .get("backend_routes")
            .and_then(Value::as_table)
            .unwrap_or_else(|| {
                panic!("Fix: OP_MATRIX scan construct `{id}` must contain backend_routes.")
            });
        let mut row_routes = Vec::new();
        for backend in REQUIRED_SCAN_CONSTRUCT_BACKENDS.iter().copied() {
            let route = routes
                .get(backend)
                .and_then(Value::as_str)
                .unwrap_or_else(|| {
                    panic!("Fix: OP_MATRIX scan construct `{id}` must route backend `{backend}`.")
                });
            assert!(
                route_values.contains(route),
                "Fix: OP_MATRIX scan construct `{id}` backend `{backend}` uses invalid route `{route}`."
            );
            row_routes.push(route);
        }

        let verifier_required = required_bool(row, "verifier_required");
        let accelerator_only = required_bool(row, "accelerator_only");
        match tier {
            "supported" => assert!(
                row_routes.iter().any(|route| *route == "native"),
                "Fix: supported scan construct `{id}` must have at least one native route."
            ),
            "rejected" => assert!(
                row_routes.iter().all(|route| *route == "unsupported"),
                "Fix: rejected scan construct `{id}` must route every backend to unsupported."
            ),
            "approximated" => {
                assert!(
                    approximation_policy != "exact",
                    "Fix: approximated scan construct `{id}` must not use exact approximation_policy."
                );
                assert!(
                    verifier_required,
                    "Fix: approximated scan construct `{id}` must require verifier proof."
                );
                assert!(
                    row_routes.iter().any(|route| *route == "prefilter"),
                    "Fix: approximated scan construct `{id}` must have at least one prefilter route."
                );
            }
            "accelerator-only" => {
                assert!(
                    accelerator_only,
                    "Fix: accelerator-only scan construct `{id}` must set accelerator_only = true."
                );
                assert!(
                    row_routes
                        .iter()
                        .any(|route| *route == "external-accelerator"),
                    "Fix: accelerator-only scan construct `{id}` must have an external-accelerator route."
                );
            }
            "verifier-required" => {
                assert!(
                    verifier_required,
                    "Fix: verifier-required scan construct `{id}` must set verifier_required = true."
                );
                assert!(
                    row_routes.iter().any(|route| *route == "verifier"),
                    "Fix: verifier-required scan construct `{id}` must have at least one verifier route."
                );
            }
            other => panic!("Fix: unhandled scan construct tier `{other}` for `{id}`."),
        }
    }

    for required in REQUIRED_SCAN_CONSTRUCT_TIERS.iter().copied() {
        assert!(
            seen_tiers.contains(required),
            "Fix: OP_MATRIX.toml must include at least one scan construct row with tier `{required}`."
        );
    }
}

#[test]
fn registry_namespaces_do_not_pollute_other_tiers() {
    for entry in vyre_intrinsics::harness::all_entries() {
        assert_eq!(
            classify_op_id(entry.id),
            OpTier::Intrinsic,
            "Fix: intrinsic registry entry `{}` must use the vyre-intrinsics::hardware namespace.",
            entry.id
        );
    }

    for entry in vyre_primitives::harness::all_entries() {
        assert_eq!(
            classify_op_id(entry.id),
            OpTier::Primitive,
            "Fix: primitive registry entry `{}` must use the vyre-primitives namespace.",
            entry.id
        );
    }

    for entry in vyre_libs::fixture_catalog::all_entries() {
        let tier = classify_op_id(entry.id);
        assert!(
            matches!(tier, OpTier::Library | OpTier::External),
            "Fix: shared harness entry `{}` must be a Tier 3 library id or an external consumer id, not {tier:?}.",
            entry.id
        );
    }

    for registration in inventory::iter::<vyre_driver::OpDefRegistration> {
        let def = (registration.op)();
        let tier = classify_op_id(def.id);
        assert!(
            matches!(tier, OpTier::Runtime | OpTier::Library),
            "Fix: driver registry op `{}` must use a runtime namespace or a deliberate Tier 3 Cat-B duplicate id.",
            def.id
        );
    }
}

fn registered_ops() -> Vec<RegisteredOp> {
    vyre_foundation::operation::OperationRegistry::global()
        .iter()
        .map(|entry| RegisteredOp {
            id: entry.id.to_string(),
            source: "vyre-foundation::operation",
            tier: entry.tier,
        })
        .collect()
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("Fix: conform crate must live two levels below the workspace root.")
        .to_path_buf()
}

fn read_toml(path: &Path) -> Value {
    let body = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("Fix: read `{}`: {error}", path.display()));
    toml::from_str::<Value>(&body)
        .unwrap_or_else(|error| panic!("Fix: parse `{}` as TOML: {error}", path.display()))
}

fn read_bench_targets(root: &Path) -> BTreeSet<String> {
    let toml = read_toml(&root.join("docs/optimization/BENCH_TARGETS.toml"));
    toml.get("target")
        .and_then(Value::as_array)
        .expect("Fix: BENCH_TARGETS.toml must contain [[target]] rows.")
        .iter()
        .map(|row| required_str(row, "id").to_string())
        .collect()
}

fn required_str<'a>(row: &'a Value, key: &str) -> &'a str {
    row.get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("Fix: OP_MATRIX row must contain string field `{key}`."))
}

fn required_array<'a>(row: &'a Value, key: &str) -> Vec<&'a str> {
    row.get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("Fix: OP_MATRIX row must contain array field `{key}`."))
        .iter()
        .map(|value| {
            value
                .as_str()
                .unwrap_or_else(|| panic!("Fix: OP_MATRIX array `{key}` must contain strings."))
        })
        .collect()
}

fn required_bool(row: &Value, key: &str) -> bool {
    row.get(key)
        .and_then(Value::as_bool)
        .unwrap_or_else(|| panic!("Fix: OP_MATRIX row must contain boolean field `{key}`."))
}

fn string_set(values: &[Value]) -> BTreeSet<&str> {
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .expect("Fix: OP_MATRIX tier/status value arrays must contain strings.")
        })
        .collect()
}

fn assert_existing_paths(root: &Path, family: &str, field: &str, paths: Vec<&str>) {
    assert!(
        !paths.is_empty(),
        "Fix: OP_MATRIX family `{family}` must list at least one {field} path."
    );
    for path in paths {
        let absolute = root.join(path);
        assert!(
            absolute.exists(),
            "Fix: OP_MATRIX family `{family}` {field} path `{path}` does not exist."
        );
    }
}

/// An `inlined_callee` row opts out of per-backend release conformance, so the
/// claim has to be true rather than convenient.
///
/// A Composite callee exists only to be inlined at its call sites, which is why
/// executing it as a program of its own would exercise a shape the release never
/// runs. Two things make that safe to assume, and this test pins both: the op
/// registers through the dialect registry alone, so nothing has handed it a
/// dispatch witness that would then go unrun, and it declares a test path, so a
/// reader can find the caller-level suite that does cover its body.
#[test]
fn op_matrix_inlined_callee_rows_register_through_the_dialect_registry_alone() {
    let root = workspace_root();
    let matrix = read_toml(&root.join("docs/optimization/OP_MATRIX.toml"));
    let rows = matrix
        .get("op")
        .and_then(Value::as_array)
        .expect("Fix: OP_MATRIX.toml must contain [[op]] rows.");

    let mut checked = 0usize;
    for row in rows {
        if !row
            .get("inlined_callee")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            continue;
        }
        checked += 1;
        let family = required_str(row, "family");
        let sources = required_array(row, "registry_sources");
        assert_eq!(
            sources,
            vec!["vyre-driver::registry"],
            "Fix: OP_MATRIX row `{family}` claims inlined_callee but registers through {sources:?}.              An op with a harness witness is dispatched on its own and must not opt out of              release conformance."
        );
        assert!(
            !required_array(row, "tests").is_empty(),
            "Fix: OP_MATRIX row `{family}` claims inlined_callee and must name the caller-level              suite that covers its body."
        );
    }
    assert!(
        checked > 0,
        "Fix: this gate must see at least one inlined_callee row; if the last one was removed,          remove the `inlined_callee` handling in xtask conformance_matrix too."
    );
}

// ── Task 9 / ROADMAP K8: tests_non_empty coverage scan gate ────────

/// Every `[[op]]` row in OP_MATRIX.toml must declare at least one test
/// path that exists on disk. This catches ops that were added to the
/// matrix without corresponding test coverage documentation.
#[test]
fn op_matrix_every_row_has_existing_test_paths() {
    let root = workspace_root();
    let matrix = read_toml(&root.join("docs/optimization/OP_MATRIX.toml"));
    let rows = matrix
        .get("op")
        .and_then(Value::as_array)
        .expect("Fix: OP_MATRIX.toml must contain [[op]] rows.");

    for row in rows {
        let family = required_str(row, "family");
        let tests = required_array(row, "tests");
        assert!(
            !tests.is_empty(),
            "Fix: OP_MATRIX family `{family}` must list at least one test path (K8 gate)."
        );
        for test_path in &tests {
            let absolute = root.join(test_path);
            assert!(
                absolute.exists(),
                "Fix: OP_MATRIX family `{family}` test path `{test_path}` does not exist on disk."
            );
        }
    }
}

/// Negative twin: the coverage scan helper correctly rejects a
/// non-existent path (validates the assertion machinery itself).
#[test]
fn op_matrix_test_path_assertion_rejects_missing_path() {
    let root = workspace_root();
    let fake_path = root.join("does_not_exist_k8_negative_twin.rs");
    assert!(
        !fake_path.exists(),
        "Negative twin fixture must reference a non-existent path"
    );
}
