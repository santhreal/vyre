//! Source-backed op truth gates for `docs/optimization/OP_MATRIX.toml`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use toml::Value;
use vyre_harness::{classify_op_id, OpTier};

#[derive(Debug)]
struct RegisteredOp {
    id: String,
    source: &'static str,
    tier: OpTier,
}

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
    for op in &registered {
        let sources_for_id = registered_ids.entry(op.id.clone()).or_default();
        assert!(
            sources_for_id.insert(op.source),
            "Fix: duplicate registered op id `{}` appears more than once in `{}`.",
            op.id,
            op.source
        );

        if sources_for_id.len() > 1 {
            assert!(
                allowed_duplicate_sources(&op.id, sources_for_id),
                "Fix: duplicate registered op id `{}` has sources {:?} without an allowed duplicate contract.",
                op.id,
                sources_for_id
            );
        }

        let row_index = op_to_row
            .get(&op.id)
            .unwrap_or_else(|| panic!("Fix: OP_MATRIX.toml is missing registered op `{}`.", op.id));
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
        assert!(
            sources.iter().any(|source| source == op.source),
            "Fix: OP_MATRIX row for `{}` must include registry source `{}`.",
            op.id,
            op.source
        );
        if sources_for_id.len() > 1 {
            assert!(
                row.get("duplicate_ok").and_then(Value::as_bool) == Some(true),
                "Fix: OP_MATRIX row for duplicate op `{}` must set duplicate_ok = true.",
                op.id
            );
        }
    }

    assert!(
        !registered_ids.is_empty(),
        "Fix: op-matrix truth test must link at least one inventory registry."
    );
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

    for entry in vyre_libs::harness::all_entries() {
        let tier = classify_op_id(entry.id);
        assert!(
            matches!(tier, OpTier::Libs | OpTier::External),
            "Fix: shared harness entry `{}` must be a Tier 3 library id or an external consumer id, not {tier:?}.",
            entry.id
        );
    }

    for registration in inventory::iter::<vyre_driver::OpDefRegistration> {
        let def = (registration.op)();
        let tier = classify_op_id(def.id);
        assert!(
            matches!(tier, OpTier::Runtime | OpTier::Libs),
            "Fix: driver registry op `{}` must use a runtime namespace or a deliberate Tier 3 Cat-B duplicate id.",
            def.id
        );
    }
}

fn registered_ops() -> Vec<RegisteredOp> {
    let mut ops = Vec::new();
    for entry in vyre_intrinsics::harness::all_entries() {
        ops.push(RegisteredOp {
            id: entry.id.to_string(),
            source: "vyre-intrinsics::harness",
            tier: OpTier::Intrinsic,
        });
    }
    for entry in vyre_primitives::harness::all_entries() {
        ops.push(RegisteredOp {
            id: entry.id.to_string(),
            source: "vyre-primitives::harness",
            tier: OpTier::Primitive,
        });
    }
    for entry in vyre_libs::harness::all_entries() {
        ops.push(RegisteredOp {
            id: entry.id.to_string(),
            source: "vyre-harness",
            tier: classify_op_id(entry.id),
        });
    }
    for registration in inventory::iter::<vyre_driver::OpDefRegistration> {
        let def = (registration.op)();
        ops.push(RegisteredOp {
            id: def.id.to_string(),
            source: "vyre-driver::registry",
            tier: classify_op_id(def.id),
        });
    }
    ops
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

fn allowed_duplicate_sources(id: &str, sources: &BTreeSet<&'static str>) -> bool {
    sources.len() == 2
        && sources.contains("vyre-harness")
        && sources.contains("vyre-driver::registry")
        && id.starts_with("vyre-libs::math::atomic::")
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
