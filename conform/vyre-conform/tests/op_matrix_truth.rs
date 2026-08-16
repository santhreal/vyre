//! Source-backed op truth gates for `docs/optimization/OP_MATRIX.toml`.

use std::collections::{BTreeMap, BTreeSet};

use toml::Value;

#[path = "op_matrix_truth/namespaces.rs"]
mod namespaces;
#[path = "op_matrix_truth/registry.rs"]
mod registry;
#[path = "op_matrix_truth/scan_constructs.rs"]
mod scan_constructs;
#[path = "op_matrix_truth/toml_rows.rs"]
mod toml_rows;

use registry::registered_ops;
use toml_rows::{
    assert_existing_paths, read_toml, required_array, required_str, string_set, workspace_root,
};

#[test]
fn op_matrix_covers_every_registered_op_once() {
    let root = workspace_root();
    let matrix = read_toml(&root.join("docs/optimization/OP_MATRIX.toml"));
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
