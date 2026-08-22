//! Reading OP_MATRIX rows and holding each field to its declared type.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use toml::Value;

pub(crate) fn workspace_root() -> PathBuf {
    structure_gate::workspace_root()
}

pub(crate) fn read_toml(path: &Path) -> Value {
    let body = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("Fix: read `{}`: {error}", path.display()));
    toml::from_str::<Value>(&body)
        .unwrap_or_else(|error| panic!("Fix: parse `{}` as TOML: {error}", path.display()))
}

pub(crate) fn read_bench_targets(root: &Path) -> BTreeSet<String> {
    let toml = read_toml(&root.join("docs/optimization/BENCH_TARGETS.toml"));
    toml.get("target")
        .and_then(Value::as_array)
        .expect("Fix: BENCH_TARGETS.toml must contain [[target]] rows.")
        .iter()
        .map(|row| required_str(row, "id").to_string())
        .collect()
}

pub(crate) fn required_str<'a>(row: &'a Value, key: &str) -> &'a str {
    row.get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("Fix: OP_MATRIX row must contain string field `{key}`."))
}

pub(crate) fn required_array<'a>(row: &'a Value, key: &str) -> Vec<&'a str> {
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

pub(crate) fn required_bool(row: &Value, key: &str) -> bool {
    row.get(key)
        .and_then(Value::as_bool)
        .unwrap_or_else(|| panic!("Fix: OP_MATRIX row must contain boolean field `{key}`."))
}

pub(crate) fn string_set(values: &[Value]) -> BTreeSet<&str> {
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .expect("Fix: OP_MATRIX tier/status value arrays must contain strings.")
        })
        .collect()
}

pub(crate) fn assert_existing_paths(root: &Path, family: &str, field: &str, paths: Vec<&str>) {
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
