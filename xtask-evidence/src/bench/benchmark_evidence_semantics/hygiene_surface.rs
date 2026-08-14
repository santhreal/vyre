//! Whether the hygiene scan covered every release surface and pattern family.
//!
//! The matrix reports which surfaces it scanned and which patterns it matched.
//! A surface flag that is absent or false, or a family missing one of its
//! required patterns, means the scan proves nothing about that surface, so both
//! are read as gaps in coverage rather than as clean results.

use serde_json::Value;

use super::data::{RELEASE_SURFACE_COVERAGE_FLAGS, RELEASE_SURFACE_REQUIRED_PATTERNS};

pub(crate) fn inspect_hygiene_release_surface_coverage(
    context: &str,
    matrix: &Value,
    failures: &mut Vec<String>,
) {
    let Some(coverage) = matrix.get("release_surface_coverage") else {
        failures.push(format!("{context}: missing release_surface_coverage"));
        return;
    };
    for field in RELEASE_SURFACE_COVERAGE_FLAGS.iter().copied() {
        if coverage.get(field).and_then(Value::as_bool) != Some(true) {
            failures.push(format!(
                "{context}: release_surface_coverage.{field} must be true"
            ));
        }
    }
    for (field, required) in RELEASE_SURFACE_REQUIRED_PATTERNS.iter().copied() {
        let values = coverage.get(field).and_then(Value::as_array);
        for required_value in required {
            if !values.is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_str() == Some(*required_value))
            }) {
                failures.push(format!(
                    "{context}: release_surface_coverage.{field} is missing `{required_value}`"
                ));
            }
        }
    }
}
