//! Apple math comparators test suite.

#![cfg(feature = "device-tests")]

use std::collections::BTreeSet;

const COMPARATORS: &str = include_str!("../../docs/optimization/APPLE_MATH_COMPARATORS.toml");
const SCHEMA: &str = "vyre.metal.apple_math_comparator.v1";
const DIGEST_PREFIX: &str = "sha256:";

fn registry() -> toml::Table {
    toml::from_str::<toml::Table>(COMPARATORS)
        .expect("Fix: APPLE_MATH_COMPARATORS.toml must parse as TOML.")
}

fn roster<'reg>(registry: &'reg toml::Table, key: &str) -> BTreeSet<&'reg str> {
    let values = registry
        .get(key)
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("Fix: the comparator registry must declare `{key}`."));
    assert!(
        !values.is_empty(),
        "Fix: `{key}` must state at least one entry."
    );
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .unwrap_or_else(|| panic!("Fix: every `{key}` entry must be a string."))
        })
        .collect()
}

fn cases(registry: &toml::Table) -> Vec<&toml::Table> {
    let rows = registry
        .get("case")
        .and_then(toml::Value::as_array)
        .expect("Fix: the comparator registry must declare [[case]] rows.");
    rows.iter()
        .map(|row| {
            row.as_table()
                .expect("Fix: every [[case]] row must be a table.")
        })
        .collect()
}

fn text<'row>(row: &'row toml::Table, key: &str, id: &str) -> &'row str {
    row.get(key)
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| panic!("Fix: comparator case `{id}` must declare `{key}` as a string."))
}

fn nanos(row: &toml::Table, key: &str, id: &str) -> i64 {
    let value = row
        .get(key)
        .and_then(toml::Value::as_integer)
        .unwrap_or_else(|| {
            panic!("Fix: comparator case `{id}` must declare `{key}` as an integer.")
        });
    assert!(
        value > 0,
        "Fix: comparator case `{id}` must record `{key}` > 0."
    );
    value
}

/// A comparator case records a selection reason that states one declared route.
///
/// WHY: this case asserted that the registry text contained nine substrings, and
/// the list conflated the route vocabulary with the metric roster. A row could
/// have recorded a selection reason for a route the registry never declares, or
/// reused one counter set for every case, and every substring would still be
/// present. The route vocabulary and the metric roster are both derived from the
/// file, so an entry added to either fails until every row records it.
///
/// Does not catch a reason that states a declared route and is wrong about why:
/// the registry states which route won, and whether the measurement supports it
/// is a separate claim.
#[test]
fn a_comparator_case_states_one_declared_route_in_its_selection_reason() {
    let registry = registry();
    assert_eq!(
        registry.get("schema").and_then(toml::Value::as_str),
        Some(SCHEMA),
        "Fix: a comparator registry schema change must be recorded in this case."
    );
    let routes = roster(&registry, "routes");
    assert!(
        routes.len() >= 2,
        "Fix: a comparator needs two routes to choose between; got {routes:?}"
    );
    let mut required = roster(&registry, "required_metrics");
    required.insert("id");
    let rows = cases(&registry);
    assert!(
        rows.len() >= 2,
        "Fix: the registry must compare at least two cases; got {} row(s).",
        rows.len()
    );

    let mut counters = BTreeSet::new();
    let mut digests = BTreeSet::new();
    for row in &rows {
        let id = text(row, "id", "<unnamed>");
        let declared: BTreeSet<&str> = row.keys().map(String::as_str).collect();
        assert_eq!(
            declared, required,
            "Fix: comparator case `{id}` must record exactly the required metrics."
        );
        let digest = text(row, "output_digest", id);
        assert!(
            digest.starts_with(DIGEST_PREFIX),
            "Fix: comparator case `{id}` must record `output_digest` as a `{DIGEST_PREFIX}` digest."
        );
        assert!(
            digests.insert(digest),
            "Fix: comparator case `{id}` repeats output digest `{digest}`, so two cases cannot \
             disagree."
        );
        let counter = text(row, "counter_evidence", id);
        assert!(
            !counter.is_empty(),
            "Fix: comparator case `{id}` must record counter evidence."
        );
        assert!(
            counters.insert(counter),
            "Fix: comparator case `{id}` repeats counter evidence `{counter}`, which cannot \
             measure two kernels."
        );
        assert!(
            !text(row, "kernel_family", id).is_empty(),
            "Fix: comparator case `{id}` must state the kernel family it compares."
        );
        nanos(row, "compile_ns", id);
        nanos(row, "gpu_ns", id);

        let reason = text(row, "selected_backend_reason", id);
        let stated: Vec<&str> = routes
            .iter()
            .copied()
            .filter(|route| reason.starts_with(&route.replace('_', "-")))
            .collect();
        assert_eq!(
            stated.len(),
            1,
            "Fix: comparator case `{id}` records reason `{reason}`, which states {} of the \
             declared routes {routes:?}.",
            stated.len()
        );
    }
}
