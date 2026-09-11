//! Metal simd scan plan registry test suite.

use std::collections::{BTreeMap, BTreeSet};

const PLANS: &str = include_str!("../../docs/optimization/METAL_SIMD_SCAN_PLANS.toml");
const PLANNER: &str = "metal-simdgroup-scan:v1";
const SCHEMA_VERSION: i64 = 1;
const EVIDENCE_PATH: &str = "vyre-driver-metal/tests/metal_simd_scan_plan_registry.rs";

fn registry() -> toml::Table {
    toml::from_str::<toml::Table>(PLANS)
        .expect("Fix: METAL_SIMD_SCAN_PLANS.toml must parse as TOML.")
}

fn rows<'reg>(registry: &'reg toml::Table, key: &str) -> Vec<&'reg toml::Table> {
    let values = registry
        .get(key)
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("Fix: the scan plan registry must declare [[{key}]] rows."));
    assert!(
        !values.is_empty(),
        "Fix: the registry must declare at least one [[{key}]] row."
    );
    values
        .iter()
        .map(|value| {
            value
                .as_table()
                .unwrap_or_else(|| panic!("Fix: every [[{key}]] row must be a table."))
        })
        .collect()
}

fn text<'row>(row: &'row toml::Table, key: &str, id: &str) -> &'row str {
    row.get(key)
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| panic!("Fix: scan plan row `{id}` must declare `{key}` as a string."))
}

fn identity(registry: &toml::Table) {
    assert_eq!(
        registry
            .get("schema_version")
            .and_then(toml::Value::as_integer),
        Some(SCHEMA_VERSION),
        "Fix: a scan plan schema change must be recorded in this case."
    );
    assert_eq!(
        registry.get("planner_id").and_then(toml::Value::as_str),
        Some(PLANNER),
        "Fix: a planner identity change must be recorded in this case."
    );
    let metrics = registry
        .get("required_metrics")
        .and_then(toml::Value::as_array)
        .expect("Fix: the scan plan registry must declare required_metrics.");
    assert!(
        !metrics.is_empty(),
        "Fix: required_metrics must state at least one metric."
    );
}

/// Every scan route requires parity and its fallback chain terminates.
///
/// WHY: this case asserted that the registry text contained six field names and
/// two route ids. A route could have stated itself as its own fallback, or two
/// routes could have pointed at each other, and every substring would still be
/// present: a fallback cycle is exactly the defect a text scan cannot see, and
/// it stalls a dispatch rather than returning a wrong answer. The route key set
/// is derived from the rows, so a field added to one route fails until every
/// route declares it.
///
/// Does not catch a terminal fallback route that no backend implements: the
/// registry states where a route hands off, and which handlers exist is a
/// separate claim.
#[test]
fn every_scan_route_requires_parity_and_its_fallback_chain_terminates() {
    let registry = registry();
    identity(&registry);
    let routes = rows(&registry, "route");

    let declared: BTreeSet<&str> = routes
        .iter()
        .flat_map(|route| route.keys().map(String::as_str))
        .collect();
    let mut fallbacks = BTreeMap::new();
    let mut divergence = BTreeSet::new();
    for route in &routes {
        let id = text(route, "route_id", "<unnamed>");
        let keys: BTreeSet<&str> = route.keys().map(String::as_str).collect();
        assert_eq!(
            keys, declared,
            "Fix: scan route `{id}` must declare every field the other routes declare."
        );
        assert_eq!(
            route.get("parity_required").and_then(toml::Value::as_bool),
            Some(true),
            "Fix: scan route `{id}` must require CPU parity."
        );
        assert!(
            !text(route, "counter_source", id).is_empty(),
            "Fix: scan route `{id}` must state where its counters come from."
        );
        assert_eq!(
            text(route, "evidence_path", id),
            EVIDENCE_PATH,
            "Fix: scan route `{id}` must point at the case that proves it."
        );
        assert!(
            divergence.insert(text(route, "divergence_class", id)),
            "Fix: scan route `{id}` repeats a divergence class, so two routes state one plan."
        );
        let fallback = text(route, "fallback_route", id);
        assert_ne!(
            fallback, id,
            "Fix: scan route `{id}` states itself as its own fallback."
        );
        assert!(
            fallbacks.insert(id, fallback).is_none(),
            "Fix: scan route id `{id}` is declared twice."
        );
    }

    for start in fallbacks.keys().copied() {
        let mut current = start;
        let mut visited = BTreeSet::from([start]);
        while let Some(next) = fallbacks.get(current).copied() {
            assert!(
                visited.insert(next),
                "Fix: the fallback chain from scan route `{start}` re-enters `{next}`, so a \
                 dispatch that exhausts its route never terminates."
            );
            current = next;
        }
    }
}

/// Every scan diagnostic states a fix and the fields a reader needs.
///
/// WHY: the substring scan proved that one diagnostic code appeared somewhere in
/// the file. A diagnostic could have required no fields, stated no fix, or
/// required a field that states a fallback route the registry never declares,
/// and the substring would still be present.
///
/// Does not catch a fix line that is stated and unhelpful: whether the wording
/// resolves the condition is not a property of the registry.
#[test]
fn every_scan_diagnostic_states_a_fix_and_a_declared_fallback() {
    let registry = registry();
    let routes = rows(&registry, "route");
    let route_ids: BTreeSet<&str> = routes
        .iter()
        .map(|route| text(route, "route_id", "<unnamed>"))
        .collect();
    let fallbacks: BTreeSet<&str> = routes
        .iter()
        .map(|route| text(route, "fallback_route", "<unnamed>"))
        .collect();

    for diagnostic in rows(&registry, "diagnostic") {
        let code = text(diagnostic, "diagnostic_code", "<unnamed>");
        assert!(
            code.chars().all(|character| character.is_ascii_uppercase()
                || character.is_ascii_digit()
                || character == '_'),
            "Fix: scan diagnostic `{code}` must state an upper-case code."
        );
        assert!(
            !text(diagnostic, "fix", code).is_empty(),
            "Fix: scan diagnostic `{code}` must state the corrective action."
        );
        assert_eq!(
            text(diagnostic, "evidence_path", code),
            EVIDENCE_PATH,
            "Fix: scan diagnostic `{code}` must point at the case that proves it."
        );
        let fields = diagnostic
            .get("required_fields")
            .and_then(toml::Value::as_array)
            .unwrap_or_else(|| {
                panic!("Fix: scan diagnostic `{code}` must declare required_fields.")
            });
        assert!(
            !fields.is_empty(),
            "Fix: scan diagnostic `{code}` must require at least one field."
        );
        let names: BTreeSet<&str> = fields
            .iter()
            .map(|field| {
                field
                    .as_str()
                    .unwrap_or_else(|| panic!("Fix: `{code}` required_fields must be strings."))
            })
            .collect();
        assert!(
            names.contains("fallback_route"),
            "Fix: scan diagnostic `{code}` must state where the dispatch goes next."
        );
        assert!(
            fallbacks.iter().any(|route| !route_ids.contains(route)),
            "Fix: every fallback resolves to another route, so `{code}` states no terminal handoff."
        );
    }
}
