//! Metal hazard certificates test suite.

use std::collections::BTreeSet;

const CERTIFICATES: &str = include_str!("../../docs/optimization/METAL_HAZARD_CERTIFICATES.toml");
const SCHEMA: &str = "vyre.metal.hazard_certificate.v1";
const NO_EVIDENCE: &str = "missing";
const NO_WAIT: &str = "none";

fn registry() -> toml::Table {
    toml::from_str::<toml::Table>(CERTIFICATES)
        .expect("Fix: METAL_HAZARD_CERTIFICATES.toml must parse as TOML.")
}

fn required_fields(registry: &toml::Table) -> BTreeSet<&str> {
    let fields = registry
        .get("required_fields")
        .and_then(toml::Value::as_array)
        .expect("Fix: the hazard registry must declare required_fields.");
    assert!(
        !fields.is_empty(),
        "Fix: required_fields must state at least one field."
    );
    fields
        .iter()
        .map(|field| {
            field
                .as_str()
                .expect("Fix: every required_fields entry must be a string.")
        })
        .collect()
}

fn certificates(registry: &toml::Table) -> Vec<&toml::Table> {
    let rows = registry
        .get("certificate")
        .and_then(toml::Value::as_array)
        .expect("Fix: the hazard registry must declare [[certificate]] rows.");
    rows.iter()
        .map(|row| {
            row.as_table()
                .expect("Fix: every [[certificate]] row must be a table.")
        })
        .collect()
}

fn field<'row>(row: &'row toml::Table, key: &str, id: &str) -> &'row toml::Value {
    row.get(key)
        .unwrap_or_else(|| panic!("Fix: certificate `{id}` must declare `{key}`."))
}

fn text<'row>(row: &'row toml::Table, key: &str, id: &str) -> &'row str {
    field(row, key, id)
        .as_str()
        .unwrap_or_else(|| panic!("Fix: certificate `{id}` must declare `{key}` as a string."))
}

/// A certificate permits dispatch exactly where its evidence discharges the
/// hazard it records.
///
/// WHY: this case asserted that the registry text contained seven substrings. A
/// row could record an untracked resource, state no synchronization evidence,
/// permit dispatch anyway, and every substring would still be present, so the
/// case certified nothing its name claims. It reads the rows now and holds each
/// one to the rule the registry exists to state. The field roster is derived
/// from the file, so a field added there fails until every row declares it.
///
/// Does not catch a certificate describing a resource the driver never binds:
/// the registry states the rule, and which resources exist is a separate claim.
#[test]
fn a_certificate_permits_dispatch_only_where_evidence_discharges_the_hazard() {
    let registry = registry();
    assert_eq!(
        registry.get("schema").and_then(toml::Value::as_str),
        Some(SCHEMA),
        "Fix: a hazard registry schema change must be recorded in this case."
    );
    let required = required_fields(&registry);
    let rows = certificates(&registry);
    assert!(
        rows.len() >= 2,
        "Fix: the registry must state both a permitted and a refused dispatch; got {} row(s).",
        rows.len()
    );

    let mut permitted = 0usize;
    let mut refused = 0usize;
    for row in &rows {
        let id = text(row, "resource_id", "<unnamed>");
        let declared: BTreeSet<&str> = row.keys().map(String::as_str).collect();
        assert_eq!(
            declared, required,
            "Fix: certificate `{id}` must declare exactly the required fields."
        );
        let evidence = text(row, "synchronization_evidence", id);
        let wait = text(row, "counter_wait_reason", id);
        let allowed = field(row, "dispatch_allowed", id)
            .as_bool()
            .unwrap_or_else(|| panic!("Fix: certificate `{id}` must declare a boolean."));
        if allowed {
            permitted += 1;
            assert_ne!(
                evidence, NO_EVIDENCE,
                "Fix: certificate `{id}` permits dispatch with no synchronization evidence."
            );
            assert_eq!(
                wait, NO_WAIT,
                "Fix: certificate `{id}` permits dispatch while a counter wait is outstanding."
            );
        } else {
            refused += 1;
            assert_ne!(
                wait, NO_WAIT,
                "Fix: certificate `{id}` refuses dispatch without stating the reason."
            );
        }
    }
    assert!(
        permitted > 0 && refused > 0,
        "Fix: the registry must exercise both polarities; permitted={permitted} refused={refused}."
    );
}
