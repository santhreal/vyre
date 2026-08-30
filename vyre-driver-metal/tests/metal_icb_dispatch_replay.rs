//! Metal icb dispatch replay test suite.

#![cfg(feature = "device-tests")]

use std::collections::BTreeSet;

const REPLAY: &str = include_str!("../../docs/optimization/METAL_ICB_DISPATCH_REPLAY.toml");
const SCHEMA: &str = "vyre.metal.icb_dispatch_replay.v1";
const DIGEST_PREFIX: &str = "sha256:";
const STABLE_DESCRIPTOR: &str = "icb-descriptor-stable";

fn registry() -> toml::Table {
    toml::from_str::<toml::Table>(REPLAY)
        .expect("Fix: METAL_ICB_DISPATCH_REPLAY.toml must parse as TOML.")
}

fn roster<'reg>(registry: &'reg toml::Table, key: &str) -> BTreeSet<&'reg str> {
    let values = registry
        .get(key)
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("Fix: the replay registry must declare `{key}`."));
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
        .expect("Fix: the replay registry must declare [[case]] rows.");
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
        .unwrap_or_else(|| panic!("Fix: replay case `{id}` must declare `{key}` as a string."))
}

fn nanos(row: &toml::Table, key: &str, id: &str) -> i64 {
    let value = row
        .get(key)
        .and_then(toml::Value::as_integer)
        .unwrap_or_else(|| panic!("Fix: replay case `{id}` must declare `{key}` as an integer."));
    assert!(
        value > 0,
        "Fix: replay case `{id}` must record `{key}` > 0."
    );
    value
}

/// Reusing a stable indirect command buffer records a lower submit cost than
/// re-encoding a dispatch whose shape varies.
///
/// WHY: this case asserted that the registry text contained seven substrings,
/// and its hardcoded list had already gone stale: the file requires
/// `route_reason` and the list omitted it. The claim the registry exists to
/// carry is that command reuse lowers host submit cost, and a row could have
/// recorded the opposite ordering, or the same output digest for two different
/// shapes, with every substring still present. The metric roster is derived from
/// the file, so a metric added there fails until every row records it.
///
/// Does not catch a submit cost that is recorded honestly and measured badly:
/// the registry states the ordering, and how a sample was taken is a separate
/// claim.
#[test]
fn command_reuse_records_a_lower_submit_cost_than_re_encoding() {
    let registry = registry();
    assert_eq!(
        registry.get("schema").and_then(toml::Value::as_str),
        Some(SCHEMA),
        "Fix: a replay registry schema change must be recorded in this case."
    );
    let routes = roster(&registry, "routes");
    assert!(
        routes.len() >= 2,
        "Fix: a replay comparison needs two routes; got {routes:?}"
    );
    let mut required = roster(&registry, "required_metrics");
    required.insert("id");
    let rows = cases(&registry);
    assert!(
        rows.len() >= 2,
        "Fix: the registry must state a reused and a re-encoded dispatch; got {} row(s).",
        rows.len()
    );

    let mut digests = BTreeSet::new();
    let mut reused = Vec::new();
    let mut re_encoded = Vec::new();
    for row in &rows {
        let id = text(row, "id", "<unnamed>");
        let declared: BTreeSet<&str> = row.keys().map(String::as_str).collect();
        assert_eq!(
            declared, required,
            "Fix: replay case `{id}` must record exactly the required metrics."
        );
        for key in ["descriptor_digest", "output_digest"] {
            let digest = text(row, key, id);
            assert!(
                digest.starts_with(DIGEST_PREFIX),
                "Fix: replay case `{id}` must record `{key}` as a `{DIGEST_PREFIX}` digest."
            );
            assert!(
                digests.insert(digest),
                "Fix: replay case `{id}` repeats digest `{digest}`, so two rows cannot disagree."
            );
        }
        assert!(
            !text(row, "route_reason", id).is_empty(),
            "Fix: replay case `{id}` must state why its route was taken."
        );
        nanos(row, "gpu_ns", id);
        let submit = nanos(row, "cpu_submit_ns", id);
        if text(row, "command_reuse_evidence", id) == STABLE_DESCRIPTOR {
            reused.push((id, submit));
        } else {
            re_encoded.push((id, submit));
        }
    }

    let slowest_reuse = reused
        .iter()
        .copied()
        .max_by_key(|(_, submit)| *submit)
        .expect("Fix: the registry must state one dispatch that reuses a stable descriptor.");
    let fastest_re_encode = re_encoded
        .iter()
        .copied()
        .min_by_key(|(_, submit)| *submit)
        .expect("Fix: the registry must state one dispatch that re-encodes its commands.");
    assert!(
        slowest_reuse.1 < fastest_re_encode.1,
        "Fix: reuse case `{}` records {} ns and re-encode case `{}` records {} ns, so the registry \
         states no submit-cost advantage for command reuse.",
        slowest_reuse.0,
        slowest_reuse.1,
        fastest_re_encode.0,
        fastest_re_encode.1
    );
}
