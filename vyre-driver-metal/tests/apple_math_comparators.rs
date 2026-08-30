//! Apple math comparators test suite.

use std::collections::BTreeSet;

mod measurement_registry;

use measurement_registry::Registry;

const COMPARATORS: &str = include_str!("../../docs/optimization/APPLE_MATH_COMPARATORS.toml");
const SCHEMA: &str = "vyre.metal.apple_math_comparator.v1";
const DIGEST_PREFIX: &str = "sha256:";

fn registry() -> Registry {
    Registry::parse(COMPARATORS, "comparator")
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
    registry.declares("schema", SCHEMA);
    let routes = registry.roster("routes");
    assert!(
        routes.len() >= 2,
        "Fix: a comparator needs two routes to choose between; got {routes:?}"
    );
    let mut required = registry.roster("required_metrics");
    required.insert("id");
    let rows = registry.rows("case", "id");
    assert!(
        rows.len() >= 2,
        "Fix: the registry must compare at least two cases; got {} row(s).",
        rows.len()
    );

    let mut counters = BTreeSet::new();
    let mut digests = BTreeSet::new();
    for row in &rows {
        row.declares_exactly(&required);
        row.digest("output_digest", DIGEST_PREFIX, &mut digests);
        let counter = row.stated("counter_evidence");
        assert!(
            counters.insert(counter),
            "Fix: comparator case `{}` repeats counter evidence `{counter}`, which cannot \
             measure two kernels.",
            row.id()
        );
        row.stated("kernel_family");
        row.nanos("compile_ns");
        row.nanos("gpu_ns");

        let reason = row.text("selected_backend_reason");
        let stated: Vec<&str> = routes
            .iter()
            .copied()
            .filter(|route| reason.starts_with(&route.replace('_', "-")))
            .collect();
        assert_eq!(
            stated.len(),
            1,
            "Fix: comparator case `{}` records reason `{reason}`, which states {} of the \
             declared routes {routes:?}.",
            row.id(),
            stated.len()
        );
    }
}
