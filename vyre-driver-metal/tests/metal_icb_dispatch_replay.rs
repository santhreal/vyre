//! Metal icb dispatch replay test suite.

use std::collections::BTreeSet;

mod measurement_registry;

use measurement_registry::Registry;

const REPLAY: &str = include_str!("../../docs/optimization/METAL_ICB_DISPATCH_REPLAY.toml");
const SCHEMA: &str = "vyre.metal.icb_dispatch_replay.v1";
const DIGEST_PREFIX: &str = "sha256:";
const STABLE_DESCRIPTOR: &str = "icb-descriptor-stable";

fn registry() -> Registry {
    Registry::parse(REPLAY, "replay")
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
    registry.declares("schema", SCHEMA);
    let routes = registry.roster("routes");
    assert!(
        routes.len() >= 2,
        "Fix: a replay comparison needs two routes; got {routes:?}"
    );
    let mut required = registry.roster("required_metrics");
    required.insert("id");
    let rows = registry.rows("case", "id");
    assert!(
        rows.len() >= 2,
        "Fix: the registry must state a reused and a re-encoded dispatch; got {} row(s).",
        rows.len()
    );

    let mut digests = BTreeSet::new();
    let mut reused = Vec::new();
    let mut re_encoded = Vec::new();
    for row in &rows {
        row.declares_exactly(&required);
        for key in ["descriptor_digest", "output_digest"] {
            row.digest(key, DIGEST_PREFIX, &mut digests);
        }
        row.stated("route_reason");
        row.nanos("gpu_ns");
        let submit = row.nanos("cpu_submit_ns");
        if row.text("command_reuse_evidence") == STABLE_DESCRIPTOR {
            reused.push((row.id(), submit));
        } else {
            re_encoded.push((row.id(), submit));
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
