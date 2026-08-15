//! The class closed here: a routing enum that offers to run a program on the
//! host.
//!
//! # What used to stand here
//!
//! `vyre_foundation::execution_plan::PolicyRoute` carried a `CpuSimd` variant,
//! `SchedulingPolicy` carried `cpu_fast_path_node_max: 64` and
//! `cpu_fast_path_static_bytes_below: 1 << 16` to feed it, and `route()` chose it
//! through `use_cpu_fast_path`, a predicate whose two parameters were both
//! underscore-prefixed and whose body was `false`. Nothing could reach the arm and
//! the two thresholds were read by nothing, so the struct advertised tuning for a
//! decision that did not exist. `vyre_runtime::routing::RoutingDecision` mirrored
//! the variant and `standard_policy` rewrote it to `PersistentMegakernel` on
//! arrival, which is a second statement that the route was not real.
//!
//! Dead is the good case. The bad case is the same shape wired up: a route that
//! quietly moves work to the host when a device capability is missing reports
//! success while delivering none of the product, and the number that would have
//! revealed it is a throughput measurement nobody takes on the failing path.
//!
//! # The property
//!
//! Vyre executes on a device. `vyre-reference` is the one crate permitted to
//! compute on a host, as the parity oracle a conformance comparison reads, and
//! the optimizer runs on the host at COMPILE time, which is not program
//! execution. Everything else routes to a device or fails closed with an error
//! naming the missing capability.
//!
//! So no routing enum may name a host execution target.
//!
//! # Why it fails by default
//!
//! Neither the enum roster nor the variant list is written here. Both are read
//! out of the workspace sources at run time: a routing enum is recognised by its
//! own contents, as an enum declaring a variant this workspace uses for a device
//! route, and then every variant it declares is checked. Add a routing enum
//! anywhere and it is measured without being registered; add a host variant to
//! one and this fails naming the file, the enum and the variant.
//!
//! The recognition rule is deliberately structural rather than a list of enum
//! names. A list of names is the thing that goes stale in silence, and a
//! `CpuSimd` variant reintroduced under a fresh enum name is the exact case a
//! name list would miss.

use std::collections::BTreeSet;
use std::path::Path;

use structure_gate::source_scan::{is_word_byte, matching_brace, rust_sources_with_text};
use structure_gate::workspace_root;

/// Variant names that mark an enum as a routing enum.
///
/// An enum declaring one of these is choosing where a program runs. This is a
/// recognition rule, not a roster: it says what a routing enum LOOKS like, and
/// any enum matching it is measured whether or not anyone remembered this file.
const DEVICE_ROUTE_MARKERS: [&str; 2] = ["PersistentMegakernel", "GpuPipeline"];

/// Fragments that name host execution in a variant.
const HOST_EXECUTION_MARKERS: [&str; 8] = [
    "Cpu", "Host", "Simd", "Scalar", "Native", "Software", "Interpret", "Emulate",
];

/// Crates whose enums are exempt, with the reason.
///
/// `vyre-reference` is the parity oracle: a routing enum there names which host
/// evaluator computes the comparison arm, which is the one legitimate host
/// execution in the workspace.
const ORACLE_CRATES: [&str; 1] = ["vyre-reference"];

/// Device routes this workspace serves, with what serves each.
///
/// `GpuPipeline` is the per-dispatch pipeline every backend implements.
/// `PersistentMegakernel` is the resident single-launch form the standard policy
/// promotes to. Both are reached by a dispatch.
///
/// A variant outside this set fails the roster test below even when its name
/// carries no host marker. That is the point: the marker test catches a route
/// back to the host, and this one catches a route nobody serves, which is how
/// the deleted `CpuSimd` survived review for as long as it did.
const SERVED_DEVICE_ROUTES: [&str; 2] = ["GpuPipeline", "PersistentMegakernel"];

/// One enum declaration found in the tree.
#[derive(Debug)]
struct RoutingEnum {
    path: String,
    name: String,
    variants: Vec<String>,
}

#[test]
fn no_routing_enum_offers_a_host_execution_target() {
    let root = workspace_root();
    let routes = routing_enums(&root);

    assert!(
        !routes.is_empty(),
        "Fix: no routing enum was recognised anywhere in the workspace, so this gate is measuring \
         nothing. Either the device route variant names in DEVICE_ROUTE_MARKERS changed, or the \
         scanner stopped reading the tree at {}",
        root.display()
    );

    let mut offenders = Vec::new();
    for route in &routes {
        if ORACLE_CRATES
            .iter()
            .any(|crate_name| route.path.starts_with(&format!("{crate_name}/")))
        {
            continue;
        }
        for variant in &route.variants {
            if let Some(marker) = HOST_EXECUTION_MARKERS
                .iter()
                .find(|marker| variant.contains(*marker))
            {
                offenders.push(format!(
                    "  {}: {}::{} (matches `{marker}`)",
                    route.path, route.name, variant
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "{} routing variant(s) name a host execution target:\n{}\n\n\
         Vyre executes on a device. A workload that cannot be placed on one is an error at the \
         point that discovers it, naming the missing capability and the corrective action, not a \
         route to somewhere slower. The only host computation in this workspace is the \
         `vyre-reference` parity oracle, which supplies the comparison arm of a conformance test \
         and is never reached by a dispatch. The optimizer also runs on the host, at compile \
         time, which is not program execution.\n\
         Fix: delete the variant, delete the predicate that selects it and any threshold field \
         that only fed that predicate, and make the surrounding decision fail closed instead.",
        offenders.len(),
        offenders.join("\n"),
    );
}

/// A routing enum declares only routes something serves.
///
/// # Why this shape
///
/// The marker test above answers "does a variant name the host". It cannot see
/// a route that is merely unserved: a `WaveOps` or `TensorPipeline` variant
/// added to a routing enum with no executor arm behind it reads to a caller as
/// a placement that exists, and the first evidence otherwise is a decision that
/// silently becomes something else on arrival.
///
/// So every variant of every recognised routing enum must be a route recorded
/// in `SERVED_DEVICE_ROUTES` with what serves it. Adding a route turns this RED
/// on arrival; the fix is to name the executor that runs it, not to extend the
/// list.
///
/// # What it does not catch
///
/// Whether the recorded executor arm still exists. This asserts a decision was
/// recorded, not that the implementation behind it is live.
#[test]
fn every_routing_variant_is_a_route_this_workspace_serves() {
    let root = workspace_root();
    let routes = routing_enums(&root);

    assert!(
        !routes.is_empty(),
        "Fix: no routing enum was recognised under {}, so this gate is measuring nothing",
        root.display()
    );

    let mut unserved = Vec::new();
    for route in &routes {
        if ORACLE_CRATES
            .iter()
            .any(|crate_name| route.path.starts_with(&format!("{crate_name}/")))
        {
            continue;
        }
        for variant in &route.variants {
            if !SERVED_DEVICE_ROUTES.contains(&variant.as_str()) {
                unserved.push(format!("  {}: {}::{}", route.path, route.name, variant));
            }
        }
    }

    assert!(
        unserved.is_empty(),
        "{} routing variant(s) name a route with no recorded executor:\n{}\n\n\
         Fix: name what serves the route and record it in SERVED_DEVICE_ROUTES beside the \
         others, or delete the variant. A route a caller can select and nothing runs is a \
         degradation path that does not exist being advertised as one.",
        unserved.len(),
        unserved.join("\n"),
    );
}

/// The gate sees a reintroduced host variant, and leaves a device-only enum alone.
///
/// Held against literal sources so a clean tree cannot make it pass by measuring
/// nothing. The first case is the deleted `PolicyRoute` verbatim.
#[test]
fn the_scanner_sees_a_reintroduced_host_route() {
    let reinjected = "\
pub enum PolicyRoute {
    /// Explicit diagnostic/reference route.
    CpuSimd,
    GpuPipeline,
    PersistentMegakernel,
}
";
    let found = enums_in(reinjected);
    let route = found
        .iter()
        .find(|item| item.1 == "PolicyRoute")
        .expect("Fix: the scanner stopped recognising an enum declaration");
    assert!(
        route.2.iter().any(|variant| variant == "CpuSimd"),
        "Fix: the scanner missed a variant sitting under a doc comment"
    );
    assert!(
        route
            .2
            .iter()
            .any(|variant| DEVICE_ROUTE_MARKERS.contains(&variant.as_str())),
        "Fix: the scanner no longer recognises this as a routing enum, so a host variant added \
         to it would go unmeasured"
    );
    assert!(
        route.2.iter().any(|variant| HOST_EXECUTION_MARKERS
            .iter()
            .any(|marker| variant.contains(marker))),
        "Fix: `CpuSimd` stopped matching the host execution markers, which is the whole check"
    );

    let device_only = "\
pub enum PolicyRoute {
    GpuPipeline,
    PersistentMegakernel,
}
";
    let clean = enums_in(device_only);
    let route = clean
        .iter()
        .find(|item| item.1 == "PolicyRoute")
        .expect("Fix: the scanner stopped recognising an enum declaration");
    assert!(
        !route.2.iter().any(|variant| HOST_EXECUTION_MARKERS
            .iter()
            .any(|marker| variant.contains(marker))),
        "Fix: the gate reports the shipped device-only enum, so it fails on correct code"
    );
}

/// Every routing enum in the workspace, recognised by its own variants.
fn routing_enums(root: &Path) -> Vec<RoutingEnum> {
    let mut found = Vec::new();
    for (relative, text) in rust_sources_with_text(root) {
        for (_, name, variants) in enums_in(&text) {
            let declared: BTreeSet<&str> = variants.iter().map(String::as_str).collect();
            if DEVICE_ROUTE_MARKERS
                .iter()
                .any(|marker| declared.contains(marker))
            {
                found.push(RoutingEnum {
                    path: relative.clone(),
                    name,
                    variants,
                });
            }
        }
    }
    found
}

/// Every `enum` declaration in `text` as (line, name, variant names).
///
/// A variant is a capitalised identifier at brace depth 1 of the enum body that
/// starts a line, which is what a rustfmt-formatted enum yields. Attributes, doc
/// comments, tuple payloads and struct payloads sit outside that shape and are
/// skipped.
fn enums_in(text: &str) -> Vec<(usize, String, Vec<String>)> {
    let mut found = Vec::new();
    let bytes = text.as_bytes();
    let mut cursor = 0;

    while let Some(offset) = text[cursor..].find("enum ") {
        let start = cursor + offset;
        cursor = start + "enum ".len();
        let is_keyword = start == 0 || !is_word_byte(bytes[start - 1]);
        if !is_keyword {
            continue;
        }
        let rest = &text[cursor..];
        let name: String = rest
            .chars()
            .take_while(|character| character.is_alphanumeric() || *character == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let Some(open) = rest.find('{') else {
            continue;
        };
        // A generic parameter list or a where clause may sit between the name
        // and the body, but a `;` or another `enum` before the brace means this
        // was not a declaration body.
        if rest[..open].contains(';') {
            continue;
        }
        let Some(close) = matching_brace(bytes, cursor + open) else {
            continue;
        };
        let body = &text[cursor + open + 1..close];
        found.push((
            text[..start].matches('\n').count() + 1,
            name,
            variants_in(body),
        ));
        cursor = close;
    }
    found
}

/// Variant names at the top level of an enum body.
fn variants_in(body: &str) -> Vec<String> {
    let mut depth = 0i32;
    let mut variants = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if depth == 0 && !trimmed.starts_with("//") && !trimmed.starts_with("#[") {
            let name: String = trimmed
                .chars()
                .take_while(|character| character.is_alphanumeric() || *character == '_')
                .collect();
            if name.chars().next().is_some_and(char::is_uppercase) {
                variants.push(name);
            }
        }
        depth += i32::try_from(trimmed.matches(['{', '(']).count()).unwrap_or(0);
        depth -= i32::try_from(trimmed.matches(['}', ')']).count()).unwrap_or(0);
        depth = depth.max(0);
    }
    variants
}

