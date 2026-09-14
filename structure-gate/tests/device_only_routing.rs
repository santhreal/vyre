//! The class closed here: a type whose values are execution routes.
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
//! the variant, `standard_policy` rewrote it to `PersistentMegakernel` on arrival,
//! and a `RoutingPolicy` trait plus a `RoutingEngine` stood behind it with no
//! production caller at all.
//!
//! Dead is the good case. The bad case is the same shape wired up: a route that
//! quietly moves work to the host when a device capability is missing reports
//! success while delivering none of the product, and the number that would have
//! revealed it is a throughput measurement nobody takes on the failing path.
//!
//! # The property
//!
//! Every production compile emits a megakernel artifact. There is one route, so
//! no type enumerates routes, and this gate asserts that absence.
//!
//! Absence is the checkable form of the property, and it is stronger than asking
//! a route type to hold only device routes. A route type is the place a host
//! route gets added: while one exists, every future variant is one edit away
//! from reaching the host, and the variant that does is indistinguishable from
//! the variants that do not until something measures the path. The enum that
//! stood here shipped `GpuPipeline` for years with nothing routing to it, which
//! is how `CpuSimd` sat beside it unnoticed.
//!
//! `vyre-reference` is the one crate permitted to compute on a host, as the
//! parity oracle a conformance comparison reads, so a route type there names
//! which host evaluator supplies the comparison arm and is exempt. The optimizer
//! also runs on the host, at compile time, which is not program execution.
//!
//! Where the rest of this class lives: `vyre-lints`' `production_cpu_fallbacks`
//! rejects a production call into the oracle, and
//! `vyre-test-support`'s `ProductionBackend` maps every backend registration the
//! registry may hold onto an `ExecutionDomain` through an exhaustive match, so a
//! registration that executes in host memory has no recorded decision and its
//! readers fail. This gate covers the third shape, which neither of those sees:
//! a route offered as a value.
//!
//! # Why it fails by default
//!
//! The roster is not written here. A route type is recognised by its own
//! contents, as an enum declaring a variant this workspace uses for a device
//! route, and the tree is read at run time. Declare one anywhere, under any
//! name, and this fails naming the file, the enum and the variant; a second
//! route therefore cannot arrive without a recorded decision.
//!
//! An empty result is only meaningful because the scanner is proven against
//! literal source below rather than against the tree, and because a file the
//! walk cannot read is a panic rather than a skip.

use std::collections::BTreeSet;
use std::path::Path;

use structure_gate::source_scan::{
    is_word_byte, matching_brace, rust_sources_with_text, SourceText,
};
use structure_gate::workspace_root;

/// Variant names that mark an enum as a routing enum.
///
/// An enum declaring one of these is choosing where a program runs. This is a
/// recognition rule, not a roster: it says what a routing enum LOOKS like, and
/// any enum matching it is measured whether or not anyone remembered this file.
const DEVICE_ROUTE_MARKERS: [&str; 2] = ["PersistentMegakernel", "GpuPipeline"];

/// Fragments that name host execution in a variant.
const HOST_EXECUTION_MARKERS: [&str; 8] = [
    "Cpu",
    "Host",
    "Simd",
    "Scalar",
    "Native",
    "Software",
    "Interpret",
    "Emulate",
];

/// Crates whose enums are exempt, with the reason.
///
/// `vyre-reference` is the parity oracle: a routing enum there names which host
/// evaluator computes the comparison arm, which is the one legitimate host
/// execution in the workspace.
const ORACLE_CRATES: [&str; 1] = ["vyre-reference"];

/// One enum declaration found in the tree.
#[derive(Debug)]
struct RoutingEnum {
    path: String,
    name: String,
    variants: Vec<String>,
}

/// No type in the workspace enumerates execution routes.
///
/// # What it does not catch
///
/// A second route expressed as something other than an enum variant: a boolean
/// field, a string id, or a trait with two implementors. Those are the shapes
/// `vyre-lints`' `production_cpu_fallbacks` and the `schedule-ownership` gate
/// read, over calls rather than declarations.
#[test]
fn no_type_enumerates_execution_routes() {
    let root = workspace_root();

    let mut offenders = Vec::new();
    for route in routing_enums(&root) {
        if ORACLE_CRATES
            .iter()
            .any(|crate_name| route.path.starts_with(&format!("{crate_name}/")))
        {
            continue;
        }
        let host = route
            .variants
            .iter()
            .filter_map(|variant| {
                HOST_EXECUTION_MARKERS
                    .iter()
                    .find(|marker| variant.contains(*marker))
                    .map(|marker| format!("{variant} matches `{marker}`"))
            })
            .collect::<Vec<_>>();
        let reached = if host.is_empty() {
            "no variant names the host yet".to_owned()
        } else {
            format!("already reaches the host: {}", host.join(", "))
        };
        offenders.push(format!(
            "  {}: {} {{ {} }}  -  {reached}",
            route.path,
            route.name,
            route.variants.join(", "),
        ));
    }

    assert!(
        offenders.is_empty(),
        "{} type(s) enumerate execution routes:\n{}\n\n\
         Every production compile emits a megakernel artifact, so there is one route and nothing \
         to select. A type whose values are routes is where a second one arrives: the variant \
         that reaches the host is indistinguishable from the variants that do not until something \
         measures the path, and a workload that cannot be placed on a device must be an error at \
         the point that discovers it, naming the missing capability, not a route to somewhere \
         slower.\n\
         Fix: delete the type and the policy trait, engine and predicate behind it, and let the \
         one route be the only thing the caller can reach. A genuine second route is a recorded \
         decision: name what serves it and what selects it here before the type exists.",
        offenders.len(),
        offenders.join("\n"),
    );
}

/// The scanner recognises a route type, and leaves an unrelated enum alone.
///
/// Held against literal sources, so the empty tree result above is a measured
/// absence rather than a scanner that stopped reading. The first case is the
/// deleted `PolicyRoute` verbatim.
#[test]
fn the_scanner_recognises_a_reintroduced_route_type() {
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
        "Fix: the scanner no longer recognises this as a route type, so reintroducing one would \
         go unmeasured"
    );
    assert!(
        route.2.iter().any(|variant| HOST_EXECUTION_MARKERS
            .iter()
            .any(|marker| variant.contains(marker))),
        "Fix: `CpuSimd` stopped matching the host execution markers, so the report would not say \
         the reintroduced type already reaches the host"
    );

    // A device-only route type is reported too: absence is the property, and
    // `GpuPipeline` shipped for years with nothing routing to it.
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
        route
            .2
            .iter()
            .any(|variant| DEVICE_ROUTE_MARKERS.contains(&variant.as_str())),
        "Fix: a route type carrying only device routes is no longer recognised, so the one shape \
         this gate exists to keep out would pass"
    );

    // An enum that uses the vocabulary for something else is not a route type.
    // `CausalPhase::MegakernelCompilation` and `PruneReason::PipelineCapacity`
    // are the live cases; an exact variant match is what separates them.
    let unrelated = "\
pub enum CausalPhase {
    MegakernelCompilation,
    DriverSubmission,
    RuntimeExecution,
}
";
    let other = enums_in(unrelated);
    let phase = other
        .iter()
        .find(|item| item.1 == "CausalPhase")
        .expect("Fix: the scanner stopped recognising an enum declaration");
    assert!(
        !phase
            .2
            .iter()
            .any(|variant| DEVICE_ROUTE_MARKERS.contains(&variant.as_str())),
        "Fix: the recognition rule widened to a substring, so every enum naming a dispatch form \
         is now reported and the gate fails on correct code"
    );
}

/// Every routing enum in the workspace, recognised by its own variants.
///
/// A source the walk cannot read is a failure rather than a gap: the rule
/// covers the tree, and a file nothing judged is a file the rule does not
/// cover.
fn routing_enums(root: &Path) -> Vec<RoutingEnum> {
    let mut found = Vec::new();
    for source in rust_sources_with_text(root) {
        let (relative, text) = match source {
            SourceText::Read { path, text } => (path, text),
            SourceText::Unread { path, reason } => {
                panic!("Fix: {path} {reason}, so no routing rule judged it")
            }
        };
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
