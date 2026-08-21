//! No shipped crate can reach a host evaluator through production dependencies.
//!
//! The AST half of this gate proves no shipped source file *contains* a host
//! oracle. That says nothing about linking one: a crate that names
//! `vyre-reference` under `[dependencies]` carries the interpreter into the
//! shipped binary whether or not any line calls it, and a later caller then
//! only has to reach for what is already there. `vyre-driver::shadow` was
//! exactly that shape before it was deleted.
//!
//! So this half reads the dependency graph rather than sources. The graph and
//! the layer each crate sits in both come from [`crate_registry`], which
//! already owns `docs/CRATE_OWNERSHIP.toml` and the manifest walk behind it: a
//! second reader here would be a second answer to what the workspace contains.
//! That walk collects `[dependencies]`, `[build-dependencies]`, and their
//! target-conditional forms, and never `[dev-dependencies]`, which is the
//! distinction this rule needs. The conformance harness is *supposed* to run
//! the interpreter, and a rule that convicted it would be switched off rather
//! than obeyed.
//!
//! [`EXEMPT_LAYERS`] names the layers that may link one; every other layer in
//! [`LAYER_ORDER`] ships and is held to the rule. Deriving it that way rather
//! than listing the shipped layers is what makes a new layer fail closed: it is
//! subject to the rule the moment `check-tier-deps` learns about it, and nobody
//! has to remember this file exists.

use std::collections::{BTreeSet, VecDeque};

use crate::gate::{Finding, GateError, Report};
use crate::gates::check_tier_deps::LAYER_ORDER;
use crate::gates::crate_registry::{self, CrateRecord, WorkspaceState};
use crate::gates::scan::Tree;

/// Crates that evaluate a user program on the host.
///
/// `vyre-reference` is the interpreter. `vyre-driver-reference` registers it as
/// a backend, so linking that crate makes host execution reachable by backend
/// id without naming the interpreter at all.
const HOST_EVALUATORS: &[&str] = &["vyre-reference", "vyre-driver-reference"];

/// The dependency kind that ends up in a shipped artifact.
///
/// A build dependency runs at compile time and a dev dependency does not link
/// at all, so neither carries an interpreter into a shipped binary.
const SHIPPED_KIND: &str = "normal";

/// Layers whose crates exist to test, measure, or register, and are expected to
/// link a host evaluator.
///
/// Every other layer ships. A layer added to [`LAYER_ORDER`] is therefore held
/// to the rule until someone decides it belongs here, which is the direction a
/// default should fail in.
const EXEMPT_LAYERS: &[&str] = &[
    "standalone-tooling",
    "test-tooling",
    "registry-link",
    "conformance",
    "tooling",
];

/// Whether a crate in `layer` ends up in a shipped artifact.
fn ships(layer: &str) -> bool {
    LAYER_ORDER.contains(&layer) && !EXEMPT_LAYERS.contains(&layer)
}

/// Findings for every shipped crate that can reach a host evaluator.
pub(crate) fn findings(tree: &Tree, report: &mut Report) -> Result<Vec<Finding>, GateError> {
    let records = crate_registry::load_registry(tree, report)?;
    let state = crate_registry::workspace_state(tree)?;
    Ok(evaluate(&records, &state))
}

/// Judge a workspace that has already been read.
///
/// Split from [`findings`] so the rule is testable against a constructed
/// workspace rather than only against this one.
fn evaluate(records: &[CrateRecord], state: &WorkspaceState) -> Vec<Finding> {
    let mut findings = Vec::new();

    for record in records {
        let layer = record.layer.as_str();
        if !LAYER_ORDER.contains(&layer) {
            findings.push(Finding::new(
                format!(
                    "`{}` declares layer `{layer}`, which is not a layer this workspace has",
                    record.package
                ),
                "declare a layer LAYER_ORDER names, or add the new layer there and decide in xtask/src/gates/host_oracle_closure.rs whether its crates ship",
            ));
            continue;
        }
        if !ships(layer) {
            continue;
        }
        // A host evaluator is allowed to be one. `vyre-driver-reference` exists
        // to register the interpreter as a backend, so it necessarily links it;
        // the rule is that nothing else does.
        if HOST_EVALUATORS.contains(&record.package.as_str()) {
            continue;
        }
        if let Some(route) = route_to_evaluator(&record.package, state) {
            let evaluator = route.last().cloned().unwrap_or_default();
            findings.push(Finding::new(
                format!(
                    "shipped crate `{}` reaches host evaluator `{evaluator}` through production dependencies: {}",
                    record.package,
                    route.join(" -> ")
                ),
                "make the edge a dev-dependency, or move the host evaluation behind the conformance harness, so a shipped binary cannot link an interpreter",
            ));
        }
    }
    findings
}

/// Destinations `package` links into a shipped artifact.
fn shipped_edges<'a>(state: &'a WorkspaceState, package: &str) -> Vec<&'a str> {
    state
        .dependencies
        .get(package)
        .map(|edges| {
            edges
                .iter()
                .filter(|(_, use_)| use_.kinds.iter().any(|kind| kind == SHIPPED_KIND))
                .map(|(destination, _)| destination.as_str())
                .collect()
        })
        .unwrap_or_default()
}

/// The shortest shipped dependency path from `start` to a host evaluator.
fn route_to_evaluator(start: &str, state: &WorkspaceState) -> Option<Vec<String>> {
    let mut seen: BTreeSet<&str> = BTreeSet::from([start]);
    let mut queue: VecDeque<Vec<String>> = VecDeque::from([vec![start.to_string()]]);
    while let Some(route) = queue.pop_front() {
        let Some(tail) = route.last() else {
            continue;
        };
        for destination in shipped_edges(state, tail) {
            let mut next = route.clone();
            next.push(destination.to_string());
            if HOST_EVALUATORS.contains(&destination) {
                return Some(next);
            }
            if seen.insert(destination) {
                queue.push_back(next);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::gates::crate_registry::DependencyUse;

    fn record(package: &str, layer: &str) -> CrateRecord {
        CrateRecord {
            package: package.to_string(),
            path: format!("crates/{package}"),
            owner: "test".to_string(),
            layer: layer.to_string(),
            responsibility: "a fixture".to_string(),
            dependencies: Vec::new(),
        }
    }

    /// A workspace where every named edge is a shipped one.
    fn shipped(edges: &[(&str, &str)]) -> WorkspaceState {
        graph(edges, SHIPPED_KIND)
    }

    fn graph(edges: &[(&str, &str)], kind: &str) -> WorkspaceState {
        let mut dependencies: BTreeMap<String, BTreeMap<String, DependencyUse>> = BTreeMap::new();
        for (from, to) in edges {
            dependencies
                .entry((*from).to_string())
                .or_default()
                .insert(
                    (*to).to_string(),
                    DependencyUse {
                        kinds: vec![kind.to_string()],
                        ..DependencyUse::default()
                    },
                );
        }
        WorkspaceState {
            members: Vec::new(),
            paths: BTreeMap::new(),
            dependencies,
        }
    }

    /// WHY: the direct form of what this rule forbids. A shipped crate naming
    /// the interpreter under `[dependencies]` links it, whatever its source
    /// calls.
    #[test]
    fn a_shipped_crate_that_links_the_interpreter_is_reported() {
        let found = evaluate(
            &[record("vyre-driver", "backend-neutral")],
            &shipped(&[("vyre-driver", "vyre-reference")]),
        );
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].message.contains("vyre-driver -> vyre-reference"),
            "{found:?}"
        );
    }

    /// WHY: the edge that made `shadow` shippable was reachable, not direct. A
    /// rule that only looked at a crate's own manifest would have passed it.
    #[test]
    fn an_indirect_route_to_the_interpreter_is_reported_with_its_path() {
        let found = evaluate(
            &[
                record("vyre-runtime", "runtime"),
                record("vyre-helper", "libraries"),
            ],
            &shipped(&[
                ("vyre-runtime", "vyre-helper"),
                ("vyre-helper", "vyre-reference"),
            ]),
        );
        assert!(
            found.iter().any(|finding| finding
                .message
                .contains("vyre-runtime -> vyre-helper -> vyre-reference")),
            "{found:?}"
        );
    }

    /// WHY: the conformance harness is supposed to run the interpreter, and a
    /// build dependency does not ship. A gate that convicted either would be
    /// turned off rather than obeyed.
    #[test]
    fn an_exempt_layer_and_a_non_shipping_kind_are_both_allowed() {
        assert!(
            evaluate(
                &[record("vyre-conform", "conformance")],
                &shipped(&[("vyre-conform", "vyre-reference")]),
            )
            .is_empty(),
            "a conformance crate runs the interpreter on purpose"
        );
        assert!(
            evaluate(
                &[record("vyre-driver", "backend-neutral")],
                &graph(&[("vyre-driver", "vyre-reference")], "build"),
            )
            .is_empty(),
            "a build dependency does not link into a shipped artifact"
        );
    }

    /// WHY: `vyre-driver-reference` registers the interpreter as a backend, so
    /// it links one by definition. Its dependents are still convicted, which is
    /// the part that matters: reaching the interpreter by backend id is still
    /// reaching the interpreter.
    #[test]
    fn a_host_evaluator_may_link_itself_but_its_dependents_may_not() {
        assert!(
            evaluate(
                &[record("vyre-driver-reference", "concrete-backend")],
                &shipped(&[("vyre-driver-reference", "vyre-reference")]),
            )
            .is_empty(),
            "a host evaluator is allowed to be one"
        );

        let found = evaluate(
            &[
                record("vyre-runtime", "runtime"),
                record("vyre-driver-reference", "concrete-backend"),
            ],
            &shipped(&[
                ("vyre-runtime", "vyre-driver-reference"),
                ("vyre-driver-reference", "vyre-reference"),
            ]),
        );
        assert!(
            found.iter().any(|finding| finding
                .message
                .contains("vyre-runtime -> vyre-driver-reference")),
            "{found:?}"
        );
    }

    /// WHY: a layer the workspace does not have is a registry defect, not a
    /// quiet pass. Waving it through would let a crate opt out of the rule by
    /// declaring a layer nobody recognises.
    #[test]
    fn a_layer_the_workspace_does_not_have_is_reported() {
        let found = evaluate(
            &[record("vyre-new", "quantum-boundary")],
            &shipped(&[("vyre-new", "vyre-reference")]),
        );
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].message.contains("not a layer this workspace has"),
            "{found:?}"
        );
    }

    /// WHY: the fail-closed direction. A layer added to the roster must be
    /// held to the rule until someone exempts it, so this asserts the default
    /// rather than a list that would have to be maintained beside the roster.
    #[test]
    fn a_new_layer_ships_until_it_is_exempted() {
        for layer in LAYER_ORDER {
            assert_eq!(
                ships(layer),
                !EXEMPT_LAYERS.contains(layer),
                "`{layer}` must ship unless it is exempt"
            );
        }
        for layer in EXEMPT_LAYERS {
            assert!(
                LAYER_ORDER.contains(layer),
                "`{layer}` is exempted but is not a layer this workspace has"
            );
        }
        assert!(!ships("quantum-boundary"), "an unknown layer is not shipped");
    }
}
