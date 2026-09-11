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
//! `EXEMPT_LAYERS` names the layers that may link one; every other layer the
//! architecture manifest declares ships and is held to the rule. Deriving it
//! that way rather than listing the shipped layers is what makes a new layer
//! fail closed: it is subject to the rule the moment a `[[layer]]` row exists,
//! and nobody has to remember this file exists.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::gate::{Finding, GateError, Report};
use crate::gates::crate_registry::{self, CrateRecord, WorkspaceState};
use crate::gates::scan::Tree;

/// Crates that evaluate a user program on the host.
///
/// `vyre-reference` is the interpreter. `vyre-driver-reference` wraps it in
/// `CpuRefEvaluator` and `ReferenceSemanticExecutor`, so linking that crate
/// makes host execution reachable without naming the interpreter at all. It
/// registers no backend, so the route is a named call rather than a backend
/// id, which is why a shipped crate still may not link it.
const HOST_EVALUATORS: &[&str] = &["vyre-reference", "vyre-driver-reference"];

/// The dependency kind that ends up in a shipped artifact.
///
/// A build dependency runs at compile time and a dev dependency does not link
/// at all, so neither carries an interpreter into a shipped binary.
const SHIPPED_KIND: &str = "normal";

/// Layers whose crates exist to test, measure, or register, and are expected to
/// link a host evaluator.
///
/// Every other declared layer ships. A layer added to the architecture manifest
/// is therefore held to the rule until someone decides it belongs here, which
/// is the direction a default should fail in.
const EXEMPT_LAYERS: &[&str] = &[
    "standalone-tooling",
    "test-tooling",
    "registry-link",
    "conformance",
    "tooling",
];

/// Whether a crate in `layer` ends up in a shipped artifact.
fn ships(layer: &str, declared: &BTreeSet<&str>) -> bool {
    declared.contains(layer) && !EXEMPT_LAYERS.contains(&layer)
}

/// Findings for every shipped crate that can reach a host evaluator.
pub(crate) fn findings(tree: &Tree, report: &mut Report) -> Result<Vec<Finding>, GateError> {
    let records = crate_registry::load_registry(tree, report)?;
    let state = crate_registry::workspace_state(tree)?;
    let ranks = crate_registry::declared_layer_ranks(tree)?;
    Ok(evaluate(&records, &state, &ranks))
}

/// The `src` directory of every crate that ships, sorted.
///
/// The AST half of this gate reads these. It used to read three literal paths,
/// and when the library and driver crates were split apart the scan silently
/// narrowed to a fraction of the workspace while still reporting a clean
/// verdict over the whole of it. Deriving the set from the same registry the
/// dependency half already reads means a crate is scanned the moment it has a
/// row, and a crate whose sources move with it stays scanned.
pub fn shipped_source_roots(tree: &Tree, report: &mut Report) -> Result<Vec<String>, GateError> {
    let records = crate_registry::load_registry(tree, report)?;
    let ranks = crate_registry::declared_layer_ranks(tree)?;
    let declared: BTreeSet<&str> = ranks.keys().map(String::as_str).collect();
    let mut roots = BTreeSet::new();
    for record in &records {
        if !ships(&record.layer, &declared) {
            continue;
        }
        if HOST_EVALUATORS.contains(&record.package.as_str()) {
            continue;
        }
        let root = format!("{}/src", record.path.trim_end_matches('/'));
        if tree.exists(&root) {
            roots.insert(root);
        }
    }
    Ok(roots.into_iter().collect())
}

/// Judge a workspace that has already been read.
///
/// Split from [`findings`] so the rule is testable against a constructed
/// workspace rather than only against this one.
fn evaluate(
    records: &[CrateRecord],
    state: &WorkspaceState,
    ranks: &BTreeMap<String, i64>,
) -> Vec<Finding> {
    let mut findings = Vec::new();
    let declared: BTreeSet<&str> = ranks.keys().map(String::as_str).collect();

    for record in records {
        let layer = record.layer.as_str();
        if !declared.contains(layer) {
            findings.push(Finding::new(
                format!(
                    "`{}` declares layer `{layer}`, which is not a layer this workspace has",
                    record.package
                ),
                "declare a layer with a `[[layer]]` row in docs/CRATE_OWNERSHIP.toml, and decide in xtask/src/gates/host_oracle_closure.rs whether its crates ship",
            ));
            continue;
        }
        if !ships(layer, &declared) {
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

    /// Layer ranks the fixtures are judged against. Names only; the rule reads
    /// which layers exist, never their order.
    fn ranks(layers: &[&str]) -> BTreeMap<String, i64> {
        layers
            .iter()
            .enumerate()
            .map(|(index, name)| ((*name).to_string(), index as i64))
            .collect()
    }

    /// The layers the fixtures name, plus every exempt layer.
    fn fixture_ranks() -> BTreeMap<String, i64> {
        let mut layers: Vec<&str> = vec![
            "backend-neutral",
            "concrete-backend",
            "libraries",
            "runtime",
        ];
        layers.extend(EXEMPT_LAYERS);
        ranks(&layers)
    }

    fn record(package: &str, layer: &str) -> CrateRecord {
        CrateRecord {
            package: package.to_string(),
            path: format!("crates/{package}"),
            layer: layer.to_string(),
            publication_class: "internal-engine".to_string(),
            seam: package.to_string(),
            interface: "a fixture seam".to_string(),
            responsibility: "a fixture".to_string(),
            facade_exported: false,
        }
    }

    /// A workspace where every named edge is a shipped one.
    fn shipped(edges: &[(&str, &str)]) -> WorkspaceState {
        graph(edges, SHIPPED_KIND)
    }

    fn graph(edges: &[(&str, &str)], kind: &str) -> WorkspaceState {
        let mut dependencies: BTreeMap<String, BTreeMap<String, DependencyUse>> = BTreeMap::new();
        for (from, to) in edges {
            dependencies.entry((*from).to_string()).or_default().insert(
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
            development: BTreeMap::new(),
        }
    }

    /// Judge a fixture against the layers the fixtures declare.
    fn judge(records: &[CrateRecord], state: &WorkspaceState) -> Vec<Finding> {
        evaluate(records, state, &fixture_ranks())
    }

    /// WHY: the direct form of what this rule forbids. A shipped crate naming
    /// the interpreter under `[dependencies]` links it, whatever its source
    /// calls.
    #[test]
    fn a_shipped_crate_that_links_the_interpreter_is_reported() {
        let found = judge(
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
        let found = judge(
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
            judge(
                &[record("vyre-conform", "conformance")],
                &shipped(&[("vyre-conform", "vyre-reference")]),
            )
            .is_empty(),
            "a conformance crate runs the interpreter on purpose"
        );
        assert!(
            judge(
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
            judge(
                &[record("vyre-driver-reference", "concrete-backend")],
                &shipped(&[("vyre-driver-reference", "vyre-reference")]),
            )
            .is_empty(),
            "a host evaluator is allowed to be one"
        );

        let found = judge(
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

    /// WHY: a layer no `[[layer]]` row declares is a manifest defect, not a
    /// quiet pass. Waving it through would let a crate opt out of the rule by
    /// declaring a layer nobody recognises.
    #[test]
    fn a_layer_the_workspace_does_not_have_is_reported() {
        let found = judge(
            &[record("vyre-new", "quantum-boundary")],
            &shipped(&[("vyre-new", "vyre-reference")]),
        );
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].message.contains("not a layer this workspace has"),
            "{found:?}"
        );
    }

    /// WHY: the fail-closed direction, against the layers this checkout
    /// actually declares rather than a list beside them. Adding a `[[layer]]`
    /// row makes its crates shipped, so the new layer is held to the rule until
    /// someone adds it to [`EXEMPT_LAYERS`]; retiring one that is exempt turns
    /// this red instead of leaving a dead exemption behind.
    #[test]
    fn a_new_layer_ships_until_it_is_exempted() {
        let tree = Tree::open(&crate::checkout::checkout_root())
            .expect("Fix: the checkout must be readable");
        let ranks = crate_registry::declared_layer_ranks(&tree)
            .expect("Fix: docs/CRATE_OWNERSHIP.toml must declare [[layer]] rows");
        let declared: BTreeSet<&str> = ranks.keys().map(String::as_str).collect();
        assert!(
            !declared.is_empty(),
            "the architecture manifest declares no layers"
        );
        for layer in &declared {
            assert_eq!(
                ships(layer, &declared),
                !EXEMPT_LAYERS.contains(layer),
                "`{layer}` must ship unless it is exempt"
            );
        }
        for layer in EXEMPT_LAYERS {
            assert!(
                declared.contains(layer),
                "`{layer}` is exempted and no [[layer]] row declares it"
            );
        }
        assert!(
            !ships("quantum-boundary", &declared),
            "an undeclared layer is not shipped"
        );
    }
}
