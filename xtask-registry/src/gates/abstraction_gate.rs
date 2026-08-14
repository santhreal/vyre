//! `cargo xtask abstraction-gate`  -  mandatory building-block enforcement.
//!
//! This is the fast local/CI gate that keeps the abstraction thesis
//! mechanical. It verifies that named composition edges point at
//! registered building blocks and that large ops are either small
//! enough to remain leaves or mostly composed from registered children.

use std::collections::BTreeSet;
use std::process;

use vyre::ir::{Node, Program};

const LOOP_BUDGET: usize = 4;
const NODE_BUDGET: usize = 200;
const COMPOSED_FRACTION_THRESHOLD: f64 = 0.6;

/// Entry point for the `abstraction-gate` subcommand.
pub(crate) fn run(_args: &[String]) {
    let ops = collect_ops();
    let ids: BTreeSet<String> = ops.iter().map(|op| op.id.clone()).collect();
    let mut failures = BTreeSet::new();

    for op in &ops {
        let mut state = WalkState::default();
        for node in op.program.entry() {
            walk(node, false, &ids, &mut state, &mut failures, &op.id);
        }

        if !within_budget(&state) {
            failures.insert(format!(
                "ABSTRACTION-BUDGET: `{}` has loops={} nodes={} registered-composed={:.1}%. Fix: extract reusable phases into registered Tier 2.5 primitives and wrap them with `region::wrap_child`.",
                op.id,
                state.loops,
                state.total_nodes,
                state.composed_fraction_pct(),
            ));
        }

        if op.id.starts_with("vyre-primitives::")
            && (op.test_inputs_missing || op.expected_output_missing)
        {
            failures.insert(format!(
                "PRIMITIVE-FIXTURE: `{}` must ship standalone test_inputs and expected_output. Fix: add an inventory fixture so the building block can be tested without its parent pipeline.",
                op.id
            ));
        }
    }

    if failures.is_empty() {
        println!(
            "abstraction-gate: {} registered building blocks checked",
            ops.len()
        );
        return;
    }

    eprintln!("abstraction-gate: {} violation(s)", failures.len());
    for failure in &failures {
        eprintln!("  - {failure}");
    }
    process::exit(1);
}

struct OpInfo {
    id: String,
    program: Program,
    test_inputs_missing: bool,
    expected_output_missing: bool,
}

fn collect_ops() -> Vec<OpInfo> {
    let mut ops = Vec::new();
    for entry in crate::live_registry::live_operation_registry().iter() {
        let program = entry.program().unwrap_or_else(|| {
            panic!(
                "Fix: canonical operation `{}` provides no neutral builder; register one or remove the registration",
                entry.id
            )
        });
        ops.push(OpInfo {
            id: entry.id.to_string(),
            program,
            test_inputs_missing: entry.test_inputs.is_none(),
            expected_output_missing: entry.expected_output.is_none(),
        });
    }
    ops
}

#[derive(Default)]
struct WalkState {
    total_nodes: usize,
    loops: usize,
    registered_composed_nodes: usize,
}

impl WalkState {
    fn composed_fraction_pct(&self) -> f64 {
        if self.total_nodes == 0 {
            return 100.0;
        }
        100.0 * self.registered_composed_nodes as f64 / self.total_nodes as f64
    }
}

fn within_budget(state: &WalkState) -> bool {
    if state.loops <= LOOP_BUDGET && state.total_nodes <= NODE_BUDGET {
        return true;
    }
    if state.total_nodes == 0 {
        return true;
    }
    let composed_fraction = state.registered_composed_nodes as f64 / state.total_nodes as f64;
    composed_fraction >= COMPOSED_FRACTION_THRESHOLD
}

fn walk(
    node: &Node,
    inside_registered_child: bool,
    ids: &BTreeSet<String>,
    state: &mut WalkState,
    failures: &mut BTreeSet<String>,
    owner_id: &str,
) {
    state.total_nodes += 1;
    if inside_registered_child {
        state.registered_composed_nodes += 1;
    }

    match node {
        Node::Region {
            generator,
            source_region,
            body,
        } => {
            let generator_name = generator.as_str();
            let is_registered_child = source_region.is_some() && ids.contains(generator_name);
            // `inline::<parent>` is what vyre-foundation names an anonymous body
            // it reparented onto its caller (algebra::composition), so the
            // prefix already states there is no operation behind it. Demanding
            // a registration for one asks for an op that must not exist.
            let is_anonymous_inline = generator_name.starts_with("inline::");
            if source_region.is_some()
                && generator_name.contains("::")
                && !is_registered_child
                && !is_anonymous_inline
            {
                failures.insert(format!(
                    "UNREGISTERED-CHILD: `{owner_id}` wraps `{generator_name}` as a child region, but no canonical SemanticOperation exists for that building block. Fix: submit it from the owning Tier 2.5/Tier 3 crate or stop marking it as a registered child."
                ));
            }
            if let Some(parent) = source_region {
                if parent.name.contains("::") && !ids.contains(parent.name.as_str()) {
                    failures.insert(format!(
                        "UNKNOWN-PARENT: `{owner_id}` child `{generator_name}` cites source_region `{}` which is not a registered op id.",
                        parent.name
                    ));
                }
            }
            for child in body.iter() {
                walk(
                    child,
                    inside_registered_child || is_registered_child,
                    ids,
                    state,
                    failures,
                    owner_id,
                );
            }
        }
        Node::Loop { body, .. } => {
            state.loops += 1;
            for child in body {
                walk(
                    child,
                    inside_registered_child,
                    ids,
                    state,
                    failures,
                    owner_id,
                );
            }
        }
        Node::Block(children) => {
            for child in children {
                walk(
                    child,
                    inside_registered_child,
                    ids,
                    state,
                    failures,
                    owner_id,
                );
            }
        }
        Node::If {
            then, otherwise, ..
        } => {
            for child in then {
                walk(
                    child,
                    inside_registered_child,
                    ids,
                    state,
                    failures,
                    owner_id,
                );
            }
            for child in otherwise {
                walk(
                    child,
                    inside_registered_child,
                    ids,
                    state,
                    failures,
                    owner_id,
                );
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vyre::ir::model::expr::GeneratorRef;
    use vyre::ir::Ident;

    use super::*;

    fn child(generator: &str, parent: &str) -> Node {
        Node::Region {
            generator: Ident::from(generator),
            source_region: Some(GeneratorRef {
                name: parent.to_string(),
            }),
            body: Arc::new(Vec::new()),
        }
    }

    fn findings(node: &Node, registered: &[&str]) -> BTreeSet<String> {
        let ids: BTreeSet<String> = registered.iter().map(|id| (*id).to_string()).collect();
        let mut state = WalkState::default();
        let mut failures = BTreeSet::new();
        walk(node, false, &ids, &mut state, &mut failures, "owner::op");
        failures
    }

    /// WHY: `inline::<parent>` is the name vyre-foundation gives an anonymous
    /// body it reparented onto its caller, so no SemanticOperation can ever
    /// exist for one. Reporting it told every caller to register an op that
    /// must not exist, and the only way to clear the finding was to stop
    /// composing. The parent it cites still has to be a real registered op.
    #[test]
    fn an_anonymous_inline_child_is_not_an_unregistered_child() {
        let node = child(
            "inline::vyre-libs::security::flows_to",
            "vyre-libs::security::flows_to",
        );
        let failures = findings(&node, &["vyre-libs::security::flows_to"]);
        assert!(
            failures.is_empty(),
            "an inline:: child must not be reported, got {failures:?}"
        );
    }

    /// WHY: the check still has to bite. A child that names a real-looking
    /// operation nobody registered is the case the gate exists for, and the
    /// inline:: exemption must not widen to cover it.
    #[test]
    fn a_child_naming_an_unregistered_operation_is_still_reported() {
        let node = child(
            "vyre-libs::security::ghost",
            "vyre-libs::security::flows_to",
        );
        let failures = findings(&node, &["vyre-libs::security::flows_to"]);
        assert!(
            failures
                .iter()
                .any(|failure| failure.starts_with("UNREGISTERED-CHILD")),
            "got {failures:?}"
        );
    }

    /// WHY: the exemption is the `inline::` prefix, not the substring. A
    /// generator that merely contains it elsewhere is an ordinary child.
    #[test]
    fn a_generator_merely_containing_inline_is_still_reported() {
        let node = child(
            "vyre-libs::security::not_inline::thing",
            "vyre-libs::security::flows_to",
        );
        let failures = findings(&node, &["vyre-libs::security::flows_to"]);
        assert!(
            failures
                .iter()
                .any(|failure| failure.starts_with("UNREGISTERED-CHILD")),
            "got {failures:?}"
        );
    }

    /// WHY: an inline child is anonymous, not composed. Counting it as
    /// registered composition would let a wrapper buy budget headroom by
    /// wrapping its own body.
    #[test]
    fn an_anonymous_inline_child_does_not_count_as_registered_composition() {
        let node = child(
            "inline::vyre-libs::security::flows_to",
            "vyre-libs::security::flows_to",
        );
        let ids: BTreeSet<String> = ["vyre-libs::security::flows_to".to_string()].into();
        let mut state = WalkState::default();
        let mut failures = BTreeSet::new();
        walk(&node, false, &ids, &mut state, &mut failures, "owner::op");
        assert_eq!(
            state.registered_composed_nodes, 0,
            "an inline child must not be counted as registered composition"
        );
    }
}
