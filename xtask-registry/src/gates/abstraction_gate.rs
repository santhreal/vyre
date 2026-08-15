//! `cargo xtask abstraction-gate`  -  mandatory building-block enforcement.
//!
//! This is the fast local/CI gate that keeps the abstraction thesis
//! mechanical. It verifies that named composition edges point at
//! registered building blocks and that large ops are either small
//! enough to remain leaves or mostly composed from registered children.

use std::collections::BTreeSet;

use vyre::ir::{Node, Program};
use vyre_foundation::composition::is_anonymous_generator;
use vyre_foundation::transform::visit::child_bodies;
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

const LOOP_BUDGET: usize = 4;
const NODE_BUDGET: usize = 200;
const COMPOSED_FRACTION_THRESHOLD: f64 = 0.6;

/// Entry point for the `abstraction-gate` subcommand.
/// Enforces the registered building-block boundaries of every registered operation.
pub struct AbstractionGate;

impl Gate for AbstractionGate {
    fn name(&self) -> &'static str {
        "abstraction-gate"
    }

    fn help(&self) -> &'static str {
        "Enforce registered building-block boundaries"
    }

    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
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

        report.note(format!(
            "{} registered building block(s) checked",
            ops.len()
        ));
        for failure in &failures {
            report.find(violation(failure));
        }
        Ok(report)
    }
}

/// Splits one violation into the problem and its corrective action.
///
/// Every violation class states the fix in the same sentence, which is how the
/// gate printed them; the contract keeps the two in separate fields so a single
/// finding read on its own is still actionable.
fn violation(text: &str) -> Finding {
    match text.split_once(" Fix: ") {
        Some((problem, fix)) => Finding::new(problem.trim(), fix),
        None => Finding::new(
            text,
            "submit the building block this violation names from its owning crate, or stop citing it as a registered child",
        ),
    }
}

struct OpInfo {
    id: String,
    program: Program,
    test_inputs_missing: bool,
    expected_output_missing: bool,
}

fn collect_ops(report: &mut Report) -> Vec<OpInfo> {
    let mut ops = Vec::new();
    for entry in vyre_registry_link::operation::live_operation_registry().iter() {
        let Some(program) = entry.program() else {
            report.find(Finding::new(
                format!(
                    "registered operation `{}` provides no neutral builder, so its composition cannot be audited",
                    entry.id
                ),
                "register a neutral builder for it, or withdraw the registration",
            ));
            continue;
        };
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

    let mut child_is_registered = false;
    match node {
        Node::Region {
            generator,
            source_region,
            ..
        } => {
            let generator_name = generator.as_str();
            child_is_registered = source_region.is_some() && ids.contains(generator_name);
            // Composition stamps `source_region` onto every entry region it
            // reparents, so a source_region by itself does not mean the author
            // declared an edge to a building block. An anonymous generator says
            // outright that no operation is behind it, and demanding a
            // registration for one asks for an op that must not exist.
            // `algebra::composition` owns which prefixes mean that.
            if source_region.is_some()
                && generator_name.contains("::")
                && !child_is_registered
                && !is_anonymous_generator(generator_name)
            {
                failures.insert(format!(
                    "UNREGISTERED-CHILD: `{owner_id}` wraps `{generator_name}` as a child region, but no canonical SemanticOperation exists for that building block. Fix: submit it from the owning Tier 2.5/Tier 3 crate, or rename it `anonymous::{generator_name}` when it is a phase boundary inside one operation rather than a building block."
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
        }
        Node::Loop { .. } => state.loops += 1,
        _ => {}
    }

    // Which variants carry children is `child_bodies`' decision. A hand-written
    // arm list here would declare every variant it was not told about a leaf,
    // so the gate would silently stop descending into a new nesting variant
    // instead of failing.
    for body in child_bodies(node) {
        for child in body {
            walk(
                child,
                inside_registered_child || child_is_registered,
                ids,
                state,
                failures,
                owner_id,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vyre::ir::model::expr::GeneratorRef;
    use vyre::ir::{Expr, Ident};

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

    /// WHY: two prefixes say "no operation is behind this region".
    /// `inline::<parent>` is what vyre-foundation names a body it reparented
    /// onto its caller; `anonymous::<label>` is what a builder names a phase
    /// boundary inside one operation. Reporting either told the caller to
    /// register an op that must not exist, and the only way to clear the
    /// finding was to stop composing. The gate knew only `inline::`, so seven
    /// `anonymous::` regions in vyre-libs and vyre-primitives were reported
    /// for a registration nobody could ever submit. Driven off
    /// `ANONYMOUS_GENERATOR_PREFIXES` so a third prefix cannot be minted
    /// without arriving here.
    #[test]
    fn an_anonymous_child_is_not_an_unregistered_child() {
        for prefix in vyre_foundation::composition::ANONYMOUS_GENERATOR_PREFIXES {
            let generator = format!("{prefix}vyre-libs::security::flows_to");
            let node = child(&generator, "vyre-libs::security::flows_to");
            let failures = findings(&node, &["vyre-libs::security::flows_to"]);
            assert!(
                failures.is_empty(),
                "a `{prefix}` child must not be reported, got {failures:?}"
            );
        }
    }

    /// WHY: the check still has to bite. A child that names a real-looking
    /// operation nobody registered is the case the gate exists for, and the
    /// anonymity exemption must not widen to cover it.
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

    /// WHY: the exemption is a prefix, not a substring. A generator that
    /// merely carries one of the words further along is an ordinary child.
    #[test]
    fn a_generator_merely_containing_an_anonymity_word_is_still_reported() {
        for word in ["not_inline", "not_anonymous"] {
            let generator = format!("vyre-libs::security::{word}::thing");
            let node = child(&generator, "vyre-libs::security::flows_to");
            let failures = findings(&node, &["vyre-libs::security::flows_to"]);
            assert!(
                failures
                    .iter()
                    .any(|failure| failure.starts_with("UNREGISTERED-CHILD")),
                "`{generator}` must still be reported, got {failures:?}"
            );
        }
    }

    /// WHY: an anonymous child is anonymous, not composed. Counting it as
    /// registered composition would let a wrapper buy budget headroom by
    /// wrapping its own body.
    #[test]
    fn an_anonymous_child_does_not_count_as_registered_composition() {
        for prefix in vyre_foundation::composition::ANONYMOUS_GENERATOR_PREFIXES {
            let generator = format!("{prefix}vyre-libs::security::flows_to");
            let node = child(&generator, "vyre-libs::security::flows_to");
            let ids: BTreeSet<String> = ["vyre-libs::security::flows_to".to_string()].into();
            let mut state = WalkState::default();
            let mut failures = BTreeSet::new();
            walk(&node, false, &ids, &mut state, &mut failures, "owner::op");
            assert_eq!(
                state.registered_composed_nodes, 0,
                "a `{prefix}` child must not be counted as registered composition"
            );
        }
    }

    /// WHY: the walk used to derive child structure itself, with a `_ => {}`
    /// arm that doubled as "this variant is a leaf". A `Node` variant that
    /// gained a body would have stopped being descended and the gate would
    /// have reported nothing about it while still exiting green. Children now
    /// come from `transform::visit::child_bodies`, so a new nesting variant
    /// fails to compile there instead. This holds the observable half: a
    /// finding buried under each nesting variant still surfaces.
    #[test]
    fn a_finding_under_every_nesting_variant_is_reported() {
        let buried = || {
            child(
                "vyre-libs::security::ghost",
                "vyre-libs::security::flows_to",
            )
        };
        let wrappers: [(&str, Node); 4] = [
            (
                "Block",
                Node::Block(vec![buried()]),
            ),
            (
                "Loop",
                Node::loop_for("i", Expr::u32(0), Expr::u32(1), vec![buried()]),
            ),
            (
                "If/then",
                Node::if_then(Expr::bool(true), vec![buried()]),
            ),
            (
                "If/otherwise",
                Node::if_then_else(Expr::bool(true), Vec::new(), vec![buried()]),
            ),
        ];
        for (name, wrapper) in wrappers {
            let outer = Node::Region {
                generator: Ident::from("vyre-libs::security::flows_to"),
                source_region: Some(GeneratorRef {
                    name: "vyre-libs::security::flows_to".to_string(),
                }),
                body: Arc::new(vec![wrapper]),
            };
            let failures = findings(&outer, &["vyre-libs::security::flows_to"]);
            assert!(
                failures
                    .iter()
                    .any(|failure| failure.starts_with("UNREGISTERED-CHILD")),
                "a child buried under {name} must still be reported, got {failures:?}"
            );
        }
    }
}
