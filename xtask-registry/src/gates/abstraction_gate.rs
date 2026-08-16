//! `cargo xtask abstraction-gate`  -  registered building-block boundaries.
//!
//! This gate reads the edges of a composition: every child region an operation
//! wraps has to name a building block that is registered, and every parent it
//! cites has to be an operation that exists. A region naming a block nobody
//! submitted is an edge to nothing, and the composition it claims cannot be
//! walked, fused or reused.
//!
//! Size is a different question and `gate1` owns it, over the same walk in
//! `composition_budget`. Reporting the budget here as well gave the tree two
//! counts of one rule, which disagreed: a phase wrapper carrying a
//! `source_region` read as composition on one side and as inlined work on the
//! other.

use std::collections::BTreeSet;

use vyre_foundation::composition::is_anonymous_generator;
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

use crate::gates::composition_budget::{self, ChildRegion};

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
        let ops = composition_budget::collect_ops(&mut report);
        let ids = composition_budget::registered_ids(&ops);
        let mut failures = BTreeSet::new();

        for op in &ops {
            composition_budget::measure(&op.program, &ids, &mut |child| {
                failures.extend(boundary_failures(&op.id, &child, &ids));
            });

            if op.id.starts_with("vyre-primitives::")
                && (op.test_inputs_missing || op.expected_output_missing)
            {
                failures.insert(format!(
                    "PRIMITIVE-FIXTURE: `{}` must ship standalone test_inputs and expected_output. Fix: add an inventory fixture so the building block can be tested without its parent pipeline.",
                    op.id
                ));
            }
        }

        report.note(format!("{} registered building block(s) checked", ops.len()));
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

/// What is wrong with the edges one child region declares.
///
/// Composition stamps `source_region` onto every entry region it reparents, so
/// a `source_region` by itself does not mean the author declared an edge to a
/// building block. An anonymous generator says outright that no operation is
/// behind it, and demanding a registration for one asks for an op that must not
/// exist; `vyre_foundation::composition` owns which prefixes mean that.
fn boundary_failures(
    owner_id: &str,
    child: &ChildRegion<'_>,
    ids: &BTreeSet<String>,
) -> Vec<String> {
    let mut failures = Vec::new();
    let Some(parent) = child.source_region else {
        return failures;
    };
    let generator = child.generator;
    if generator.contains("::") && !child.registered && !is_anonymous_generator(generator) {
        failures.push(format!(
            "UNREGISTERED-CHILD: `{owner_id}` wraps `{generator}` as a child region, but no canonical SemanticOperation exists for that building block. Fix: submit it from the owning Tier 2.5/Tier 3 crate, or rename it `anonymous::{generator}` when it is a phase boundary inside one operation rather than a building block."
        ));
    }
    if parent.as_str().contains("::") && !ids.contains(parent.as_str()) {
        failures.push(format!(
            "UNKNOWN-PARENT: `{owner_id}` child `{generator}` cites source_region `{}` which is not a registered op id.",
            parent.as_str()
        ));
    }
    failures
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vyre::ir::{Expr, Ident, Node, Program};

    use super::*;

    fn child(generator: &str, parent: &str) -> Node {
        Node::Region {
            generator: Ident::from(generator),
            source_region: Some(Ident::from(parent)),
            body: Arc::new(Vec::new()),
        }
    }

    /// Every boundary failure the walk finds under `node`.
    fn findings(node: &Node, registered: &[&str]) -> BTreeSet<String> {
        let ids: BTreeSet<String> = registered.iter().map(|id| (*id).to_string()).collect();
        let program = Program::wrapped(Vec::new(), [1, 1, 1], vec![node.clone()]);
        let mut failures = BTreeSet::new();
        composition_budget::measure(&program, &ids, &mut |region| {
            failures.extend(boundary_failures("owner::op", &region, &ids));
        });
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

    /// WHY: a parent nobody registered means the composition cites an operation
    /// that does not exist, which is unwalkable in the other direction.
    #[test]
    fn a_child_citing_an_unregistered_parent_is_reported() {
        let node = child("vyre-libs::security::flows_to", "vyre-libs::security::ghost");
        let failures = findings(&node, &["vyre-libs::security::flows_to"]);
        assert!(
            failures
                .iter()
                .any(|failure| failure.starts_with("UNKNOWN-PARENT")),
            "got {failures:?}"
        );
    }

    /// WHY: the walk used to derive child structure itself, with a `_ => {}`
    /// arm that doubled as "this variant is a leaf". A `Node` variant that
    /// gained a body would have stopped being descended and the gate would
    /// have reported nothing about it while still exiting green. Children now
    /// come from `visit::child_bodies`, so a new nesting variant
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
            ("Block", Node::Block(vec![buried()])),
            (
                "Loop",
                Node::loop_for("i", Expr::u32(0), Expr::u32(1), vec![buried()]),
            ),
            ("If/then", Node::if_then(Expr::bool(true), vec![buried()])),
            (
                "If/otherwise",
                Node::if_then_else(Expr::bool(true), Vec::new(), vec![buried()]),
            ),
        ];
        for (name, wrapper) in wrappers {
            let outer = Node::Region {
                generator: Ident::from("vyre-libs::security::flows_to"),
                source_region: Some(Ident::from("vyre-libs::security::flows_to")),
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
