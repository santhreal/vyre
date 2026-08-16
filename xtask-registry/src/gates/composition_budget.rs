//! What Gate 1 counts, for both gates that read a registered operation.
//!
//! The budget and the boundary are two questions about the same region tree:
//! how much of an operation is composed from registered building blocks, and
//! whether every child region it wraps names one. Each was measured by its own
//! walk, and the two answers disagreed. One counted a node as composed whenever
//! it sat inside a region carrying a `source_region`, which a phase wrapper
//! around inlined code also carries, so `dominator_tree` read as 91.1 percent
//! composed against 0.0 percent from the other walk, and the reading that could
//! not fail was the one wired to a pin. That walk also listed the node variants
//! it descended into by hand, so a nesting variant added later would have been
//! counted as a leaf.
//!
//! Composition is a call to another registered operation. A node counts toward
//! it when it is the child region naming that operation, or sits inside one.
//! An operation's own entry region names itself and carries no parent, so an
//! operation is never composed of itself, and a phase boundary inside one
//! operation is anonymous by construction and never counts.

use std::collections::BTreeSet;

use vyre::ir::{GeneratorRef, Node, Program};
use vyre_foundation::visit::child_bodies;
use xtask::gate::{Finding, Report};

/// Loops an operation may hold before it has to be composed.
pub const LOOP_BUDGET: usize = 4;
/// Nodes an operation may hold before it has to be composed.
pub const NODE_BUDGET: usize = 200;
/// Share of an over-budget operation's nodes that must be composed.
pub const COMPOSED_FRACTION_THRESHOLD: f64 = 0.6;

/// One registered operation and the program its neutral builder produces.
pub struct Op {
    /// The canonical operation id.
    pub id: String,
    /// The program the neutral builder produces.
    pub program: Program,
    /// Whether the registration ships no standalone inputs.
    pub test_inputs_missing: bool,
    /// Whether the registration ships no standalone expected output.
    pub expected_output_missing: bool,
}

/// Every registered operation whose builder can be audited.
///
/// An entry with no neutral builder is reported rather than skipped: neither
/// its composition nor its complexity can be read, so skipping it would count
/// an unmeasurable operation as a clean one.
pub fn collect_ops(report: &mut Report) -> Vec<Op> {
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
        ops.push(Op {
            id: entry.id.to_string(),
            program,
            test_inputs_missing: entry.test_inputs.is_none(),
            expected_output_missing: entry.expected_output.is_none(),
        });
    }
    ops
}

/// The identifiers a child region has to name to count as composition.
#[must_use]
pub fn registered_ids(ops: &[Op]) -> BTreeSet<String> {
    ops.iter().map(|op| op.id.clone()).collect()
}

/// What one operation's program costs, and how much of it is composed.
#[derive(Default)]
pub struct Counts {
    /// Nodes in the operation's whole region tree.
    pub total_nodes: usize,
    /// Loops in that tree, at any depth.
    pub loops: usize,
    /// Nodes that are a call to another registered operation, or sit inside one.
    pub composed_nodes: usize,
    /// The inline work an author would have to extract to pass the budget.
    pub inline_hot_spots: Vec<String>,
}

impl Counts {
    /// Whether the operation is small enough to read whole, or composed enough
    /// that its size is other operations' size.
    #[must_use]
    pub fn passes(&self) -> bool {
        if self.loops <= LOOP_BUDGET && self.total_nodes <= NODE_BUDGET {
            return true;
        }
        if self.total_nodes == 0 {
            return true;
        }
        self.composed_nodes as f64 / self.total_nodes as f64 >= COMPOSED_FRACTION_THRESHOLD
    }

    /// The composed share, as a percentage for a report line.
    #[must_use]
    pub fn composed_fraction_pct(&self) -> f64 {
        if self.total_nodes == 0 {
            return 100.0;
        }
        100.0 * self.composed_nodes as f64 / self.total_nodes as f64
    }
}

/// A child region the walk passed through.
pub struct ChildRegion<'a> {
    /// The generator the region names.
    pub generator: &'a str,
    /// The operation the region declares itself part of, when it declares one.
    pub source_region: Option<&'a GeneratorRef>,
    /// Whether the generator is a registered operation.
    pub registered: bool,
}

/// Count `program` and hand every region it holds to `regions`.
pub fn measure(
    program: &Program,
    ids: &BTreeSet<String>,
    regions: &mut impl FnMut(ChildRegion<'_>),
) -> Counts {
    let mut counts = Counts::default();
    for node in program.entry() {
        walk(node, false, ids, &mut counts, regions);
    }
    counts
}

/// Count `node`, then descend into whatever bodies it holds.
///
/// `inside_composition` propagates downward from a region naming a registered
/// operation: the nodes under it are that operation's size, not this one's.
/// Which variants carry children is `child_bodies`' decision, so a nesting
/// variant added to the IR is descended into rather than counted as a leaf.
fn walk(
    node: &Node,
    inside_composition: bool,
    ids: &BTreeSet<String>,
    counts: &mut Counts,
    regions: &mut impl FnMut(ChildRegion<'_>),
) {
    counts.total_nodes += 1;
    let mut composes = false;
    match node {
        Node::Region {
            generator,
            source_region,
            body,
        } => {
            let generator = generator.as_str();
            composes = source_region.is_some() && ids.contains(generator);
            regions(ChildRegion {
                generator,
                source_region: source_region.as_ref(),
                registered: composes,
            });
            if !inside_composition && !composes && body.len() > INLINE_REGION_NODES {
                counts.inline_hot_spots.push(format!(
                    "anonymous region `{generator}` with {} top-level body nodes",
                    body.len()
                ));
            }
        }
        Node::Loop { body, .. } => {
            counts.loops += 1;
            if !inside_composition {
                counts
                    .inline_hot_spots
                    .push(format!("inline loop with {} body nodes", body.len()));
            }
        }
        _ => {}
    }
    if inside_composition || composes {
        counts.composed_nodes += 1;
    }
    let composed_below = inside_composition || composes;
    for body in child_bodies(node) {
        for child in body {
            walk(child, composed_below, ids, counts, regions);
        }
    }
}

/// Body nodes an uncomposed region may hold before it is reported as the work
/// that should have been a call.
const INLINE_REGION_NODES: usize = 50;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vyre::ir::{Expr, Ident};

    use super::*;

    fn ids(registered: &[&str]) -> BTreeSet<String> {
        registered.iter().map(|id| (*id).to_string()).collect()
    }

    fn region(generator: &str, parent: Option<&str>, body: Vec<Node>) -> Node {
        Node::Region {
            generator: Ident::from(generator),
            source_region: parent.map(|name| GeneratorRef {
                name: name.to_string(),
            }),
            body: Arc::new(body),
        }
    }

    fn counts(entry: Vec<Node>, registered: &[&str]) -> Counts {
        let program = Program::wrapped(Vec::new(), [1, 1, 1], entry);
        measure(&program, &ids(registered), &mut |_| {})
    }

    fn work() -> Vec<Node> {
        vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(4),
            vec![Node::let_bind("x", Expr::u32(1))],
        )]
    }

    /// WHY: a phase wrapper carries a `source_region` naming its own operation,
    /// so counting any wrapped node as composed made inlined work read as
    /// composition and let a bespoke split satisfy the budget.
    #[test]
    fn a_phase_wrapper_naming_its_own_operation_composes_nothing() {
        let measured = counts(
            vec![region(
                "anonymous::phase",
                Some("owner::op"),
                work(),
            )],
            &["owner::op"],
        );

        assert_eq!(measured.composed_nodes, 0);
        assert_eq!(measured.loops, 1);
        assert_eq!(measured.total_nodes, 3);
    }

    /// WHY: a call to another registered operation is the composition the
    /// budget credits, and the whole subtree it brings is that operation's
    /// size.
    #[test]
    fn a_region_naming_a_registered_operation_composes_its_subtree() {
        let measured = counts(
            vec![region("other::op", Some("owner::op"), work())],
            &["owner::op", "other::op"],
        );

        assert_eq!(measured.composed_nodes, 3, "the region and its two nodes");
        assert_eq!(measured.composed_fraction_pct(), 100.0);
    }

    /// WHY: an entry region names its own operation and cites no parent, so
    /// crediting it would report every operation as composed of itself.
    #[test]
    fn an_entry_region_is_not_composition() {
        let measured = counts(
            vec![region("owner::op", None, work())],
            &["owner::op"],
        );

        assert_eq!(measured.composed_nodes, 0);
    }

    /// WHY: the budget passes on size alone, and only an over-budget operation
    /// has to answer for its composed share.
    #[test]
    fn a_small_uncomposed_operation_passes_and_a_looping_one_does_not() {
        let small = counts(vec![region("owner::op", None, work())], &["owner::op"]);
        assert!(small.passes(), "one loop is inside the budget");

        let mut looping = Counts {
            loops: LOOP_BUDGET + 1,
            total_nodes: 10,
            composed_nodes: 5,
            inline_hot_spots: Vec::new(),
        };
        assert!(!looping.passes(), "half composed is under the threshold");
        looping.composed_nodes = 6;
        assert!(looping.passes(), "the threshold is a share, not a count");
    }

    /// WHY: an operation with no nodes has nothing to compose, and dividing by
    /// its size would report it as the worst in the registry.
    #[test]
    fn a_program_with_no_nodes_passes_at_a_full_fraction() {
        let measured = Counts::default();

        assert!(measured.passes());
        assert_eq!(measured.composed_fraction_pct(), 100.0);
    }

    /// WHY: the boundary gate asks about every region, including the phase
    /// wrappers the budget refuses to credit, so the walk reports them rather
    /// than filtering to the ones it counts.
    #[test]
    fn every_region_reaches_the_observer() {
        let program = Program::wrapped(
            Vec::new(),
            [1, 1, 1],
            vec![region(
                "owner::op",
                None,
                vec![region("other::op", Some("owner::op"), Vec::new())],
            )],
        );
        let mut seen = Vec::new();
        let measured = measure(&program, &ids(&["other::op"]), &mut |child| {
            seen.push((child.generator.to_string(), child.registered));
        });

        assert_eq!(
            seen,
            vec![
                ("owner::op".to_string(), false),
                ("other::op".to_string(), true)
            ]
        );
        assert_eq!(measured.composed_nodes, 1);
    }
}
