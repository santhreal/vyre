//! Program → data-derived loop bound analysis.
//!
//! `Node::Loop` runs for as many iterations as its `to` expression states. When
//! that expression reaches a buffer element, the iteration count is whatever
//! ran before it wrote that element, so one out-of-contract `u32` asks for four
//! billion iterations: hours in the reference interpreter, a watchdog reset on
//! a device. An out-of-bounds guard inside the body does not bound the loop,
//! because it discards the result of an iteration that already ran.
//!
//! The correction is a clamp: a bound read from a buffer is intersected with
//! the extents of the buffers the body indexes, which keeps the emitted result
//! identical for in-contract input because a producer contract already holds
//! the value at or below that extent. This module answers whether a program
//! still carries a bound that was never clamped, so the property is checked at
//! the IR, where every builder and every dialect meets, rather than at each
//! call site that happens to construct one.
//!
//! The per-variant decisions come from [`crate::visit::expr_magnitude`], which
//! has no catch-all arm over `Expr`, so a new expression variant cannot enter
//! the IR and be read here as fixed when the program is built.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::ir::{Expr, Ident, Node, Program};
use crate::visit::{
    expr_children, expr_magnitude, for_each_node, node_scalars, ExprMagnitude, NameBinding,
};

/// One loop whose iteration count is not fixed when the program is built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataDerivedLoopBound {
    /// Induction variable of the loop, which names it in a diagnostic.
    pub var: Ident,
    /// Buffer whose contents the bound reaches, when the analysis can attribute
    /// it to one. `None` is a bound that reaches a value core cannot attribute
    /// at all: a call result, an out-of-tree extension, an operator carrying no
    /// recorded magnitude decision, or a name no enclosing scope binds.
    pub source: Option<Ident>,
}

/// Every loop in `program` whose iteration count is read from buffer contents.
///
/// An empty result is the contract a program is expected to satisfy: every
/// bound is a literal, a launch-geometry fact, a declared buffer extent, or a
/// value clamped against one.
#[must_use]
pub fn data_derived_loop_bounds(program: &Program) -> Vec<DataDerivedLoopBound> {
    data_derived_loop_bounds_in(program.entry())
}

/// Every loop in `nodes` and in every nested body whose iteration count is read
/// from buffer contents.
///
/// The body-taking form of [`data_derived_loop_bounds`], for a caller holding a
/// fragment rather than a whole program. Names are resolved against the
/// bindings `nodes` carries, so a fragment that reads a name declared outside
/// it reports that name as unattributable rather than assuming it is fixed.
///
/// The walk is iterative in both namespaces, over statements and over
/// expressions, so an adversarially deep program costs heap rather than native
/// stack.
#[must_use]
pub fn data_derived_loop_bounds_in(nodes: &[Node]) -> Vec<DataDerivedLoopBound> {
    let mut resolver = Resolver::default();
    let mut loops: Vec<(&Ident, &Expr)> = Vec::new();
    for_each_node(nodes, |node| {
        let scalars = node_scalars(node);
        let Some((binding, name)) = scalars.binding else {
            return;
        };
        match binding {
            // A declaration and a rebinding both put a value into the name. The
            // name is bounded only when every value it can carry is, so both
            // are recorded and the resolver requires all of them.
            NameBinding::Declare | NameBinding::Reassign => {
                if let Some(value) = scalars.operands[0] {
                    resolver.bindings.entry(name).or_default().push(value);
                }
            }
            // An induction variable is below its own loop's bound, so the two
            // share one classification.
            NameBinding::Induction => {
                if let Some(to) = scalars.operands[1] {
                    resolver.bindings.entry(name).or_default().push(to);
                    loops.push((name, to));
                }
            }
        }
    });

    let mut findings = Vec::new();
    for (var, bound) in loops {
        if let Provenance::Data(source) = resolver.classify(bound) {
            findings.push(DataDerivedLoopBound {
                var: var.clone(),
                source: source.cloned(),
            });
        }
    }
    findings
}

/// Whether a value is fixed when the program is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Provenance<'a> {
    /// Derived only from literals, launch geometry and declared extents.
    Host,
    /// Reaches buffer contents, or a value core cannot attribute.
    Data(Option<&'a Ident>),
}

impl<'a> Provenance<'a> {
    /// The provenance of a value derived from every operand: data-derived as
    /// soon as one operand is, and the first such operand names the source.
    fn all(operands: impl IntoIterator<Item = Self>) -> Self {
        operands
            .into_iter()
            .find(|operand| matches!(operand, Self::Data(_)))
            .unwrap_or(Self::Host)
    }

    /// The provenance of a value at most the smallest operand: fixed as soon as
    /// one operand is fixed, which is what a clamp against a buffer extent
    /// makes true of a bound read from a buffer.
    fn least(operands: impl IntoIterator<Item = Self>) -> Self {
        let mut first_data = Self::Data(None);
        for operand in operands {
            match operand {
                Self::Host => return Self::Host,
                data @ Self::Data(Some(_)) => {
                    if first_data == Self::Data(None) {
                        first_data = data;
                    }
                }
                Self::Data(None) => {}
            }
        }
        first_data
    }
}

/// One step of the iterative classification.
enum Step<'a> {
    /// Classify this expression, pushing its operands first when it has any.
    Enter(&'a Expr),
    /// Combine the top `arity` results according to `magnitude`.
    Combine {
        magnitude: ExprMagnitude<'a>,
        arity: usize,
    },
    /// Combine the top `arity` results into the classification of `name`, which
    /// is bounded only when every value it carries is.
    CloseName { name: &'a Ident, arity: usize },
}

#[derive(Default)]
struct Resolver<'a> {
    bindings: FxHashMap<&'a Ident, SmallVec<[&'a Expr; 2]>>,
    resolved: FxHashMap<&'a Ident, Provenance<'a>>,
    open: FxHashSet<&'a Ident>,
}

impl<'a> Resolver<'a> {
    fn classify(&mut self, expr: &'a Expr) -> Provenance<'a> {
        let mut work: Vec<Step<'a>> = vec![Step::Enter(expr)];
        let mut results: Vec<Provenance<'a>> = Vec::new();
        while let Some(step) = work.pop() {
            match step {
                Step::Enter(current) => self.enter(current, &mut work, &mut results),
                Step::Combine { magnitude, arity } => {
                    let operands = results.split_off(results.len() - arity);
                    results.push(match magnitude {
                        ExprMagnitude::LeastOperand => Provenance::least(operands),
                        _ => Provenance::all(operands),
                    });
                }
                Step::CloseName { name, arity } => {
                    let values = results.split_off(results.len() - arity);
                    let provenance = Provenance::all(values);
                    self.open.remove(name);
                    self.resolved.insert(name, provenance);
                    results.push(provenance);
                }
            }
        }
        debug_assert_eq!(results.len(), 1, "one result per classified expression");
        results.pop().unwrap_or(Provenance::Data(None))
    }

    fn enter(
        &mut self,
        current: &'a Expr,
        work: &mut Vec<Step<'a>>,
        results: &mut Vec<Provenance<'a>>,
    ) {
        let magnitude = expr_magnitude(current);
        match magnitude {
            ExprMagnitude::HostFact
            | ExprMagnitude::BufferExtent(_)
            | ExprMagnitude::BitCount
            | ExprMagnitude::Predicate => results.push(Provenance::Host),
            ExprMagnitude::BufferElement(buffer) => results.push(Provenance::Data(Some(buffer))),
            ExprMagnitude::Unknown => results.push(Provenance::Data(None)),
            ExprMagnitude::Binding(name) => self.enter_name(name, work, results),
            ExprMagnitude::LeastOperand | ExprMagnitude::AllOperands => {
                let children = expr_children(current);
                let arity = children.iter().count();
                work.push(Step::Combine { magnitude, arity });
                work.extend(children.iter().rev().map(Step::Enter));
            }
        }
    }

    fn enter_name(
        &mut self,
        name: &'a Ident,
        work: &mut Vec<Step<'a>>,
        results: &mut Vec<Provenance<'a>>,
    ) {
        if let Some(provenance) = self.resolved.get(name) {
            results.push(*provenance);
            return;
        }
        // A name reached while its own values are still being classified is a
        // cycle, which an accumulator written inside the loop it bounds
        // produces. Nothing in the cycle fixes the value, so the cycle is the
        // finding rather than a reason to recurse.
        if !self.open.insert(name) {
            results.push(Provenance::Data(None));
            return;
        }
        // A name no statement binds is rejected by validation before this
        // analysis is meaningful, so it is reported rather than assumed fixed.
        // The value list is copied out because the `else` arm writes back into
        // `self`, and every element is a shared reference.
        let Some(values) = self.bindings.get(name).cloned() else {
            self.open.remove(name);
            self.resolved.insert(name, Provenance::Data(None));
            results.push(Provenance::Data(None));
            return;
        };
        work.push(Step::CloseName {
            name,
            arity: values.len(),
        });
        work.extend(values.into_iter().rev().map(Step::Enter));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType};

    fn program(body: Vec<Node>) -> Program {
        Program::wrapped(
            vec![
                BufferDecl::read("counts", 0, DataType::U32),
                BufferDecl::output("out", 1, DataType::U32),
            ],
            [64, 1, 1],
            body,
        )
    }

    fn store_one() -> Node {
        Node::store("out", Expr::var("i"), Expr::u32(1))
    }

    #[test]
    fn a_literal_bound_is_fixed_when_the_program_is_built() {
        let found = data_derived_loop_bounds(&program(vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(64),
            vec![store_one()],
        )]));
        assert_eq!(found, Vec::new());
    }

    #[test]
    fn a_buffer_extent_bound_is_fixed_when_the_program_is_built() {
        let found = data_derived_loop_bounds(&program(vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::buf_len("out"),
            vec![store_one()],
        )]));
        assert_eq!(found, Vec::new());
    }

    #[test]
    fn a_bound_loaded_from_a_buffer_names_that_buffer() {
        let found = data_derived_loop_bounds(&program(vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::load("counts", Expr::u32(0)),
            vec![store_one()],
        )]));
        assert_eq!(
            found,
            vec![DataDerivedLoopBound {
                var: Ident::from("i"),
                source: Some(Ident::from("counts")),
            }]
        );
    }

    #[test]
    fn a_loaded_bound_reached_through_a_binding_is_still_reported() {
        let found = data_derived_loop_bounds(&program(vec![
            Node::let_bind("limit", Expr::load("counts", Expr::u32(0))),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::add(Expr::var("limit"), Expr::u32(1)),
                vec![store_one()],
            ),
        ]));
        assert_eq!(
            found,
            vec![DataDerivedLoopBound {
                var: Ident::from("i"),
                source: Some(Ident::from("counts")),
            }]
        );
    }

    #[test]
    fn clamping_a_loaded_bound_against_an_extent_fixes_it() {
        let found = data_derived_loop_bounds(&program(vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::min(Expr::load("counts", Expr::u32(0)), Expr::buf_len("out")),
            vec![store_one()],
        )]));
        assert_eq!(found, Vec::new());
    }

    #[test]
    fn clamping_a_loaded_bound_against_another_load_does_not_fix_it() {
        let found = data_derived_loop_bounds(&program(vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::min(
                Expr::load("counts", Expr::u32(0)),
                Expr::load("counts", Expr::u32(1)),
            ),
            vec![store_one()],
        )]));
        assert_eq!(
            found,
            vec![DataDerivedLoopBound {
                var: Ident::from("i"),
                source: Some(Ident::from("counts")),
            }]
        );
    }

    #[test]
    fn a_population_count_of_a_load_is_bounded_by_the_element_width() {
        let found = data_derived_loop_bounds(&program(vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::popcount(Expr::load("counts", Expr::u32(0))),
            vec![store_one()],
        )]));
        assert_eq!(found, Vec::new());
    }

    #[test]
    fn a_bound_rebound_from_a_load_is_reported_through_the_reassignment() {
        let found = data_derived_loop_bounds(&program(vec![
            Node::let_bind("limit", Expr::u32(4)),
            Node::assign("limit", Expr::load("counts", Expr::u32(0))),
            Node::loop_for("i", Expr::u32(0), Expr::var("limit"), vec![store_one()]),
        ]));
        assert_eq!(
            found,
            vec![DataDerivedLoopBound {
                var: Ident::from("i"),
                source: Some(Ident::from("counts")),
            }]
        );
    }

    #[test]
    fn a_bound_a_loop_body_feeds_back_into_terminates_the_walk() {
        let found = data_derived_loop_bounds(&program(vec![
            Node::let_bind("limit", Expr::u32(4)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::var("limit"),
                vec![Node::assign(
                    "limit",
                    Expr::add(Expr::var("limit"), Expr::u32(1)),
                )],
            ),
        ]));
        assert_eq!(
            found,
            vec![DataDerivedLoopBound {
                var: Ident::from("i"),
                source: None,
            }]
        );
    }

    #[test]
    fn a_loop_nested_under_a_data_derived_loop_inherits_the_finding() {
        let found = data_derived_loop_bounds(&program(vec![Node::loop_for(
            "outer",
            Expr::u32(0),
            Expr::load("counts", Expr::u32(0)),
            vec![Node::loop_for(
                "inner",
                Expr::u32(0),
                Expr::var("outer"),
                vec![store_one()],
            )],
        )]));
        assert_eq!(
            found,
            vec![
                DataDerivedLoopBound {
                    var: Ident::from("outer"),
                    source: Some(Ident::from("counts")),
                },
                DataDerivedLoopBound {
                    var: Ident::from("inner"),
                    source: Some(Ident::from("counts")),
                },
            ]
        );
    }

    #[test]
    fn a_call_result_is_reported_without_a_buffer() {
        let found = data_derived_loop_bounds(&program(vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::call("vyre.op.unresolved", vec![Expr::u32(1)]),
            vec![store_one()],
        )]));
        assert_eq!(
            found,
            vec![DataDerivedLoopBound {
                var: Ident::from("i"),
                source: None,
            }]
        );
    }
}
