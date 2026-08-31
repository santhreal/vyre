#![allow(clippy::unwrap_used)]
//! Top-level validation entry point.
//!
//! This module runs the complete validation pipeline on a `Program`:
//! buffer declarations, node structure, expression types, depth limits,
//! and output markers. Every error is returned as a `ValidationError`
//! with an actionable `Fix:` hint.

mod node_visitor;

use super::expr_rules::validate_output_markers;
use crate::validate::{ValidationLocation, ValidationPhase};
use std::borrow::Cow;
// Self-composition (duplicate self-exclusive regions) is enforced in
// `PreorderValidator::run` via `self_comp_counts`  -  do not add a second
// `duplicate_self_exclusive_regions` walk here.
use super::{depth, err, node_rules, ValidationError, ValidationOptions, ValidationReport};
use crate::ir_inner::model::expr::{Expr, Ident};
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::op_signature::{BufferAccess, DataType};
use crate::ir_inner::model::program::Program;
use crate::visit::child_bodies;
use crate::visit::node_visitor::dispatch_node;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::convert::Infallible;
use std::ops::ControlFlow;

/// Validate a program for structural and semantic correctness.
///
/// The validator checks the stable rules of the IR semantics:
/// workgroup dimensions must be positive,
/// buffer names and bindings must be unique, workgroup buffers must have
/// a positive element count, and the node tree must respect depth limits.
/// A successful validation (empty error vector) means the program is
/// safe to lower to any backend.
///
/// # Examples
///
/// ```
/// use vyre::ir::Program;
/// use vyre::validate;
///
/// let program = Program::wrapped(Vec::new(), [1, 1, 1], Vec::new());
/// let errors = validate(&program);
/// assert!(errors.is_empty());
/// ```
#[inline]
#[must_use]
pub fn validate(program: &Program) -> Vec<ValidationError> {
    validate_with_options(program, ValidationOptions::default()).errors
}

/// Report ONLY the `Fma`-operand f32 violations (rule `V028`) in `program`.
///
/// This is the focused subset that emit backends must run before lowering.
/// An `Fma` node with non-f32 operands is unique among IR-validity hazards in
/// that it BOTH silently miscompiles (integer operands lower to `a*b+c`, not
/// fused-multiply-add) AND emits successfully (no downstream stage rejects it).
/// Every other validation rule corresponds to a program that either emits
/// correctly or fails with a dedicated, more-specific downstream diagnostic, so
/// emit boundaries must not run full [`validate`] (it would preempt those
/// messages and trip rules unrelated to silent miscompilation).
///
/// Reuses the full validator so `Fma` type inference stays single-sourced with
/// [`validate`]. Selection is by stable typed rule identity, never rendered
/// prose.
#[must_use]
pub fn fma_f32_violations(program: &Program) -> Vec<ValidationError> {
    validate(program)
        .into_iter()
        .filter(|error| error.code().as_str() == "V028")
        .collect()
}

/// Validate a program with explicit backend/shadowing options.
///
/// `ValidationOptions::default()` performs best-effort universal validation:
/// it enforces backend-independent structural rules but does not reject
/// backend-specific cast targets unless a concrete backend capability contract
/// is supplied.
#[inline]
#[must_use]
pub fn validate_with_options(
    program: &Program,
    options: ValidationOptions<'_>,
) -> ValidationReport {
    let (mut report, buffer_map) = validate_program_level(program);

    let mut validator = PreorderValidator::new(options, buffer_map);
    validator.run(program.entry());
    report.errors.append(&mut validator.errors);
    report.warnings.append(&mut validator.warnings);

    // V116 has exactly one implementation, and it is a whole-program pass, not
    // a rule the node walk can carry: the hazard is a relation between two
    // nodes, and threading it through the walk's frame stack is what let the
    // walk record a transfer's `source`/`destination` while dropping the
    // `offset`/`size` operands.
    super::fusion_safety::validate_fusion_alias_hazards(program.entry(), &mut report.errors);

    // P-1.0-V2.2: linear-type discipline checker. Reports buffers
    // whose `LinearType` declaration is violated by the actual usage
    // count in the IR.
    report
        .errors
        .extend(crate::validate::linear_type::check_linear_types(program));

    // P-1.0-V3.2: shape-predicate refinement checker. Reports buffers
    // whose static `count` violates the declared `ShapePredicate`.
    report
        .errors
        .extend(crate::validate::shape_predicate::check_shape_predicates(
            program,
        ));

    // V131/V132/V133: async copy tag discipline. Reports a tag started while
    // it is already in flight, a wait with nothing to wait for, and a transfer
    // left in flight where the invocation ends. All three are relations
    // between nodes on a path, which the node walk cannot see. The
    // differential property in `tests` runs this pass on its legacy arm too,
    // because a pass only one arm runs makes every async program a mismatch.
    report
        .errors
        .extend(crate::validate::async_pipeline::check_async_pipeline(
            program,
        ));

    for (ordinal, issue) in report.errors.iter_mut().enumerate() {
        if matches!(issue.location(), ValidationLocation::Program) {
            issue.set_location(ValidationLocation::Traversal {
                ordinal: ordinal as u64,
            });
        }
    }

    report
        .trace
        .extend(report.errors.iter().map(ValidationError::trace_event));

    report
}

/// Every rule that reads the program header rather than the node tree, plus the
/// buffer lookup a node walk needs and the report the walk appends to.
///
/// Both the production single walk and the multi-walk validator that the
/// differential property test keeps as its second arm start here. Stating it
/// once is why a correction to a buffer-table diagnostic cannot make that
/// property fail for a reason that has nothing to do with the walk it compares.
pub(crate) fn validate_program_level<'p>(
    program: &'p Program,
) -> (
    ValidationReport,
    FxHashMap<&'p str, &'p crate::ir_inner::model::program::BufferDecl>,
) {
    let mut report = ValidationReport {
        errors: Vec::with_capacity(program.buffers().len() + program.entry().len()),
        warnings: Vec::new(),
        trace: Vec::new(),
    };

    if let Some(message) = program.top_level_region_violation_cause() {
        report.errors.push(err(
            "V105",
            ValidationPhase::Program,
            ValidationLocation::Program,
            message,
            "construct runnable programs with `Program::wrapped(...)` or wrap the body in `Node::Region` before validation, interpretation, or dispatch",
        ));
    }

    for (axis, &size) in program.workgroup_size.iter().enumerate() {
        if size == 0 {
            report.errors.push(err(
                "V106",
                ValidationPhase::Program,
                ValidationLocation::WorkgroupAxis(axis as u8),
                format!("workgroup_size[{axis}] is 0"),
                "all workgroup dimensions must be >= 1.",
            ));
        }
    }

    let mut seen_names = FxHashSet::default();
    seen_names.reserve(program.buffers().len());
    let mut seen_bindings = FxHashSet::default();
    seen_bindings.reserve(program.buffers().len());
    for buf in program.buffers() {
        if !seen_names.insert(&buf.name) {
            report.errors.push(err(
                "V107",
                ValidationPhase::Program,
                ValidationLocation::Buffer(Cow::Owned(buf.name.to_string())),
                format!("duplicate buffer name `{}`", buf.name),
                "each buffer must have a unique name",
            ));
        }
        if buf.access != BufferAccess::Workgroup && !seen_bindings.insert(buf.binding) {
            report.errors.push(err(
                "V108",
                ValidationPhase::Program,
                ValidationLocation::Buffer(Cow::Owned(buf.name.to_string())),
                format!(
                    "duplicate binding slot {} (buffer `{}`)",
                    buf.binding, buf.name
                ),
                "each buffer must have a unique binding",
            ));
        }
        if buf.access == BufferAccess::Workgroup && buf.count == 0 {
            report.errors.push(err(
                "V109",
                ValidationPhase::Program,
                ValidationLocation::Buffer(Cow::Owned(buf.name.to_string())),
                format!("workgroup buffer `{}` has count 0", buf.name),
                "declare a positive element count",
            ));
        }
        validate_output_buffer_contract(buf, &mut report.errors);
    }
    validate_output_markers(program.buffers(), &mut report.errors);

    let mut buffer_map: FxHashMap<&str, &crate::ir_inner::model::program::BufferDecl> =
        FxHashMap::default();
    buffer_map.reserve(program.buffers().len());
    buffer_map.extend(program.buffers().iter().map(|b| (b.name.as_ref(), b)));

    (report, buffer_map)
}

fn validate_output_buffer_contract(
    buf: &crate::ir_inner::model::program::BufferDecl,
    errors: &mut Vec<ValidationError>,
) {
    if !buf.is_output() {
        return;
    }

    if matches!(buf.element(), DataType::Array { .. } | DataType::Tensor) {
        errors.push(err(
            "V110",
            ValidationPhase::Program,
            ValidationLocation::Buffer(Cow::Owned(buf.name().to_string())),
            format!(
                "output buffer `{}` uses unsupported element type `{}`",
                buf.name(),
                buf.element()
            ),
            "output buffers must use fixed-width scalar or vector element types, not Array or Tensor",
        ));
    }
    if buf.is_backend_allocated_output() && buf.count() == 0 && buf.output_byte_range().is_none() {
        errors.push(err(
            "V130",
            ValidationPhase::Program,
            ValidationLocation::Buffer(Cow::Owned(buf.name().to_string())),
            format!(
                "backend-allocated output buffer `{}` has no static element count or output byte range",
                buf.name()
            ),
            "declare the output with `.with_count(n)`, or use `.with_output_byte_range(0..0)` for a genuinely empty output",
        ));
    }
}

// ------------------------------------------------------------------
// PreorderValidator  -  single-pass explicit-stack traversal
// ------------------------------------------------------------------

use super::binding::Binding;
use super::expr_rules;
use super::uniformity::is_uniform;
// use super::report::warn;

/// Scope frame pushed for every nested node sequence.
pub(super) struct ScopeFrame<'p> {
    pub(super) scope_log: node_rules::ScopeLog,
    pub(super) region_bindings: FxHashSet<Ident>,
    pub(super) divergent: bool,
    pub(super) depth: usize,
    pub(super) nodes: &'p [Node],
}

/// Stack frames for the explicit traversal.
enum Frame<'p> {
    /// Visit a single node (pre-order).
    Child(&'p Node),
    /// Enter a new scope.
    PushScope {
        divergent: bool,
        depth: usize,
        nodes: &'p [Node],
    },
    /// Leave the current scope and check `Return` position.
    PopScope,
    /// Inject the loop variable binding into the current scope. The
    /// `uniform` flag mirrors the loop's bound uniformity: in a
    /// uniform-bound loop every invocation walks the same iteration
    /// count with the same counter value, so the loop var is itself
    /// uniform.
    InsertLoopVar { var: Ident, uniform: bool },
    /// Inject tile input scalar bindings into the current scope.
    InsertTileInputs { inputs: Vec<Ident> },
}

/// Single-pass validator that performs all node-tree checks in one
/// explicit-stack traversal.
pub(super) struct PreorderValidator<'p, 'o> {
    pub(super) options: ValidationOptions<'o>,
    pub(super) buffers: FxHashMap<&'p str, &'p crate::ir_inner::model::program::BufferDecl>,
    pub(super) scope: FxHashMap<Ident, Binding>,
    pub(super) scope_stack: SmallVec<[ScopeFrame<'p>; 16]>,
    pub(super) limits: depth::LimitState,
    pub(super) self_comp_counts: hashbrown::HashMap<String, usize>,
    pub(super) errors: Vec<ValidationError>,
    pub(super) warnings: Vec<super::ValidationWarning>,
    pub(super) current_node: u32,
    pub(super) next_node: u32,
    /// HOT PATH (`PreorderValidator::validate_expr`): reuse one report buffer per expression so we do not allocate fresh error/warning vectors for every `validate_expr` invocation while traversing the IR tree.
    pub(super) expr_report_scratch: ValidationReport,
}

impl<'p, 'o> PreorderValidator<'p, 'o> {
    fn new(
        options: ValidationOptions<'o>,
        buffers: FxHashMap<&'p str, &'p crate::ir_inner::model::program::BufferDecl>,
    ) -> Self {
        Self {
            options,
            buffers,
            scope: FxHashMap::default(),
            scope_stack: SmallVec::new(),
            limits: depth::LimitState::default(),
            self_comp_counts: hashbrown::HashMap::default(),
            errors: Vec::new(),
            current_node: 0,
            next_node: 0,
            warnings: Vec::new(),
            expr_report_scratch: ValidationReport::default(),
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "single-pass validator run loop is an explicit stack machine; keeping frames together preserves stack-safety and validation-order invariants"
    )]
    fn run(&mut self, nodes: &'p [Node]) {
        let mut stack: SmallVec<[Frame<'p>; 128]> = SmallVec::new();
        stack.push(Frame::PopScope);
        for node in nodes.iter().rev() {
            stack.push(Frame::Child(node));
        }
        stack.push(Frame::PushScope {
            divergent: false,
            depth: 0,
            nodes,
        });

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Child(node) => {
                    let synthetic_root = self.next_node == 0
                        && matches!(
                            node,
                            Node::Region { generator, .. }
                                if generator.as_ref() == Program::ROOT_REGION_GENERATOR
                        );
                    self.current_node = self.next_node;
                    if !synthetic_root {
                        self.next_node = self.next_node.saturating_add(1);
                    }
                    let first_new_error = self.errors.len();
                    if dispatch_node(self, node).is_break() {
                        break;
                    }
                    match node {
                        Node::If {
                            cond,
                            then,
                            otherwise,
                            ..
                        } => {
                            let depth = self.current_depth();
                            // Branches stay non-divergent only when the
                            // parent scope is already uniform AND the
                            // condition is uniform across the workgroup.
                            // A non-uniform cond splits invocations
                            // across the two branches, so any barrier
                            // inside is reached by only some lanes.
                            let parent_divergent = self.current_divergent();
                            let branch_divergent =
                                parent_divergent || !is_uniform(cond, &self.scope);
                            push_nested_sequence(
                                &mut stack,
                                otherwise,
                                branch_divergent,
                                depth + 1,
                                None,
                            );
                            push_nested_sequence(
                                &mut stack,
                                then,
                                branch_divergent,
                                depth + 1,
                                None,
                            );
                        }
                        Node::Loop {
                            var,
                            from,
                            to,
                            body,
                        } => {
                            let depth = self.current_depth();
                            // The loop body is divergent only when its
                            // parent already is OR when either bound
                            // varies across the workgroup. Uniform
                            // bounds keep every invocation in lockstep
                            //  -  same iteration count, same loop-var
                            // value at each step  -  so a barrier inside
                            // is reached by every lane simultaneously.
                            let parent_divergent = self.current_divergent();
                            let bounds_uniform =
                                is_uniform(from, &self.scope) && is_uniform(to, &self.scope);
                            let body_divergent = parent_divergent || !bounds_uniform;
                            // Loop var inherits the bounds' uniformity
                            // when the parent is also uniform; if the
                            // parent is divergent the var only matters
                            // within already-divergent context.
                            let var_uniform = bounds_uniform && !parent_divergent;
                            push_nested_sequence(
                                &mut stack,
                                body,
                                body_divergent,
                                depth + 1,
                                Some(Frame::InsertLoopVar {
                                    var: var.clone(),
                                    uniform: var_uniform,
                                }),
                            );
                        }
                        Node::TileElementwise { inputs, body, .. } => {
                            let depth = self.current_depth();
                            let divergent = self.current_divergent();
                            push_nested_sequence(
                                &mut stack,
                                body,
                                divergent,
                                depth + 1,
                                Some(Frame::InsertTileInputs {
                                    inputs: inputs.clone(),
                                }),
                            );
                        }
                        // Every other variant descends through the one owner
                        // of which variants nest bodies. A body scopes its
                        // bindings like a block: a `let` inside one does not
                        // outlive it. `region_inline_scope` pins that, because
                        // `region_inline` flattening a Region into its parent
                        // is only sound while the parent cannot already see the
                        // flattened bindings. A leaf yields no body and pushes
                        // nothing, and a body-bearing variant added tomorrow
                        // descends here instead of having its children skipped
                        // by every rule in the pipeline.
                        other => {
                            let depth = self.current_depth();
                            let divergent = self.current_divergent();
                            for body in child_bodies(other).into_iter().rev() {
                                if body.is_empty() {
                                    continue;
                                }
                                push_nested_sequence(&mut stack, body, divergent, depth + 1, None);
                            }
                        }
                    }
                    for issue in &mut self.errors[first_new_error..] {
                        match issue.location() {
                            ValidationLocation::Program => {
                                issue.set_location(ValidationLocation::Node(self.current_node));
                            }
                            ValidationLocation::Expression { depth, .. } => {
                                issue.set_location(ValidationLocation::Expression {
                                    node: self.current_node,
                                    depth: *depth,
                                });
                            }
                            ValidationLocation::Operand { operand, .. } => {
                                issue.set_location(ValidationLocation::Operand {
                                    node: self.current_node,
                                    operand: *operand,
                                });
                            }
                            ValidationLocation::Node(_) => {
                                issue.set_location(ValidationLocation::Node(self.current_node));
                            }
                            _ => {}
                        }
                    }
                }
                Frame::PushScope {
                    divergent,
                    depth,
                    nodes,
                } => {
                    self.scope_stack.push(ScopeFrame {
                        scope_log: Vec::new(),
                        region_bindings: FxHashSet::default(),
                        divergent,
                        depth,
                        nodes,
                    });
                }
                Frame::PopScope => {
                    let Some(frame) = self.scope_stack.pop() else {
                        self.errors.push(err("V111", ValidationPhase::Node, ValidationLocation::Program, "malformed validation frame stream: PopScope without matching PushScope".to_string(), "rebuild the program through the structured IR builder before validation.".to_string()));
                        continue;
                    };
                    node_rules::restore_scope(&mut self.scope, frame.scope_log);
                    node_rules::check_unreachable_after_return(frame.nodes, &mut self.errors);
                }
                Frame::InsertLoopVar { var, uniform } => {
                    let Some(frame) = self.scope_stack.last_mut() else {
                        self.errors.push(err("V114", ValidationPhase::Node, ValidationLocation::Program, format!(
                            "malformed validation frame stream: loop variable `{var}` inserted outside any scope"
                        ), "rebuild the program through the structured IR builder before validation.".to_string()));
                        continue;
                    };
                    node_rules::insert_binding(
                        &mut self.scope,
                        var.clone(),
                        node_rules::loop_var_binding(uniform),
                        Some(&mut frame.scope_log),
                    );
                }
                Frame::InsertTileInputs { inputs } => {
                    let Some(frame) = self.scope_stack.last_mut() else {
                        continue;
                    };
                    for input in inputs {
                        node_rules::insert_binding(
                            &mut self.scope,
                            input.clone(),
                            Binding {
                                ty: DataType::F32,
                                ty_known: true,
                                mutable: false,
                                uniform: false,
                            },
                            Some(&mut frame.scope_log),
                        );
                    }
                }
            }
        }

        // Emit self-composition errors deterministically.
        let mut duplicates: Vec<String> = self
            .self_comp_counts
            .drain()
            .filter_map(|(generator, count)| (count > 1).then_some(generator))
            .collect();
        duplicates.sort_unstable();
        for generator in duplicates {
            self.errors.push(err("V115", ValidationPhase::Composition, ValidationLocation::Program, format!(
                "region `{generator}` is marked non-composable with itself but appears multiple times in one fused program"
            ), "split the parser into separate dispatches, or give each instance distinct scratch storage before fusion.".to_string()));
        }
    }

    #[inline]
    pub(super) fn current_divergent(&self) -> bool {
        self.scope_stack.last().is_some_and(|f| f.divergent)
    }

    #[inline]
    pub(super) fn current_depth(&self) -> usize {
        self.scope_stack.last().map_or(0, |f| f.depth)
    }

    /// Run the legacy `validate_expr` helper and merge its diagnostics.
    pub(super) fn validate_expr(&mut self, expr: &Expr, depth_level: usize) {
        self.expr_report_scratch.errors.clear();
        self.expr_report_scratch.warnings.clear();
        expr_rules::validate_expr(
            expr,
            &self.buffers,
            &self.scope,
            self.options,
            &mut self.expr_report_scratch,
            depth_level,
        );
        for issue in &mut self.expr_report_scratch.errors {
            match issue.location() {
                ValidationLocation::Program => {
                    issue.set_location(ValidationLocation::Expression {
                        node: self.current_node,
                        depth: u32::try_from(depth_level).unwrap_or(u32::MAX),
                    });
                }
                ValidationLocation::Expression { depth, .. } => {
                    issue.set_location(ValidationLocation::Expression {
                        node: self.current_node,
                        depth: *depth,
                    });
                }
                ValidationLocation::Operand { operand, .. } => {
                    issue.set_location(ValidationLocation::Operand {
                        node: self.current_node,
                        operand: *operand,
                    });
                }
                ValidationLocation::Node(_) => {
                    issue.set_location(ValidationLocation::Node(self.current_node));
                }
                _ => {}
            }
        }
        self.errors.append(&mut self.expr_report_scratch.errors);
        for warning in &mut self.expr_report_scratch.warnings {
            if warning.location.as_ref().is_some_and(|location| {
                location.op_id.as_ref() == "program"
                    && location.operand_idx.is_none()
                    && location.attr_name.is_none()
                    && location.graph_node.is_none()
                    && location.graph_value.is_none()
                    && location.path.is_none()
                    && location.source_span.is_none()
            }) {
                warning.location = Some(
                    ValidationLocation::Expression {
                        node: self.current_node,
                        depth: u32::try_from(depth_level).unwrap_or(u32::MAX),
                    }
                    .diagnostic_location(),
                );
            }
        }
        self.warnings.append(&mut self.expr_report_scratch.warnings);
    }

    /// An async transfer carries the same empty-tag rule as the wait that pairs
    /// with it, and validates `offset` and `size` as expressions like any other.
    ///
    /// It reported the tag rule as `V117` while `visit_async_wait` reported the
    /// identical condition as `V128`, so the same defect had two stable
    /// identities depending on which end of the transfer carried it. `V128` is
    /// the per-node rule family this belongs to; `V117` sat in the
    /// malformed-frame-stream range and is gone.
    ///
    /// `source` and `destination` are storage-tier tags first: a transfer may
    /// name an endpoint outside the dispatch's buffers, so neither is required
    /// to resolve. `extension_adversarial::async_extension_tags_remain_structural`
    /// pins that. A destination that DOES resolve is a buffer the target
    /// compilers store through, so it carries the writability rule every other
    /// write carries; `bytes_rejection::check_async_destination` owns it.
    ///
    /// What this does NOT do is record alias accesses. `V116` is owned by
    /// `super::fusion_safety`, which records all four operands; recording two of
    /// them here as well is what produced two answers for one rule.
    pub(super) fn validate_async_transfer(
        &mut self,
        destination: &Ident,
        offset: &Expr,
        size: &Expr,
        tag: &Ident,
    ) -> ControlFlow<Infallible> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_async_transfer(
            destination,
            offset,
            size,
            tag,
            &self.buffers,
            &self.scope,
            &mut self.errors,
        );
        // The offset and size operands are expressions like any other, and
        // going unvalidated meant a load from an undeclared buffer inside a
        // transfer size was accepted while the same load in a store index was
        // rejected.
        self.validate_expr(offset, 0);
        self.validate_expr(size, 0);

        ControlFlow::Continue(())
    }
}

/// Push the stack frames needed to process a nested node sequence.
fn push_nested_sequence<'p>(
    stack: &mut SmallVec<[Frame<'p>; 128]>,
    nodes: &'p [Node],
    divergent: bool,
    depth: usize,
    pre_children: Option<Frame<'p>>,
) {
    stack.push(Frame::PopScope);
    for child in nodes.iter().rev() {
        stack.push(Frame::Child(child));
    }
    if let Some(pre) = pre_children {
        stack.push(pre);
    }
    stack.push(Frame::PushScope {
        divergent,
        depth,
        nodes,
    });
}

#[cfg(test)]
#[path = "../../../tests/internal/validate/rule_pipeline/mod.rs"]
mod tests;
