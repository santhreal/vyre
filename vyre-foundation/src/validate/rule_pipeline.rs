#![allow(clippy::unwrap_used)]
//! Top-level validation entry point.
//!
//! This module runs the complete validation pipeline on a `Program`:
//! buffer declarations, node structure, expression types, depth limits,
//! and output markers. Every error is returned as a `ValidationError`
//! with an actionable `Fix:` hint.

pub use super::depth::{
    DEFAULT_MAX_CALL_DEPTH, DEFAULT_MAX_EXPR_DEPTH, DEFAULT_MAX_NESTING_DEPTH,
    DEFAULT_MAX_NODE_COUNT,
};
use super::expr_rules::validate_output_markers;
use super::fusion_safety::{collect_expr_accesses, NodeAccesses};
use crate::validate::{ValidationLocation, ValidationPhase};
use std::borrow::Cow;
// Self-composition (duplicate self-exclusive regions) is enforced in
// `PreorderValidator::run` via `self_comp_counts`  -  do not add a second
// `duplicate_self_exclusive_regions` walk here.
use super::{depth, err, node_rules, ValidationError, ValidationOptions, ValidationReport};
use crate::composition::self_exclusive_region_key;
use crate::ir_inner::model::expr::{Expr, Ident};
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::program::Program;
use crate::ir_inner::model::spec_types::{BufferAccess, DataType};
use crate::visit::traits::{dispatch_node, NodeVisitor};
use hashbrown::hash_map::RawEntryMut;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::convert::Infallible;
use std::ops::ControlFlow;

/// Validate a program for structural and semantic correctness.
///
/// The validator checks the stable rules documented in
/// `docs/ir-semantics.md`: workgroup dimensions must be positive,
/// buffer names and bindings must be unique, workgroup buffers must have
/// a positive element count, and the node tree must respect depth limits.
/// A successful validation (empty error vector) means the program is
/// safe to lower to any backend.
///
/// # Examples
///
/// ```
/// use vyre::ir::{Program, validate};
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

use super::barrier;
use super::binding::{check_sibling_duplicate, Binding};
use super::expr_rules;
use super::shadowing;
use super::typecheck::expr_type;
use super::uniformity::is_uniform;
// use super::report::warn;

/// Scope frame pushed for every nested node sequence.
struct ScopeFrame<'p> {
    scope_log: node_rules::ScopeLog,
    region_bindings: FxHashSet<Ident>,
    divergent: bool,
    depth: usize,
    nodes: &'p [Node],
}

/// Stack frames for the explicit traversal.
enum Frame<'p> {
    /// Visit a single node (pre-order).
    Child(&'p Node),
    /// Post-order action for `If`: extend parent alias state with cond accesses.
    PostIf,
    /// Post-order action for `Loop`: extend parent alias state with from/to accesses.
    PostLoop,
    /// Enter a new scope.
    PushScope {
        divergent: bool,
        depth: usize,
        nodes: &'p [Node],
    },
    /// Leave the current scope and check `Return` position.
    PopScope,
    /// Enter a fresh alias tracking frame.
    PushAlias,
    /// Restore the parent alias tracking frame.
    PopAlias,
    /// Inject the loop variable binding into the current scope. The
    /// `uniform` flag mirrors the loop's bound uniformity: in a
    /// uniform-bound loop every invocation walks the same iteration
    /// count with the same counter value, so the loop var is itself
    /// uniform.
    InsertLoopVar { var: Ident, uniform: bool },
}

/// Single-pass validator that performs all node-tree checks in one
/// explicit-stack traversal.
struct PreorderValidator<'p, 'o> {
    options: ValidationOptions<'o>,
    buffers: FxHashMap<&'p str, &'p crate::ir_inner::model::program::BufferDecl>,
    scope: FxHashMap<Ident, Binding>,
    scope_stack: SmallVec<[ScopeFrame<'p>; 16]>,
    limits: depth::LimitState,
    alias_reads: FxHashSet<Ident>,
    alias_atomics: FxHashSet<Ident>,
    alias_stack: SmallVec<[(FxHashSet<Ident>, FxHashSet<Ident>); 8]>,
    pending_alias_extensions: SmallVec<[NodeAccesses; 8]>,
    self_comp_counts: hashbrown::HashMap<String, usize>,
    errors: Vec<ValidationError>,
    warnings: Vec<super::ValidationWarning>,
    current_node: u32,
    next_node: u32,
    /// HOT PATH (`PreorderValidator::validate_expr`): reuse one report buffer per expression so we do not allocate fresh error/warning vectors for every `validate_expr` invocation while traversing the IR tree.
    expr_report_scratch: ValidationReport,
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
            alias_reads: FxHashSet::default(),
            alias_atomics: FxHashSet::default(),
            alias_stack: SmallVec::new(),
            pending_alias_extensions: SmallVec::new(),
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
        stack.push(Frame::PushAlias);
        stack.push(Frame::PushScope {
            divergent: false,
            depth: 0,
            nodes,
        });

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Child(node) => {
                    self.current_node = self.next_node;
                    self.next_node = self.next_node.saturating_add(1);
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
                            stack.push(Frame::PostIf);
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
                            stack.push(Frame::PostLoop);
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
                        Node::Block(body) => {
                            let depth = self.current_depth();
                            let divergent = self.current_divergent();
                            push_nested_sequence(&mut stack, body, divergent, depth + 1, None);
                        }
                        Node::Region { body, .. } => {
                            let depth = self.current_depth();
                            let divergent = self.current_divergent();
                            push_nested_sequence(&mut stack, body, divergent, depth + 1, None);
                        }
                        _ => {}
                    }
                    for issue in &mut self.errors[first_new_error..] {
                        if matches!(issue.location(), ValidationLocation::Program) {
                            issue.set_location(ValidationLocation::Node(self.current_node));
                        }
                    }
                }
                Frame::PostIf | Frame::PostLoop => {
                    if let Some(accesses) = self.pending_alias_extensions.pop() {
                        self.extend_alias(&accesses);
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
                Frame::PushAlias => {
                    let reads = std::mem::take(&mut self.alias_reads);
                    let atomics = std::mem::take(&mut self.alias_atomics);
                    self.alias_stack.push((reads, atomics));
                    self.alias_reads = FxHashSet::default();
                    self.alias_atomics = FxHashSet::default();
                }
                Frame::PopAlias => {
                    let Some((reads, atomics)) = self.alias_stack.pop() else {
                        self.errors.push(err("V113", ValidationPhase::Node, ValidationLocation::Program, "malformed validation frame stream: PopAlias without matching PushAlias".to_string(), "rebuild the program through the structured IR builder before validation.".to_string()));
                        continue;
                    };
                    let _ = std::mem::take(&mut self.alias_reads);
                    let _ = std::mem::take(&mut self.alias_atomics);
                    self.alias_reads = reads;
                    self.alias_atomics = atomics;
                }
                Frame::InsertLoopVar { var, uniform } => {
                    let Some(frame) = self.scope_stack.last_mut() else {
                        self.errors.push(err("V114", ValidationPhase::Node, ValidationLocation::Program, format!(
                            "malformed validation frame stream: loop variable `{var}` inserted outside any scope"
                        ), format!(
                            "rebuild the program through the structured IR builder before validation."
                        )));
                        continue;
                    };
                    node_rules::insert_binding(
                        &mut self.scope,
                        var.clone(),
                        node_rules::loop_var_binding(uniform),
                        Some(&mut frame.scope_log),
                    );
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
            ), format!(
                "split the parser into separate dispatches, or give each instance distinct scratch storage before fusion."
            )));
        }
    }

    #[inline]
    fn current_divergent(&self) -> bool {
        self.scope_stack.last().is_some_and(|f| f.divergent)
    }

    #[inline]
    fn current_depth(&self) -> usize {
        self.scope_stack.last().map_or(0, |f| f.depth)
    }

    /// Run the legacy `validate_expr` helper and merge its diagnostics.
    fn validate_expr(&mut self, expr: &Expr, depth_level: usize) {
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
            if matches!(issue.location(), ValidationLocation::Program) {
                issue.set_location(ValidationLocation::Expression {
                    node: self.current_node,
                    depth: u32::try_from(depth_level).unwrap_or(u32::MAX),
                });
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

    /// Report fusion-alias hazards between `accesses` and the current linear state.
    fn report_alias_hazards(&mut self, accesses: &NodeAccesses) {
        let mut hazards = accesses
            .atomic_buffers
            .intersection(&self.alias_reads)
            .cloned()
            .collect::<SmallVec<[Ident; 8]>>();
        hazards.extend(
            accesses
                .read_buffers
                .intersection(&self.alias_atomics)
                .cloned(),
        );
        hazards.sort_unstable_by(|a, b| a.as_str().cmp(b.as_str()));
        hazards.dedup();

        for buffer in hazards {
            self.errors.push(err("V116", ValidationPhase::Composition, ValidationLocation::Program, format!(
                "fusion hazard on buffer `{buffer}`: one node reads it non-atomically while another issues an atomic access without an explicit barrier"
            ), format!(
                "insert `Node::barrier()` between the read path and the atomic path, or rename the buffers before fusion."
            )));
        }
    }

    /// Extend the current alias frame with `accesses`.
    fn extend_alias(&mut self, accesses: &NodeAccesses) {
        self.alias_reads
            .extend(accesses.read_buffers.iter().cloned());
        self.alias_atomics
            .extend(accesses.atomic_buffers.iter().cloned());
    }

    /// An async transfer carries the same empty-tag rule as the wait that pairs
    /// with it, plus the two buffers it touches.
    ///
    /// It reported that rule as `V117` while `visit_async_wait` reported the
    /// identical condition as `V128`, so the same defect had two stable
    /// identities depending on which end of the transfer carried it. `V128` is
    /// the per-node rule family this belongs to (`V119`-`V130`); `V117` sat in
    /// the malformed-frame-stream range and is gone.
    fn validate_async_transfer(
        &mut self,
        source: &Ident,
        destination: &Ident,
        offset: &Expr,
        size: &Expr,
        tag: &Ident,
    ) -> ControlFlow<Infallible> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_async_tag(tag, &mut self.errors);
        // The offset and size operands are expressions like any other, and
        // going unvalidated meant a load from an undeclared buffer inside a
        // transfer size was accepted while the same load in a store index was
        // rejected.
        self.validate_expr(offset, 0);
        self.validate_expr(size, 0);

        let mut accesses = NodeAccesses::default();
        accesses.read_buffers.insert(source.clone());
        accesses.read_buffers.insert(destination.clone());
        self.report_alias_hazards(&accesses);
        self.extend_alias(&accesses);

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
    stack.push(Frame::PopAlias);
    for child in nodes.iter().rev() {
        stack.push(Frame::Child(child));
    }
    if let Some(pre) = pre_children {
        stack.push(pre);
    }
    stack.push(Frame::PushAlias);
    stack.push(Frame::PushScope {
        divergent,
        depth,
        nodes,
    });
}

macro_rules! async_transfer_visitors {
    () => {
        fn visit_async_load(
            &mut self,
            _node: &Node,
            source: &Ident,
            destination: &Ident,
            offset: &Expr,
            size: &Expr,
            tag: &Ident,
        ) -> ControlFlow<Self::Break> {
            self.validate_async_transfer(source, destination, offset, size, tag)
        }

        fn visit_async_store(
            &mut self,
            _node: &Node,
            source: &Ident,
            destination: &Ident,
            offset: &Expr,
            size: &Expr,
            tag: &Ident,
        ) -> ControlFlow<Self::Break> {
            self.validate_async_transfer(source, destination, offset, size, tag)
        }
    };
}

// ------------------------------------------------------------------
// NodeVisitor implementation
// ------------------------------------------------------------------

impl NodeVisitor for PreorderValidator<'_, '_> {
    type Break = Infallible;

    fn visit_let(&mut self, _node: &Node, name: &Ident, value: &Expr) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        self.validate_expr(value, 0);

        let Some(frame) = self.scope_stack.last_mut() else {
            self.errors.push(err(
                "V118",
                ValidationPhase::Node,
                ValidationLocation::Program,
                format!(
                "malformed validation frame stream: let binding `{name}` appeared outside any scope"
            ),
                format!("rebuild the program through the structured IR builder before validation."),
            ));
            return ControlFlow::Continue(());
        };
        // Same-region duplicate Lets are always invalid, even when
        // shadowing is allowed for nested scopes  -  the V032 contract
        // covered by `sibling_duplicate_lets_are_rejected_even_when_shadowing_is_allowed`.
        // `allow_shadowing` only opens nested scopes; siblings collide
        // unconditionally.
        let duplicate_sibling = check_sibling_duplicate(
            name,
            &mut frame.region_bindings,
            /*allow_duplicate_siblings=*/ false,
            &mut self.errors,
        );
        if !duplicate_sibling {
            shadowing::check_local(name, &self.scope, self.options, &mut self.errors);
        }
        let ty_opt = expr_type(value, &self.buffers, &self.scope);
        let ty = ty_opt.clone().unwrap_or(DataType::U32);
        let ty_known = ty_opt.is_some();
        let uniform = is_uniform(value, &self.scope);
        node_rules::insert_binding(
            &mut self.scope,
            name.clone(),
            Binding {
                ty,
                ty_known,
                mutable: true,
                uniform,
            },
            Some(&mut frame.scope_log),
        );

        let mut accesses = NodeAccesses::default();
        collect_expr_accesses(value, &mut accesses);
        self.report_alias_hazards(&accesses);
        self.extend_alias(&accesses);

        ControlFlow::Continue(())
    }

    fn visit_assign(
        &mut self,
        _node: &Node,
        name: &Ident,
        value: &Expr,
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_assign(
            name,
            value,
            &self.buffers,
            &self.scope,
            &mut self.errors,
        );
        self.validate_expr(value, 0);

        // Reassigning with a divergent rhs taints the binding's
        // uniformity for the remainder of its lifetime.
        let new_uniform = is_uniform(value, &self.scope);
        if let Some(binding) = self.scope.get_mut(name.as_str()) {
            binding.uniform = binding.uniform && new_uniform;
        }

        let mut accesses = NodeAccesses::default();
        collect_expr_accesses(value, &mut accesses);
        self.report_alias_hazards(&accesses);
        self.extend_alias(&accesses);

        ControlFlow::Continue(())
    }

    fn visit_store(
        &mut self,
        _node: &Node,
        buffer: &Ident,
        index: &Expr,
        value: &Expr,
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_store(
            buffer,
            index,
            value,
            &self.buffers,
            &self.scope,
            &mut self.errors,
        );
        self.validate_expr(index, 0);
        self.validate_expr(value, 0);

        let mut accesses = NodeAccesses::default();
        accesses.read_buffers.insert(buffer.clone());
        collect_expr_accesses(index, &mut accesses);
        collect_expr_accesses(value, &mut accesses);
        self.report_alias_hazards(&accesses);
        self.extend_alias(&accesses);

        ControlFlow::Continue(())
    }

    fn visit_if(
        &mut self,
        _node: &Node,
        cond: &Expr,
        _then: &[Node],
        _otherwise: &[Node],
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        self.validate_expr(cond, 0);
        node_rules::check_if_condition(cond, &self.buffers, &self.scope, &mut self.errors);

        let mut accesses = NodeAccesses::default();
        collect_expr_accesses(cond, &mut accesses);
        self.report_alias_hazards(&accesses);
        self.pending_alias_extensions.push(accesses);

        ControlFlow::Continue(())
    }

    fn visit_loop(
        &mut self,
        _node: &Node,
        var: &Ident,
        from: &Expr,
        to: &Expr,
        body: &[Node],
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        self.validate_expr(from, 0);
        self.validate_expr(to, 0);
        node_rules::check_loop_bounds(from, to, &self.buffers, &self.scope, &mut self.errors);
        shadowing::check_local(var, &self.scope, self.options, &mut self.errors);
        let bounds_uniform = is_uniform(from, &self.scope) && is_uniform(to, &self.scope);
        let var_uniform = bounds_uniform && !self.current_divergent();
        let mut back_edge_scope = self.scope.clone();
        back_edge_scope.insert(var.clone(), node_rules::loop_var_binding(var_uniform));
        barrier::check_loop_back_edge(body, &back_edge_scope, &mut self.errors);

        let mut accesses = NodeAccesses::default();
        collect_expr_accesses(from, &mut accesses);
        collect_expr_accesses(to, &mut accesses);
        self.report_alias_hazards(&accesses);
        self.pending_alias_extensions.push(accesses);

        ControlFlow::Continue(())
    }

    fn visit_indirect_dispatch(
        &mut self,
        _node: &Node,
        count_buffer: &Ident,
        count_offset: u64,
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_indirect_dispatch(
            count_buffer,
            count_offset,
            &self.buffers,
            &mut self.errors,
        );

        let mut accesses = NodeAccesses::default();
        accesses.read_buffers.insert(count_buffer.clone());
        self.report_alias_hazards(&accesses);
        self.extend_alias(&accesses);

        ControlFlow::Continue(())
    }

    async_transfer_visitors!();

    fn visit_async_wait(&mut self, _node: &Node, tag: &Ident) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_async_tag(tag, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_trap(
        &mut self,
        _node: &Node,
        address: &Expr,
        _tag: &Ident,
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        self.validate_expr(address, 0);
        ControlFlow::Continue(())
    }

    fn visit_resume(&mut self, _node: &Node, _tag: &Ident) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_return(&mut self, _node: &Node) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_barrier(&mut self, node: &Node) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        let divergent = self.current_divergent();
        let Node::Barrier { ordering } = node else {
            self.errors.push(err(
                "V129",
                ValidationPhase::Memory,
                ValidationLocation::Program,
                "malformed barrier visitor dispatch".to_string(),
                "rebuild the program through the structured IR builder before validation."
                    .to_string(),
            ));
            return ControlFlow::Continue(());
        };
        barrier::check_barrier(divergent, *ordering, &mut self.errors);
        self.alias_reads.clear();
        self.alias_atomics.clear();
        ControlFlow::Continue(())
    }

    fn visit_collective(&mut self, node: &Node) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_collective(node, self.options, &self.buffers, &mut self.errors);

        let mut accesses = NodeAccesses::default();
        for name in node_rules::collective_buffers(node).into_iter().flatten() {
            accesses.read_buffers.insert(name.clone());
        }
        self.report_alias_hazards(&accesses);
        self.extend_alias(&accesses);
        ControlFlow::Continue(())
    }

    fn visit_block(&mut self, _node: &Node, _body: &[Node]) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        ControlFlow::Continue(())
    }

    fn visit_region(
        &mut self,
        _node: &Node,
        generator: &Ident,
        _source_region: &Option<crate::ir_inner::model::expr::GeneratorRef>,
        _body: &[Node],
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        if let Some(base) = self_exclusive_region_key(generator.as_str()) {
            match self.self_comp_counts.raw_entry_mut().from_key(base) {
                RawEntryMut::Occupied(mut o) => *o.get_mut() += 1,
                RawEntryMut::Vacant(v) => {
                    v.insert(base.to_string(), 1);
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn visit_opaque_node(
        &mut self,
        _node: &Node,
        extension: &dyn crate::ir_inner::model::node::NodeExtension,
    ) -> ControlFlow<Self::Break> {
        let depth = self.current_depth();
        depth::check_limits(&mut self.limits, depth, &mut self.errors);
        node_rules::check_opaque_node_extension(extension, &mut self.errors);
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
#[path = "rule_pipeline_tests.rs"]
mod tests;
