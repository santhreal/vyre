//! The recursive node walk the single-pass `PreorderValidator` replaced, kept
//! as the second arm of the differential property test in
//! `rule_pipeline_tests`.
//!
//! The property compares two traversals, so this walk descends independently:
//! its own recursion, its own scope frames, its own divergence propagation.
//! What it does NOT restate is the rules. Every rule it applies at a node comes
//! from `super::node_rules`, and the program-header rules come from
//! `rule_pipeline::validate_program_level`, so a corrected diagnostic cannot
//! fail the property for a reason that has nothing to do with the walk.
//!
//! Test-only: `validate` and every production caller go through
//! `rule_pipeline`.

use rustc_hash::FxHashSet;

use super::binding::check_sibling_duplicate;
use super::depth::{self, LimitState};
use super::expr_rules::validate_expr;
use super::node_rules::{self, insert_binding, restore_scope, BufferTable, Scope, ScopeLog};
use super::typecheck::{expr_type, ScopeTypes};
use super::uniformity::is_uniform;
use super::{barrier, shadowing, Binding, ValidationOptions, ValidationReport};
use crate::ir_inner::model::expr::Ident;
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::op_signature::DataType;

#[inline]
pub(crate) fn validate_nodes(
    nodes: &[Node],
    buffers: &BufferTable<'_>,
    scope: &mut Scope,
    divergent: bool,
    depth: usize,
    limits: &mut LimitState,
    options: ValidationOptions<'_>,
    report: &mut ValidationReport,
) {
    let mut region_bindings = FxHashSet::default();
    validate_nodes_inner(
        nodes,
        buffers,
        scope,
        divergent,
        depth,
        limits,
        options,
        report,
        &mut region_bindings,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn validate_nodes_inner(
    nodes: &[Node],
    buffers: &BufferTable<'_>,
    scope: &mut Scope,
    divergent: bool,
    depth: usize,
    limits: &mut LimitState,
    options: ValidationOptions<'_>,
    report: &mut ValidationReport,
    region_bindings: &mut FxHashSet<Ident>,
    mut scope_log: Option<&mut ScopeLog>,
) {
    for node in nodes {
        validate_node_inner(
            node,
            buffers,
            scope,
            divergent,
            depth,
            limits,
            options,
            report,
            region_bindings,
            scope_log.as_deref_mut(),
        );
    }

    node_rules::check_unreachable_after_return(nodes, &mut report.errors);
}

#[allow(clippy::too_many_arguments, clippy::unnested_or_patterns)]
fn validate_node_inner(
    node: &Node,
    buffers: &BufferTable<'_>,
    scope: &mut Scope,
    divergent: bool,
    depth: usize,
    limits: &mut LimitState,
    options: ValidationOptions<'_>,
    report: &mut ValidationReport,
    region_bindings: &mut FxHashSet<Ident>,
    scope_log: Option<&mut ScopeLog>,
) {
    depth::check_limits(limits, depth, &mut report.errors);

    match node {
        Node::Let { name, value } => {
            validate_expr(value, buffers, scope, options, report, 0);
            let duplicate_sibling = check_sibling_duplicate(
                name,
                region_bindings,
                options.allow_shadowing,
                &mut report.errors,
            );
            if !duplicate_sibling {
                shadowing::check_local(name, scope, options, &mut report.errors);
            }
            let ty_opt = expr_type(value, &mut ScopeTypes::new(buffers, scope));
            let ty = ty_opt.clone().unwrap_or(DataType::U32);
            let ty_known = ty_opt.is_some();
            let uniform = is_uniform(value, scope);
            insert_binding(
                scope,
                name.clone(),
                Binding {
                    ty,
                    ty_known,
                    mutable: true,
                    uniform,
                },
                scope_log,
            );
        }
        Node::Assign { name, value } => {
            node_rules::check_assign(name, value, buffers, scope, &mut report.errors);
            validate_expr(value, buffers, scope, options, report, 0);
            // Reassignment with a divergent rhs taints the binding's
            // uniformity for the remainder of its lifetime.
            let new_uniform = is_uniform(value, scope);
            if let Some(binding) = scope.get_mut(name.as_str()) {
                binding.uniform = binding.uniform && new_uniform;
            }
        }
        Node::Store {
            buffer,
            index,
            value,
        } => {
            node_rules::check_store(buffer, index, value, buffers, scope, &mut report.errors);
            validate_expr(index, buffers, scope, options, report, 0);
            validate_expr(value, buffers, scope, options, report, 0);
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            validate_expr(cond, buffers, scope, options, report, 0);
            node_rules::check_if_condition(cond, buffers, scope, &mut report.errors);
            // Branches stay non-divergent only when the parent scope is
            // already uniform AND the condition is uniform across the
            // workgroup. A non-uniform cond splits invocations across
            // the two branches; a divergent parent already failed the
            // uniformity precondition so we conservatively propagate.
            let branch_divergent = divergent || !is_uniform(cond, scope);
            validate_scoped_nested_nodes(
                then,
                buffers,
                scope,
                branch_divergent,
                depth,
                limits,
                options,
                report,
                |_, _| {},
            );
            validate_scoped_nested_nodes(
                otherwise,
                buffers,
                scope,
                branch_divergent,
                depth,
                limits,
                options,
                report,
                |_, _| {},
            );
        }
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            validate_expr(from, buffers, scope, options, report, 0);
            validate_expr(to, buffers, scope, options, report, 0);
            node_rules::check_loop_bounds(from, to, buffers, scope, &mut report.errors);
            shadowing::check_local(var, scope, options, &mut report.errors);
            // The loop body is divergent only when its parent already is
            // OR when either bound varies across the workgroup. Uniform
            // bounds keep every invocation in lockstep  -  same iteration
            // count, same loop-var value at each step  -  so a barrier
            // inside is reached by every lane simultaneously.
            let bounds_uniform = is_uniform(from, scope) && is_uniform(to, scope);
            let body_divergent = divergent || !bounds_uniform;
            // The loop counter inherits the bounds' uniformity; in a
            // uniform-bound loop every lane sees the same counter value
            // at the same source position.
            let var_uniform = bounds_uniform && !divergent;
            let mut back_edge_scope = scope.clone();
            back_edge_scope.insert(var.clone(), node_rules::loop_var_binding(var_uniform));
            barrier::check_loop_back_edge(body, &back_edge_scope, &mut report.errors);
            validate_scoped_nested_nodes(
                body,
                buffers,
                scope,
                body_divergent,
                depth,
                limits,
                options,
                report,
                |scope, scope_log| {
                    insert_binding(
                        scope,
                        var.clone(),
                        node_rules::loop_var_binding(var_uniform),
                        Some(scope_log),
                    );
                },
            );
        }
        Node::Return => {}
        Node::Block(nodes) => {
            validate_scoped_nested_nodes(
                nodes,
                buffers,
                scope,
                divergent,
                depth,
                limits,
                options,
                report,
                |_, _| {},
            );
        }
        Node::Barrier { ordering } => {
            barrier::check_barrier(divergent, *ordering, &mut report.errors);
        }
        Node::IndirectDispatch {
            count_buffer,
            count_offset,
        } => {
            node_rules::check_indirect_dispatch(
                count_buffer,
                *count_offset,
                buffers,
                &mut report.errors,
            );
        }
        Node::AsyncLoad {
            destination,
            offset,
            size,
            tag,
            ..
        }
        | Node::AsyncStore {
            destination,
            offset,
            size,
            tag,
            ..
        } => {
            node_rules::check_async_transfer(destination, tag, buffers, &mut report.errors);
            validate_expr(offset, buffers, scope, options, report, 0);
            validate_expr(size, buffers, scope, options, report, 0);
        }
        Node::AsyncWait { tag } => {
            node_rules::check_async_tag(tag, &mut report.errors);
        }
        Node::Trap { address, .. } => {
            validate_expr(address, buffers, scope, options, report, 0);
        }
        Node::Resume { .. } => {}
        Node::AllReduce { .. }
        | Node::Broadcast { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. } => {
            node_rules::check_collective(node, options, buffers, &mut report.errors);
        }
        Node::Region { body, .. } => {
            // A region is a grouping marker for traces and op-id breadcrumbs,
            // but it scopes its body exactly like a `Block`: the pass that
            // flattens a small region into its parent sequence re-wraps it in a
            // `Node::Block` when a name would collide, which is sound only while
            // a region body's bindings die at the region boundary.
            //
            // This walk passed no scope log at all, so a region-body `let` was
            // never recorded and never undone: it leaked out of the region, out
            // of the `If` branch holding the region, and stayed live for the rest
            // of the program.
            validate_scoped_nested_nodes(
                body,
                buffers,
                scope,
                divergent,
                depth,
                limits,
                options,
                report,
                |_, _| {},
            );
        }
        Node::Opaque(extension) => {
            node_rules::check_opaque_node_extension(extension.as_ref(), &mut report.errors);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_scoped_nested_nodes(
    nodes: &[Node],
    buffers: &BufferTable<'_>,
    scope: &mut Scope,
    divergent: bool,
    depth: usize,
    limits: &mut LimitState,
    options: ValidationOptions<'_>,
    report: &mut ValidationReport,
    configure_scope: impl FnOnce(&mut Scope, &mut ScopeLog),
) {
    let mut scope_log = Vec::new();
    let mut region_bindings = FxHashSet::default();
    configure_scope(scope, &mut scope_log);
    validate_nodes_inner(
        nodes,
        buffers,
        scope,
        divergent,
        depth.saturating_add(1),
        limits,
        options,
        report,
        &mut region_bindings,
        Some(&mut scope_log),
    );
    restore_scope(scope, scope_log);
}
