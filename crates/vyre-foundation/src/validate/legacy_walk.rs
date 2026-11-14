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

struct WalkCtx<'a, 'b> {
    buffers: &'a BufferTable<'b>,
    scope: &'a mut Scope,
    limits: &'a mut LimitState,
    options: ValidationOptions<'b>,
    report: &'a mut ValidationReport,
}

#[inline]
pub(crate) fn validate_nodes<'b>(
    nodes: &[Node],
    buffers: &BufferTable<'b>,
    scope: &mut Scope,
    divergent: bool,
    depth: usize,
    limits: &mut LimitState,
    options: ValidationOptions<'b>,
    report: &mut ValidationReport,
) {
    let mut region_bindings = FxHashSet::default();
    let mut ctx = WalkCtx {
        buffers,
        scope,
        limits,
        options,
        report,
    };
    ctx.validate_nodes_inner(nodes, divergent, depth, &mut region_bindings, None);
}

impl WalkCtx<'_, '_> {
    fn validate_expr(&mut self, expr: &crate::ir_inner::model::expr::Expr) {
        validate_expr(expr, self.buffers, self.scope, self.options, self.report, 0);
    }

    fn validate_nodes_inner(
        &mut self,
        nodes: &[Node],
        divergent: bool,
        depth: usize,
        region_bindings: &mut FxHashSet<Ident>,
        mut scope_log: Option<&mut ScopeLog>,
    ) {
        for node in nodes {
            self.validate_node_inner(
                node,
                divergent,
                depth,
                region_bindings,
                scope_log.as_deref_mut(),
            );
        }
        node_rules::check_unreachable_after_return(nodes, &mut self.report.errors);
    }

    fn validate_scoped(
        &mut self,
        nodes: &[Node],
        divergent: bool,
        depth: usize,
        configure_scope: impl FnOnce(&mut Scope, &mut ScopeLog),
    ) {
        let mut scope_log = Vec::new();
        let mut region_bindings = FxHashSet::default();
        configure_scope(self.scope, &mut scope_log);
        self.validate_nodes_inner(
            nodes,
            divergent,
            depth.saturating_add(1),
            &mut region_bindings,
            Some(&mut scope_log),
        );
        restore_scope(self.scope, scope_log);
    }

    #[allow(clippy::unnested_or_patterns)]
    fn validate_node_inner(
        &mut self,
        node: &Node,
        divergent: bool,
        depth: usize,
        region_bindings: &mut FxHashSet<Ident>,
        scope_log: Option<&mut ScopeLog>,
    ) {
        depth::check_limits(self.limits, depth, &mut self.report.errors);

        match node {
            Node::Let { name, value } => {
                self.validate_expr(value);
                let duplicate_sibling =
                    check_sibling_duplicate(name, region_bindings, false, &mut self.report.errors);
                if !duplicate_sibling {
                    shadowing::check_local(name, self.scope, self.options, &mut self.report.errors);
                }
                let ty_opt = expr_type(value, &mut ScopeTypes::new(self.buffers, self.scope));
                let ty_known = ty_opt.is_some();
                let ty = ty_opt.unwrap_or(DataType::U32);
                let uniform = is_uniform(value, self.scope);
                insert_binding(
                    self.scope,
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
                node_rules::check_assign(
                    name,
                    value,
                    self.buffers,
                    self.scope,
                    &mut self.report.errors,
                );
                self.validate_expr(value);
                let new_uniform = is_uniform(value, self.scope);
                if let Some(binding) = self.scope.get_mut(name.as_str()) {
                    binding.uniform = binding.uniform && new_uniform;
                }
            }
            Node::Store {
                buffer,
                index,
                value,
            } => {
                node_rules::check_store(
                    buffer,
                    index,
                    value,
                    self.buffers,
                    self.scope,
                    &mut self.report.errors,
                );
                self.validate_expr(index);
                self.validate_expr(value);
            }
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                self.validate_expr(cond);
                node_rules::check_if_condition(
                    cond,
                    self.buffers,
                    self.scope,
                    &mut self.report.errors,
                );
                let branch_divergent = divergent || !is_uniform(cond, self.scope);
                self.validate_scoped(then, branch_divergent, depth, |_, _| {});
                self.validate_scoped(otherwise, branch_divergent, depth, |_, _| {});
            }
            Node::Loop {
                var,
                from,
                to,
                body,
            } => {
                self.validate_expr(from);
                self.validate_expr(to);
                node_rules::check_loop_bounds(
                    from,
                    to,
                    self.buffers,
                    self.scope,
                    &mut self.report.errors,
                );
                shadowing::check_local(var, self.scope, self.options, &mut self.report.errors);
                let bounds_uniform = is_uniform(from, self.scope) && is_uniform(to, self.scope);
                let body_divergent = divergent || !bounds_uniform;
                let var_uniform = bounds_uniform && !divergent;
                let mut back_edge_scope = self.scope.clone();
                back_edge_scope.insert(var.clone(), node_rules::loop_var_binding(var_uniform));
                barrier::check_loop_back_edge(body, &back_edge_scope, &mut self.report.errors);
                self.validate_scoped(body, body_divergent, depth, |scope, scope_log| {
                    insert_binding(
                        scope,
                        var.clone(),
                        node_rules::loop_var_binding(var_uniform),
                        Some(scope_log),
                    );
                });
            }
            Node::Return | Node::Resume { .. } | Node::TileReduce { .. } => {}
            Node::Block(nodes) => {
                self.validate_scoped(nodes, divergent, depth, |_, _| {});
            }
            Node::Barrier { ordering } | Node::LogicalBarrier { ordering } => {
                barrier::check_barrier(divergent, *ordering, &mut self.report.errors);
            }
            Node::IndirectDispatch {
                count_buffer,
                count_offset,
            } => {
                node_rules::check_indirect_dispatch(
                    count_buffer,
                    *count_offset,
                    self.buffers,
                    &mut self.report.errors,
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
                node_rules::check_async_transfer(
                    destination,
                    offset,
                    size,
                    tag,
                    self.buffers,
                    self.scope,
                    &mut self.report.errors,
                );
                self.validate_expr(offset);
                self.validate_expr(size);
            }
            Node::AsyncWait { tag } => {
                node_rules::check_async_tag(tag, &mut self.report.errors);
            }
            Node::Trap { address, .. } => {
                self.validate_expr(address);
            }
            Node::AllReduce { .. }
            | Node::Broadcast { .. }
            | Node::AllGather { .. }
            | Node::ReduceScatter { .. } => {
                node_rules::check_collective(
                    node,
                    self.options,
                    self.buffers,
                    &mut self.report.errors,
                );
            }
            Node::Region { body, .. } => {
                self.validate_scoped(body, divergent, depth, |_, _| {});
            }
            Node::TileLoad {
                tile,
                tile_type,
                buffer,
                origin,
                ..
            } => {
                for expr in origin {
                    self.validate_expr(expr);
                }
                node_rules::check_tile_load(
                    tile,
                    tile_type,
                    buffer,
                    origin,
                    self.buffers,
                    self.options,
                    &mut self.report.errors,
                );
            }
            Node::TileStore {
                buffer,
                origin,
                tile,
            } => {
                for expr in origin {
                    self.validate_expr(expr);
                }
                node_rules::check_tile_store(
                    buffer,
                    origin,
                    tile,
                    self.buffers,
                    &mut self.report.errors,
                );
            }
            Node::TileMatmul { acc, a, b } => {
                node_rules::check_tile_matmul(acc, a, b, self.options, &mut self.report.errors);
            }
            Node::TileElementwise { inputs, body, .. } => {
                self.validate_scoped(body, divergent, depth, |nested_scope, scope_log| {
                    for input in inputs {
                        insert_binding(
                            nested_scope,
                            input.clone(),
                            Binding {
                                ty: DataType::F32,
                                ty_known: true,
                                mutable: false,
                                uniform: false,
                            },
                            Some(scope_log),
                        );
                    }
                });
            }
            Node::TileDecl { name, tile } => {
                node_rules::check_tile_residency(name, tile, self.options, &mut self.report.errors);
            }
            Node::Opaque(extension) => {
                node_rules::check_opaque_node_extension(
                    extension.as_ref(),
                    &mut self.report.errors,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_inner::model::expr::Expr;
    use rustc_hash::FxHashMap;

    fn run_allow_shadowing(nodes: &[Node]) -> ValidationReport {
        let buffers: BufferTable<'_> = FxHashMap::default();
        let mut scope: Scope = FxHashMap::default();
        let mut limits = LimitState::default();
        let mut report = ValidationReport::default();
        let options = ValidationOptions::default().with_shadowing(true);
        validate_nodes(
            nodes,
            &buffers,
            &mut scope,
            false,
            0,
            &mut limits,
            options,
            &mut report,
        );
        report
    }

    fn let_u32(name: &str, v: u32) -> Node {
        Node::Let {
            name: name.into(),
            value: Expr::u32(v),
        }
    }

    fn count_v032(report: &ValidationReport) -> usize {
        report
            .errors
            .iter()
            .filter(|e| e.code().as_str() == "V032")
            .count()
    }

    #[test]
    fn same_scope_duplicate_lets_rejected_even_with_allow_shadowing() {
        let report = run_allow_shadowing(&[let_u32("x", 1), let_u32("x", 2)]);
        assert!(
            count_v032(&report) >= 1,
            "same-scope duplicate let must be rejected with V032 even when allow_shadowing is true: {:?}",
            report.errors
        );
    }

    #[test]
    fn nested_scope_shadowing_accepted_with_allow_shadowing() {
        let report = run_allow_shadowing(&[
            let_u32("x", 1),
            Node::If {
                cond: Expr::bool(true),
                then: vec![let_u32("x", 2)],
                otherwise: Vec::new(),
            },
        ]);
        assert!(
            report.errors.is_empty(),
            "nested-scope shadowing must be accepted when allow_shadowing is true: {:?}",
            report.errors
        );
    }

    #[test]
    fn repeated_sibling_lets_rejected_with_multiple_v032_under_allow_shadowing() {
        let report = run_allow_shadowing(&[let_u32("x", 1), let_u32("x", 2), let_u32("x", 3)]);
        assert_eq!(
            count_v032(&report),
            2,
            "three sibling let bindings in the same region must emit two V032 errors even when allow_shadowing is true: {:?}",
            report.errors
        );
    }

    #[test]
    fn nested_region_repeated_sibling_lets_rejected_under_allow_shadowing() {
        let report = run_allow_shadowing(&[
            Node::If {
                cond: Expr::bool(true),
                then: vec![let_u32("y", 10), let_u32("y", 20)],
                otherwise: vec![Node::Block(vec![let_u32("z", 30), let_u32("z", 40)])],
            },
            Node::Loop {
                var: "i".into(),
                from: Expr::u32(0),
                to: Expr::u32(2),
                body: vec![let_u32("w", 50), let_u32("w", 60)],
            },
        ]);
        assert_eq!(
            count_v032(&report),
            3,
            "each nested region with duplicate sibling lets must emit V032 even when allow_shadowing is true: {:?}",
            report.errors
        );
    }
}
