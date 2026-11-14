use super::UseFacts;
use crate::ir::{Expr, Ident, Node};
use crate::optimizer::rewrite::push_expr_children;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::sync::Arc;

#[derive(Default)]
struct UseFactBuilder {
    var_counts: FxHashMap<Ident, usize>,
    buffer_reads: FxHashMap<Ident, usize>,
    buffer_writes: FxHashMap<Ident, usize>,
    buffer_index_axes: FxHashMap<Ident, [usize; 3]>,
    var_buffer_deps: FxHashMap<Ident, FxHashSet<Ident>>,
    buffer_write_deps: FxHashMap<Ident, FxHashSet<Ident>>,
    indirect_dispatch_buffers: FxHashSet<Ident>,
    scratch_deps: FxHashSet<Ident>,
    has_opaque: bool,
}

impl UseFactBuilder {
    fn finish(self) -> UseFacts {
        UseFacts {
            var_counts: Arc::new(self.var_counts),
            buffer_reads: self.buffer_reads,
            buffer_writes: self.buffer_writes,
            buffer_index_axes: self.buffer_index_axes,
            var_buffer_deps: self.var_buffer_deps,
            buffer_write_deps: self.buffer_write_deps,
            indirect_dispatch_buffers: self.indirect_dispatch_buffers,
            has_opaque: self.has_opaque,
        }
    }

    #[inline]
    fn take_scratch(&mut self) -> FxHashSet<Ident> {
        let mut deps = std::mem::take(&mut self.scratch_deps);
        deps.clear();
        deps
    }

    #[inline]
    fn restore_scratch(&mut self, mut deps: FxHashSet<Ident>) {
        deps.clear();
        if deps.capacity() > self.scratch_deps.capacity() {
            self.scratch_deps = deps;
        }
    }
}

#[inline]
fn bump_count(map: &mut FxHashMap<Ident, usize>, key: &Ident) {
    if let Some(slot) = map.get_mut(key) {
        *slot += 1;
    } else {
        map.insert(key.clone(), 1);
    }
}

#[inline]
fn bump_axis(map: &mut FxHashMap<Ident, [usize; 3]>, key: &Ident, axis: u8) {
    let idx = usize::from(axis).min(2);
    if let Some(axes) = map.get_mut(key) {
        axes[idx] += 1;
    } else {
        let mut axes = [0; 3];
        axes[idx] = 1;
        map.insert(key.clone(), axes);
    }
}

#[inline]
fn insert_ident(set: &mut FxHashSet<Ident>, key: &Ident) {
    if !set.contains(key) {
        set.insert(key.clone());
    }
}

#[inline]
fn extend_idents<'a>(dst: &mut FxHashSet<Ident>, src: impl IntoIterator<Item = &'a Ident>) {
    for item in src {
        insert_ident(dst, item);
    }
}

fn merge_into_map_set(
    map: &mut FxHashMap<Ident, FxHashSet<Ident>>,
    key: &Ident,
    deps: &mut FxHashSet<Ident>,
) {
    if deps.is_empty() {
        return;
    }
    if let Some(existing) = map.get_mut(key) {
        existing.extend(deps.drain());
    } else {
        let mut fresh = FxHashSet::default();
        fresh.reserve(deps.len());
        fresh.extend(deps.drain());
        map.insert(key.clone(), fresh);
    }
}

pub(super) fn derive_use_facts(program: &crate::ir::Program) -> UseFacts {
    let mut facts = UseFactBuilder::default();
    derive_nodes_uses(program.entry(), &mut facts, &FxHashSet::default());
    facts.finish()
}

fn derive_nodes_uses(nodes: &[Node], facts: &mut UseFactBuilder, control_deps: &FxHashSet<Ident>) {
    for node in nodes {
        match node {
            Node::Let { name, value } | Node::Assign { name, value } => {
                let mut deps = facts.take_scratch();
                record_expr_uses_and_buffer_deps_into(&mut deps, value, facts);
                extend_idents(&mut deps, control_deps);
                if deps.is_empty() {
                    if !facts.var_buffer_deps.contains_key(name) {
                        facts
                            .var_buffer_deps
                            .insert(name.clone(), FxHashSet::default());
                    }
                } else {
                    merge_into_map_set(&mut facts.var_buffer_deps, name, &mut deps);
                }
                facts.restore_scratch(deps);
            }
            Node::Store {
                buffer,
                index,
                value,
            } => {
                bump_count(&mut facts.buffer_writes, buffer);
                let mut deps = facts.take_scratch();
                record_expr_uses_and_buffer_deps_into(&mut deps, index, facts);
                count_index_axes(index, buffer, facts);
                record_expr_uses_and_buffer_deps_into(&mut deps, value, facts);
                extend_idents(&mut deps, control_deps);
                merge_into_map_set(&mut facts.buffer_write_deps, buffer, &mut deps);
                facts.restore_scratch(deps);
            }
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                let mut cond_deps = facts.take_scratch();
                record_expr_uses_and_buffer_deps_into(&mut cond_deps, cond, facts);
                if cond_deps.is_empty() {
                    facts.restore_scratch(cond_deps);
                    derive_nodes_uses(then, facts, control_deps);
                    derive_nodes_uses(otherwise, facts, control_deps);
                } else {
                    let branch_control = union_deps(control_deps, &cond_deps);
                    facts.restore_scratch(cond_deps);
                    derive_nodes_uses(then, facts, &branch_control);
                    derive_nodes_uses(otherwise, facts, &branch_control);
                }
            }
            Node::Loop { from, to, body, .. } => {
                let mut loop_deps = facts.take_scratch();
                record_expr_uses_and_buffer_deps_into(&mut loop_deps, from, facts);
                record_expr_uses_and_buffer_deps_into(&mut loop_deps, to, facts);
                if loop_deps.is_empty() {
                    facts.restore_scratch(loop_deps);
                    derive_nodes_uses(body, facts, control_deps);
                } else {
                    let loop_control = union_deps(control_deps, &loop_deps);
                    facts.restore_scratch(loop_deps);
                    derive_nodes_uses(body, facts, &loop_control);
                }
            }
            Node::Block(nodes) => {
                derive_nodes_uses(nodes, facts, control_deps);
            }
            Node::Region { body: nodes, .. } => {
                derive_nodes_uses(nodes, facts, control_deps);
            }
            Node::AsyncLoad {
                source,
                destination,
                offset,
                size,
                ..
            }
            | Node::AsyncStore {
                source,
                destination,
                offset,
                size,
                ..
            } => {
                bump_count(&mut facts.buffer_reads, source);
                bump_count(&mut facts.buffer_writes, destination);
                let mut deps = facts.take_scratch();
                record_expr_uses_and_buffer_deps_into(&mut deps, offset, facts);
                record_expr_uses_and_buffer_deps_into(&mut deps, size, facts);
                extend_idents(&mut deps, control_deps);
                insert_ident(&mut deps, source);
                merge_into_map_set(&mut facts.buffer_write_deps, destination, &mut deps);
                facts.restore_scratch(deps);
            }
            Node::Trap { address, .. } => {
                let mut deps = facts.take_scratch();
                record_expr_uses_and_buffer_deps_into(&mut deps, address, facts);
                facts.restore_scratch(deps);
            }
            Node::IndirectDispatch { count_buffer, .. } => {
                insert_ident(&mut facts.indirect_dispatch_buffers, count_buffer);
                bump_count(&mut facts.buffer_reads, count_buffer);
            }
            Node::AllReduce { buffer, .. } | Node::Broadcast { buffer, .. } => {
                bump_count(&mut facts.buffer_reads, buffer);
                bump_count(&mut facts.buffer_writes, buffer);
                let mut deps = facts.take_scratch();
                extend_idents(&mut deps, control_deps);
                insert_ident(&mut deps, buffer);
                merge_into_map_set(&mut facts.buffer_write_deps, buffer, &mut deps);
                facts.restore_scratch(deps);
            }
            Node::AllGather { input, output, .. } | Node::ReduceScatter { input, output, .. } => {
                bump_count(&mut facts.buffer_reads, input);
                bump_count(&mut facts.buffer_writes, output);
                let mut deps = facts.take_scratch();
                extend_idents(&mut deps, control_deps);
                insert_ident(&mut deps, input);
                merge_into_map_set(&mut facts.buffer_write_deps, output, &mut deps);
                facts.restore_scratch(deps);
            }
            Node::Opaque(_) => {
                facts.has_opaque = true;
            }
            Node::TileLoad {
                tile,
                buffer,
                origin,
                ..
            } => {
                bump_count(&mut facts.buffer_reads, buffer);
                let mut deps = facts.take_scratch();
                insert_ident(&mut deps, buffer);
                for off in origin {
                    record_expr_uses_and_buffer_deps_into(&mut deps, off, facts);
                }
                extend_idents(&mut deps, control_deps);
                merge_into_map_set(&mut facts.var_buffer_deps, tile, &mut deps);
                facts.restore_scratch(deps);
            }
            Node::TileStore {
                buffer,
                origin,
                tile,
            } => {
                bump_count(&mut facts.buffer_writes, buffer);
                bump_count(&mut facts.var_counts, tile);
                let mut deps = facts.take_scratch();
                insert_ident(&mut deps, tile);
                for off in origin {
                    record_expr_uses_and_buffer_deps_into(&mut deps, off, facts);
                }
                extend_idents(&mut deps, control_deps);
                if let Some(tile_deps) = facts.var_buffer_deps.get(tile) {
                    extend_idents(&mut deps, tile_deps);
                }
                merge_into_map_set(&mut facts.buffer_write_deps, buffer, &mut deps);
                facts.restore_scratch(deps);
            }
            Node::TileElementwise { out, inputs, body } => {
                for inp in inputs {
                    bump_count(&mut facts.var_counts, inp);
                }
                derive_nodes_uses(body, facts, control_deps);
                let mut deps = facts.take_scratch();
                for inp in inputs {
                    insert_ident(&mut deps, inp);
                    if let Some(inp_deps) = facts.var_buffer_deps.get(inp) {
                        extend_idents(&mut deps, inp_deps);
                    }
                }
                extend_idents(&mut deps, control_deps);
                if deps.is_empty() {
                    if !facts.var_buffer_deps.contains_key(out) {
                        facts
                            .var_buffer_deps
                            .insert(out.clone(), FxHashSet::default());
                    }
                } else {
                    merge_into_map_set(&mut facts.var_buffer_deps, out, &mut deps);
                }
                facts.restore_scratch(deps);
            }
            Node::TileMatmul { acc, a, b } => {
                bump_count(&mut facts.var_counts, acc);
                bump_count(&mut facts.var_counts, a);
                bump_count(&mut facts.var_counts, b);
                let mut deps = facts.take_scratch();
                insert_ident(&mut deps, acc);
                insert_ident(&mut deps, a);
                insert_ident(&mut deps, b);
                extend_idents(&mut deps, control_deps);
                merge_into_map_set(&mut facts.var_buffer_deps, acc, &mut deps);
                facts.restore_scratch(deps);
            }
            Node::TileReduce { out, tile, .. } => {
                bump_count(&mut facts.var_counts, tile);
                let mut deps = facts.take_scratch();
                insert_ident(&mut deps, tile);
                extend_idents(&mut deps, control_deps);
                merge_into_map_set(&mut facts.var_buffer_deps, out, &mut deps);
                facts.restore_scratch(deps);
            }
            Node::TileDecl { .. }
            | Node::Return
            | Node::Barrier { .. }
            | Node::LogicalBarrier { .. }
            | Node::AsyncWait { .. }
            | Node::Resume { .. } => {}
        }
    }
}

fn record_expr_uses_and_buffer_deps_into(
    deps: &mut FxHashSet<Ident>,
    expr: &Expr,
    facts: &mut UseFactBuilder,
) {
    let mut stack: SmallVec<[&Expr; 16]> = SmallVec::new();
    stack.push(expr);
    while let Some(expr) = stack.pop() {
        match expr {
            Expr::Var(name) => {
                bump_count(&mut facts.var_counts, name);
                if let Some(var_deps) = facts.var_buffer_deps.get(name) {
                    extend_idents(deps, var_deps);
                }
            }
            Expr::Load { buffer, index } => {
                bump_count(&mut facts.buffer_reads, buffer);
                count_index_axes(index, buffer, facts);
                insert_ident(deps, buffer);
            }
            Expr::BufLen { buffer } => {
                bump_count(&mut facts.buffer_reads, buffer);
                insert_ident(deps, buffer);
            }
            Expr::BufferRef { buffer } => {
                insert_ident(deps, buffer);
            }
            Expr::Atomic { buffer, index, .. } => {
                bump_count(&mut facts.buffer_reads, buffer);
                bump_count(&mut facts.buffer_writes, buffer);
                count_index_axes(index, buffer, facts);
                insert_ident(deps, buffer);
            }
            Expr::Opaque(_) => {
                facts.has_opaque = true;
            }
            _ => {}
        }
        push_expr_children(expr, &mut stack);
    }
}

fn union_deps(a: &FxHashSet<Ident>, b: &FxHashSet<Ident>) -> FxHashSet<Ident> {
    if a.is_empty() {
        return b.clone();
    }
    if b.is_empty() {
        return a.clone();
    }
    let mut out = FxHashSet::default();
    out.reserve(a.len().saturating_add(b.len()));
    out.extend(a.iter().cloned());
    out.extend(b.iter().cloned());
    out
}

fn count_index_axes(index: &Expr, buffer: &Ident, facts: &mut UseFactBuilder) {
    let mut stack: SmallVec<[&Expr; 16]> = SmallVec::new();
    stack.push(index);
    while let Some(expr) = stack.pop() {
        if let Expr::InvocationId { axis }
        | Expr::LocalId { axis }
        | Expr::LogicalIndex { axis }
        | Expr::LogicalWithinTileId { axis } = expr
        {
            bump_axis(&mut facts.buffer_index_axes, buffer, *axis);
        }
        crate::optimizer::rewrite::push_expr_children(expr, &mut stack);
    }
}
