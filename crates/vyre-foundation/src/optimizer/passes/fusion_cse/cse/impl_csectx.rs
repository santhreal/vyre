#![allow(clippy::expect_used)]
use super::expr_key::CseExprId;
use super::{expr_has_effect, CseCtx, ScopeFrame, ScopedBinding};
use crate::ir::{Expr, Ident, Node};
use crate::optimizer::rewrite::{rewrite_binary, rewrite_fma, rewrite_select};
use std::borrow::Cow;

impl CseCtx {
    /// Reset all accumulated state while preserving allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.values.clear();
        self.undo_log.clear();
        self.scope_stack.clear();
        self.current_epoch = 0;
        self.arena.clear();
        self.deduplication.clear();
        self.subgroup_counter = 0;
        self.intern_calls
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn enter_scope(&mut self) {
        self.scope_stack.push(ScopeFrame {
            undo_len: self.undo_log.len(),
            epoch: self.current_epoch,
        });
    }

    #[inline]
    pub(crate) fn leave_scope(&mut self) {
        let Some(frame) = self.scope_stack.pop() else {
            return;
        };
        while self.undo_log.len() > frame.undo_len {
            let Some((key, old)) = self.undo_log.pop() else {
                break;
            };
            match old {
                Some(value) => {
                    self.values.insert(key, value);
                }
                None => {
                    self.values.remove(&key);
                }
            }
        }
        self.current_epoch = frame.epoch;
    }

    #[inline]
    pub(crate) fn clear_observed_state(&mut self) {
        if self.scope_stack.is_empty() {
            self.values.clear();
        } else {
            self.current_epoch = self.current_epoch.wrapping_add(1);
        }
    }

    #[inline]
    fn record_insert(&mut self, key: CseExprId, value: Ident) {
        let old = self.values.insert(
            key,
            ScopedBinding {
                name: value,
                epoch: self.current_epoch,
            },
        );
        if !self.scope_stack.is_empty() {
            self.undo_log.push((key, old));
        }
    }

    #[inline]
    fn visible_value(&self, key: CseExprId) -> Option<&Ident> {
        let value = self.values.get(&key)?;
        (value.epoch == self.current_epoch).then_some(&value.name)
    }

    #[inline]
    pub(crate) fn nodes_cow<'a>(&mut self, nodes: &'a [Node]) -> Cow<'a, [Node]> {
        for (idx, node) in nodes.iter().enumerate() {
            if let Cow::Owned(first) = self.node_cow(node) {
                let mut out = Vec::with_capacity(nodes.len());
                out.extend_from_slice(&nodes[..idx]);
                out.push(first);
                for rest in &nodes[idx + 1..] {
                    out.push(self.node_cow(rest).into_owned());
                }
                return Cow::Owned(out);
            }
        }
        Cow::Borrowed(nodes)
    }

    #[inline]
    fn exprs_cow<'a>(&mut self, exprs: &'a [Expr]) -> Cow<'a, [Expr]> {
        for (idx, expr) in exprs.iter().enumerate() {
            if let Cow::Owned(first) = self.expr(expr) {
                let mut out = Vec::with_capacity(exprs.len());
                out.extend_from_slice(&exprs[..idx]);
                out.push(first);
                for rest in &exprs[idx + 1..] {
                    out.push(self.expr(rest).into_owned());
                }
                return Cow::Owned(out);
            }
        }
        Cow::Borrowed(exprs)
    }

    #[inline]
    #[expect(
        clippy::too_many_lines,
        reason = "CSE node rewrite keeps visible-value state transitions adjacent to Node variant handling"
    )]
    pub(crate) fn node_cow<'a>(&mut self, node: &'a Node) -> Cow<'a, Node> {
        match node {
            Node::Let { name, value } => {
                let value_cow = self.expr(value);
                if expr_has_effect(value_cow.as_ref()) {
                    self.clear_observed_state();
                    return match value_cow {
                        Cow::Borrowed(_) => Cow::Borrowed(node),
                        Cow::Owned(v) => Cow::Owned(Node::Let {
                            name: name.clone(),
                            value: v,
                        }),
                    };
                }

                if matches!(
                    value_cow.as_ref(),
                    Expr::Var(_)
                        | Expr::LitU32(_)
                        | Expr::LitI32(_)
                        | Expr::LitF32(_)
                        | Expr::LitBool(_)
                ) {
                    return match value_cow {
                        Cow::Borrowed(_) => Cow::Borrowed(node),
                        Cow::Owned(v) => Cow::Owned(Node::Let {
                            name: name.clone(),
                            value: v,
                        }),
                    };
                }
                let key = self.intern_expr(value_cow.as_ref());
                if let Some(existing) = self.visible_value(key) {
                    Cow::Owned(Node::Let {
                        name: name.clone(),
                        value: Expr::Var(existing.clone()),
                    })
                } else {
                    self.record_insert(key, name.clone());
                    match value_cow {
                        Cow::Borrowed(_) => Cow::Borrowed(node),
                        Cow::Owned(v) => Cow::Owned(Node::Let {
                            name: name.clone(),
                            value: v,
                        }),
                    }
                }
            }
            Node::Assign { name, value } => {
                let value_cow = self.expr(value);
                self.clear_observed_state();
                match value_cow {
                    Cow::Borrowed(_) => Cow::Borrowed(node),
                    Cow::Owned(v) => Cow::Owned(Node::Assign {
                        name: name.clone(),
                        value: v,
                    }),
                }
            }
            Node::Store {
                buffer,
                index,
                value,
            } => {
                let index_cow = self.expr(index);
                let value_cow = self.expr(value);
                self.clear_observed_state();
                match (index_cow, value_cow) {
                    (Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(node),
                    (idx, val) => Cow::Owned(Node::Store {
                        buffer: buffer.clone(),
                        index: idx.into_owned(),
                        value: val.into_owned(),
                    }),
                }
            }
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                let cond_cow = self.expr(cond);
                self.enter_scope();
                let then_cow = self.nodes_cow(then);
                self.leave_scope();
                self.enter_scope();
                let otherwise_cow = self.nodes_cow(otherwise);
                self.leave_scope();
                self.clear_observed_state();
                match (cond_cow, then_cow, otherwise_cow) {
                    (Cow::Borrowed(_), Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(node),
                    (c, t, o) => Cow::Owned(Node::If {
                        cond: c.into_owned(),
                        then: t.into_owned(),
                        otherwise: o.into_owned(),
                    }),
                }
            }
            Node::Loop {
                var,
                from,
                to,
                body,
            } => {
                let from_cow = self.expr(from);
                let to_cow = self.expr(to);
                let mut body_ctx = CseCtx::default();
                let body_cow = body_ctx.nodes_cow(body);
                self.clear_observed_state();
                match (from_cow, to_cow, body_cow) {
                    (Cow::Borrowed(_), Cow::Borrowed(_), Cow::Borrowed(_)) => Cow::Borrowed(node),
                    (f, t, b) => Cow::Owned(Node::Loop {
                        var: var.clone(),
                        from: f.into_owned(),
                        to: t.into_owned(),
                        body: b.into_owned(),
                    }),
                }
            }
            Node::Return
            | Node::TileMatmul { .. }
            | Node::TileReduce { .. }
            | Node::TileDecl { .. } => Cow::Borrowed(node),
            Node::Block(nodes) => {
                self.enter_scope();
                let nodes_cow = self.nodes_cow(nodes);
                self.leave_scope();
                self.clear_observed_state();
                match nodes_cow {
                    Cow::Borrowed(_) => Cow::Borrowed(node),
                    Cow::Owned(n) => Cow::Owned(Node::Block(n)),
                }
            }
            Node::Barrier { .. }
            | Node::LogicalBarrier { .. }
            | Node::IndirectDispatch { .. }
            | Node::AsyncLoad { .. }
            | Node::AsyncStore { .. }
            | Node::AsyncWait { .. }
            | Node::Trap { .. }
            | Node::Resume { .. }
            | Node::AllReduce { .. }
            | Node::AllGather { .. }
            | Node::ReduceScatter { .. }
            | Node::Broadcast { .. }
            | Node::Opaque(_) => {
                self.clear_observed_state();
                Cow::Borrowed(node)
            }
            Node::Region {
                generator,
                source_region,
                body,
            } => {
                self.enter_scope();
                let body_cow = self.nodes_cow(body);
                self.leave_scope();
                self.clear_observed_state();
                match body_cow {
                    Cow::Borrowed(_) => Cow::Borrowed(node),
                    Cow::Owned(nodes) => Cow::Owned(Node::Region {
                        generator: generator.clone(),
                        source_region: source_region.clone(),
                        body: std::sync::Arc::new(nodes),
                    }),
                }
            }
            Node::TileLoad {
                tile,
                tile_type,
                buffer,
                origin,
                layout,
            } => {
                self.clear_observed_state();
                match self.exprs_cow(origin) {
                    Cow::Borrowed(_) => Cow::Borrowed(node),
                    Cow::Owned(origin) => Cow::Owned(Node::TileLoad {
                        tile: tile.clone(),
                        tile_type: tile_type.clone(),
                        buffer: buffer.clone(),
                        origin,
                        layout: layout.clone(),
                    }),
                }
            }
            Node::TileStore {
                buffer,
                origin,
                tile,
            } => {
                self.clear_observed_state();
                match self.exprs_cow(origin) {
                    Cow::Borrowed(_) => Cow::Borrowed(node),
                    Cow::Owned(origin) => Cow::Owned(Node::TileStore {
                        buffer: buffer.clone(),
                        origin,
                        tile: tile.clone(),
                    }),
                }
            }
            Node::TileElementwise { out, inputs, body } => {
                self.enter_scope();
                let body_cow = self.nodes_cow(body);
                self.leave_scope();
                match body_cow {
                    Cow::Borrowed(_) => Cow::Borrowed(node),
                    Cow::Owned(body) => Cow::Owned(Node::TileElementwise {
                        out: out.clone(),
                        inputs: inputs.clone(),
                        body,
                    }),
                }
            }
        }
    }

    #[inline]
    pub(crate) fn expr<'a>(&mut self, expr: &'a Expr) -> Cow<'a, Expr> {
        let rewritten = match expr {
            Expr::Load { buffer, index } => match self.expr(index) {
                Cow::Borrowed(_) => Cow::Borrowed(expr),
                Cow::Owned(index) => Cow::Owned(Expr::Load {
                    buffer: buffer.clone(),
                    index: Box::new(index),
                }),
            },
            Expr::BinOp { op, left, right } => {
                rewrite_binary(expr, *op, self.expr(left), self.expr(right))
            }
            Expr::UnOp { op, operand } => match self.expr(operand) {
                Cow::Borrowed(_) => Cow::Borrowed(expr),
                Cow::Owned(operand) => Cow::Owned(Expr::UnOp {
                    op: op.clone(),
                    operand: Box::new(operand),
                }),
            },
            Expr::Fma { a, b, c } => rewrite_fma(expr, self.expr(a), self.expr(b), self.expr(c)),
            Expr::Call { op_id, args } => match rewrite_args(self, args) {
                Cow::Borrowed(_) => Cow::Borrowed(expr),
                Cow::Owned(args) => Cow::Owned(Expr::Call {
                    op_id: op_id.clone(),
                    args,
                }),
            },
            Expr::Select {
                cond,
                true_val,
                false_val,
            } => rewrite_select(
                expr,
                self.expr(cond),
                self.expr(true_val),
                self.expr(false_val),
            ),
            Expr::Cast { target, value } => match self.expr(value) {
                Cow::Borrowed(_) => Cow::Borrowed(expr),
                Cow::Owned(value) => Cow::Owned(Expr::Cast {
                    target: target.clone(),
                    value: Box::new(value),
                }),
            },
            Expr::Atomic {
                op,
                buffer,
                index,
                expected,
                value,
                ordering,
            } => {
                let index = self.expr(index).into_owned();
                let expected = expected
                    .as_deref()
                    .map(|expr| Box::new(self.expr(expr).into_owned()));
                let value = self.expr(value).into_owned();
                self.clear_observed_state();
                Cow::Owned(Expr::Atomic {
                    op: *op,
                    buffer: buffer.clone(),
                    index: Box::new(index),
                    expected,
                    value: Box::new(value),
                    ordering: *ordering,
                })
            }
            _ => Cow::Borrowed(expr),
        };

        if matches!(rewritten.as_ref(), Expr::Var(_)) || expr_has_effect(rewritten.as_ref()) {
            return rewritten;
        }

        // Soundness fix (S19): the previous pointer cache mapped
        // `*const Expr → CseExprId`, claiming the IR is immutable during
        // a single CSE pass. That isn't safe  -  Box<Expr> sub-trees
        // freed by `Cow::Owned` rewrites can be reallocated at the
        // same address by later sub-trees, so a stale CseExprId from a
        // prior expression flows through `values.get(stale_id)` and
        // CSE merges semantically distinct expressions (caught by
        // `full_optimize_is_idempotent_on_canonical_wire` regression
        //  -  `BitAnd(46, Mul(BitAnd(888, X), 0))` collapsed into the
        // unrelated `BitOr(invocation, 1)`). Drop the cache; the
        // underlying `deduplication` map already gives O(1) intern
        // dedup by structural key.
        let key = self.intern_expr(rewritten.as_ref());
        // Never replace a literal with a variable reference  -  literals are
        // already minimal, and the variable may be mutated later (e.g. loop
        // counters or state accumulators). Substituting `0u` with `var state`
        // when `state` was initially `0u` is unsound if `state` is reassigned.
        if matches!(
            rewritten.as_ref(),
            Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitBool(_)
        ) {
            return rewritten;
        }
        self.visible_value(key).map_or(rewritten, |existing| {
            Cow::Owned(Expr::var(existing.clone()))
        })
    }
}

#[inline]
fn rewrite_args<'a>(ctx: &mut CseCtx, args: &'a [Expr]) -> Cow<'a, [Expr]> {
    let mut rewritten: Option<Vec<Expr>> = None;
    for (index, arg) in args.iter().enumerate() {
        match ctx.expr(arg) {
            Cow::Borrowed(_) if rewritten.is_none() => {}
            Cow::Borrowed(borrowed) => {
                if let Some(out) = rewritten.as_mut() {
                    out.push(Expr::clone(borrowed));
                }
            }
            Cow::Owned(owned) => {
                let out = rewritten.get_or_insert_with(|| {
                    let mut out = Vec::with_capacity(args.len());
                    out.extend_from_slice(&args[..index]);
                    out
                });
                out.push(owned);
            }
        }
    }
    rewritten.map_or(Cow::Borrowed(args), Cow::Owned)
}
