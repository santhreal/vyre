//! The one structural `Node` rewrite.
//!
//! Four passes used to carry their own `match node { .. }` that rebuilt every
//! variant with new operands, new binding names, or new child bodies:
//! induction-variable substitution (`transform::subst`), fusion alpha-renaming
//! (`execution_plan::fusion::alpha_rename`), cache-key canonicalization
//! (`Program::canonicalized`), and the pass engine's encoded-order rewrite.
//! They differed only in what they did at each position, never in which
//! positions exist, so a variant added to `Node` had to be threaded through
//! four independent matches and the first one missed silently dropped the new
//! variant's operands.
//!
//! `rewrite_node` is now the only rewriting enumeration of `Node`. It offers
//! every rewritable position to a `NodeRewrite` policy, in the order the
//! program is written: identifiers and operand expressions first, then child
//! bodies, matching the operand/body split `visit::node_shape` records
//! and the body groups `visit::child_bodies` enumerates. The pass
//! engine's expression arena numbers expressions in exactly that order, so a
//! pass that consumes a per-expression GPU verdict can drive this walk and stay
//! aligned with the encoder.
//!
//! The walk is borrow-preserving. A position that reports no change is not
//! rebuilt, and a node whose positions all report no change returns `None` so
//! the caller keeps the original rather than deep-cloning an identical tree.
//! Substitution of an absent variable, renaming under an empty rename set, and
//! canonicalization of an already-canonical body therefore allocate nothing.

use std::sync::Arc;

use crate::ir::{Expr, Ident, Node, Program};

/// What a rewriting walk does at each position of a [`Node`].
///
/// Every hook answers with `None` for "unchanged", which is what keeps
/// [`rewrite_node`] from cloning a subtree it did not touch.
pub trait NodeRewrite {
    /// Called once per node, before any of that node's positions.
    ///
    /// A policy whose decision depends on the node as a whole, rather than on
    /// one position of it, reads it here: which binding an operand belongs to,
    /// or which slot of an external per-node table the node occupies. The pass
    /// engine's CSE passes index a GPU-built expression arena that way, and
    /// the arena numbers nodes in exactly this order.
    fn enter(&mut self, node: &Node) {
        let _ = node;
    }

    /// Rewrite one operand expression of a node.
    ///
    /// Called once per operand, in source order, before any child body of the
    /// same node. A stateful policy can rely on that order: it is the order
    /// the expression arena assigns identifiers in.
    fn operand(&mut self, expr: &Expr) -> Option<Expr>;

    /// Rewrite one name a node binds in the value namespace.
    ///
    /// Covers the `Let` and `Assign` targets and the `Loop` induction
    /// variable, which is exactly what
    /// [`visit::node_scalars`](crate::visit::node_scalars) reports as a
    /// binding. Buffer names are a separate namespace declared in the
    /// program's buffer table, not bound by a node, and are carried through
    /// unchanged; a pass that renames a buffer rewrites that table with
    /// `Program::with_rewritten_buffers`.
    fn binding(&mut self, name: &Ident) -> Option<Ident> {
        let _ = name;
        None
    }

    /// Rewrite one stream tag.
    ///
    /// Covers the async copy, wait, trap, and resume tags, which is exactly
    /// what [`visit::node_tag`](crate::visit::node_tag) reports. A tag names an
    /// in-flight transfer rather than a value, and the start that opens it and
    /// the wait that closes it must carry the same name, so a policy that
    /// renames values leaves this hook alone.
    ///
    /// One hook used to be offered both namespaces, and a value renamer then
    /// had to re-derive which position it had been called for:
    /// `transform::inline` carried a per-node copy of the binding
    /// `node_scalars` reports for no other purpose, and a renamer that skipped
    /// that step renamed a tag that happened to share a variable's name.
    fn tag(&mut self, name: &Ident) -> Option<Ident> {
        let _ = name;
        None
    }

    /// Rewrite one child body of `parent`.
    ///
    /// `parent` is the node that owns the body, so a policy can act on the
    /// binding it introduces: substitution stops at a `Loop` whose induction
    /// variable shadows the substituted name. The default recurses with the
    /// same policy.
    fn body(&mut self, parent: &Node, body: &[Node]) -> Option<Vec<Node>>
    where
        Self: Sized,
    {
        let _ = parent;
        rewrite_body(body, self)
    }
}

/// Rewrite every node of `body` under `rewrite`.
///
/// Returns `None` when no node changed. The output is only allocated once some
/// node reports a change, and it is sized for the whole body at that point so
/// the remaining nodes append without reallocating.
pub fn rewrite_body<R: NodeRewrite>(body: &[Node], rewrite: &mut R) -> Option<Vec<Node>> {
    let mut out: Option<Vec<Node>> = None;
    for (index, node) in body.iter().enumerate() {
        match rewrite_node(node, rewrite) {
            None => {
                if let Some(out) = out.as_mut() {
                    out.push(node.clone());
                }
            }
            Some(rewritten) => {
                out.get_or_insert_with(|| {
                    let mut sink = Vec::with_capacity(body.len());
                    sink.extend_from_slice(&body[..index]);
                    sink
                })
                .push(rewritten);
            }
        }
    }
    out
}

/// Rewrite one node under `rewrite`, returning `None` when nothing changed.
///
/// Exhaustive with no catch-all arm, deliberately, for the same reason
/// `visit::child_bodies` is: adding a `Node` variant fails to compile
/// here, and that failure forces the author to say which of the new variant's
/// positions a rewrite owes a visit. A catch-all would classify the new variant
/// as inert and silently leave its operands unrewritten, which for a
/// substitution is a stale variable reference rather than a missed
/// optimization.
pub fn rewrite_node<R: NodeRewrite>(node: &Node, rewrite: &mut R) -> Option<Node> {
    rewrite.enter(node);
    match node {
        Node::Let { name, value } => {
            let new_name = rewrite.binding(name);
            let new_value = rewrite.operand(value);
            (new_name.is_some() || new_value.is_some()).then(|| Node::Let {
                name: keep(new_name, name),
                value: keep(new_value, value),
            })
        }
        Node::Assign { name, value } => {
            let new_name = rewrite.binding(name);
            let new_value = rewrite.operand(value);
            (new_name.is_some() || new_value.is_some()).then(|| Node::Assign {
                name: keep(new_name, name),
                value: keep(new_value, value),
            })
        }
        Node::Store {
            buffer,
            index,
            value,
        } => {
            let new_index = rewrite.operand(index);
            let new_value = rewrite.operand(value);
            (new_index.is_some() || new_value.is_some()).then(|| Node::Store {
                buffer: buffer.clone(),
                index: keep(new_index, index),
                value: keep(new_value, value),
            })
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let new_cond = rewrite.operand(cond);
            let new_then = rewrite.body(node, then);
            let new_otherwise = rewrite.body(node, otherwise);
            (new_cond.is_some() || new_then.is_some() || new_otherwise.is_some()).then(|| {
                Node::If {
                    cond: keep(new_cond, cond),
                    then: keep_body(new_then, then),
                    otherwise: keep_body(new_otherwise, otherwise),
                }
            })
        }
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            let new_var = rewrite.binding(var);
            let new_from = rewrite.operand(from);
            let new_to = rewrite.operand(to);
            let new_body = rewrite.body(node, body);
            (new_var.is_some() || new_from.is_some() || new_to.is_some() || new_body.is_some())
                .then(|| Node::Loop {
                    var: keep(new_var, var),
                    from: keep(new_from, from),
                    to: keep(new_to, to),
                    body: keep_body(new_body, body),
                })
        }
        Node::Block(body) => rewrite.body(node, body).map(Node::Block),
        Node::Region {
            generator,
            source_region,
            body,
        } => rewrite.body(node, body).map(|body| Node::Region {
            generator: generator.clone(),
            source_region: source_region.clone(),
            body: Arc::new(body),
        }),
        Node::AsyncLoad {
            source,
            destination,
            offset,
            size,
            tag,
        } => {
            let parts = rewrite_async_copy(rewrite, offset, size, tag);
            parts.map(|(offset, size, tag)| Node::AsyncLoad {
                source: source.clone(),
                destination: destination.clone(),
                offset,
                size,
                tag,
            })
        }
        Node::AsyncStore {
            source,
            destination,
            offset,
            size,
            tag,
        } => {
            let parts = rewrite_async_copy(rewrite, offset, size, tag);
            parts.map(|(offset, size, tag)| Node::AsyncStore {
                source: source.clone(),
                destination: destination.clone(),
                offset,
                size,
                tag,
            })
        }
        Node::Trap { address, tag } => {
            let new_address = rewrite.operand(address);
            let new_tag = rewrite.tag(tag);
            (new_address.is_some() || new_tag.is_some()).then(|| Node::Trap {
                address: Box::new(keep(new_address, address.as_ref())),
                tag: keep(new_tag, tag),
            })
        }
        Node::TileLoad {
            tile,
            tile_type,
            buffer,
            origin,
            layout,
        } => {
            let mut changed = false;
            let new_origin: Vec<Expr> = origin
                .iter()
                .map(|e| {
                    if let Some(ne) = rewrite.operand(e) {
                        changed = true;
                        ne
                    } else {
                        e.clone()
                    }
                })
                .collect();
            changed.then(|| Node::TileLoad {
                tile: tile.clone(),
                tile_type: tile_type.clone(),
                buffer: buffer.clone(),
                origin: new_origin,
                layout: layout.clone(),
            })
        }
        Node::TileStore {
            buffer,
            origin,
            tile,
        } => {
            let mut changed = false;
            let new_origin: Vec<Expr> = origin
                .iter()
                .map(|e| {
                    if let Some(ne) = rewrite.operand(e) {
                        changed = true;
                        ne
                    } else {
                        e.clone()
                    }
                })
                .collect();
            changed.then(|| Node::TileStore {
                buffer: buffer.clone(),
                origin: new_origin,
                tile: tile.clone(),
            })
        }
        Node::TileElementwise { out, inputs, body } => {
            let new_body = rewrite.body(node, body);
            new_body.map(|body| Node::TileElementwise {
                out: out.clone(),
                inputs: inputs.clone(),
                body,
            })
        }
        Node::AsyncWait { tag } => rewrite.tag(tag).map(|tag| Node::AsyncWait { tag }),
        Node::Resume { tag } => rewrite.tag(tag).map(|tag| Node::Resume { tag }),
        Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileDecl { .. }
        | Node::Return
        | Node::Barrier { .. }
        | Node::IndirectDispatch { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::Opaque(_) => None,
    }
}

/// `offset`, `size`, `tag` of an async copy, or `None` when all three held.
#[inline]
fn rewrite_async_copy<R: NodeRewrite>(
    rewrite: &mut R,
    offset: &Expr,
    size: &Expr,
    tag: &Ident,
) -> Option<(Box<Expr>, Box<Expr>, Ident)> {
    let new_offset = rewrite.operand(offset);
    let new_size = rewrite.operand(size);
    let new_tag = rewrite.tag(tag);
    (new_offset.is_some() || new_size.is_some() || new_tag.is_some()).then(|| {
        (
            Box::new(keep(new_offset, offset)),
            Box::new(keep(new_size, size)),
            keep(new_tag, tag),
        )
    })
}

#[inline]
fn keep<T: Clone>(rewritten: Option<T>, original: &T) -> T {
    rewritten.unwrap_or_else(|| original.clone())
}

#[inline]
fn keep_body(rewritten: Option<Vec<Node>>, original: &[Node]) -> Vec<Node> {
    rewritten.unwrap_or_else(|| original.to_vec())
}

/// The nodes of one scope up to and including the first `Node::Return`.
///
/// Everything after a `Return` is unreachable, so no rewrite carries it
/// forward and no analysis indexes a verdict for it. One owner, because four
/// walks repeated the truncation beside their own loop and a fifth forgot it.
#[must_use]
pub fn reachable_prefix(body: &[Node]) -> &[Node] {
    let end = body
        .iter()
        .position(|node| matches!(node, Node::Return))
        .map_or(body.len(), |index| index + 1);
    &body[..end]
}

/// Drive `rewrite` over one scope for its effects, discarding the rebuilt nodes.
///
/// A counting pass reads its answer out of the policy, not out of the tree, so
/// it must still visit exactly the positions the rewriting pass will visit.
pub fn visit_scope<R: NodeRewrite>(body: &[Node], rewrite: &mut R) {
    for node in reachable_prefix(body) {
        rewrite_node(node, rewrite);
    }
}

/// Append one rewritten scope onto `out`.
///
/// A node the policy reports unchanged is cloned rather than rebuilt, which is
/// what keeps an untouched scope from being deep-copied.
pub fn extend_with_rewritten_scope<R: NodeRewrite>(
    body: &[Node],
    rewrite: &mut R,
    out: &mut Vec<Node>,
) {
    let reachable = reachable_prefix(body);
    out.reserve(reachable.len());
    for node in reachable {
        out.push(rewrite_node(node, rewrite).unwrap_or_else(|| node.clone()));
    }
}

/// Rewrite one scope, reporting `None` when nothing in it changed.
///
/// The node walk is borrow-preserving: a node whose positions all report no
/// change returns `None` rather than a rebuilt clone. A scope walk that
/// discards that answer and rebuilds anyway deep-copies the whole subtree on
/// every pass, including the passes that rewrote nothing. Truncation counts as
/// a change, because the unreachable tail must not survive.
pub fn rewrite_scope_opt<R: NodeRewrite>(body: &[Node], rewrite: &mut R) -> Option<Vec<Node>> {
    let reachable = reachable_prefix(body);
    let mut out: Option<Vec<Node>> = None;
    for (index, node) in reachable.iter().enumerate() {
        match rewrite_node(node, rewrite) {
            None => {
                if let Some(out) = out.as_mut() {
                    out.push(node.clone());
                }
            }
            Some(rewritten) => {
                out.get_or_insert_with(|| {
                    let mut sink = Vec::with_capacity(reachable.len());
                    sink.extend_from_slice(&reachable[..index]);
                    sink
                })
                .push(rewritten);
            }
        }
    }
    if out.is_none() && reachable.len() != body.len() {
        return Some(reachable.to_vec());
    }
    out
}

/// Rewrite one scope into a fresh body.
pub fn rewrite_scope<R: NodeRewrite>(body: &[Node], rewrite: &mut R) -> Vec<Node> {
    rewrite_scope_opt(body, rewrite).unwrap_or_else(|| reachable_prefix(body).to_vec())
}

/// Rewrite the program's entry scope, descending through a single wrapping
/// `Node::Region`.
///
/// A composition-built Program wraps its whole body in one Region carrying the
/// generator id, and a rewrite that replaced the entry with the rewritten body
/// would drop that identity. Every scope rewrite therefore enters through here
/// rather than calling `Program::with_rewritten_entry` directly.
#[must_use]
pub fn rewrite_program_entry(
    program: &Program,
    rewrite: impl FnOnce(&[Node]) -> Vec<Node>,
) -> Program {
    let new_entry = match program.entry() {
        [Node::Region {
            generator,
            source_region,
            body,
        }] => vec![Node::Region {
            generator: generator.clone(),
            source_region: source_region.clone(),
            body: Arc::new(rewrite(body)),
        }],
        entry => rewrite(entry),
    };
    program.with_rewritten_entry(new_entry)
}
