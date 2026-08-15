//! The one statement walk both inlining sides drive.
//!
//! Inlining rewrites statements twice. Once in the caller, where an
//! `Expr::Call` operand becomes a value plus the statements that produce it,
//! and once in the callee body being pasted in, where the same thing happens
//! plus alpha-renaming, argument substitution, and the redirect of the callee's
//! output store into the caller's result binding. Each side used to carry its
//! own exhaustive `match node { .. }`, and the pair had diverged: the callee
//! walk expanded an async copy's `offset` and `size` and a trap address, the
//! caller walk cloned all three verbatim, so a call in one of those positions
//! survived inlining on the caller side. Under `UnresolvedCalls::Reject` that
//! left behind the very call inlining exists to refuse.
//!
//! Which positions a statement has is [`rewrite_walk::rewrite_node`]'s
//! decision, the one rewriting enumeration of `Node`. It has no catch-all arm,
//! so a new variant is a compile error there rather than a statement this walk
//! copies with its operands unexpanded. What to do at each position is an
//! [`ExpandPolicy`], which is the only thing the two sides disagree about.
//!
//! The walk is borrow-preserving. A statement whose positions all report no
//! change and which hoists nothing is not rebuilt, so a subtree with no call in
//! it costs nothing to walk past.

use std::borrow::Cow;
use std::slice;

use crate::error::{IrError as Error, IrResult as Result};
use crate::ir::{Expr, Ident, Node};
use crate::optimizer::rewrite::rewrite_node_slices;
use crate::transform::rewrite_walk::{self, NodeRewrite};
use crate::visit::{node_scalars, NameBinding};

/// What an inlining walk does at the positions a statement owns.
///
/// Every hook answers `None` for "unchanged", which is what keeps
/// [`expand_body`] from rebuilding a statement it did not touch.
pub(super) trait ExpandPolicy {
    /// One operand expression, plus the statements its value needs in front of
    /// the statement that reads it.
    ///
    /// Called once per operand, in source order, before any child body of the
    /// same statement, so a value hoisted out of an operand lands after
    /// everything the previous operand hoisted and before the statement that
    /// consumes it.
    ///
    /// # Errors
    ///
    /// Whatever the policy refuses to express in the caller.
    fn operand(&mut self, expr: &Expr, prefix: &mut Vec<Node>) -> Result<Option<Expr>>;

    /// One name a statement binds, in the value namespace.
    ///
    /// `binding` distinguishes a fresh declaration from a rebinding, which is
    /// the difference between recording a rename and looking one up.
    fn binding(&mut self, binding: NameBinding, name: &Ident) -> Option<Ident> {
        let _ = (binding, name);
        None
    }

    /// The statement this policy writes in place of `node`, whose own positions
    /// are already expanded.
    ///
    /// This is where a statement changes kind rather than contents: the callee
    /// side turns a store to the callee's output buffer into an assignment to
    /// the caller's result binding, because the caller has no such buffer.
    fn replace(&mut self, node: &Node) -> Option<Node> {
        let _ = node;
        None
    }
}

/// `nodes` with every statement expanded under `policy`.
///
/// Returns the input borrowed when nothing in it changed.
///
/// # Errors
///
/// The first refusal any position reports. Positions after it are not visited.
pub(super) fn expand_body<'a, P: ExpandPolicy>(
    nodes: &'a [Node],
    policy: &mut P,
) -> Result<Cow<'a, [Node]>> {
    let mut failure: Option<Error> = None;
    let expanded = rewrite_node_slices(nodes, |node| {
        if failure.is_some() {
            return Cow::Borrowed(slice::from_ref(node));
        }
        match expand_node(node, policy) {
            Ok(expanded) => expanded,
            Err(error) => {
                failure = Some(error);
                Cow::Borrowed(slice::from_ref(node))
            }
        }
    });
    match failure {
        Some(error) => Err(error),
        None => Ok(expanded),
    }
}

/// One statement expanded into the statements that replace it.
fn expand_node<'a, P: ExpandPolicy>(node: &'a Node, policy: &mut P) -> Result<Cow<'a, [Node]>> {
    // An opaque statement is an out-of-tree extension. Neither side can rename
    // inside it, substitute an argument into it, or find the calls it holds, so
    // both refuse it rather than emitting it into a caller whose namespace it
    // was never written against.
    if let Node::Opaque(extension) = node {
        return Err(Error::lowering(format!(
            "inliner cannot expand opaque statement extension `{}`/`{}`. Fix: lower the extension to core Node variants before inlining.",
            extension.extension_kind(),
            extension.debug_identity()
        )));
    }

    let mut prefix = Vec::new();
    let mut failure: Option<Error> = None;
    let rewritten = {
        let mut positions = Positions {
            policy: &mut *policy,
            prefix: &mut prefix,
            failure: &mut failure,
            binding: None,
        };
        rewrite_walk::rewrite_node(node, &mut positions)
    };
    if let Some(error) = failure {
        return Err(error);
    }

    let replacement = policy.replace(rewritten.as_ref().unwrap_or(node));
    if prefix.is_empty() && rewritten.is_none() && replacement.is_none() {
        return Ok(Cow::Borrowed(slice::from_ref(node)));
    }

    prefix.push(match replacement {
        Some(replacement) => replacement,
        None => rewritten.unwrap_or_else(|| node.clone()),
    });
    Ok(Cow::Owned(prefix))
}

/// Offers each position of one statement to an [`ExpandPolicy`].
struct Positions<'a, P> {
    policy: &'a mut P,
    /// Statements hoisted out of this statement's operands.
    prefix: &'a mut Vec<Node>,
    /// The first refusal, which stops every later position.
    failure: &'a mut Option<Error>,
    /// What the statement being visited does to the name it binds, or `None`
    /// when it binds nothing in the value namespace.
    binding: Option<NameBinding>,
}

impl<P: ExpandPolicy> NodeRewrite for Positions<'_, P> {
    fn enter(&mut self, node: &Node) {
        self.binding = node_scalars(node).binding.map(|(binding, _)| binding);
    }

    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        if self.failure.is_some() {
            return None;
        }
        match self.policy.operand(expr, self.prefix) {
            Ok(rewritten) => rewritten,
            Err(error) => {
                *self.failure = Some(error);
                None
            }
        }
    }

    fn ident(&mut self, name: &Ident) -> Option<Ident> {
        // Which name a statement binds in the value namespace is
        // `node_scalars`'s decision. A statement that binds none reaches this
        // hook only for an async copy, trap, or resume tag, which is a
        // different namespace: renaming a tag would break its pairing with the
        // matching wait, so tags are carried through.
        let binding = self.binding?;
        self.policy.binding(binding, name)
    }

    fn body(&mut self, parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        let _ = parent;
        if self.failure.is_some() {
            return None;
        }
        match expand_body(body, self.policy) {
            Ok(Cow::Borrowed(_)) => None,
            Ok(Cow::Owned(expanded)) => Some(expanded),
            Err(error) => {
                *self.failure = Some(error);
                None
            }
        }
    }
}
