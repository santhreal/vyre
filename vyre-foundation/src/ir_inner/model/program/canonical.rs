use std::borrow::Cow;

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::spec_types::BinOp;
use crate::transform::rewrite_walk::{self, NodeRewrite};

use super::{meta::buffer_decl_canonical_key, BufferDecl, Program};

impl Program {
    /// Return the canonical IR shape used for security-sensitive cache keys.
    ///
    /// Canonicalization preserves executable semantics while removing
    /// authoring-order noise: buffer declarations are sorted by their stable
    /// wire key, commutative expression operands are normalized, and `Block`
    /// wrappers that do not own local bindings are flattened.
    #[must_use]
    pub fn canonicalized(&self) -> Self {
        let mut buffers = self.buffers().to_vec();
        sort_buffers(&mut buffers);
        let mut ctx = CanonicalCtx::default();
        let entry = ctx
            .canonicalize_nodes(self.entry())
            .unwrap_or_else(|| self.entry().to_vec());
        self.with_rewritten_entry(entry)
            .with_rewritten_buffers(buffers)
    }

    /// Serialize the canonical IR shape into stable VIR0 wire bytes.
    ///
    /// # Errors
    ///
    /// Returns the same wire-format validation errors as [`Self::to_wire`],
    /// but after canonical normalization has been applied.
    #[must_use]
    pub fn canonical_wire_bytes(&self) -> Result<Vec<u8>, crate::error::IrError> {
        let canonical = self.canonicalized();
        // Pre-size: VIR0 wire encoding lands in the ballpark of ~32
        // bytes per IR node + a fixed program header. Over-sizing is
        // free at this stage and avoids the typical 4-7 reallocations
        // a fresh Vec<u8> would do while the encoder pushes header
        // tags + buffer table + node tree.
        let stats = canonical.stats();
        let estimate = 256
            + stats.node_count.saturating_mul(48)
            + canonical.buffers().len().saturating_mul(64);
        let mut out = Vec::with_capacity(estimate);
        crate::serial::wire::encode::to_wire_into(&canonical, &mut out)
            .map_err(|message| crate::error::IrError::WireFormatValidation { message })?;
        Ok(out)
    }

    /// BLAKE3 digest of [`Self::canonical_wire_bytes`].
    ///
    /// # Errors
    ///
    /// Returns a wire-format validation error if the canonical program cannot
    /// be represented by the current VIR0 encoder.
    pub fn canonical_wire_hash(&self) -> Result<blake3::Hash, crate::error::IrError> {
        self.canonical_wire_bytes()
            .map(|bytes| blake3::hash(&bytes))
    }
}

fn sort_buffers(buffers: &mut [BufferDecl]) {
    buffers.sort_by_cached_key(buffer_decl_canonical_key);
}

#[derive(Default)]
struct CanonicalCtx {
    left_key: Vec<u8>,
    right_key: Vec<u8>,
}

impl CanonicalCtx {
    /// Canonicalize every node of `nodes`, splicing out `Block` wrappers that
    /// own no local binding. `None` when the body was already canonical.
    ///
    /// This is also the [`NodeRewrite::body`] hook, so the splice applies at
    /// every depth rather than only at the entry.
    fn canonicalize_nodes(&mut self, nodes: &[Node]) -> Option<Vec<Node>> {
        let mut out: Option<Vec<Node>> = None;
        for (index, node) in nodes.iter().enumerate() {
            let rewritten = rewrite_walk::rewrite_node(node, self);
            let splices = matches!(
                rewritten.as_ref().unwrap_or(node),
                Node::Block(children) if can_splice_block(children)
            );
            if out.is_none() && rewritten.is_none() && !splices {
                continue;
            }
            let sink = out.get_or_insert_with(|| nodes[..index].to_vec());
            push_canonical_node(sink, rewritten.unwrap_or_else(|| node.clone()));
        }
        out
    }
}

impl NodeRewrite for CanonicalCtx {
    /// Normalize commutative operand order. The expression rewrite reports no
    /// change when nothing swapped, so an already-canonical operand is neither
    /// rebuilt nor re-encoded.
    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        match crate::optimizer::rewrite::rewrite_expr(expr, &mut |candidate| {
            let Expr::BinOp { op, left, right } = candidate else {
                return None;
            };
            if !should_swap_operands(*op, left, right, &mut self.left_key, &mut self.right_key) {
                return None;
            }
            Some(Expr::BinOp {
                op: *op,
                left: Box::new((**right).clone()),
                right: Box::new((**left).clone()),
            })
        }) {
            Cow::Borrowed(_) => None,
            Cow::Owned(rewritten) => Some(rewritten),
        }
    }

    fn body(&mut self, _parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        self.canonicalize_nodes(body)
    }
}

fn push_canonical_node(out: &mut Vec<Node>, node: Node) {
    match node {
        Node::Block(children) if can_splice_block(&children) => out.extend(children),
        other => out.push(other),
    }
}

fn can_splice_block(nodes: &[Node]) -> bool {
    nodes.iter().all(|node| !matches!(node, Node::Let { .. }))
}

fn should_swap_operands(
    op: BinOp,
    left: &Expr,
    right: &Expr,
    left_key: &mut Vec<u8>,
    right_key: &mut Vec<u8>,
) -> bool {
    if !is_commutative_binop(op) {
        return false;
    }
    match (is_literal(left), is_literal(right)) {
        (true, false) => true,
        (false, true) => false,
        (true, true) => {
            // Both literals: every commutative op is observably-safe
            // to canonicalize because the literal pair folds to the
            // same value regardless of order. The float-sensitivity
            // contract (Add/Mul reassociation changes rounding) only
            // applies when at least one operand is non-literal.
            expr_wire_key_cmp(left, right, left_key, right_key).is_gt()
        }
        (false, false) => {
            can_sort_all_operands(op) && expr_wire_key_cmp(left, right, left_key, right_key).is_gt()
        }
    }
}

fn expr_wire_key_cmp(
    left: &Expr,
    right: &Expr,
    left_key: &mut Vec<u8>,
    right_key: &mut Vec<u8>,
) -> std::cmp::Ordering {
    left_key.clear();
    right_key.clear();
    append_expr_wire_key(left_key, left);
    append_expr_wire_key(right_key, right);
    left_key.as_slice().cmp(right_key.as_slice())
}

fn append_expr_wire_key(key: &mut Vec<u8>, expr: &Expr) {
    if let Err(error) = crate::serial::wire::encode::put_expr(key, expr) {
        key.clear();
        key.extend_from_slice(b"VYRE-CANONICAL-EXPR-WIRE-ERROR\0");
        key.extend_from_slice(error.as_bytes());
    }
}

fn is_commutative_binop(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add
            | BinOp::WrappingAdd
            | BinOp::SaturatingAdd
            | BinOp::Mul
            | BinOp::SaturatingMul
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::BitXor
            | BinOp::Eq
            | BinOp::Ne
            | BinOp::And
            | BinOp::Or
            | BinOp::Min
            | BinOp::Max
            | BinOp::AbsDiff
    )
}

fn can_sort_all_operands(op: BinOp) -> bool {
    // Ops whose operand swap is observably safe even when both
    // operands are arbitrary non-literal expressions. Excludes Add /
    // Mul because float reassociation changes rounding for non-literal
    // chains; `should_swap_operands` handles the both-literal case
    // separately so the canonical fingerprint still normalises
    // `Add(1, 2)` vs `Add(2, 1)`.
    matches!(
        op,
        BinOp::WrappingAdd
            | BinOp::SaturatingAdd
            | BinOp::SaturatingMul
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::BitXor
            | BinOp::Eq
            | BinOp::Ne
            | BinOp::And
            | BinOp::Or
            | BinOp::AbsDiff
    )
}

fn is_literal(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_)
    )
}
