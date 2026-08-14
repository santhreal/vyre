use std::borrow::Cow;
use std::slice;
use std::sync::Arc;

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::spec_types::BinOp;
use crate::optimizer::rewrite::{rewrite_node_slices, rewrite_nodes_cow};

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
        self.canonical_form().into_owned()
    }

    /// Canonical shape of this program, borrowed when it is already canonical.
    ///
    /// Fingerprints are taken after every optimizer pass, and by then the
    /// program is normally canonical already, so the walk reports what it
    /// changed rather than rebuilding the whole tree to discover it changed
    /// nothing.
    fn canonical_form(&self) -> Cow<'_, Self> {
        match (
            canonical_entry(self.entry()),
            canonical_buffers(self.buffers()),
        ) {
            (Cow::Borrowed(_), None) => Cow::Borrowed(self),
            (Cow::Borrowed(_), Some(buffers)) => Cow::Owned(self.with_rewritten_buffers(buffers)),
            (Cow::Owned(entry), None) => Cow::Owned(self.with_rewritten_entry(entry)),
            (Cow::Owned(entry), Some(buffers)) => Cow::Owned(
                self.with_rewritten_entry(entry)
                    .with_rewritten_buffers(buffers),
            ),
        }
    }

    /// Serialize the canonical IR shape into stable VIR0 wire bytes.
    ///
    /// # Errors
    ///
    /// Returns the same wire-format validation errors as [`Self::to_wire`],
    /// but after canonical normalization has been applied.
    #[must_use]
    pub fn canonical_wire_bytes(&self) -> Result<Vec<u8>, crate::error::IrError> {
        let canonical = self.canonical_form();
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

/// Canonically ordered buffer table, or `None` when it is already ordered.
fn canonical_buffers(buffers: &[BufferDecl]) -> Option<Vec<BufferDecl>> {
    let keys: Vec<Vec<u8>> = buffers.iter().map(buffer_decl_canonical_key).collect();
    if keys.windows(2).all(|pair| pair[0] <= pair[1]) {
        return None;
    }
    let mut order: Vec<usize> = (0..buffers.len()).collect();
    order.sort_by(|&left, &right| keys[left].cmp(&keys[right]));
    Some(
        order
            .into_iter()
            .map(|index| buffers[index].clone())
            .collect(),
    )
}

/// Canonical entry body: commutative operands normalized, then transparent
/// `Block` wrappers flattened. Borrowed when neither step changes anything.
fn canonical_entry(nodes: &[Node]) -> Cow<'_, [Node]> {
    let mut ctx = CanonicalCtx::default();
    match rewrite_nodes_cow(nodes, &mut |candidate| {
        ctx.swap_commutative_operands(candidate)
    }) {
        Cow::Borrowed(nodes) => splice_transparent_blocks(nodes),
        Cow::Owned(nodes) => {
            let spliced = match splice_transparent_blocks(&nodes) {
                Cow::Borrowed(_) => None,
                Cow::Owned(spliced) => Some(spliced),
            };
            Cow::Owned(spliced.unwrap_or(nodes))
        }
    }
}

/// Flatten every `Block` that owns no local binding, bottom up. A flattened
/// block loses its wrapper, so it always reports an owned rewrite.
fn splice_transparent_blocks(nodes: &[Node]) -> Cow<'_, [Node]> {
    rewrite_node_slices(nodes, |node| match node {
        Node::Block(children) => {
            let children = splice_transparent_blocks(children);
            if can_splice_block(children.as_ref()) {
                Cow::Owned(children.into_owned())
            } else {
                match children {
                    Cow::Borrowed(_) => Cow::Borrowed(slice::from_ref(node)),
                    Cow::Owned(children) => Cow::Owned(vec![Node::Block(children)]),
                }
            }
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let then_body = splice_transparent_blocks(then);
            let otherwise_body = splice_transparent_blocks(otherwise);
            if matches!(
                (&then_body, &otherwise_body),
                (Cow::Borrowed(_), Cow::Borrowed(_))
            ) {
                Cow::Borrowed(slice::from_ref(node))
            } else {
                Cow::Owned(vec![Node::if_then_else(
                    cond.clone(),
                    then_body.into_owned(),
                    otherwise_body.into_owned(),
                )])
            }
        }
        Node::Loop {
            var,
            from,
            to,
            body,
        } => match splice_transparent_blocks(body) {
            Cow::Borrowed(_) => Cow::Borrowed(slice::from_ref(node)),
            Cow::Owned(body) => Cow::Owned(vec![Node::loop_for(
                var.clone(),
                from.clone(),
                to.clone(),
                body,
            )]),
        },
        Node::Region {
            generator,
            source_region,
            body,
        } => match splice_transparent_blocks(body) {
            Cow::Borrowed(_) => Cow::Borrowed(slice::from_ref(node)),
            Cow::Owned(body) => Cow::Owned(vec![Node::Region {
                generator: generator.clone(),
                source_region: source_region.clone(),
                body: Arc::new(body),
            }]),
        },
        // Every remaining statement carries no nested node list, so there is
        // nothing to flatten. Listed rather than wildcarded: a new statement
        // node that owns a body must be routed above, and this match is what
        // refuses to compile until it is.
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::Store { .. }
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
        | Node::Return
        | Node::Barrier { .. }
        | Node::Opaque(_) => Cow::Borrowed(slice::from_ref(node)),
    })
}

#[derive(Default)]
struct CanonicalCtx {
    left_key: Vec<u8>,
    right_key: Vec<u8>,
}

impl CanonicalCtx {
    /// Normalize one commutative `BinOp` operand pair. `None` leaves the
    /// expression untouched, which is what keeps an already-canonical subtree
    /// borrowed instead of cloned.
    fn swap_commutative_operands(&mut self, candidate: &Expr) -> Option<Expr> {
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
