//! An e-graph mirror of the scalar expression tree.
//!
//! The e-graph substrate is generic over a language whose nodes are `Eq + Hash`
//! (the hashcons keys on them). `Expr` is neither: it carries `f32` literals
//! and `Arc<dyn ExprExtension>` payloads. The mirror keeps the shape the laws
//! act on, a binary operator over two child classes, and interns everything
//! else as an opaque leaf. A leaf is sound to carry unexamined because no
//! derived rewrite looks inside one.

use smallvec::smallvec;

use crate::ir::{BinOp, Expr};
use crate::optimizer::eqsat::{
    try_extract_best, EChildren, EClassId, EGraph, EGraphError, ENodeLang,
};

/// One node of the mirrored expression language.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExprLang {
    /// An expression the mirror does not decompose, by position in the leaf
    /// table of the [`ExprMirror`] that interned it.
    Leaf(usize),
    /// A binary operator over two child classes.
    Bin {
        /// Operator applied to the two children.
        op: BinOp,
        /// Left operand class.
        left: EClassId,
        /// Right operand class.
        right: EClassId,
    },
}

impl ENodeLang for ExprLang {
    fn children(&self) -> EChildren {
        match self {
            Self::Leaf(_) => EChildren::new(),
            Self::Bin { left, right, .. } => smallvec![*left, *right],
        }
    }

    fn with_children(&self, children: &[EClassId]) -> Self {
        match self {
            Self::Leaf(index) => Self::Leaf(*index),
            Self::Bin { op, .. } => Self::Bin {
                op: *op,
                left: children[0],
                right: children[1],
            },
        }
    }
}

/// A mirrored expression: the e-graph, its leaf table, and the root class.
#[derive(Debug)]
pub struct ExprMirror {
    egraph: EGraph<ExprLang>,
    leaves: Vec<Expr>,
    root: EClassId,
}

impl ExprMirror {
    /// Mirror `expr` into a fresh e-graph.
    ///
    /// # Errors
    ///
    /// Returns the substrate's allocation and class-id errors.
    pub fn of(expr: &Expr) -> Result<Self, EGraphError> {
        let mut mirror = Self {
            egraph: EGraph::new(),
            leaves: Vec::new(),
            root: EClassId(0),
        };
        mirror.root = mirror.add_expr(expr)?;
        Ok(mirror)
    }

    /// The class the whole mirrored expression lives in.
    #[must_use]
    pub fn root(&self) -> EClassId {
        self.egraph.find_immut(self.root)
    }

    /// The e-graph, for a rewrite loop that reads its nodes.
    #[must_use]
    pub fn egraph(&self) -> &EGraph<ExprLang> {
        &self.egraph
    }

    /// The e-graph, for a rewrite loop that adds law-derived nodes.
    pub fn egraph_mut(&mut self) -> &mut EGraph<ExprLang> {
        &mut self.egraph
    }

    /// Whether the derived equalities already prove `expr` equal to the root.
    ///
    /// Mirroring `expr` into the same graph is how the question is asked: a
    /// term the derivation reached is hashconsed into the class it was unioned
    /// with, so adding it again returns that class. A term the derivation never
    /// reached lands in a fresh class instead, which is the negative answer.
    ///
    /// # Errors
    ///
    /// Returns the substrate's allocation and class-id errors.
    pub fn holds_equivalent(&mut self, expr: &Expr) -> Result<bool, EGraphError> {
        let root = self.root();
        let added = self.add_expr(expr)?;
        let added = self.egraph.try_find(added)?;
        let root = self.egraph.try_find(root)?;
        Ok(added == root)
    }

    /// The unsigned literal a class denotes, when every node in it is that
    /// literal leaf.
    ///
    /// The identity and absorbing laws state their element as a `u32`, so a
    /// derived rewrite recognises the element only in the literal form whose
    /// value that number states. An `i32` or `f32` literal holding the same
    /// bits is a different value under a different law and is not matched.
    #[must_use]
    pub fn literal_u32(&self, class: EClassId) -> Option<u32> {
        let class = self.egraph.class(self.egraph.find_immut(class))?;
        class.nodes.iter().find_map(|node| match node {
            ExprLang::Leaf(index) => match self.leaves.get(*index) {
                Some(Expr::LitU32(value)) => Some(*value),
                _ => None,
            },
            ExprLang::Bin { .. } => None,
        })
    }

    /// Rebuild the lowest-cost expression the root class represents.
    ///
    /// Cost is one per node, so extraction prefers the smallest equivalent
    /// term. `depth_budget` bounds the reconstruction, because a class can
    /// reach itself through a derived equality and a term walk over a cyclic
    /// class would not terminate.
    ///
    /// # Errors
    ///
    /// Returns the substrate's allocation and class-id errors.
    pub fn extract(&self, depth_budget: usize) -> Result<Option<Expr>, EGraphError> {
        self.extract_class(self.root(), depth_budget)
    }

    fn extract_class(
        &self,
        class: EClassId,
        depth_budget: usize,
    ) -> Result<Option<Expr>, EGraphError> {
        if depth_budget == 0 {
            return Ok(None);
        }
        let Some((node, _)) = try_extract_best(&self.egraph, class, node_cost)? else {
            return Ok(None);
        };
        match node {
            ExprLang::Leaf(index) => Ok(self.leaves.get(index).cloned()),
            ExprLang::Bin { op, left, right } => {
                let Some(left) = self.extract_class(left, depth_budget - 1)? else {
                    return Ok(None);
                };
                let Some(right) = self.extract_class(right, depth_budget - 1)? else {
                    return Ok(None);
                };
                Ok(Some(Expr::BinOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                }))
            }
        }
    }

    fn add_expr(&mut self, expr: &Expr) -> Result<EClassId, EGraphError> {
        match expr {
            Expr::BinOp { op, left, right } => {
                let left = self.add_expr(left)?;
                let right = self.add_expr(right)?;
                self.egraph.try_add(ExprLang::Bin {
                    op: *op,
                    left,
                    right,
                })
            }
            leaf => {
                let index = self.intern_leaf(leaf);
                self.egraph.try_add(ExprLang::Leaf(index))
            }
        }
    }

    fn intern_leaf(&mut self, leaf: &Expr) -> usize {
        if let Some(index) = self.leaves.iter().position(|held| held == leaf) {
            return index;
        }
        self.leaves.push(leaf.clone());
        self.leaves.len() - 1
    }
}

/// One per node, so extraction prefers the term with the fewest nodes.
fn node_cost(_node: &ExprLang) -> u64 {
    1
}
