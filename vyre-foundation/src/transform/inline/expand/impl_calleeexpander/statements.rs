use super::super::CalleeExpander;
use crate::error::IrResult as Result;
use crate::ir::{Expr, Ident, Node};
use crate::transform::inline::expand_walk::{expand_body, ExpandPolicy};
use crate::visit::NameBinding;

impl CalleeExpander<'_> {
    /// The callee's statements rewritten into the caller's namespace.
    ///
    /// Which positions a statement has is
    /// [`crate::transform::rewrite_walk::rewrite_node`]'s decision, driven from
    /// here through [`expand_body`]. This walk used to be a second exhaustive
    /// match over `Node` beside the caller's, and the two had diverged over
    /// which positions hold a callee-local expression.
    ///
    /// # Errors
    ///
    /// Whatever expanding one of the statements reports.
    #[inline]
    pub(crate) fn nodes(&mut self, nodes: &[Node]) -> Result<Vec<Node>> {
        Ok(expand_body(nodes, self)?.into_owned())
    }
}

impl ExpandPolicy for CalleeExpander<'_> {
    /// Every operand is alpha-renamed and has its calls hoisted, including an
    /// async copy's `offset` and `size` and a trap address: each of those can
    /// reference a callee-local, and a callee-local left unrenamed dangles
    /// against its renamed declaration.
    fn operand(&mut self, expr: &Expr, prefix: &mut Vec<Node>) -> Result<Option<Expr>> {
        let (statements, value) = self.expr(expr)?;
        prefix.extend(statements);
        Ok(Some(value))
    }

    /// A declaration and a loop induction variable get a fresh prefixed name
    /// recorded for the rest of the callee body. A rebinding resolves against
    /// the name already recorded for it, because an `Assign` names a binding
    /// some enclosing statement declared.
    fn binding(&mut self, binding: NameBinding, name: &Ident) -> Option<Ident> {
        Some(Ident::from(match binding {
            NameBinding::Declare | NameBinding::Induction => self.rename_decl(name),
            NameBinding::Reassign => self.rename_use(name),
        }))
    }

    /// A store to the callee's output buffer becomes an assignment to the
    /// caller's result binding, because the caller declares no such buffer.
    fn replace(&mut self, node: &Node) -> Option<Node> {
        let Node::Store { buffer, value, .. } = node else {
            return None;
        };
        if self.output_name != *buffer {
            return None;
        }
        self.saw_output = true;
        Some(Node::assign(&self.result_name, value.clone()))
    }
}
