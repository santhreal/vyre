use super::TypeFacts;
use crate::ir::{DataType, Expr, Ident, Node, Program};
use crate::validate::typecheck::{expr_type, TypeEnv};
use rustc_hash::FxHashMap;

pub(super) fn derive(program: &Program) -> TypeFacts {
    let mut ctx = TypeFactCtx {
        facts: TypeFacts::default(),
        buffer_types: program
            .buffers()
            .iter()
            .map(|buffer| (Ident::from(buffer.name()), buffer.element().clone()))
            .collect(),
        expr_key: Vec::with_capacity(64),
    };
    ctx.infer_nodes_types(program.entry());
    ctx.facts
}

struct TypeFactCtx {
    facts: TypeFacts,
    buffer_types: FxHashMap<Ident, DataType>,
    expr_key: Vec<u8>,
}

/// The optimizer reads the same answer validation does.
///
/// Type inference itself is [`expr_type`]; this only supplies the two free-name
/// lookups and harvests the type of every subexpression the walk resolves, so
/// the fact map is filled from a single traversal.
impl TypeEnv for TypeFactCtx {
    fn var_type(&self, name: &str) -> Option<DataType> {
        self.facts.var_types.get(name).cloned()
    }

    fn buffer_element(&self, name: &str) -> Option<DataType> {
        self.buffer_types.get(name).cloned()
    }

    fn on_typed(&mut self, expr: &Expr, ty: Option<&DataType>) {
        let Some(ty) = ty else {
            return;
        };
        let key = self.expr_structural_key(expr);
        self.facts.expr_types.insert(key, ty.clone());
    }
}

impl TypeFactCtx {
    fn infer_nodes_types(&mut self, nodes: &[Node]) {
        let mut stack = Vec::with_capacity(nodes.len());
        stack.extend(nodes.iter().rev());
        while let Some(node) = stack.pop() {
            match node {
                Node::Let { name, value } | Node::Assign { name, value } => {
                    if let Some(ty) = expr_type(value, self) {
                        self.facts.var_types.insert(name.clone(), ty);
                    }
                }
                Node::Store { index, value, .. } => {
                    self.record_expr_type(index);
                    self.record_expr_type(value);
                }
                Node::If {
                    cond,
                    then,
                    otherwise,
                } => {
                    self.record_expr_type(cond);
                    stack.extend(otherwise.iter().rev());
                    stack.extend(then.iter().rev());
                }
                Node::Loop { from, to, body, .. } => {
                    self.record_expr_type(from);
                    self.record_expr_type(to);
                    stack.extend(body.iter().rev());
                }
                Node::Block(nodes) => {
                    stack.extend(nodes.iter().rev());
                }
                Node::Region { body, .. } => {
                    stack.extend(body.iter().rev());
                }
                Node::AsyncLoad { offset, size, .. } | Node::AsyncStore { offset, size, .. } => {
                    self.record_expr_type(offset);
                    self.record_expr_type(size);
                }
                Node::Trap { address, .. } => {
                    self.record_expr_type(address);
                }
                Node::Return
                | Node::Barrier { .. }
                | Node::IndirectDispatch { .. }
                | Node::AllReduce { .. }
                | Node::AllGather { .. }
                | Node::ReduceScatter { .. }
                | Node::Broadcast { .. }
                | Node::AsyncWait { .. }
                | Node::Resume { .. }
                | Node::Opaque(_) => {}
            }
        }
    }

    /// Type an expression that binds no name, purely for the facts the walk
    /// deposits through [`TypeEnv::on_typed`].
    fn record_expr_type(&mut self, expr: &Expr) {
        drop(expr_type(expr, self));
    }

    fn expr_structural_key(&mut self, expr: &Expr) -> u64 {
        self.expr_key.clear();
        if let Err(error) = crate::serial::wire::encode::put_expr(&mut self.expr_key, expr) {
            self.expr_key.clear();
            self.expr_key
                .extend_from_slice(b"VYRE-TYPE-FACT-EXPR-WIRE-ERROR\0");
            self.expr_key.extend_from_slice(error.as_bytes());
        }
        let digest = blake3::hash(&self.expr_key);
        u64::from_le_bytes([
            digest.as_bytes()[0],
            digest.as_bytes()[1],
            digest.as_bytes()[2],
            digest.as_bytes()[3],
            digest.as_bytes()[4],
            digest.as_bytes()[5],
            digest.as_bytes()[6],
            digest.as_bytes()[7],
        ])
    }
}
