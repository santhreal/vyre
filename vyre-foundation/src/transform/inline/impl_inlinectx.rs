use std::borrow::Cow;

use super::expand::CalleeExpander;
use super::expand_walk::{expand_body, ExpandPolicy};
use super::{
    input_arg_map, input_buffers, output_buffer, zero_value, Error, Expr, HashMap, Ident,
    InlineCtx, Node, OpResolver, Program, Result, UnresolvedCalls,
};
use crate::optimizer::rewrite::rewrite_expr;

impl InlineCtx {
    #[inline]
    pub(crate) fn new(resolver: OpResolver) -> Self {
        Self::new_with_mode(resolver, UnresolvedCalls::Reject)
    }

    #[inline]
    pub(crate) fn new_with_mode(resolver: OpResolver, unresolved: UnresolvedCalls) -> Self {
        Self {
            resolver,
            unresolved,
            stack: Vec::new(),
            next_call_id: 0,
        }
    }

    /// `nodes` with every `Expr::Call` in them expanded.
    ///
    /// Which positions a statement has is
    /// [`crate::transform::rewrite_walk::rewrite_node`]'s decision, driven from
    /// here through [`expand_body`]. The match this replaces enumerated them
    /// itself and cloned an async copy's `offset` and `size` and a trap address
    /// verbatim, so a call in one of those positions reached a backend under
    /// `UnresolvedCalls::Reject`, which is the case inlining exists to refuse.
    ///
    /// # Errors
    ///
    /// Whatever expanding one of the calls reports.
    pub(crate) fn inline_nodes(&mut self, nodes: &[Node]) -> Result<Vec<Node>> {
        let mut policy = CallerInline(self);
        Ok(expand_body(nodes, &mut policy)?.into_owned())
    }

    /// `expr` with every `Expr::Call` replaced by the value its callee produces,
    /// plus the statements that value needs in front of it.
    ///
    /// Operand positions come from [`rewrite_expr`], the one owner. The match
    /// this replaces enumerated them itself and put `SubgroupBallot`,
    /// `SubgroupShuffle` and `SubgroupReduce` in its inert arm, so a call inside
    /// a subgroup operand was handed back verbatim: under
    /// `UnresolvedCalls::Reject` the program kept an `Expr::Call` that inlining
    /// exists to refuse, and under `Keep` that call's own arguments were never
    /// inlined either.
    ///
    /// The walk is bottom-up, so a call reaches [`Self::expand_call`] with its
    /// arguments already inlined and no argument is walked twice.
    pub(crate) fn inline_expr(&mut self, expr: &Expr) -> Result<(Vec<Node>, Expr)> {
        let mut prefix = Vec::new();
        let inlined = self.inline_expr_into(expr, &mut prefix)?;
        Ok((prefix, inlined.unwrap_or_else(|| expr.clone())))
    }

    /// `expr` with every `Expr::Call` replaced by the value its callee
    /// produces, hoisting the statements that value needs onto `prefix`.
    ///
    /// Reports `None` when the expression held no call, so a call-free operand
    /// is not cloned.
    fn inline_expr_into(&mut self, expr: &Expr, prefix: &mut Vec<Node>) -> Result<Option<Expr>> {
        let mut failure = None;
        let inlined = rewrite_expr(expr, &mut |candidate| {
            if failure.is_some() {
                return None;
            }
            let Expr::Call { op_id, args } = candidate else {
                return None;
            };
            match self.expand_call(op_id, args) {
                Ok((statements, value)) => {
                    prefix.extend(statements);
                    Some(value)
                }
                Err(error) => {
                    failure = Some(error);
                    None
                }
            }
        });
        match failure {
            Some(error) => Err(error),
            None => Ok(match inlined {
                Cow::Borrowed(_) => None,
                Cow::Owned(value) => Some(value),
            }),
        }
    }

    /// One call site expanded, with `args` already expanded by whichever side
    /// owns them: the caller's own operands, or a callee body's nested call.
    ///
    /// # Errors
    ///
    /// [`Error::InlineCycle`] for a recursive composition, and
    /// [`Error::InlineUnknownOp`] for a call the resolver cannot expand under
    /// [`UnresolvedCalls::Reject`].
    #[inline]
    pub(in crate::transform::inline) fn expand_call(
        &mut self,
        op_id: &str,
        args: &[Expr],
    ) -> Result<(Vec<Node>, Expr)> {
        if self.stack.iter().any(|active| active == op_id) {
            return Err(Error::InlineCycle {
                op_id: op_id.to_string(),
            });
        }

        let callee = match (self.resolver)(op_id) {
            Some(callee) => callee,
            // An op with no composition body is an intrinsic. Under `Keep`
            // the caller executes it directly, so hand the call back with
            // its arguments already inlined.
            None if self.unresolved == UnresolvedCalls::Keep => {
                return Ok((
                    Vec::new(),
                    Expr::Call {
                        op_id: op_id.into(),
                        args: args.to_vec(),
                    },
                ));
            }
            None => {
                return Err(Error::InlineUnknownOp {
                    op_id: op_id.to_string(),
                })
            }
        };
        self.stack.push(op_id.to_string());
        let result = self.expand_callee(op_id, &callee, args.to_vec());
        self.stack.pop();
        result
    }

    #[inline]
    pub(crate) fn expand_callee(
        &mut self,
        op_id: &str,
        callee: &Program,
        args: Vec<Expr>,
    ) -> Result<(Vec<Node>, Expr)> {
        let call_id = self.next_call_id;
        self.next_call_id = self.next_call_id.saturating_add(1);
        let prefix = format!("_vyre_inl{call_id}_");
        let expected_args = input_buffers(callee).len();
        if args.len() != expected_args {
            return Err(Error::InlineArgCountMismatch {
                op_id: op_id.to_string(),
                expected: expected_args,
                got: args.len(),
            });
        }
        let output = output_buffer(op_id, callee)?;
        let result_name = format!("{prefix}result");
        let mut expander = CalleeExpander {
            ctx: self,
            prefix,
            vars: HashMap::default(),
            input_args: input_arg_map(callee, args),
            output_name: Ident::from(output.name()),
            result_name: result_name.clone(),
            saw_output: false,
        };

        let mut nodes = Vec::with_capacity(callee.entry().len() + 1);
        nodes.push(Node::let_bind(&result_name, zero_value(&output.element())));
        nodes.extend(expander.nodes(callee.entry())?);

        if !expander.saw_output {
            return Err(Error::InlineNoOutput {
                op_id: op_id.to_string(),
            });
        }

        Ok((nodes, Expr::var(&result_name)))
    }
}

/// Caller-side inlining as a policy over the one statement walk.
///
/// Nothing is renamed here: the caller's statements are already written in the
/// caller's namespace, so the only position that changes is an operand holding
/// a call.
struct CallerInline<'a>(&'a mut InlineCtx);

impl ExpandPolicy for CallerInline<'_> {
    fn operand(&mut self, expr: &Expr, prefix: &mut Vec<Node>) -> Result<Option<Expr>> {
        self.0.inline_expr_into(expr, prefix)
    }
}
