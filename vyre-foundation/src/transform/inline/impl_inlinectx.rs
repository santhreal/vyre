use super::expand::CalleeExpander;
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

    #[inline]
    pub(crate) fn inline_nodes(&mut self, nodes: &[Node]) -> Result<Vec<Node>> {
        let mut out = Vec::with_capacity(nodes.len());
        for node in nodes {
            out.extend(self.inline_node(node)?);
        }
        Ok(out)
    }

    #[inline]
    #[expect(
        clippy::too_many_lines,
        reason = "exhaustive Node inlining dispatch keeps each IR variant's rewrite contract visible"
    )]
    pub(crate) fn inline_node(&mut self, node: &Node) -> Result<Vec<Node>> {
        match node {
            Node::Let { name, value } => {
                let (mut prefix, value) = self.inline_expr(value)?;
                prefix.push(Node::let_bind(name, value));
                Ok(prefix)
            }
            Node::Assign { name, value } => {
                let (mut prefix, value) = self.inline_expr(value)?;
                prefix.push(Node::assign(name, value));
                Ok(prefix)
            }
            Node::Store {
                buffer,
                index,
                value,
            } => {
                let (mut prefix, index) = self.inline_expr(index)?;
                let (value_prefix, value) = self.inline_expr(value)?;
                prefix.extend(value_prefix);
                prefix.push(Node::store(buffer, index, value));
                Ok(prefix)
            }
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                let (mut prefix, cond) = self.inline_expr(cond)?;
                prefix.push(Node::if_then_else(
                    cond,
                    self.inline_nodes(then)?,
                    self.inline_nodes(otherwise)?,
                ));
                Ok(prefix)
            }
            Node::Loop {
                var,
                from,
                to,
                body,
            } => {
                let (mut prefix, from) = self.inline_expr(from)?;
                let (to_prefix, to) = self.inline_expr(to)?;
                prefix.extend(to_prefix);
                prefix.push(Node::loop_for(var, from, to, self.inline_nodes(body)?));
                Ok(prefix)
            }
            Node::Return => Ok(vec![Node::Return]),
            Node::Block(nodes) => Ok(vec![Node::Block(self.inline_nodes(nodes)?)]),
            Node::Barrier { ordering } => Ok(vec![Node::barrier_with_ordering(*ordering)]),
            Node::IndirectDispatch {
                count_buffer,
                count_offset,
            } => Ok(vec![Node::IndirectDispatch {
                count_buffer: count_buffer.clone(),
                count_offset: *count_offset,
            }]),
            Node::AsyncLoad {
                source,
                destination,
                offset,
                size,
                tag,
            } => Ok(vec![Node::async_load_gpu_driven(
                source.clone(),
                destination.clone(),
                (**offset).clone(),
                (**size).clone(),
                tag.clone(),
            )]),
            Node::AsyncStore {
                source,
                destination,
                offset,
                size,
                tag,
            } => Ok(vec![Node::async_store(
                source.clone(),
                destination.clone(),
                (**offset).clone(),
                (**size).clone(),
                tag.clone(),
            )]),
            Node::AsyncWait { tag } => Ok(vec![Node::async_wait(tag)]),
            Node::Trap { .. }
            | Node::Resume { .. }
            | Node::AllReduce { .. }
            | Node::AllGather { .. }
            | Node::ReduceScatter { .. }
            | Node::Broadcast { .. } => Ok(vec![node.clone()]),
            Node::Region {
                generator,
                source_region,
                body,
            } => Ok(vec![Node::Region {
                generator: generator.clone(),
                source_region: source_region.clone(),
                body: std::sync::Arc::new(self.inline_nodes(body)?),
            }]),
            Node::Opaque(extension) => Err(Error::lowering(format!(
                "inliner cannot rewrite opaque statement extension `{}`/`{}`. Fix: lower the extension to core Node variants before inlining.",
                extension.extension_kind(),
                extension.debug_identity()
            ))),
        }
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
            None => Ok((prefix, inlined.into_owned())),
        }
    }

    /// One call site expanded, with `args` already inlined by the caller.
    #[inline]
    fn expand_call(&mut self, op_id: &str, args: &[Expr]) -> Result<(Vec<Node>, Expr)> {
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
