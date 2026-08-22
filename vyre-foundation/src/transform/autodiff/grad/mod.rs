//! Reverse-mode autodiff IR transform.
//!
//! The main entry point is [`grad`]: given a forward `Program`, a set of
//! output buffer names, and a set of input buffer names, it emits a new
//! `Program` whose stores write the gradients of the outputs w.r.t. the
//! inputs into `grad_<input>` buffers.

use std::ops::ControlFlow;

use rustc_hash::FxHashMap;

use crate::ir::{
    node_variant_name, BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program,
};
use crate::validate::typecheck::{expr_type, TypeEnv};
use crate::visit::{for_each_node, node_scalars, try_for_each_expr, NameBinding};

use super::error::AutodiffError;
mod expr;
use expr::{emit_adjoint_expr, insert_pullback};

/// Per-forward-node pullback expression metadata.
///
/// Keys are stable per-transform pullback node ids in reverse-walk emission
/// order. Values are the adjoint expression consumed by that forward statement.
pub type PullbackMap = FxHashMap<usize, Expr>;

/// Compute reverse-mode gradients for a forward Program.
///
/// # Arguments
///
/// * `program`  -  the forward-pass Program.
/// * `outputs`  -  buffer names whose values to differentiate (the "loss").
/// * `inputs`  -  buffer names to compute gradients w.r.t. Gradient buffers
///   `grad_<name>` are added to the output Program.
///
/// # Returns
///
/// A new `Program` that:
/// 1. Re-declares all forward buffers (inputs as `ReadOnly`, outputs as `ReadOnly`).
/// 2. Declares fresh `grad_<input>` `ReadWrite` buffers for each input in `inputs`.
/// 3. Seeds `grad_<output> = 1.0` for each output in `outputs`.
/// 4. Walks the forward body in reverse, emitting adjoint accumulation stores.
///
/// # Errors
///
/// Returns `AutodiffError` if the Program contains non-differentiable ops in
/// the gradient path, or if an output/input buffer name is not found.
pub fn grad(
    program: &Program,
    outputs: &[&str],
    inputs: &[&str],
) -> Result<Program, AutodiffError> {
    grad_with_pullback(program, outputs, inputs).map(|(program, _pullbacks)| program)
}

/// Compute reverse-mode gradients and return top-level pullback metadata.
///
/// # Errors
///
/// Returns `AutodiffError` if the Program contains unsupported control flow,
/// non-differentiable expression nodes, or unknown input/output buffer names.
pub fn grad_with_pullback(
    program: &Program,
    outputs: &[&str],
    inputs: &[&str],
) -> Result<(Program, PullbackMap), AutodiffError> {
    validate_buffer_names(program, outputs, inputs)?;
    let (back_buffers, output_set, input_set, grad_source_set) =
        build_backward_buffers(program, outputs, inputs);

    // Build the backward body.
    let mut body: Vec<Node> = Vec::new();
    let i_expr = Expr::InvocationId { axis: 0 };
    let mut pullbacks = PullbackMap::default();
    let forward_nodes = program.entry();
    let mut adjoint_env: AdjointEnv = AdjointEnv::new(&grad_source_set, program);

    // Reverse-mode execution must declare every local adjoint before any
    // reverse contribution assigns to it. A previous implementation declared
    // `_adj_*` inside the reversed `Let` handler, which meant downstream
    // `Store` pullbacks could assign a local before declaration and then the
    // later `Let` reset it to zero. Predeclaring locals makes the generated IR
    // SSA-validator-friendly and preserves accumulated adjoints.
    let mut local_targets = Vec::new();
    collect_adjoint_targets(forward_nodes, &mut local_targets);
    for name in &local_targets {
        let adj_name = adjoint_env.ensure_adjoint_var(name.as_str());
        body.push(Node::Let {
            name: adj_name.into(),
            value: Expr::f32(0.0),
        });
    }

    // Phase 0: Gradient buffers are read during accumulation. Make the
    // generated backward Program self-contained by clearing every declared
    // gradient lane before any seed or pullback store runs; callers must not
    // have to provide pre-zeroed scratch to get correct gradients.
    for source_name in &grad_source_set {
        let grad_name = format!("grad_{source_name}");
        body.push(Node::Store {
            buffer: grad_name.into(),
            index: i_expr.clone(),
            value: Expr::f32(0.0),
        });
    }

    // Phase 1: Seed  -  store 1.0 into each grad_<output>[i].
    for out_name in &output_set {
        let grad_name = format!("grad_{out_name}");
        body.push(Node::Store {
            buffer: grad_name.into(),
            index: i_expr.clone(),
            value: Expr::f32(1.0),
        });
    }

    // Phase 2: Reverse walk of forward body.
    // Collect the forward nodes, then process them in reverse order.
    let mut next_pullback_id = 0usize;
    for node in forward_nodes.iter().rev() {
        emit_adjoint_node(
            node,
            &mut body,
            &mut adjoint_env,
            &output_set,
            &mut pullbacks,
            &mut next_pullback_id,
        )?;
    }

    // Phase 3: Flush accumulated adjoints to grad_<input> buffers.
    for inp_name in &input_set {
        let grad_name = format!("grad_{inp_name}");
        if let Some(accum_var) = adjoint_env.get_accumulator(inp_name) {
            body.push(Node::Store {
                buffer: grad_name.into(),
                index: i_expr.clone(),
                value: Expr::Var(accum_var.into()),
            });
        }
    }

    // Fail closed on dangling forward-local references.
    //
    // Reverse-mode adjoints for nonlinear ops carry the forward operand's value
    // (e.g. `d(a*a)` accumulates `adjoint * a`, referencing the forward local
    // `a`). The backward Program re-declares forward BUFFERS as ReadOnly so
    // forward buffer loads are recoverable, but it does NOT re-materialize
    // forward LOCALS (`let`/`assign` results): there is no forward-value
    // recompute or tape. So if any adjoint expression embeds a `Var` naming a
    // forward local, that reference dangles -- the emitted backward Program
    // would fail validation ("reference to undeclared variable").
    //
    // A local used only LINEARLY (its adjoint never multiplies in the local's
    // value -- e.g. `out = xw + x`) never embeds `Var(xw)` in an emitted adjoint
    // expression, so it is NOT flagged: those programs still differentiate. We
    // refuse only when the forward value is genuinely required, turning a silent
    // invalid-Program into an honest, actionable error at `grad()` time rather
    // than a validation failure when the caller later runs the backward pass.
    if let Some(local) = first_dangling_forward_local(&body, &local_targets) {
        return Err(AutodiffError::NotDifferentiable {
            op: format!("forward local `{local}`"),
            fix: "reverse-mode autodiff needs the forward value of this local for a nonlinear \
                  adjoint, but cannot recompute it. Inline the local's definition into its uses \
                  (so the adjoint reads buffers directly), or keep it out of the gradient path"
                .into(),
        });
    }

    Ok((
        Program::wrapped(back_buffers, program.workgroup_size(), body),
        pullbacks,
    ))
}

/// Find the first forward-local name (a `let`/`assign` target) that appears as a
/// `Var` reference anywhere in the emitted backward `body`.
///
/// The backward Program never declares forward locals, so any such reference is
/// a dangling use of an un-recomputed forward value (see the call site). Adjoint
/// accumulator bindings are named `_adj_*` and forward loop variables are bound
/// by the backward loop, so neither is in `local_targets`; only genuine
/// forward-local value references match.
///
/// The walk is [`try_for_each_expr`], which takes node nesting, operand
/// positions, and sub-expressions from their exhaustive owners. The hand-written
/// descent this replaces recorded `Node::Trap`, `Node::AsyncLoad`, and
/// `Node::AsyncStore` as carrying no operand at all, contradicting
/// `node_operands`, which reports `Trap.address` and both async copies'
/// `offset` / `size`. A dangling reference in one of those positions read as
/// clean, so this fail-closed guard failed OPEN and `grad` returned a backward
/// Program naming an undeclared local instead of an `AutodiffError`. No adjoint
/// rule emits those three nodes today (`emit_adjoint_node` refuses them in the
/// forward direction), so the hole was latent rather than live.
fn first_dangling_forward_local(body: &[Node], local_targets: &[Ident]) -> Option<String> {
    let flow = try_for_each_expr(body, |expr| {
        let Expr::Var(name) = expr else {
            return ControlFlow::Continue(());
        };
        match local_targets
            .iter()
            .find(|local| local.as_str() == name.as_str())
        {
            Some(local) => ControlFlow::Break(local.as_str().to_string()),
            None => ControlFlow::Continue(()),
        }
    });
    match flow {
        ControlFlow::Break(local) => Some(local),
        ControlFlow::Continue(()) => None,
    }
}

fn validate_buffer_names(
    program: &Program,
    outputs: &[&str],
    inputs: &[&str],
) -> Result<(), AutodiffError> {
    for name in outputs.iter().chain(inputs.iter()) {
        if program
            .buffers()
            .iter()
            .all(|buffer| buffer.name() != *name)
        {
            return Err(AutodiffError::BufferNotFound {
                name: (*name).to_string(),
            });
        }
    }
    Ok(())
}

fn build_backward_buffers(
    program: &Program,
    outputs: &[&str],
    inputs: &[&str],
) -> (Vec<BufferDecl>, Vec<String>, Vec<String>, Vec<String>) {
    let mut back_buffers = Vec::new();
    let mut next_binding = 0u32;
    for fwd_buf in program.buffers() {
        back_buffers.push(
            BufferDecl::storage(
                fwd_buf.name(),
                next_binding,
                BufferAccess::ReadOnly,
                fwd_buf.element(),
            )
            .with_count(fwd_buf.count()),
        );
        next_binding += 1;
    }

    let output_set: Vec<String> = outputs.iter().map(ToString::to_string).collect();
    let input_set: Vec<String> = inputs.iter().map(ToString::to_string).collect();
    let grad_source_set = grad_buffer_source_names(program, &output_set, &input_set);
    let mut grad_buf_binding = FxHashMap::default();
    for name in &grad_source_set {
        let grad_name = format!("grad_{name}");
        if grad_buf_binding.contains_key(&grad_name) {
            continue;
        }
        let Some(fwd_buf) = program
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == name.as_str())
        else {
            continue;
        };
        let mut grad_decl = BufferDecl::read_write(&grad_name, next_binding, DataType::F32)
            .with_count(fwd_buf.count());
        if output_set.iter().any(|candidate| candidate == name)
            || input_set.iter().any(|candidate| candidate == name)
        {
            grad_decl = grad_decl.with_pipeline_live_out(true);
        }
        back_buffers.push(grad_decl);
        grad_buf_binding.insert(grad_name, next_binding);
        next_binding += 1;
    }
    (back_buffers, output_set, input_set, grad_source_set)
}

fn grad_buffer_source_names(
    program: &Program,
    outputs: &[String],
    inputs: &[String],
) -> Vec<String> {
    let mut names = Vec::new();
    for name in outputs.iter().chain(inputs) {
        push_unique_string(&mut names, name.as_str());
    }
    for buffer in program.buffers() {
        if buffer.element() == DataType::F32 {
            push_unique_string(&mut names, buffer.name());
        }
    }
    names
}

fn push_unique_string(out: &mut Vec<String>, name: &str) {
    if !out.iter().any(|existing| existing == name) {
        out.push(name.to_string());
    }
}

/// Environment tracking adjoint accumulation for each variable / buffer load.
struct AdjointEnv {
    /// Maps variable name → current adjoint expression accumulator variable name.
    var_adjoints: FxHashMap<String, String>,
    /// Counter for generating fresh adjoint variable names.
    fresh_counter: u32,
    /// Buffers with declared gradient storage in the backward Program.
    grad_buffers: Vec<String>,
    /// Declared element type for every forward buffer.
    buffer_types: FxHashMap<String, DataType>,
    /// Inferred type for forward locals.
    var_types: FxHashMap<String, DataType>,
}

impl AdjointEnv {
    fn new(grad_buffers: &[String], program: &Program) -> Self {
        let mut env = Self {
            var_adjoints: FxHashMap::default(),
            fresh_counter: 0,
            grad_buffers: grad_buffers.to_vec(),
            buffer_types: program
                .buffers()
                .iter()
                .map(|buffer| (buffer.name().to_string(), buffer.element()))
                .collect(),
            var_types: FxHashMap::default(),
        };
        env.record_forward_types(program.entry());
        env
    }

    /// Get or create an accumulator variable for the adjoint of `var_name`.
    fn ensure_adjoint_var(&mut self, var_name: &str) -> String {
        if let Some(existing) = self.var_adjoints.get(var_name) {
            return existing.clone();
        }
        let adj_name = format!("_adj_{var_name}_{}", self.fresh_counter);
        self.fresh_counter += 1;
        self.var_adjoints
            .insert(var_name.to_string(), adj_name.clone());
        adj_name
    }

    /// Get the accumulator variable name for a buffer input, if one was created.
    fn get_accumulator(&self, buf_name: &str) -> Option<String> {
        self.var_adjoints.get(buf_name).cloned()
    }

    fn has_grad_buffer(&self, buf_name: &str) -> bool {
        self.grad_buffers.iter().any(|b| b == buf_name)
    }

    fn buffer_type(&self, buf_name: &str) -> Option<DataType> {
        self.buffer_types.get(buf_name).cloned()
    }

    /// Record the static type of every forward local, in forward source order.
    ///
    /// Descent is [`for_each_node`], the one exhaustive owner of which variants
    /// nest, and its order is the same depth-first source order the recursive
    /// walk this replaces produced, which matters because a later `Let`'s type
    /// is inferred from the types recorded by the earlier ones. Which name a
    /// statement binds, and what it does to it, comes from
    /// [`node_scalars`](crate::visit::node_scalars), so a `Node`
    /// variant that gains a binding position cannot leave a forward local
    /// untyped here and silently lose its adjoint.
    fn record_forward_types(&mut self, nodes: &[Node]) {
        for_each_node(nodes, |node| {
            let scalars = node_scalars(node);
            let Some((binding, name)) = scalars.binding else {
                return;
            };
            match binding {
                NameBinding::Declare | NameBinding::Reassign => {
                    match scalars.operands[0].and_then(|value| expr_type(value, self)) {
                        Some(ty) => {
                            self.var_types.insert(name.as_str().to_string(), ty);
                        }
                        None => {
                            self.var_types.remove(name.as_str());
                        }
                    }
                }
                NameBinding::Induction => {
                    self.var_types
                        .insert(name.as_str().to_string(), DataType::U32);
                }
            }
        });
    }
}

/// Autodiff reads the same answer validation does.
///
/// Type inference itself is [`expr_type`]; only the two free-name lookups are
/// local to the forward pass. Subexpression types are not recorded, so the
/// default [`TypeEnv::on_typed`] applies.
impl TypeEnv for AdjointEnv {
    fn var_type(&self, name: &str) -> Option<DataType> {
        self.var_types.get(name).cloned()
    }

    fn buffer_element(&self, name: &str) -> Option<DataType> {
        self.buffer_types.get(name).cloned()
    }
}

/// Every forward local the backward pass needs an adjoint accumulator for, in
/// first-appearance order.
///
/// Descent is [`for_each_node`], the one exhaustive owner of which variants
/// nest, and the per-node answer is
/// [`node_scalars`](crate::visit::node_scalars), the one exhaustive
/// owner of the scalar namespace. Only the declaring and rebinding forms need
/// an accumulator; [`NameBinding::Induction`] names a loop counter the backward
/// loop re-binds itself. A variant that later gains a binding position fails to
/// compile in `node_scalars` rather than dropping a forward local whose adjoint
/// then reads as zero.
fn collect_adjoint_targets(nodes: &[Node], out: &mut Vec<Ident>) {
    for_each_node(nodes, |node| match node_scalars(node).binding {
        Some((NameBinding::Declare | NameBinding::Reassign, name)) => {
            push_unique_ident(out, name);
        }
        Some((NameBinding::Induction, _)) | None => {}
    });
}

fn push_unique_ident(out: &mut Vec<Ident>, name: &Ident) {
    if !out.iter().any(|existing| existing == name) {
        out.push(name.clone());
    }
}

/// Emit adjoint nodes for a single forward Node.
#[expect(
    clippy::too_many_lines,
    reason = "autodiff node lowering is an exhaustive IR-node dispatch table; splitting it would scatter unsupported-control-flow errors"
)]
fn emit_adjoint_node(
    node: &Node,
    body: &mut Vec<Node>,
    env: &mut AdjointEnv,
    output_set: &[String],
    pullbacks: &mut PullbackMap,
    next_pullback_id: &mut usize,
) -> Result<(), AutodiffError> {
    match node {
        // Forward: let x = value
        // Backward: propagate adjoint of x through value
        Node::Let { name, value } => {
            let var_name = name.as_str();
            let adj_var = env.ensure_adjoint_var(var_name);
            let adj_expr = Expr::Var(adj_var.into());
            insert_pullback(pullbacks, next_pullback_id, adj_expr.clone());
            // Propagate adjoint through the expression tree.
            emit_adjoint_expr(value, &adj_expr, body, env)?;
        }
        // Forward: store buf[idx] = value
        // Backward: adjoint of value comes from grad_buf[idx]
        Node::Store {
            buffer,
            index,
            value,
        } => {
            let buf_name = buffer.as_str();
            let has_grad_buffer =
                output_set.iter().any(|o| o == buf_name) || env.has_grad_buffer(buf_name);
            let grad_buf = format!("grad_{buf_name}");
            let adj_expr = if has_grad_buffer {
                Expr::Load {
                    buffer: grad_buf.clone().into(),
                    index: Box::new(index.clone()),
                }
            } else {
                Expr::f32(0.0)
            };
            insert_pullback(pullbacks, next_pullback_id, adj_expr.clone());
            emit_adjoint_expr(value, &adj_expr, body, env)?;
            if has_grad_buffer {
                body.push(Node::Store {
                    buffer: grad_buf.into(),
                    index: index.clone(),
                    value: Expr::f32(0.0),
                });
            }
        }
        // Forward: x = value (reassignment)
        // Same as Let for adjoint purposes.
        Node::Assign { name, value } => {
            let adj_var = env.ensure_adjoint_var(name.as_str());
            let adj_expr = Expr::Var(adj_var.into());
            insert_pullback(pullbacks, next_pullback_id, adj_expr.clone());
            emit_adjoint_expr(value, &adj_expr, body, env)?;
        }
        // Forward: if cond { then } else { otherwise }
        // Backward: route adjoint through the branch that was taken.
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let mut then_body = Vec::new();
            for n in then.iter().rev() {
                emit_adjoint_node(
                    n,
                    &mut then_body,
                    env,
                    output_set,
                    pullbacks,
                    next_pullback_id,
                )?;
            }
            let mut else_body = Vec::new();
            for n in otherwise.iter().rev() {
                emit_adjoint_node(
                    n,
                    &mut else_body,
                    env,
                    output_set,
                    pullbacks,
                    next_pullback_id,
                )?;
            }
            body.push(Node::If {
                cond: cond.clone(),
                then: then_body,
                otherwise: else_body,
            });
        }
        // Forward: for var in from..to { loop_body }
        // Backward: run the adjoint of loop_body in reverse iteration order.
        Node::Loop {
            var,
            from,
            to,
            body: loop_body,
        } => {
            let mut adj_body = Vec::new();
            for n in loop_body.iter().rev() {
                emit_adjoint_node(
                    n,
                    &mut adj_body,
                    env,
                    output_set,
                    pullbacks,
                    next_pullback_id,
                )?;
            }
            // Reverse iteration: for var in from..to, but with the
            // induction variable remapped to the reversed index so that
            // adj_body[0] corresponds to the *last* forward iteration.
            //
            // reversed_var = (to - 1) - (var - from)
            //              = to - 1 - var + from
            //
            // We keep the loop bounds as from..to (same iteration count)
            // and substitute every occurrence of the original `var` in
            // adj_body with the reversed expression so that the body
            // addresses elements in reverse order at runtime.
            let reversed_index = Expr::sub(
                Expr::sub(to.clone(), Expr::u32(1)),
                Expr::sub(Expr::var(var.as_str()), from.clone()),
            );
            let adj_body =
                crate::transform::subst::substitute_nodes(&adj_body, var, &reversed_index);
            body.push(Node::Loop {
                var: var.clone(),
                from: from.clone(),
                to: to.clone(),
                body: adj_body,
            });
        }
        // Barrier  -  pass through.
        Node::Barrier { ordering } => {
            body.push(Node::barrier_with_ordering(*ordering));
        }
        // Block  -  unwrap and recurse.
        Node::Block(nodes) => {
            for n in nodes.iter().rev() {
                emit_adjoint_node(n, body, env, output_set, pullbacks, next_pullback_id)?;
            }
        }
        // Region  -  recurse into body.
        Node::Region {
            generator,
            source_region,
            body: region_body,
        } => {
            let mut adj_region_body = Vec::new();
            for n in region_body.iter().rev() {
                emit_adjoint_node(
                    n,
                    &mut adj_region_body,
                    env,
                    output_set,
                    pullbacks,
                    next_pullback_id,
                )?;
            }
            body.push(Node::Region {
                generator: generator.clone(),
                source_region: source_region.clone(),
                body: std::sync::Arc::new(adj_region_body),
            });
        }
        // Not differentiable: control flow with no adjoint, buffer-level
        // collectives and async copies, the trap/resume pair, and the opaque
        // extension node whose semantics this crate does not know.
        //
        // One arm, and the kind string is `node_variant_name`, the registry's
        // own name for the variant. The `Debug` rendering this replaces was
        // truncated at 60 characters, so two rejected nodes of the same variant
        // produced two different error strings and one long variant produced a
        // string with no variant name left in it at all.
        Node::Return
        | Node::IndirectDispatch { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncStore { .. }
        | Node::AsyncWait { .. }
        | Node::Trap { .. }
        | Node::Resume { .. }
        | Node::TileLoad { .. }
        | Node::TileStore { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileElementwise { .. }
        | Node::TileDecl { .. }
        | Node::Opaque(_) => {
            return Err(AutodiffError::UnsupportedNode {
                kind: node_variant_name(node).to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/internal/transform/autodiff/grad/mod.rs"]
mod tests;
