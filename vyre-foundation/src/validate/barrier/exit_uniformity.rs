//! Barrier-settled workgroup-uniform exit analysis for V055.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ir_inner::model::expr::{Expr, Ident};
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::types::DataType;
use crate::memory_model::MemoryOrdering;
use crate::validate::binding::Binding;
use crate::validate::uniformity::is_uniform_with_load_policy;

#[derive(Clone)]
struct ExitUniformityState {
    scope: FxHashMap<Ident, Binding>,
    dirty_buffers: FxHashSet<Ident>,
    loads_settled: bool,
    unknown_write: bool,
}

#[derive(Clone, Copy)]
struct ExitProof {
    has_return: bool,
    all_collective: bool,
}

impl ExitProof {
    const NONE: Self = Self {
        has_return: false,
        all_collective: true,
    };

    fn merge(&mut self, other: Self) {
        self.has_return |= other.has_return;
        self.all_collective &= other.all_collective;
    }
}

pub(super) fn exits_after_last_barrier_are_uniform(
    steps: &[&Node],
    scope: &FxHashMap<Ident, Binding>,
) -> bool {
    let Some(last_barrier) = steps
        .iter()
        .rposition(|node| matches!(node, Node::Barrier { .. }))
    else {
        return false;
    };
    let Node::Barrier { ordering } = steps[last_barrier] else {
        return false;
    };
    let mut state = ExitUniformityState {
        scope: scope.clone(),
        dirty_buffers: FxHashSet::default(),
        loads_settled: barrier_acquires(*ordering),
        unknown_write: false,
    };
    let mut proof = ExitProof::NONE;
    for node in &steps[last_barrier + 1..] {
        proof.merge(analyze_exit_node(node, &mut state, true));
    }
    proof.has_return && proof.all_collective
}

fn analyze_exit_sequence(
    nodes: &[Node],
    state: &mut ExitUniformityState,
    path_uniform: bool,
) -> ExitProof {
    let mut proof = ExitProof::NONE;
    for node in nodes {
        proof.merge(analyze_exit_node(node, state, path_uniform));
    }
    proof
}

fn analyze_exit_node(
    node: &Node,
    state: &mut ExitUniformityState,
    path_uniform: bool,
) -> ExitProof {
    match node {
        Node::Let { name, value } => {
            let uniform = exit_expr_is_uniform(value, state);
            invalidate_expr_atomics(value, state);
            state.scope.insert(
                name.clone(),
                Binding {
                    ty: DataType::U32,
                    ty_known: false,
                    mutable: true,
                    uniform,
                },
            );
            ExitProof::NONE
        }
        Node::Assign { name, value } => {
            let uniform = exit_expr_is_uniform(value, state);
            invalidate_expr_atomics(value, state);
            if let Some(binding) = state.scope.get_mut(name.as_str()) {
                binding.uniform = uniform;
            }
            ExitProof::NONE
        }
        Node::Store {
            buffer,
            index,
            value,
        } => {
            invalidate_expr_atomics(index, state);
            invalidate_expr_atomics(value, state);
            state.dirty_buffers.insert(buffer.clone());
            ExitProof::NONE
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let cond_uniform = exit_expr_is_uniform(cond, state);
            invalidate_expr_atomics(cond, state);
            let before = state.clone();
            let mut then_state = before.clone();
            let then_proof =
                analyze_exit_sequence(then, &mut then_state, path_uniform && cond_uniform);
            let mut else_state = before;
            let else_proof =
                analyze_exit_sequence(otherwise, &mut else_state, path_uniform && cond_uniform);
            *state = merge_exit_states(then_state, else_state);
            let mut proof = then_proof;
            proof.merge(else_proof);
            proof
        }
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            let bounds_uniform =
                exit_expr_is_uniform(from, state) && exit_expr_is_uniform(to, state);
            invalidate_expr_atomics(from, state);
            invalidate_expr_atomics(to, state);
            let before = state.clone();
            let mut body_state = before.clone();
            body_state.scope.insert(
                var.clone(),
                Binding {
                    ty: DataType::U32,
                    ty_known: true,
                    mutable: false,
                    uniform: bounds_uniform,
                },
            );
            let proof =
                analyze_exit_sequence(body, &mut body_state, path_uniform && bounds_uniform);
            body_state.scope.remove(var.as_str());
            *state = merge_exit_states(before, body_state);
            proof
        }
        Node::IndirectDispatch { .. } | Node::AsyncWait { .. } | Node::Resume { .. } => {
            ExitProof::NONE
        }
        Node::AsyncLoad {
            destination,
            offset,
            size,
            ..
        }
        | Node::AsyncStore {
            destination,
            offset,
            size,
            ..
        } => {
            invalidate_expr_atomics(offset, state);
            invalidate_expr_atomics(size, state);
            state.dirty_buffers.insert(destination.clone());
            ExitProof::NONE
        }
        Node::Trap { address, .. } => {
            invalidate_expr_atomics(address, state);
            ExitProof::NONE
        }
        Node::AllReduce { buffer, .. } | Node::Broadcast { buffer, .. } => {
            state.dirty_buffers.insert(buffer.clone());
            ExitProof::NONE
        }
        Node::AllGather { output, .. } | Node::ReduceScatter { output, .. } => {
            state.dirty_buffers.insert(output.clone());
            ExitProof::NONE
        }
        Node::Return => ExitProof {
            has_return: true,
            all_collective: path_uniform,
        },
        Node::Barrier { ordering } => {
            if barrier_acquires(*ordering) {
                state.dirty_buffers.clear();
                state.loads_settled = true;
                state.unknown_write = false;
            }
            ExitProof::NONE
        }
        Node::Block(nodes) => analyze_exit_sequence(nodes, state, path_uniform),
        Node::Region { body, .. } => analyze_exit_sequence(body, state, path_uniform),
        Node::Opaque(_) => {
            state.unknown_write = true;
            ExitProof::NONE
        }
    }
}

fn exit_expr_is_uniform(expr: &Expr, state: &ExitUniformityState) -> bool {
    is_uniform_with_load_policy(expr, &state.scope, |buffer| {
        state.loads_settled
            && !state.unknown_write
            && !state.dirty_buffers.contains(buffer.as_str())
    })
}

fn invalidate_expr_atomics(expr: &Expr, state: &mut ExitUniformityState) {
    crate::visit::visit_expr_buffer_accesses(expr, |access, buffer| {
        if access == crate::visit::ExprBufferAccess::Atomic {
            state.dirty_buffers.insert(buffer.clone());
        }
    });
}

fn merge_exit_states(
    mut left: ExitUniformityState,
    right: ExitUniformityState,
) -> ExitUniformityState {
    left.dirty_buffers.extend(right.dirty_buffers);
    left.loads_settled &= right.loads_settled;
    left.unknown_write |= right.unknown_write;
    left.scope.retain(|name, binding| {
        let Some(other) = right.scope.get(name.as_str()) else {
            return false;
        };
        binding.uniform &= other.uniform;
        true
    });
    left
}

fn barrier_acquires(ordering: MemoryOrdering) -> bool {
    matches!(
        ordering,
        MemoryOrdering::Acquire | MemoryOrdering::AcqRel | MemoryOrdering::SeqCst
    )
}
