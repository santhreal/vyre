//! Validation and AST structural footprint estimation for [`ProgramGraph`](super::ProgramGraph).

use super::types::{ProgramGraphError, ShapeDim, ValueContract};
use crate::ir_inner::model::op_signature::BufferAccess;
use crate::ir_inner::model::program::Program;

#[derive(Debug, Clone, Copy)]
pub(super) enum PortRole {
    Input,
    Output,
}

pub(super) fn validate_buffer(
    node: &str,
    program: &Program,
    buffer_name: &str,
    contract: &ValueContract,
    role: PortRole,
) -> Result<(), ProgramGraphError> {
    let buffer = program
        .buffers()
        .iter()
        .find(|buffer| buffer.name() == buffer_name)
        .ok_or_else(|| ProgramGraphError::MissingBuffer {
            node: node.to_string(),
            buffer: buffer_name.to_string(),
        })?;
    if buffer.element() != contract.dtype {
        return Err(ProgramGraphError::BufferContract {
            node: node.to_string(),
            buffer: buffer_name.to_string(),
            reason: format!(
                "Program uses {:?}, graph uses {:?}",
                buffer.element(),
                contract.dtype
            ),
        });
    }
    if let Some(elements) = static_element_count(&contract.shape).map_err(|reason| {
        ProgramGraphError::BufferContract {
            node: node.to_string(),
            buffer: buffer_name.to_string(),
            reason,
        }
    })? {
        if buffer.count() != 0 && elements != u64::from(buffer.count()) {
            return Err(ProgramGraphError::BufferContract {
                node: node.to_string(),
                buffer: buffer_name.to_string(),
                reason: format!(
                    "Program declares {} elements, graph shape requires {elements}",
                    buffer.count()
                ),
            });
        }
    }
    let access_satisfies_contract = match contract.access {
        BufferAccess::ReadOnly => matches!(
            buffer.access(),
            BufferAccess::ReadOnly | BufferAccess::ReadWrite | BufferAccess::Uniform
        ),
        BufferAccess::ReadWrite => buffer.access() == BufferAccess::ReadWrite,
        BufferAccess::WriteOnly => {
            matches!(
                buffer.access(),
                BufferAccess::WriteOnly | BufferAccess::ReadWrite
            )
        }
        BufferAccess::Uniform => buffer.access() == BufferAccess::Uniform,
        _ => false,
    };
    if !access_satisfies_contract {
        return Err(ProgramGraphError::BufferContract {
            node: node.to_string(),
            buffer: buffer_name.to_string(),
            reason: format!(
                "Program access {:?} does not satisfy graph access {:?}",
                buffer.access(),
                contract.access
            ),
        });
    }
    let access = buffer.access();
    let compatible = match role {
        PortRole::Input => matches!(
            access,
            BufferAccess::ReadOnly | BufferAccess::ReadWrite | BufferAccess::Uniform
        ),
        PortRole::Output => matches!(access, BufferAccess::ReadWrite | BufferAccess::WriteOnly),
    };
    if !compatible {
        return Err(ProgramGraphError::BufferContract {
            node: node.to_string(),
            buffer: buffer_name.to_string(),
            reason: format!("{role:?} port cannot use {access:?} access"),
        });
    }
    Ok(())
}

pub(super) fn static_element_count(shape: &[ShapeDim]) -> Result<Option<u64>, String> {
    let mut elements = 1_u64;
    for dimension in shape {
        match dimension {
            ShapeDim::Known(extent) => {
                elements = elements.checked_mul(*extent).ok_or_else(|| {
                    "graph shape element count overflows u64; reduce or shard dimensions"
                        .to_string()
                })?;
            }
            ShapeDim::Unresolved | ShapeDim::Symbol(_) | ShapeDim::Expr(_) => return Ok(None),
        }
    }
    Ok(Some(elements))
}

pub(super) fn estimate_program_bytes(program: &Program) -> usize {
    let base = std::mem::size_of::<Program>();
    let buffer_bytes = program
        .buffers()
        .iter()
        .map(|b| std::mem::size_of::<crate::ir::BufferDecl>() + b.name().len())
        .sum::<usize>();
    let mut node_bytes = program.entry().len() * std::mem::size_of::<crate::ir::Node>();
    for node in program.entry() {
        estimate_node_bytes(node, &mut node_bytes);
    }
    base + buffer_bytes + node_bytes
}

fn estimate_node_bytes(node: &crate::ir::Node, bytes: &mut usize) {
    use crate::ir::Node;
    *bytes += std::mem::size_of::<Node>();
    match node {
        Node::Let { name, value, .. } | Node::Assign { name, value, .. } => {
            *bytes += name.as_str().len();
            estimate_expr_bytes(value, bytes);
        }
        Node::Store {
            buffer,
            index,
            value,
            ..
        } => {
            *bytes += buffer.as_str().len();
            estimate_expr_bytes(index, bytes);
            estimate_expr_bytes(value, bytes);
        }
        Node::If { cond, .. } => estimate_expr_bytes(cond, bytes),
        Node::Loop { from, to, .. } => {
            estimate_expr_bytes(from, bytes);
            estimate_expr_bytes(to, bytes);
        }
        Node::AsyncLoad { offset, size, .. } | Node::AsyncStore { offset, size, .. } => {
            estimate_expr_bytes(offset, bytes);
            estimate_expr_bytes(size, bytes);
        }
        Node::Trap { address, .. } => estimate_expr_bytes(address, bytes),
        Node::Opaque(_) => *bytes += 64,
        _ => {}
    }
    // Nested statements come from the one owner of `Node` child structure, so
    // a nesting variant added to `Node` is measured here rather than counted
    // as its own header and nothing else.
    for body in crate::visit::child_bodies(node) {
        for child in body {
            estimate_node_bytes(child, bytes);
        }
    }
}

fn estimate_expr_bytes(expr: &crate::ir::Expr, bytes: &mut usize) {
    use crate::ir::Expr;
    *bytes += std::mem::size_of::<Expr>();
    match expr {
        Expr::Load { buffer, index, .. } => {
            *bytes += buffer.as_str().len();
            estimate_expr_bytes(index, bytes);
        }
        Expr::BinOp { left, right, .. } => {
            estimate_expr_bytes(left, bytes);
            estimate_expr_bytes(right, bytes);
        }
        Expr::UnOp { operand, .. }
        | Expr::Cast { value: operand, .. }
        | Expr::SubgroupBallot { cond: operand }
        | Expr::SubgroupReduce { value: operand, .. } => {
            estimate_expr_bytes(operand, bytes);
        }
        Expr::Call { args, .. } => {
            for arg in args {
                estimate_expr_bytes(arg, bytes);
            }
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            estimate_expr_bytes(cond, bytes);
            estimate_expr_bytes(true_val, bytes);
            estimate_expr_bytes(false_val, bytes);
        }
        Expr::Fma { a, b, c } => {
            estimate_expr_bytes(a, bytes);
            estimate_expr_bytes(b, bytes);
            estimate_expr_bytes(c, bytes);
        }
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            estimate_expr_bytes(index, bytes);
            if let Some(expected) = expected {
                estimate_expr_bytes(expected, bytes);
            }
            estimate_expr_bytes(value, bytes);
        }
        Expr::SubgroupShuffle { value, lane } => {
            estimate_expr_bytes(value, bytes);
            estimate_expr_bytes(lane, bytes);
        }
        Expr::Var(name) => {
            *bytes += name.as_str().len();
        }
        Expr::BufLen { buffer, .. } => {
            *bytes += buffer.as_str().len();
        }
        Expr::Opaque(_) => {
            *bytes += 64;
        }
        _ => {}
    }
}
