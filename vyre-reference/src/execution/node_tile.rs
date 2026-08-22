//! Tile operation evaluation for reference node execution.

use vyre_foundation::ir::{Expr, Ident, Layout, Node, Program, SubgroupReduceOp, Tile};

use crate::execution::expr as eval_expr;
use crate::execution::node::execute_node;
use crate::execution::tile;
use crate::value::Value;
use crate::workgroup::{Invocation, Memory};
use crate::ReferenceError;

pub(crate) fn eval_tile_load(
    tile_name: &str,
    tile_type: &Tile,
    buffer: &str,
    origin: &[Expr],
    layout: &Layout,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), ReferenceError> {
    let mut origin_coords = Vec::with_capacity(origin.len());
    for expr in origin {
        let v = eval_expr::eval(expr, invocation, memory, program)?;
        let coord = v
            .try_as_u32()
            .ok_or_else(|| ReferenceError::new("tile load origin coord must be u32".to_string()))?;
        origin_coords.push(coord);
    }
    let target = eval_expr::buffer(memory, program, buffer)?;
    let elements = tile::load_elements(target, &origin_coords, tile_type, layout);
    invocation.bind(tile_name, Value::Array(elements))
}

pub(crate) fn eval_tile_store(
    buffer: &str,
    origin: &[Expr],
    tile_name: &str,
    invocation: &mut Invocation<'_>,
    memory: &mut Memory,
    program: &Program,
) -> Result<(), ReferenceError> {
    let mut origin_coords = Vec::with_capacity(origin.len());
    for expr in origin {
        let v = eval_expr::eval(expr, invocation, memory, program)?;
        let coord = v.try_as_u32().ok_or_else(|| {
            ReferenceError::new("tile store origin coord must be u32".to_string())
        })?;
        origin_coords.push(coord);
    }
    let tile_val = invocation.local(tile_name).ok_or_else(|| {
        ReferenceError::new(format!(
            "tile `{tile_name}` not found in scope for tile store"
        ))
    })?;
    let elements = match tile_val {
        Value::Array(elems) => elems.clone(),
        single => vec![single.clone()],
    };
    let target = eval_expr::buffer_mut(memory, program, buffer)?;
    tile::store_elements(target, &origin_coords, &elements);
    Ok(())
}

pub(crate) fn eval_tile_matmul(
    acc_name: &str,
    a_name: &str,
    b_name: &str,
    invocation: &mut Invocation<'_>,
) -> Result<(), ReferenceError> {
    let acc_val = invocation
        .local(acc_name)
        .cloned()
        .unwrap_or(Value::Array(Vec::new()));
    let a_val = invocation
        .local(a_name)
        .cloned()
        .ok_or_else(|| ReferenceError::new(format!("tile `{a_name}` not found for matmul")))?;
    let b_val = invocation
        .local(b_name)
        .cloned()
        .ok_or_else(|| ReferenceError::new(format!("tile `{b_name}` not found for matmul")))?;
    let a_elems = tile::to_elements(&a_val);
    let b_elems = tile::to_elements(&b_val);
    let mut acc_elems = tile::to_elements(&acc_val);
    tile::matmul(&mut acc_elems, &a_elems, &b_elems);
    invocation.assign(acc_name, Value::Array(acc_elems))
}

pub(crate) fn eval_tile_reduce(
    out_name: &str,
    tile_name: &str,
    op: SubgroupReduceOp,
    axis: u32,
    invocation: &mut Invocation<'_>,
) -> Result<(), ReferenceError> {
    let tile_val = invocation
        .local(tile_name)
        .cloned()
        .ok_or_else(|| ReferenceError::new(format!("tile `{tile_name}` not found for reduce")))?;
    let elements = match tile_val {
        Value::Array(e) => e,
        s => vec![s],
    };
    let res = tile::reduce(&elements, op, axis);
    invocation.bind(out_name, Value::Array(res))
}

pub(crate) fn eval_tile_elementwise<'a>(
    out_name: &str,
    inputs: &[Ident],
    body: &'a [Node],
    invocation: &mut Invocation<'a>,
    memory: &mut Memory,
    program: &'a Program,
) -> Result<(), ReferenceError> {
    let mut input_arrays = Vec::with_capacity(inputs.len());
    let mut max_len = 0;
    let mut saved_inputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let val = invocation
            .local(input.as_str())
            .cloned()
            .ok_or_else(|| ReferenceError::new(format!("tile input `{input}` not found")))?;
        let elems = match &val {
            Value::Array(e) => e.clone(),
            s => vec![s.clone()],
        };
        max_len = max_len.max(elems.len());
        input_arrays.push(elems);
        saved_inputs.push(val);
    }
    for input in inputs {
        invocation.unbind(input.as_str());
    }
    let mut out_elems = Vec::with_capacity(max_len);
    for idx in 0..max_len {
        invocation.push_scope();
        for (i, input) in inputs.iter().enumerate() {
            let elem = input_arrays[i]
                .get(idx)
                .cloned()
                .unwrap_or(Value::Float(0.0));
            invocation.bind(input.as_str(), elem)?;
        }
        for node in body {
            execute_node(node, invocation, memory, program)?;
        }
        let out_val = invocation
            .local(out_name)
            .cloned()
            .unwrap_or(Value::Float(0.0));
        out_elems.push(out_val);
        invocation.pop_scope();
    }
    for (input, val) in inputs.iter().zip(saved_inputs) {
        let _ = invocation.bind(input.as_str(), val);
    }
    invocation.bind(out_name, Value::Array(out_elems))
}
