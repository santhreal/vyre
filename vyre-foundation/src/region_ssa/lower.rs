//! Bidirectional lowering between statement-based `Program` and typed `RegionModule` SSA.

use rustc_hash::FxHashMap;
use vyre_spec::{BinOp, DataType};

use crate::ir::{BufferAccess, BufferDecl, Expr, Node, Program};
use crate::visit::child_bodies;
use super::builder::{DominanceError, RegionBuilder};
use super::{
    GlobalDecl, RegionKind, RegionModule, RegionOp, RegionOpKind, ScalarLiteral, ValueId,
};

/// Error produced when lowering between Program and Region SSA.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegionSsaError {
    /// Dominance or typing error during SSA construction.
    #[error("Dominance error in Region SSA: {0}")]
    Dominance(#[from] DominanceError),
    /// Unhandled or unsupported node kind during lowering.
    #[error("Unsupported IR construct during lowering: {0}")]
    Unsupported(String),
    /// Unbound variable reference.
    #[error("Unbound variable in statement IR: {0}")]
    UnboundVariable(String),
}

/// Lower a statement-based [`Program`] into a typed [`RegionModule`].
pub fn lower_program_to_region_ssa(program: &Program) -> Result<RegionModule, RegionSsaError> {
    let mut module = RegionModule::new("lowered_module");

    // Convert BufferDecls to GlobalDecls
    for buf in program.buffers.iter() {
        module.globals.push(GlobalDecl {
            name: buf.name.to_string(),
            ty: buf.element.clone(),
            size_bytes: buf.count as u64,
            binding: Some(buf.binding),
        });
    }

    let mut builder = RegionBuilder::new_function("main", Vec::new(), Vec::new());
    let mut var_map: FxHashMap<String, ValueId> = FxHashMap::default();

    for node in program.entry.iter() {
        lower_node_to_ssa(node, &mut builder, &mut var_map)?;
    }

    builder.terminate_return(Vec::new())?;
    let func = builder.build()?;
    module.add_function(func);

    Ok(module)
}

fn lower_node_to_ssa(
    node: &Node,
    builder: &mut RegionBuilder,
    var_map: &mut FxHashMap<String, ValueId>,
) -> Result<(), RegionSsaError> {
    match node {
        Node::Let { name, value } => {
            let val = lower_expr_to_ssa(value, builder, var_map)?;
            var_map.insert(name.to_string(), val);
        }
        Node::Assign { name, value } => {
            let val = lower_expr_to_ssa(value, builder, var_map)?;
            var_map.insert(name.to_string(), val);
        }
        Node::Store {
            buffer,
            index,
            value,
        } => {
            let idx_val = lower_expr_to_ssa(index, builder, var_map)?;
            let val = lower_expr_to_ssa(value, builder, var_map)?;
            builder.emit_buffer_store(buffer.to_string(), idx_val, val, None)?;
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let _cond_val = lower_expr_to_ssa(cond, builder, var_map)?;
            let mut then_vars = var_map.clone();
            for n in then {
                lower_node_to_ssa(n, builder, &mut then_vars)?;
            }
            if !otherwise.is_empty() {
                let mut else_vars = var_map.clone();
                for n in otherwise {
                    lower_node_to_ssa(n, builder, &mut else_vars)?;
                }
            }
        }
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            let _from_val = lower_expr_to_ssa(from, builder, var_map)?;
            let _to_val = lower_expr_to_ssa(to, builder, var_map)?;

            // Build a structured recurrence region with loop-carried variable
            let iv = builder.emit_constant(ScalarLiteral::U32(0))?;
            let res = builder.build_recurrence_region(
                vec![iv],
                vec![DataType::U32],
                1024,
                |b, inner_iv, _state| {
                    let mut loop_vars = var_map.clone();
                    loop_vars.insert(var.to_string(), inner_iv);
                    for n in body {
                        lower_node_to_ssa(n, b, &mut loop_vars).map_err(|e| match e {
                            RegionSsaError::Dominance(d) => d,
                            _ => DominanceError::InvalidTerminator(e.to_string()),
                        })?;
                    }
                    let one = b.emit_constant(ScalarLiteral::U32(1))?;
                    let next_iv = b.emit_binary(
                        BinOp::Add,
                        inner_iv,
                        one,
                        DataType::U32,
                    )?;
                    Ok(vec![next_iv])
                },
            )?;
            var_map.insert(var.to_string(), res[0]);
        }
        Node::Block(nodes) => {
            for n in nodes {
                lower_node_to_ssa(n, builder, var_map)?;
            }
        }
        Node::Return => {
            builder.terminate_return(Vec::new())?;
        }
        other => {
            // A variant with no SSA of its own still nests statements that
            // have one. Taking the children from `child_bodies` means a
            // nesting variant added to `Node` is lowered here instead of
            // dropped by a catch-all that descends into nothing.
            for body in child_bodies(other) {
                for child in body {
                    lower_node_to_ssa(child, builder, var_map)?;
                }
            }
        }
    }
    Ok(())
}

fn lower_expr_to_ssa(
    expr: &Expr,
    builder: &mut RegionBuilder,
    var_map: &FxHashMap<String, ValueId>,
) -> Result<ValueId, RegionSsaError> {
    match expr {
        Expr::LitU32(v) => Ok(builder.emit_constant(ScalarLiteral::U32(*v))?),
        Expr::LitI32(v) => Ok(builder.emit_constant(ScalarLiteral::I32(*v))?),
        Expr::LitF32(v) => Ok(builder.emit_constant(ScalarLiteral::F32(*v))?),
        Expr::LitBool(v) => Ok(builder.emit_constant(ScalarLiteral::Bool(*v))?),
        Expr::Var(name) => var_map
            .get(name.as_ref())
            .copied()
            .ok_or_else(|| RegionSsaError::UnboundVariable(name.to_string())),
        Expr::BinOp { op, left, right } => {
            let l_val = lower_expr_to_ssa(left, builder, var_map)?;
            let r_val = lower_expr_to_ssa(right, builder, var_map)?;
            Ok(builder.emit_binary(*op, l_val, r_val, DataType::U32)?)
        }
        Expr::UnOp { op, operand } => {
            let in_val = lower_expr_to_ssa(operand, builder, var_map)?;
            Ok(builder.emit_unary(op.clone(), in_val, DataType::U32)?)
        }
        Expr::Cast { target, value } => {
            let in_val = lower_expr_to_ssa(value, builder, var_map)?;
            Ok(builder.emit_cast(in_val, target.clone())?)
        }
        Expr::Load { buffer, index } => {
            let idx_val = lower_expr_to_ssa(index, builder, var_map)?;
            Ok(builder.emit_buffer_load(buffer.to_string(), idx_val, DataType::U32, None)?)
        }
        Expr::InvocationId { axis } => {
            let name = match axis {
                0 => "global_id_x",
                1 => "global_id_y",
                _ => "global_id_z",
            };
            Ok(builder.emit_coordinate_query(name, DataType::U32)?)
        }
        Expr::LocalId { axis } => {
            let name = match axis {
                0 => "local_id_x",
                1 => "local_id_y",
                _ => "local_id_z",
            };
            Ok(builder.emit_coordinate_query(name, DataType::U32)?)
        }
        Expr::WorkgroupId { axis } => {
            let name = match axis {
                0 => "workgroup_id_x",
                1 => "workgroup_id_y",
                _ => "workgroup_id_z",
            };
            Ok(builder.emit_coordinate_query(name, DataType::U32)?)
        }
        _ => {
            // Default constant for other expressions
            Ok(builder.emit_constant(ScalarLiteral::U32(0))?)
        }
    }
}

/// Lower a typed [`RegionModule`] back into an executable statement-based [`Program`].
pub fn lower_region_ssa_to_program(module: &RegionModule) -> Result<Program, RegionSsaError> {
    let mut buffers = Vec::new();
    for g in &module.globals {
        buffers.push(BufferDecl::storage(
            g.name.as_str(),
            g.binding.unwrap_or(0),
            BufferAccess::ReadWrite,
            g.ty.clone(),
        ));
    }

    let mut body = Vec::new();
    if let Some(func) = module.functions.first() {
        let mut val_to_expr: FxHashMap<ValueId, Expr> = FxHashMap::default();
        for blk in &func.blocks {
            for op in &blk.ops {
                lower_ssa_op_to_nodes(op, &mut body, &mut val_to_expr)?;
            }
        }
    }

    Ok(Program::wrapped(buffers, [64, 1, 1], body))
}

fn lower_ssa_op_to_nodes(
    op: &RegionOp,
    body: &mut Vec<Node>,
    val_to_expr: &mut FxHashMap<ValueId, Expr>,
) -> Result<(), RegionSsaError> {
    match &op.kind {
        RegionOpKind::Constant(lit) => {
            let expr = match lit {
                ScalarLiteral::U32(v) => Expr::LitU32(*v),
                ScalarLiteral::I32(v) => Expr::LitI32(*v),
                ScalarLiteral::U64(v) => Expr::LitU32(*v as u32),
                ScalarLiteral::I64(v) => Expr::LitI32(*v as i32),
                ScalarLiteral::F32(v) => Expr::LitF32(*v),
                ScalarLiteral::F64(v) => Expr::LitF32(*v as f32),
                ScalarLiteral::Bool(v) => Expr::LitBool(*v),
            };
            if let Some(res) = op.results.first() {
                val_to_expr.insert(res.id, expr.clone());
                let var_name = format!("v_{}", res.id.0);
                body.push(Node::let_bind(var_name, expr));
            }
        }
        RegionOpKind::Binary { op: binop, left, right } => {
            let l_expr = val_to_expr
                .get(left)
                .cloned()
                .unwrap_or_else(|| Expr::var(format!("v_{}", left.0)));
            let r_expr = val_to_expr
                .get(right)
                .cloned()
                .unwrap_or_else(|| Expr::var(format!("v_{}", right.0)));
            let expr = Expr::BinOp {
                op: *binop,
                left: Box::new(l_expr),
                right: Box::new(r_expr),
            };
            if let Some(res) = op.results.first() {
                val_to_expr.insert(res.id, expr.clone());
                let var_name = format!("v_{}", res.id.0);
                body.push(Node::let_bind(var_name, expr));
            }
        }
        RegionOpKind::Unary { op: unop, input } => {
            let in_expr = val_to_expr
                .get(input)
                .cloned()
                .unwrap_or_else(|| Expr::var(format!("v_{}", input.0)));
            let expr = Expr::UnOp {
                op: unop.clone(),
                operand: Box::new(in_expr),
            };
            if let Some(res) = op.results.first() {
                val_to_expr.insert(res.id, expr.clone());
                let var_name = format!("v_{}", res.id.0);
                body.push(Node::let_bind(var_name, expr));
            }
        }
        RegionOpKind::Cast { input, target_type } => {
            let in_expr = val_to_expr
                .get(input)
                .cloned()
                .unwrap_or_else(|| Expr::var(format!("v_{}", input.0)));
            let expr = Expr::Cast {
                target: target_type.clone(),
                value: Box::new(in_expr),
            };
            if let Some(res) = op.results.first() {
                val_to_expr.insert(res.id, expr.clone());
                let var_name = format!("v_{}", res.id.0);
                body.push(Node::let_bind(var_name, expr));
            }
        }
        RegionOpKind::CoordinateQuery { name } => {
            let expr = if name.contains("global") {
                Expr::gid_x()
            } else if name.contains("local") {
                Expr::LocalId { axis: 0 }
            } else {
                Expr::WorkgroupId { axis: 0 }
            };
            if let Some(res) = op.results.first() {
                val_to_expr.insert(res.id, expr.clone());
                let var_name = format!("v_{}", res.id.0);
                body.push(Node::let_bind(var_name, expr));
            }
        }
        RegionOpKind::BufferLoad { buffer, index, .. } => {
            let idx_expr = val_to_expr
                .get(index)
                .cloned()
                .unwrap_or_else(|| Expr::var(format!("v_{}", index.0)));
            let expr = Expr::load(buffer.as_str(), idx_expr);
            if let Some(res) = op.results.first() {
                val_to_expr.insert(res.id, expr.clone());
                let var_name = format!("v_{}", res.id.0);
                body.push(Node::let_bind(var_name, expr));
            }
        }
        RegionOpKind::BufferStore {
            buffer,
            index,
            value,
            ..
        } => {
            let idx_expr = val_to_expr
                .get(index)
                .cloned()
                .unwrap_or_else(|| Expr::var(format!("v_{}", index.0)));
            let val_expr = val_to_expr
                .get(value)
                .cloned()
                .unwrap_or_else(|| Expr::var(format!("v_{}", value.0)));
            body.push(Node::store(buffer.as_str(), idx_expr, val_expr));
        }
        RegionOpKind::Region(region) => {
            match &region.kind {
                RegionKind::Recurrence { trip_count_bound, .. } => {
                    let mut inner_body = Vec::new();
                    for b in &region.blocks {
                        for inner_op in &b.ops {
                            lower_ssa_op_to_nodes(inner_op, &mut inner_body, val_to_expr)?;
                        }
                    }
                    let var_name = "i";
                    body.push(Node::loop_(
                        var_name,
                        Expr::LitU32(0),
                        Expr::LitU32(*trip_count_bound as u32),
                        inner_body,
                    ));
                }
                _ => {
                    for b in &region.blocks {
                        for inner_op in &b.ops {
                            lower_ssa_op_to_nodes(inner_op, body, val_to_expr)?;
                        }
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}
