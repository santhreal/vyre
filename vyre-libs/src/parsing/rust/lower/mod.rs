//! Rust AST to Vyre IR lowering (reusable substrate, Tier 3).
//!
//! Mirrors `vyre-libs::parsing::c::lower`: lowering a resolved language AST to a
//! `vyre::ir::Program` is a Tier-3 concern and lives in the library, not the
//! frontend driver. Lowering borrows the AST plus its resolution.
//!
//! ## Model (nano-subset)
//!
//! The last function in the module is the entry kernel. Each `i32`/`bool`
//! parameter becomes a `ReadOnly` input buffer (one element); the return value
//! is stored into a single `out` output buffer. Local bindings are
//! alpha-renamed to `v{binding_id}` so Rust's legal shadowing never trips
//! Vyre's no-shadowing validator (V008). `return e` lowers to
//! `store(out, 0, e)`; `if`/`else` whose arms both return is a terminal
//! statement. Anything outside the wired subset returns a loud
//! [`RustLowerError::Unsupported`] rather than miscompiling.

use std::collections::HashMap;

use thiserror::Error;
use vyre::ir::{BufferDecl, DataType, Expr as IrExpr, Node, Program};

use super::lex::tokens::{EQ, LT, MINUS, PLUS, SLASH, STAR};
use super::parse::{Expr, Module, Stmt, Type};
use super::sema::{BindingId, Resolution};

/// Errors from Rust to Vyre IR lowering.
#[derive(Debug, Clone, Error)]
pub enum RustLowerError {
    /// The module declares no function to use as the entry kernel.
    #[error("Rust lowering needs at least one function to use as the entry kernel")]
    NoEntryFunction,
    /// A construct outside the wired lowering subset was encountered.
    #[error("Rust to Vyre IR lowering does not support {0} yet; not emitting a miscompiled Program")]
    Unsupported(String),
}

/// Lower a resolved module to a Vyre IR program (the last function is the entry).
///
/// # Errors
/// Returns [`RustLowerError::NoEntryFunction`] for an empty module, or
/// [`RustLowerError::Unsupported`] for any construct outside the wired subset.
pub fn lower(module: &Module, resolution: &Resolution) -> Result<Program, RustLowerError> {
    let entry_index = module
        .functions
        .len()
        .checked_sub(1)
        .ok_or(RustLowerError::NoEntryFunction)?;
    let func = &module.functions[entry_index];

    let def_to_id: HashMap<u32, BindingId> = resolution
        .bindings
        .iter()
        .enumerate()
        .filter(|(_, b)| b.function == entry_index)
        .map(|(id, b)| (b.def_offset, id))
        .collect();

    let mut buffers = Vec::with_capacity(func.params.len() + 1);
    let mut entry_nodes = Vec::new();
    for (i, (offset, ty)) in func.params.iter().enumerate() {
        let dtype = scalar_dtype(ty)?;
        let buf = format!("p{i}");
        buffers.push(BufferDecl::read(&buf, i as u32, dtype).with_count(1));
        let binding = def_to_id
            .get(offset)
            .copied()
            .ok_or_else(|| RustLowerError::Unsupported("unresolved parameter".to_string()))?;
        entry_nodes.push(Node::let_bind(
            format!("v{binding}"),
            IrExpr::load(buf, IrExpr::u32(0)),
        ));
    }
    let out_dtype = scalar_dtype(&func.ret)?;
    buffers.push(BufferDecl::output("out", func.params.len() as u32, out_dtype).with_count(1));

    let ctx = LowerCtx { resolution, def_to_id: &def_to_id };
    entry_nodes.extend(ctx.lower_stmts(&func.body)?);
    Ok(Program::wrapped(buffers, [1, 1, 1], entry_nodes))
}

/// The Vyre element type for a nano scalar type (only `i32`/`bool` so far).
fn scalar_dtype(ty: &Type) -> Result<DataType, RustLowerError> {
    match ty {
        Type::I32 => Ok(DataType::I32),
        Type::Bool => Ok(DataType::Bool),
        Type::Unit => Err(RustLowerError::Unsupported("unit-typed parameter or return".to_string())),
        Type::Ref { .. } => Err(RustLowerError::Unsupported("reference-typed parameter or return".to_string())),
    }
}

struct LowerCtx<'a> {
    resolution: &'a Resolution,
    def_to_id: &'a HashMap<u32, BindingId>,
}

impl LowerCtx<'_> {
    fn lower_stmts(&self, stmts: &[Stmt]) -> Result<Vec<Node>, RustLowerError> {
        let mut nodes = Vec::new();
        for stmt in stmts {
            match stmt {
                Stmt::Let { name, init, .. } => {
                    let binding = self.def_to_id.get(name).copied().ok_or_else(|| {
                        RustLowerError::Unsupported("unresolved let binding".to_string())
                    })?;
                    nodes.push(Node::let_bind(format!("v{binding}"), self.lower_expr(init)?));
                }
                Stmt::Return(Some(expr)) => {
                    nodes.push(Node::store("out", IrExpr::u32(0), self.lower_expr(expr)?));
                    return Ok(nodes);
                }
                Stmt::Return(None) => return Ok(nodes),
                Stmt::Expr(Expr::If { cond, then_block, else_block }) => {
                    let then_nodes = self.lower_stmts(block_stmts(then_block))?;
                    let else_nodes = match else_block {
                        Some(block) => self.lower_stmts(block_stmts(block))?,
                        None => Vec::new(),
                    };
                    nodes.push(Node::if_then_else(self.lower_expr(cond)?, then_nodes, else_nodes));
                    let then_div = stmts_diverge(block_stmts(then_block));
                    let else_div = else_block.as_ref().is_some_and(|b| stmts_diverge(block_stmts(b)));
                    if then_div && else_div {
                        return Ok(nodes);
                    }
                }
                // Pure expression statements (the nano-subset has no assignment)
                // have no observable effect; drop them.
                Stmt::Expr(_) => {}
            }
        }
        Ok(nodes)
    }

    fn lower_expr(&self, expr: &Expr) -> Result<IrExpr, RustLowerError> {
        match expr {
            Expr::LiteralInt(_, value) => Ok(IrExpr::i32(*value as i32)),
            Expr::LiteralBool(_, value) => Ok(IrExpr::bool(*value)),
            Expr::Var(offset) => {
                let binding = self.resolution.uses.get(offset).copied().ok_or_else(|| {
                    RustLowerError::Unsupported("unresolved variable use".to_string())
                })?;
                Ok(IrExpr::var(format!("v{binding}")))
            }
            Expr::Binary { op, lhs, rhs } => {
                let l = self.lower_expr(lhs)?;
                let r = self.lower_expr(rhs)?;
                Ok(match *op {
                    PLUS => IrExpr::add(l, r),
                    MINUS => IrExpr::sub(l, r),
                    STAR => IrExpr::mul(l, r),
                    SLASH => IrExpr::div(l, r),
                    EQ => IrExpr::eq(l, r),
                    LT => IrExpr::lt(l, r),
                    other => return Err(RustLowerError::Unsupported(format!("binary operator {other}"))),
                })
            }
            Expr::Borrow { .. } => Err(RustLowerError::Unsupported("borrow expression".to_string())),
            Expr::Deref(_) => Err(RustLowerError::Unsupported("dereference expression".to_string())),
            Expr::Call { .. } => Err(RustLowerError::Unsupported("function call".to_string())),
            Expr::Block(_) | Expr::If { .. } => {
                Err(RustLowerError::Unsupported("block/if used as a value".to_string()))
            }
        }
    }
}

/// Statement list of a block expression (empty for anything else).
fn block_stmts(expr: &Expr) -> &[Stmt] {
    match expr {
        Expr::Block(stmts) => stmts,
        _ => &[],
    }
}

/// Whether a statement sequence returns on every path (so trailing code is
/// unreachable). Mirrors the typeck divergence check.
fn stmts_diverge(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Return(_) => true,
        Stmt::Expr(Expr::If { then_block, else_block: Some(else_block), .. }) => {
            stmts_diverge(block_stmts(then_block)) && stmts_diverge(block_stmts(else_block))
        }
        _ => false,
    })
}
