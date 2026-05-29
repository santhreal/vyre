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

use super::lex::tokens::{ANDAND, EQ, GE, GT, LE, LT, MINUS, NE, OROR, PERCENT, PLUS, SLASH, STAR};
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

    let ctx = LowerCtx { module, resolution, def_to_id: &def_to_id };
    entry_nodes.extend(ctx.lower_stmts(&func.body)?);
    Ok(Program::wrapped(buffers, [1, 1, 1], entry_nodes))
}

/// The Vyre element type for a nano scalar type (only `i32`/`bool` so far).
fn scalar_dtype(ty: &Type) -> Result<DataType, RustLowerError> {
    match ty {
        Type::I32 => Ok(DataType::I32),
        Type::Bool => Ok(DataType::Bool),
        Type::Unit => Err(RustLowerError::Unsupported("unit-typed parameter or return".to_string())),
        // A reference parameter carries its pointee value: in the
        // assignment-free nano-subset a `&T` is a pure read-alias, so it lowers
        // to a buffer of the pointee's element type.
        Type::Ref { inner, .. } => scalar_dtype(inner),
    }
}

struct LowerCtx<'a> {
    module: &'a Module,
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
                    nodes.push(Node::let_bind(format!("v{binding}"), self.lower_value(init, None)?));
                }
                Stmt::Return(Some(expr)) => {
                    nodes.push(Node::store("out", IrExpr::u32(0), self.lower_value(expr, None)?));
                    return Ok(nodes);
                }
                Stmt::Return(None) => return Ok(nodes),
                Stmt::Expr(Expr::If { cond, then_block, else_block }) => {
                    let then_nodes = self.lower_stmts(block_stmts(then_block))?;
                    let else_nodes = match else_block {
                        Some(block) => self.lower_stmts(block_stmts(block))?,
                        None => Vec::new(),
                    };
                    nodes.push(Node::if_then_else(self.lower_value(cond, None)?, then_nodes, else_nodes));
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

    /// Lower a value expression. `subst` is `None` for the entry scope (a
    /// variable lowers to its alpha-renamed local `v{id}`), or `Some(map)` while
    /// inlining a callee (a callee variable lowers to its substituted argument).
    fn lower_value(
        &self,
        expr: &Expr,
        subst: Option<&HashMap<BindingId, IrExpr>>,
    ) -> Result<IrExpr, RustLowerError> {
        match expr {
            Expr::LiteralInt(_, value) => Ok(IrExpr::i32(*value as i32)),
            Expr::LiteralBool(_, value) => Ok(IrExpr::bool(*value)),
            Expr::Var(offset) => {
                let binding = self.resolution.uses.get(offset).copied().ok_or_else(|| {
                    RustLowerError::Unsupported("unresolved variable use".to_string())
                })?;
                match subst {
                    Some(map) => map.get(&binding).cloned().ok_or_else(|| {
                        RustLowerError::Unsupported("callee variable not substituted".to_string())
                    }),
                    None => Ok(IrExpr::var(format!("v{binding}"))),
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let l = self.lower_value(lhs, subst)?;
                let r = self.lower_value(rhs, subst)?;
                Ok(match *op {
                    PLUS => IrExpr::add(l, r),
                    MINUS => IrExpr::sub(l, r),
                    STAR => IrExpr::mul(l, r),
                    SLASH => IrExpr::div(l, r),
                    // Vyre types `Mod`'s result as u32 even for i32 operands;
                    // the value is signed-correct, so cast back to i32 to keep
                    // composition and the i32 store well-typed.
                    PERCENT => IrExpr::cast(DataType::I32, IrExpr::rem(l, r)),
                    EQ => IrExpr::eq(l, r),
                    NE => IrExpr::ne(l, r),
                    LT => IrExpr::lt(l, r),
                    GT => IrExpr::gt(l, r),
                    LE => IrExpr::le(l, r),
                    GE => IrExpr::ge(l, r),
                    ANDAND => IrExpr::and(l, r),
                    OROR => IrExpr::or(l, r),
                    other => return Err(RustLowerError::Unsupported(format!("binary operator {other}"))),
                })
            }
            Expr::Call { name, args } => self.lower_call(name, args, subst),
            // In the assignment-free nano-subset a reference is a pure read
            // alias: `&e` evaluates to e's value and `*r` reads that value back,
            // so borrow and dereference are value-transparent.
            Expr::Borrow { expr, .. } => self.lower_value(expr, subst),
            Expr::Deref(inner) => self.lower_value(inner, subst),
            Expr::Not(inner) => Ok(IrExpr::not(self.lower_value(inner, subst)?)),
            Expr::Block(_) | Expr::If { .. } => {
                Err(RustLowerError::Unsupported("block/if used as a value".to_string()))
            }
        }
    }

    /// Inline a call to a straight-line single-return callee: substitute its
    /// parameters with the (caller-scope) argument expressions, fold its `let`
    /// bindings, and return its lowered return expression. A callee with control
    /// flow or no terminal return is a loud `Unsupported` (never miscompiled).
    fn lower_call(
        &self,
        name: &u32,
        args: &[Expr],
        caller_subst: Option<&HashMap<BindingId, IrExpr>>,
    ) -> Result<IrExpr, RustLowerError> {
        let callee_index = self
            .resolution
            .calls
            .get(name)
            .copied()
            .ok_or_else(|| RustLowerError::Unsupported("unresolved call".to_string()))?;
        let callee = &self.module.functions[callee_index];
        if args.len() != callee.params.len() {
            return Err(RustLowerError::Unsupported("call arity mismatch".to_string()));
        }
        let mut subst: HashMap<BindingId, IrExpr> = HashMap::new();
        for (i, (offset, _)) in callee.params.iter().enumerate() {
            let binding = self.def_to_id.get(offset).copied().ok_or_else(|| {
                RustLowerError::Unsupported("unresolved callee parameter".to_string())
            })?;
            subst.insert(binding, self.lower_value(&args[i], caller_subst)?);
        }
        for stmt in &callee.body {
            match stmt {
                Stmt::Let { name: offset, init, .. } => {
                    let value = self.lower_value(init, Some(&subst))?;
                    let binding = self.def_to_id.get(offset).copied().ok_or_else(|| {
                        RustLowerError::Unsupported("unresolved callee binding".to_string())
                    })?;
                    subst.insert(binding, value);
                }
                Stmt::Return(Some(expr)) => return self.lower_value(expr, Some(&subst)),
                _ => {
                    return Err(RustLowerError::Unsupported(
                        "call to a callee with control flow or no terminal return".to_string(),
                    ))
                }
            }
        }
        Err(RustLowerError::Unsupported("call to a callee with no return".to_string()))
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
