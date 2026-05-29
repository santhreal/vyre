//! Rust semantic analysis: name resolution, type inference, borrow checking.
//!
//! Reusable substrate (Tier 3), mirroring `vyre-libs::parsing::c::sema`. The
//! algorithms live here so any consumer can run typed Rust analysis without
//! depending on the `vyre-frontend-rust` driver crate. The driver orchestrates.
//!
//! `resolve` is implemented for the nano-subset. `typeck` and `borrow_check`
//! are not yet wired and return loud, actionable errors rather than a fake
//! success, so a caller never consumes an unchecked module as if it were
//! verified.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use super::parse::{Expr, Module, Stmt, Type};

/// Stable id for a resolved binding (index into [`ResolvedModule::bindings`]).
pub type BindingId = usize;

/// A resolved binding: a function parameter or a `let` declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    /// Recovered identifier text.
    pub name: String,
    /// Whether the binding was declared `mut`.
    pub mutable: bool,
    /// Declared type.
    pub ty: Type,
    /// Source byte offset of the defining identifier.
    pub def_offset: u32,
    /// Index of the enclosing function in `Module::functions`.
    pub function: usize,
}

/// A module with names resolved: a binding table plus a use -> binding map.
#[derive(Debug, Clone)]
pub struct ResolvedModule {
    /// The parsed module (unchanged).
    pub module: Module,
    /// Every parameter and `let` binding, in declaration order.
    pub bindings: Vec<Binding>,
    /// Map from each variable-use source offset to its resolved binding.
    pub uses: HashMap<u32, BindingId>,
}

/// Module after type inference (alias until a distinct typed form exists).
pub type TypedModule = ResolvedModule;
/// Module after borrow checking (alias until a distinct verified form exists).
pub type VerifiedModule = ResolvedModule;

/// Errors from the Rust semantic-analysis stages.
#[derive(Debug, Clone, Error)]
pub enum RustSemaError {
    /// A variable use did not resolve to any in-scope binding (rustc E0425).
    #[error("cannot find value `{name}` in this scope (byte {offset})")]
    UnresolvedName {
        /// The unresolved identifier text.
        name: String,
        /// Source byte offset of the use.
        offset: u32,
    },
    /// A call referenced a function that is not defined (rustc E0425).
    #[error("cannot find function `{name}` in this scope (byte {offset})")]
    UnknownFunction {
        /// The unresolved function name.
        name: String,
        /// Source byte offset of the call.
        offset: u32,
    },
    /// Type inference and checking are not implemented for the nano-subset.
    #[error("type checking is not wired to the Rust nano-subset type environment yet; use parse-only API calls until semantic analysis is enabled")]
    TypeckUnavailable,
    /// Borrow checking is not implemented for the nano-subset.
    #[error("borrow checking is not wired to a Rust CFG yet; disable borrow checking for parse-only pipeline runs")]
    BorrowUnavailable,
}

/// Recover the identifier text that begins at `offset` in `source`.
///
/// The AST stores only the start offset of each identifier, so resolution
/// re-scans the contiguous identifier bytes (`[A-Za-z0-9_]`) from that offset.
fn ident_at(source: &[u8], offset: u32) -> String {
    let start = offset as usize;
    let mut end = start;
    while end < source.len() && (source[end].is_ascii_alphanumeric() || source[end] == b'_') {
        end += 1;
    }
    String::from_utf8_lossy(&source[start..end.min(source.len())]).into_owned()
}

struct Resolver<'a> {
    source: &'a [u8],
    fn_names: &'a HashSet<String>,
    bindings: Vec<Binding>,
    uses: HashMap<u32, BindingId>,
    /// Lexical scope stack of name -> binding id; innermost frame is last.
    scopes: Vec<HashMap<String, BindingId>>,
    function: usize,
}

impl Resolver<'_> {
    fn declare(&mut self, name: String, mutable: bool, ty: Type, def_offset: u32) {
        let id = self.bindings.len();
        self.bindings.push(Binding {
            name: name.clone(),
            mutable,
            ty,
            def_offset,
            function: self.function,
        });
        self.scopes
            .last_mut()
            .expect("Fix: resolver scope stack must be non-empty when declaring a binding")
            .insert(name, id);
    }

    fn lookup(&self, name: &str) -> Option<BindingId> {
        self.scopes.iter().rev().find_map(|frame| frame.get(name).copied())
    }

    fn resolve_expr(&mut self, expr: &Expr) -> Result<(), RustSemaError> {
        match expr {
            Expr::LiteralInt(..) | Expr::LiteralBool(..) => Ok(()),
            Expr::Var(offset) => {
                let name = ident_at(self.source, *offset);
                match self.lookup(&name) {
                    Some(id) => {
                        self.uses.insert(*offset, id);
                        Ok(())
                    }
                    None => Err(RustSemaError::UnresolvedName { name, offset: *offset }),
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.resolve_expr(lhs)?;
                self.resolve_expr(rhs)
            }
            Expr::Borrow { expr, .. } => self.resolve_expr(expr),
            Expr::Deref(inner) => self.resolve_expr(inner),
            Expr::Call { name, args } => {
                let fname = ident_at(self.source, *name);
                if !self.fn_names.contains(&fname) {
                    return Err(RustSemaError::UnknownFunction { name: fname, offset: *name });
                }
                for arg in args {
                    self.resolve_expr(arg)?;
                }
                Ok(())
            }
            Expr::Block(stmts) => {
                self.scopes.push(HashMap::new());
                let result = self.resolve_block(stmts);
                self.scopes.pop();
                result
            }
            Expr::If { cond, then_block, else_block } => {
                self.resolve_expr(cond)?;
                self.resolve_expr(then_block)?;
                if let Some(else_block) = else_block {
                    self.resolve_expr(else_block)?;
                }
                Ok(())
            }
        }
    }

    fn resolve_block(&mut self, stmts: &[Stmt]) -> Result<(), RustSemaError> {
        for stmt in stmts {
            match stmt {
                Stmt::Let { mutable, name, ty, init } => {
                    // The initializer is resolved against the scope as it exists
                    // before this binding, so `let x = x + 1` resolves the RHS
                    // `x` to the outer binding; the new binding then shadows it.
                    self.resolve_expr(init)?;
                    let recovered = ident_at(self.source, *name);
                    self.declare(recovered, *mutable, ty.clone(), *name);
                }
                Stmt::Expr(expr) => self.resolve_expr(expr)?,
                Stmt::Return(Some(expr)) => self.resolve_expr(expr)?,
                Stmt::Return(None) => {}
            }
        }
        Ok(())
    }
}

/// Resolve names in a parsed module against its `source`.
///
/// Recovers identifier text from source offsets, tracks lexical scope
/// (parameters plus block-nested `let`, with shadowing), maps every variable
/// use to its binding, and rejects uses of names not in scope (rustc E0425).
/// Function calls are resolved against the set of module function names
/// (forward references allowed).
///
/// # Errors
/// Returns [`RustSemaError::UnresolvedName`] or [`RustSemaError::UnknownFunction`]
/// for a use with no in-scope definition.
pub fn resolve(module: &Module, source: &[u8]) -> Result<ResolvedModule, RustSemaError> {
    let fn_names: HashSet<String> =
        module.functions.iter().map(|f| ident_at(source, f.name)).collect();

    let mut resolver = Resolver {
        source,
        fn_names: &fn_names,
        bindings: Vec::new(),
        uses: HashMap::new(),
        scopes: Vec::new(),
        function: 0,
    };

    for (index, func) in module.functions.iter().enumerate() {
        resolver.function = index;
        resolver.scopes = vec![HashMap::new()];
        for (offset, ty) in &func.params {
            let name = ident_at(source, *offset);
            resolver.declare(name, false, ty.clone(), *offset);
        }
        resolver.resolve_block(&func.body)?;
    }

    Ok(ResolvedModule {
        module: module.clone(),
        bindings: resolver.bindings,
        uses: resolver.uses,
    })
}

/// Infer and check types for a resolved module.
///
/// # Errors
/// Returns [`RustSemaError::TypeckUnavailable`] until type checking is wired.
pub fn typeck(module: &ResolvedModule) -> Result<TypedModule, RustSemaError> {
    let _ = module;
    Err(RustSemaError::TypeckUnavailable)
}

/// Borrow-check a typed module.
///
/// When implemented this composes the shared dataflow engine over a Rust CFG.
///
/// # Errors
/// Returns [`RustSemaError::BorrowUnavailable`] until borrow checking is wired.
pub fn borrow_check(module: &TypedModule) -> Result<VerifiedModule, RustSemaError> {
    let _ = module;
    Err(RustSemaError::BorrowUnavailable)
}
