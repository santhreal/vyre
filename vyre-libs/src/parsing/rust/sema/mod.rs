//! Rust semantic analysis: name resolution, type checking, borrow checking.
//!
//! Reusable substrate (Tier 3), mirroring `vyre-libs::parsing::c::sema`. The
//! algorithms live here so any consumer can run typed Rust analysis without
//! depending on the `vyre-frontend-rust` driver crate. The driver orchestrates.
//!
//! Analyses are side tables over the borrowed AST: a pass is
//! `pass(&Module, &Resolution) -> Result<...>` and never clones the AST. This
//! keeps the pipeline allocation-lean and extends cleanly as the compiler grows
//! (new passes add new side tables, not new copies of the program).
//!
//! `resolve` and `typeck` are implemented for the nano-subset. `borrow_check`
//! implements the mutability rule (E0596) and reports the conflicting-borrow
//! rules (E0499/E0502) as not-yet-wired rather than faking a complete pass.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use super::lex::tokens::{EQ, LT, MINUS, PLUS, SLASH, STAR};
use super::parse::{Expr, Module, Stmt, Type};

/// Stable id for a resolved binding (index into [`Resolution::bindings`]).
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

/// Name-resolution result: a binding table plus a use -> binding map over the
/// AST. Holds no copy of the AST; analyses borrow the `Module` alongside it.
#[derive(Debug, Clone, Default)]
pub struct Resolution {
    /// Every parameter and `let` binding, in declaration order.
    pub bindings: Vec<Binding>,
    /// Map from each variable-use source offset to its resolved binding.
    pub uses: HashMap<u32, BindingId>,
}

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
    /// A `&mut` borrow targeted an immutable place (rustc E0596).
    #[error("cannot borrow `{name}` as mutable, as it is not declared as mutable (byte {offset})")]
    CannotBorrowImmutableAsMutable {
        /// The immutable binding being borrowed mutably.
        name: String,
        /// Source byte offset of the borrowed place.
        offset: u32,
    },
    /// A function returned a reference to a call-local value (rustc E0597).
    #[error("cannot return a reference to a local value; it does not live long enough (byte {offset})")]
    ReturnsReferenceToLocal {
        /// Source byte offset of the offending place.
        offset: u32,
    },
    /// An expression's type did not match the expected type (rustc E0308).
    #[error("mismatched types in {context}: expected `{expected}`, found `{found}`")]
    TypeMismatch {
        /// Where the mismatch occurred (let binding, return, operand, argument).
        context: String,
        /// The expected type.
        expected: String,
        /// The type actually found.
        found: String,
    },
    /// A dereference was applied to a non-reference type (rustc E0614).
    #[error("type `{found}` cannot be dereferenced; only references can")]
    CannotDeref {
        /// The non-reference type.
        found: String,
    },
    /// An `if` condition was not `bool` (rustc E0308).
    #[error("`if` condition must be `bool`, found `{found}`")]
    NonBooleanCondition {
        /// The non-boolean condition type.
        found: String,
    },
    /// A call passed the wrong number of arguments (rustc E0061).
    #[error("function `{function}` expects {expected} argument(s), found {found}")]
    ArgCountMismatch {
        /// The called function name.
        function: String,
        /// The declared parameter count.
        expected: usize,
        /// The supplied argument count.
        found: usize,
    },
    /// A non-unit function body does not return on all paths (rustc E0308).
    #[error("function `{function}` must return `{expected}` on all paths")]
    MissingReturn {
        /// The function name.
        function: String,
        /// The declared return type.
        expected: String,
    },
    /// Borrow checking is incomplete: the conflicting-borrow rules require weir.
    #[error("borrow checking is incomplete: the conflicting-borrow rules (E0499/E0502) require the weir dataflow analysis and are not yet wired")]
    BorrowUnavailable,
}

/// Recover the identifier text that begins at `offset` in `source`.
///
/// The AST stores only the start offset of each identifier, so resolution and
/// type checking re-scan the contiguous identifier bytes (`[A-Za-z0-9_]`).
fn ident_at(source: &[u8], offset: u32) -> String {
    let start = (offset as usize).min(source.len());
    let mut end = start;
    while end < source.len() && (source[end].is_ascii_alphanumeric() || source[end] == b'_') {
        end += 1;
    }
    String::from_utf8_lossy(&source[start..end]).into_owned()
}

/// Render a type for diagnostics, matching Rust surface syntax.
fn type_str(ty: &Type) -> String {
    match ty {
        Type::I32 => "i32".to_string(),
        Type::Bool => "bool".to_string(),
        Type::Unit => "()".to_string(),
        Type::Ref { mutable, inner } => {
            format!("&{}{}", if *mutable { "mut " } else { "" }, type_str(inner))
        }
    }
}

// ----------------------------------------------------------------------------
// Name resolution
// ----------------------------------------------------------------------------

struct Resolver<'a> {
    source: &'a [u8],
    fn_names: &'a HashSet<String>,
    bindings: Vec<Binding>,
    uses: HashMap<u32, BindingId>,
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
///
/// # Errors
/// Returns [`RustSemaError::UnresolvedName`] or [`RustSemaError::UnknownFunction`]
/// for a use with no in-scope definition.
pub fn resolve(module: &Module, source: &[u8]) -> Result<Resolution, RustSemaError> {
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

    Ok(Resolution { bindings: resolver.bindings, uses: resolver.uses })
}

// ----------------------------------------------------------------------------
// Type checking
// ----------------------------------------------------------------------------

struct FnSig {
    params: Vec<Type>,
    ret: Type,
}

struct TypeCk<'a> {
    source: &'a [u8],
    resolution: &'a Resolution,
    sigs: &'a HashMap<String, FnSig>,
    ret: &'a Type,
}

impl TypeCk<'_> {
    fn type_of(&self, expr: &Expr) -> Result<Type, RustSemaError> {
        match expr {
            Expr::LiteralInt(..) => Ok(Type::I32),
            Expr::LiteralBool(..) => Ok(Type::Bool),
            Expr::Var(offset) => {
                let id = *self
                    .resolution
                    .uses
                    .get(offset)
                    .expect("Fix: resolve must record every variable use before typeck runs");
                Ok(self.resolution.bindings[id].ty.clone())
            }
            Expr::Binary { op, lhs, rhs } => {
                let lt = self.type_of(lhs)?;
                let rt = self.type_of(rhs)?;
                match *op {
                    PLUS | MINUS | STAR | SLASH => {
                        self.require(&lt, &Type::I32, "arithmetic operand")?;
                        self.require(&rt, &Type::I32, "arithmetic operand")?;
                        Ok(Type::I32)
                    }
                    LT => {
                        self.require(&lt, &Type::I32, "comparison operand")?;
                        self.require(&rt, &Type::I32, "comparison operand")?;
                        Ok(Type::Bool)
                    }
                    EQ => {
                        if lt != rt {
                            return Err(RustSemaError::TypeMismatch {
                                context: "equality operands".to_string(),
                                expected: type_str(&lt),
                                found: type_str(&rt),
                            });
                        }
                        Ok(Type::Bool)
                    }
                    _ => Ok(Type::I32),
                }
            }
            Expr::Borrow { mutable, expr } => {
                let inner = self.type_of(expr)?;
                Ok(Type::Ref { mutable: *mutable, inner: Box::new(inner) })
            }
            Expr::Deref(inner) => match self.type_of(inner)? {
                Type::Ref { inner, .. } => Ok(*inner),
                other => Err(RustSemaError::CannotDeref { found: type_str(&other) }),
            },
            Expr::Call { name, args } => {
                let fname = ident_at(self.source, *name);
                let sig = self.sigs.get(&fname).ok_or(RustSemaError::UnknownFunction {
                    name: fname.clone(),
                    offset: *name,
                })?;
                if args.len() != sig.params.len() {
                    return Err(RustSemaError::ArgCountMismatch {
                        function: fname,
                        expected: sig.params.len(),
                        found: args.len(),
                    });
                }
                for (arg, param_ty) in args.iter().zip(&sig.params) {
                    let at = self.type_of(arg)?;
                    self.require(&at, param_ty, "function argument")?;
                }
                Ok(sig.ret.clone())
            }
            Expr::Block(stmts) => {
                self.check_block(stmts)?;
                Ok(Type::Unit)
            }
            Expr::If { cond, then_block, else_block } => {
                let ct = self.type_of(cond)?;
                if ct != Type::Bool {
                    return Err(RustSemaError::NonBooleanCondition { found: type_str(&ct) });
                }
                let tt = self.type_of(then_block)?;
                let et = match else_block {
                    Some(else_block) => self.type_of(else_block)?,
                    None => Type::Unit,
                };
                if tt != et {
                    return Err(RustSemaError::TypeMismatch {
                        context: "if/else branches".to_string(),
                        expected: type_str(&tt),
                        found: type_str(&et),
                    });
                }
                Ok(tt)
            }
        }
    }

    fn require(&self, found: &Type, expected: &Type, context: &str) -> Result<(), RustSemaError> {
        if found == expected {
            Ok(())
        } else {
            Err(RustSemaError::TypeMismatch {
                context: context.to_string(),
                expected: type_str(expected),
                found: type_str(found),
            })
        }
    }

    fn check_block(&self, stmts: &[Stmt]) -> Result<(), RustSemaError> {
        for stmt in stmts {
            match stmt {
                Stmt::Let { ty, init, .. } => {
                    let it = self.type_of(init)?;
                    self.require(&it, ty, "let binding")?;
                }
                Stmt::Expr(expr) => {
                    self.type_of(expr)?;
                }
                Stmt::Return(Some(expr)) => {
                    let rt = self.type_of(expr)?;
                    self.require(&rt, self.ret, "return value")?;
                }
                Stmt::Return(None) => {
                    self.require(&Type::Unit, self.ret, "return value")?;
                }
            }
        }
        Ok(())
    }
}

/// Type-check a resolved module against its `source` (rustc E0308 / E0061 / E0614).
///
/// Checks `let` initializer types, return-value types, binary-operator operand
/// types, dereference of references only, boolean `if` conditions, call arity
/// and argument types, and that a non-unit function returns on all paths.
///
/// # Errors
/// Returns the matching [`RustSemaError`] variant on a type error.
pub fn typeck(module: &Module, source: &[u8], resolution: &Resolution) -> Result<(), RustSemaError> {
    let sigs: HashMap<String, FnSig> = module
        .functions
        .iter()
        .map(|f| {
            (
                ident_at(source, f.name),
                FnSig {
                    params: f.params.iter().map(|(_, t)| t.clone()).collect(),
                    ret: f.ret.clone(),
                },
            )
        })
        .collect();

    for func in &module.functions {
        let ck = TypeCk { source, resolution, sigs: &sigs, ret: &func.ret };
        ck.check_block(&func.body)?;
        if func.ret != Type::Unit && !block_diverges(&func.body) {
            return Err(RustSemaError::MissingReturn {
                function: ident_at(source, func.name),
                expected: type_str(&func.ret),
            });
        }
    }
    Ok(())
}

/// Whether a statement sequence is guaranteed to return on all paths.
fn block_diverges(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_diverges)
}

fn stmt_diverges(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Return(_) => true,
        Stmt::Expr(expr) => expr_diverges(expr),
        Stmt::Let { init, .. } => expr_diverges(init),
    }
}

fn expr_diverges(expr: &Expr) -> bool {
    match expr {
        Expr::Block(stmts) => block_diverges(stmts),
        Expr::If { then_block, else_block: Some(else_block), .. } => {
            expr_diverges(then_block) && expr_diverges(else_block)
        }
        _ => false,
    }
}

// ----------------------------------------------------------------------------
// Borrow checking
// ----------------------------------------------------------------------------

/// Borrow-check a resolved module.
///
/// Implemented rules: mutability (rustc E0596) via [`check_mutability`], and
/// dangling-reference / escape (rustc E0597) via [`check_escape`]. The
/// conflicting-borrow rules (E0499/E0502) require liveness from the weir
/// dataflow analysis and are not yet wired, so `borrow_check` surfaces a real
/// mutability or escape error when one exists and otherwise reports
/// [`RustSemaError::BorrowUnavailable`] rather than claiming a complete borrow
/// check it has not performed.
///
/// # Errors
/// Returns [`RustSemaError::CannotBorrowImmutableAsMutable`] (E0596) or
/// [`RustSemaError::ReturnsReferenceToLocal`] (E0597) for a violation, or
/// [`RustSemaError::BorrowUnavailable`] while the conflicting-borrow rules
/// remain unimplemented.
pub fn borrow_check(module: &Module, resolution: &Resolution) -> Result<(), RustSemaError> {
    check_mutability(module, resolution)?;
    check_escape(module, resolution)?;
    Err(RustSemaError::BorrowUnavailable)
}

/// Check the mutability borrow rule (rustc E0596) over a resolved module.
///
/// A `&mut` borrow is rejected when the borrowed place is an immutable binding
/// or a dereference of a shared (`&T`) reference. Borrowing a temporary is
/// allowed.
///
/// # Errors
/// Returns [`RustSemaError::CannotBorrowImmutableAsMutable`] on a violation.
pub fn check_mutability(module: &Module, resolution: &Resolution) -> Result<(), RustSemaError> {
    for func in &module.functions {
        check_mut_stmts(&func.body, resolution)?;
    }
    Ok(())
}

fn check_mut_stmts(stmts: &[Stmt], resolution: &Resolution) -> Result<(), RustSemaError> {
    for stmt in stmts {
        match stmt {
            Stmt::Let { init, .. } => check_mut_expr(init, resolution)?,
            Stmt::Expr(expr) => check_mut_expr(expr, resolution)?,
            Stmt::Return(Some(expr)) => check_mut_expr(expr, resolution)?,
            Stmt::Return(None) => {}
        }
    }
    Ok(())
}

fn check_mut_expr(expr: &Expr, resolution: &Resolution) -> Result<(), RustSemaError> {
    match expr {
        Expr::Borrow { mutable, expr } => {
            if *mutable {
                check_mutable_place(expr, resolution)?;
            }
            check_mut_expr(expr, resolution)
        }
        Expr::Binary { lhs, rhs, .. } => {
            check_mut_expr(lhs, resolution)?;
            check_mut_expr(rhs, resolution)
        }
        Expr::Deref(inner) => check_mut_expr(inner, resolution),
        Expr::Call { args, .. } => {
            for arg in args {
                check_mut_expr(arg, resolution)?;
            }
            Ok(())
        }
        Expr::Block(stmts) => check_mut_stmts(stmts, resolution),
        Expr::If { cond, then_block, else_block } => {
            check_mut_expr(cond, resolution)?;
            check_mut_expr(then_block, resolution)?;
            if let Some(else_block) = else_block {
                check_mut_expr(else_block, resolution)?;
            }
            Ok(())
        }
        Expr::LiteralInt(..) | Expr::LiteralBool(..) | Expr::Var(..) => Ok(()),
    }
}

/// Verify that `place` denotes a mutable place for a `&mut` borrow.
fn check_mutable_place(place: &Expr, resolution: &Resolution) -> Result<(), RustSemaError> {
    match place {
        Expr::Var(offset) => {
            if let Some(&id) = resolution.uses.get(offset) {
                let binding = &resolution.bindings[id];
                if !binding.mutable {
                    return Err(RustSemaError::CannotBorrowImmutableAsMutable {
                        name: binding.name.clone(),
                        offset: *offset,
                    });
                }
            }
            Ok(())
        }
        Expr::Deref(inner) => {
            if let Expr::Var(offset) = inner.as_ref() {
                if let Some(&id) = resolution.uses.get(offset) {
                    let binding = &resolution.bindings[id];
                    if let Type::Ref { mutable: false, .. } = binding.ty {
                        return Err(RustSemaError::CannotBorrowImmutableAsMutable {
                            name: binding.name.clone(),
                            offset: *offset,
                        });
                    }
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Check that no function returns a reference to a call-local value (rustc E0597).
///
/// A reference escapes only if it borrows a binding's storage (`&x`, for any
/// local binding) or transitively points at such a borrow. A reference derived
/// from a `&T` parameter's pointee (`return r`, `&*r`) is allowed. The check
/// never rejects a reference whose provenance it cannot prove local, so it never
/// diverges from rustc on acceptance.
///
/// # Errors
/// Returns [`RustSemaError::ReturnsReferenceToLocal`] when a returned reference
/// provably borrows a local value.
pub fn check_escape(module: &Module, resolution: &Resolution) -> Result<(), RustSemaError> {
    let def_to_id: HashMap<u32, BindingId> = resolution
        .bindings
        .iter()
        .enumerate()
        .map(|(id, b)| (b.def_offset, id))
        .collect();

    for func in &module.functions {
        let returns_ref = matches!(func.ret, Type::Ref { .. });
        // A reference's borrows-local provenance: true if the reference value
        // ultimately points at a call-local binding's storage. Parameters point
        // outside the call, so they start false.
        let mut borrows_local: HashMap<BindingId, bool> = HashMap::new();
        for (offset, _ty) in &func.params {
            if let Some(&id) = def_to_id.get(offset) {
                borrows_local.insert(id, false);
            }
        }
        walk_escape(&func.body, returns_ref, &def_to_id, resolution, &mut borrows_local)?;
    }
    Ok(())
}

fn walk_escape(
    stmts: &[Stmt],
    returns_ref: bool,
    def_to_id: &HashMap<u32, BindingId>,
    resolution: &Resolution,
    borrows_local: &mut HashMap<BindingId, bool>,
) -> Result<(), RustSemaError> {
    for stmt in stmts {
        match stmt {
            Stmt::Let { name, ty, init, .. } => {
                if let Some(&id) = def_to_id.get(name) {
                    let escapes = matches!(ty, Type::Ref { .. })
                        && escapes_offset(init, resolution, borrows_local).is_some();
                    borrows_local.insert(id, escapes);
                }
                descend_escape(init, returns_ref, def_to_id, resolution, borrows_local)?;
            }
            Stmt::Expr(expr) => {
                descend_escape(expr, returns_ref, def_to_id, resolution, borrows_local)?;
            }
            Stmt::Return(Some(expr)) => {
                if returns_ref {
                    if let Some(offset) = escapes_offset(expr, resolution, borrows_local) {
                        return Err(RustSemaError::ReturnsReferenceToLocal { offset });
                    }
                }
                descend_escape(expr, returns_ref, def_to_id, resolution, borrows_local)?;
            }
            Stmt::Return(None) => {}
        }
    }
    Ok(())
}

fn descend_escape(
    expr: &Expr,
    returns_ref: bool,
    def_to_id: &HashMap<u32, BindingId>,
    resolution: &Resolution,
    borrows_local: &mut HashMap<BindingId, bool>,
) -> Result<(), RustSemaError> {
    match expr {
        Expr::Block(stmts) => {
            walk_escape(stmts, returns_ref, def_to_id, resolution, borrows_local)
        }
        Expr::If { cond, then_block, else_block } => {
            descend_escape(cond, returns_ref, def_to_id, resolution, borrows_local)?;
            descend_escape(then_block, returns_ref, def_to_id, resolution, borrows_local)?;
            if let Some(else_block) = else_block {
                descend_escape(else_block, returns_ref, def_to_id, resolution, borrows_local)?;
            }
            Ok(())
        }
        Expr::Binary { lhs, rhs, .. } => {
            descend_escape(lhs, returns_ref, def_to_id, resolution, borrows_local)?;
            descend_escape(rhs, returns_ref, def_to_id, resolution, borrows_local)
        }
        Expr::Borrow { expr, .. } => {
            descend_escape(expr, returns_ref, def_to_id, resolution, borrows_local)
        }
        Expr::Deref(inner) => {
            descend_escape(inner, returns_ref, def_to_id, resolution, borrows_local)
        }
        Expr::Call { args, .. } => {
            for arg in args {
                descend_escape(arg, returns_ref, def_to_id, resolution, borrows_local)?;
            }
            Ok(())
        }
        Expr::Var(..) | Expr::LiteralInt(..) | Expr::LiteralBool(..) => Ok(()),
    }
}

/// If the reference value `expr` provably borrows a call-local binding, return
/// the source offset of the offending place; otherwise `None`.
fn escapes_offset(
    expr: &Expr,
    resolution: &Resolution,
    borrows_local: &HashMap<BindingId, bool>,
) -> Option<u32> {
    match expr {
        Expr::Borrow { expr, .. } => match expr.as_ref() {
            // `&x` borrows the storage of a call-local binding; it escapes.
            Expr::Var(offset) => Some(*offset),
            // `&*r` points at r's pointee; it escapes iff that pointee is local.
            Expr::Deref(inner) => {
                if let Expr::Var(offset) = inner.as_ref() {
                    let id = resolution.uses.get(offset)?;
                    if *borrows_local.get(id).unwrap_or(&false) {
                        Some(*offset)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        },
        // Returning a reference binding escapes iff it holds a local borrow.
        Expr::Var(offset) => {
            let id = resolution.uses.get(offset)?;
            if *borrows_local.get(id).unwrap_or(&false) {
                Some(*offset)
            } else {
                None
            }
        }
        _ => None,
    }
}
