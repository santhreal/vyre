//! AST expression and mutation extraction helpers.

use std::collections::BTreeSet;

use super::host_oracle_elimination_records::assigns_left_operand;

pub(super) fn extract_pat_bindings(pat: &syn::Pat, out: &mut BTreeSet<String>) {
    match pat {
        syn::Pat::Ident(pi) => {
            out.insert(pi.ident.to_string());
            if let Some((_, subpat)) = &pi.subpat {
                extract_pat_bindings(subpat, out);
            }
        }
        syn::Pat::Tuple(pt) => {
            for elem in &pt.elems {
                extract_pat_bindings(elem, out);
            }
        }
        syn::Pat::TupleStruct(pts) => {
            for elem in &pts.elems {
                extract_pat_bindings(elem, out);
            }
        }
        syn::Pat::Struct(ps) => {
            for field in &ps.fields {
                extract_pat_bindings(&field.pat, out);
            }
        }
        syn::Pat::Slice(ps) => {
            for elem in &ps.elems {
                extract_pat_bindings(elem, out);
            }
        }
        syn::Pat::Reference(pr) => {
            extract_pat_bindings(&pr.pat, out);
        }
        syn::Pat::Type(pt) => {
            extract_pat_bindings(&pt.pat, out);
        }
        syn::Pat::Paren(pp) => {
            extract_pat_bindings(&pp.pat, out);
        }
        syn::Pat::Or(po) => {
            for case in &po.cases {
                extract_pat_bindings(case, out);
            }
        }
        _ => {}
    }
}

pub(super) fn extract_root_ident_from_expr(expr: &syn::Expr, out: &mut BTreeSet<String>) {
    match expr {
        syn::Expr::Path(p) => {
            if let Some(ident) = p.path.get_ident() {
                out.insert(ident.to_string());
            } else if let Some(seg) = p.path.segments.first() {
                out.insert(seg.ident.to_string());
            }
        }
        syn::Expr::Field(f) => {
            extract_root_ident_from_expr(&f.base, out);
            let field_str = quote::quote!(#expr).to_string().replace(' ', "");
            out.insert(field_str);
        }
        syn::Expr::MethodCall(mc) => {
            extract_root_ident_from_expr(&mc.receiver, out);
        }
        syn::Expr::Index(idx) => {
            extract_root_ident_from_expr(&idx.expr, out);
        }
        syn::Expr::Reference(r) => {
            extract_root_ident_from_expr(&r.expr, out);
        }
        syn::Expr::Paren(p) => {
            extract_root_ident_from_expr(&p.expr, out);
        }
        syn::Expr::Unary(u) => {
            extract_root_ident_from_expr(&u.expr, out);
        }
        syn::Expr::Try(t) => {
            extract_root_ident_from_expr(&t.expr, out);
        }
        _ => {}
    }
}

#[derive(Default)]
pub(super) struct IdentReadCollector {
    pub(super) idents: BTreeSet<String>,
}

impl<'ast> syn::visit::Visit<'ast> for IdentReadCollector {
    fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
        if let Some(ident) = p.path.get_ident() {
            self.idents.insert(ident.to_string());
        } else if let Some(seg) = p.path.segments.first() {
            self.idents.insert(seg.ident.to_string());
        }
        syn::visit::visit_expr_path(self, p);
    }

    fn visit_expr_field(&mut self, f: &'ast syn::ExprField) {
        extract_root_ident_from_expr(&f.base, &mut self.idents);
        let field_str = quote::quote!(#f).to_string().replace(' ', "");
        self.idents.insert(field_str);
        syn::visit::visit_expr_field(self, f);
    }
}

pub(super) fn extract_read_idents_from_stmt(stmt: &syn::Stmt) -> BTreeSet<String> {
    let mut collector = IdentReadCollector::default();
    syn::visit::visit_stmt(&mut collector, stmt);
    collector.idents
}

pub(super) fn extract_read_idents_from_expr(expr: &syn::Expr) -> BTreeSet<String> {
    let mut collector = IdentReadCollector::default();
    syn::visit::visit_expr(&mut collector, expr);
    collector.idents
}

#[derive(Default)]
pub(super) struct MutatedStorageCollector {
    pub(super) mutated: BTreeSet<String>,
}

impl<'ast> syn::visit::Visit<'ast> for MutatedStorageCollector {
    fn visit_stmt(&mut self, stmt: &'ast syn::Stmt) {
        if let syn::Stmt::Local(local) = stmt {
            extract_pat_bindings(&local.pat, &mut self.mutated);
        }
        if let syn::Stmt::Expr(syn::Expr::MethodCall(call), Some(_)) = stmt {
            extract_root_ident_from_expr(&call.receiver, &mut self.mutated);
        }
        syn::visit::visit_stmt(self, stmt);
    }

    fn visit_expr_assign(&mut self, expr: &'ast syn::ExprAssign) {
        extract_root_ident_from_expr(&expr.left, &mut self.mutated);
        syn::visit::visit_expr_assign(self, expr);
    }

    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        if assigns_left_operand(&expr.op) {
            extract_root_ident_from_expr(&expr.left, &mut self.mutated);
        }
        syn::visit::visit_expr_binary(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        let mname = expr.method.to_string();
        if matches!(
            mname.as_str(),
            "push" | "extend" | "extend_from_slice" | "insert" | "append" | "clear"
        ) {
            extract_root_ident_from_expr(&expr.receiver, &mut self.mutated);
        }
        for arg in &expr.args {
            if let syn::Expr::Reference(r) = arg {
                if r.mutability.is_some() {
                    extract_root_ident_from_expr(&r.expr, &mut self.mutated);
                }
            }
        }
        syn::visit::visit_expr_method_call(self, expr);
    }

    fn visit_expr_call(&mut self, expr: &'ast syn::ExprCall) {
        for arg in &expr.args {
            if let syn::Expr::Reference(r) = arg {
                if r.mutability.is_some() {
                    extract_root_ident_from_expr(&r.expr, &mut self.mutated);
                }
            } else {
                extract_root_ident_from_expr(arg, &mut self.mutated);
            }
        }
        syn::visit::visit_expr_call(self, expr);
    }
}

pub(super) fn extract_mutated_storage_from_stmt(stmt: &syn::Stmt) -> BTreeSet<String> {
    let mut collector = MutatedStorageCollector::default();
    if let syn::Stmt::Expr(syn::Expr::MethodCall(call), _) = stmt {
        extract_root_ident_from_expr(&call.receiver, &mut collector.mutated);
    }
    syn::visit::visit_stmt(&mut collector, stmt);
    collector.mutated
}

pub(super) fn is_reduction_or_arithmetic_method(method_name: &str) -> bool {
    matches!(
        method_name,
        "any"
            | "all"
            | "sum"
            | "fold"
            | "rfold"
            | "reduce"
            | "count"
            | "count_ones"
            | "count_zeros"
            | "leading_zeros"
            | "trailing_zeros"
            | "rotate_left"
            | "rotate_right"
            | "reverse_bits"
            | "wrapping_add"
            | "wrapping_sub"
            | "wrapping_mul"
            | "wrapping_div"
            | "wrapping_rem"
            | "saturating_add"
            | "saturating_sub"
            | "saturating_mul"
            | "overflowing_add"
            | "overflowing_sub"
            | "overflowing_mul"
            | "checked_add"
            | "checked_sub"
            | "checked_mul"
            | "checked_div"
    )
}

pub(super) fn is_matching_width_type(type_ident: &str, chunk_len: u32) -> bool {
    match chunk_len {
        1 => matches!(type_ident, "u8" | "i8"),
        2 => matches!(type_ident, "u16" | "i16"),
        4 => matches!(type_ident, "u32" | "i32" | "f32"),
        8 => matches!(type_ident, "u64" | "i64" | "f64"),
        16 => matches!(type_ident, "u128" | "i128"),
        _ => false,
    }
}

pub(super) fn is_exact_chunk_receiver(expr: &syn::Expr, chunk_ident: &str) -> bool {
    match expr {
        syn::Expr::Path(p) => p.path.get_ident().is_some_and(|id| id == chunk_ident),
        syn::Expr::Reference(r) => is_exact_chunk_receiver(&r.expr, chunk_ident),
        syn::Expr::Paren(p) => is_exact_chunk_receiver(&p.expr, chunk_ident),
        _ => false,
    }
}

pub(super) fn is_exact_chunk_to_array(expr: &syn::Expr, chunk_ident: &str) -> bool {
    match expr {
        syn::Expr::MethodCall(mc) => {
            let mname = mc.method.to_string();
            if mname == "unwrap" || mname == "expect" {
                if let syn::Expr::MethodCall(inner_mc) = &*mc.receiver {
                    if inner_mc.method == "try_into" && inner_mc.args.is_empty() {
                        return is_exact_chunk_receiver(&inner_mc.receiver, chunk_ident);
                    }
                }
                false
            } else if mname == "try_into" && mc.args.is_empty() {
                is_exact_chunk_receiver(&mc.receiver, chunk_ident)
            } else {
                false
            }
        }
        syn::Expr::Try(t) => {
            if let syn::Expr::MethodCall(mc) = &*t.expr {
                if mc.method == "try_into" && mc.args.is_empty() {
                    return is_exact_chunk_receiver(&mc.receiver, chunk_ident);
                }
            }
            false
        }
        syn::Expr::Path(p) => p.path.get_ident().is_some_and(|id| id == chunk_ident),
        syn::Expr::Reference(r) => is_exact_chunk_to_array(&r.expr, chunk_ident),
        syn::Expr::Paren(p) => is_exact_chunk_to_array(&p.expr, chunk_ident),
        _ => false,
    }
}

pub(super) fn is_exact_from_le_bytes_expr(
    expr: &syn::Expr,
    chunk_ident: &str,
    chunk_len: u32,
) -> bool {
    match expr {
        syn::Expr::Call(c) => {
            if let syn::Expr::Path(p) = &*c.func {
                let segs: Vec<String> = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                let is_valid_builtin_path = if segs.len() == 2 {
                    segs[1] == "from_le_bytes" && is_matching_width_type(&segs[0], chunk_len)
                } else if segs.len() == 4 {
                    (segs[0] == "core" || segs[0] == "std")
                        && segs[1] == "primitive"
                        && segs[3] == "from_le_bytes"
                        && is_matching_width_type(&segs[2], chunk_len)
                } else {
                    false
                };
                if is_valid_builtin_path && c.args.len() == 1 {
                    return is_exact_chunk_to_array(&c.args[0], chunk_ident);
                }
            }
            false
        }
        syn::Expr::Paren(p) => is_exact_from_le_bytes_expr(&p.expr, chunk_ident, chunk_len),
        _ => false,
    }
}

/// Whether a `for` loop only routes each output slot to a decoder.
///
/// A wrapper that reads back several named buffers walks its own list of output
/// slots, one iteration per slot, and each iteration resolves the slot's bytes
/// and hands them to a byte decoder that writes into the slot's own sink. Every
/// byte reaches exactly one word of exactly one caller-owned output, so nothing
/// is accumulated, compared, folded or gathered: the loop is the readback
/// plumbing, and the number of iterations is the number of outputs rather than
/// the number of elements. Judging every post-dispatch loop a reduction
/// convicted that plumbing, so the shape it has is stated here.
///
/// The admitted shape is a closed whitelist, which is what keeps it from
/// admitting a reduction that happens to be written as calls. The sequence is a
/// named collection, every statement is a call or a `let` bound from one, and
/// every operand is a path, a field, a literal, a reference or another such
/// call. An arithmetic or comparison operator, a compound assignment, an index
/// into data, a method call, a closure, a branch, a `match` or a nested loop is
/// none of those, so each one leaves the shape and the loop is reported. A
/// callee that computes is convicted where it is defined, by the same walk that
/// reads this file.
pub(super) fn is_output_slot_transport_loop(expr: &syn::ExprForLoop) -> bool {
    if !iterates_named_collection(&expr.expr) {
        return false;
    }
    let mut bound = BTreeSet::new();
    extract_pat_bindings(&expr.pat, &mut bound);
    if bound.is_empty() {
        return false;
    }
    let mut transported = 0usize;
    for stmt in &expr.body.stmts {
        match stmt {
            syn::Stmt::Local(local) => {
                let Some(init) = &local.init else {
                    return false;
                };
                if init.diverge.is_some() {
                    return false;
                }
                if !is_transport_call(&init.expr) {
                    return false;
                }
                let mut names = BTreeSet::new();
                extract_pat_bindings(&local.pat, &mut names);
                if names.is_empty() {
                    return false;
                }
                transported += 1;
            }
            syn::Stmt::Expr(value, _) => {
                if !is_transport_call(value) {
                    return false;
                }
                transported += 1;
            }
            syn::Stmt::Item(_) | syn::Stmt::Macro(_) => return false,
        }
    }
    transported > 0
}

/// Whether the loop walks a named collection rather than a computed sequence.
///
/// A slot list is a binding, a field of one, or one of the three iterator
/// constructors over either. A range, a `zip`, an `enumerate` or a chain that
/// derives its sequence from data is a different loop and is judged as one.
fn iterates_named_collection(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Path(_) => true,
        syn::Expr::Field(field) => iterates_named_collection(&field.base),
        syn::Expr::Reference(reference) => iterates_named_collection(&reference.expr),
        syn::Expr::Paren(paren) => iterates_named_collection(&paren.expr),
        syn::Expr::Group(group) => iterates_named_collection(&group.expr),
        syn::Expr::Unary(unary) => {
            matches!(unary.op, syn::UnOp::Deref(_)) && iterates_named_collection(&unary.expr)
        }
        syn::Expr::MethodCall(call) => {
            matches!(
                call.method.to_string().as_str(),
                "iter" | "iter_mut" | "into_iter"
            ) && call.args.is_empty()
                && iterates_named_collection(&call.receiver)
        }
        _ => false,
    }
}

/// Whether the expression is a plain function call over transport operands.
///
/// A method call is not one of these. A reduction spells itself as a method far
/// more often than as a free function (`.sum()`, `.max()`, `.wrapping_add()`),
/// and a transport loop needs none, so refusing the whole form closes that
/// class without a list of method names to keep current.
fn is_transport_call(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Try(inner) => is_transport_call(&inner.expr),
        syn::Expr::Paren(paren) => is_transport_call(&paren.expr),
        syn::Expr::Group(group) => is_transport_call(&group.expr),
        syn::Expr::Call(call) => {
            matches!(&*call.func, syn::Expr::Path(_)) && call.args.iter().all(is_transport_operand)
        }
        _ => false,
    }
}

/// Whether the expression names a value without computing one.
fn is_transport_operand(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Path(_) | syn::Expr::Lit(_) => true,
        syn::Expr::Field(field) => is_transport_operand(&field.base),
        syn::Expr::Reference(reference) => is_transport_operand(&reference.expr),
        syn::Expr::Paren(paren) => is_transport_operand(&paren.expr),
        syn::Expr::Group(group) => is_transport_operand(&group.expr),
        syn::Expr::Unary(unary) => {
            matches!(unary.op, syn::UnOp::Deref(_)) && is_transport_operand(&unary.expr)
        }
        syn::Expr::Try(inner) => is_transport_call(&inner.expr),
        syn::Expr::Call(_) => is_transport_call(expr),
        _ => false,
    }
}

pub(super) fn is_pure_decoder_loop(expr: &syn::ExprForLoop) -> bool {
    let chunk_len = match &*expr.expr {
        syn::Expr::MethodCall(mc) => {
            if mc.method != "chunks_exact" || mc.args.len() != 1 {
                return false;
            }
            match mc.args.first() {
                Some(syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(li),
                    ..
                })) => {
                    let val = li.base10_parse::<u32>().unwrap_or(0);
                    if val == 1 || val == 2 || val == 4 || val == 8 || val == 16 {
                        val
                    } else {
                        return false;
                    }
                }
                Some(syn::Expr::Reference(r)) => {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(li),
                        ..
                    }) = &*r.expr
                    {
                        let val = li.base10_parse::<u32>().unwrap_or(0);
                        if val == 1 || val == 2 || val == 4 || val == 8 || val == 16 {
                            val
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        _ => return false,
    };

    let chunk_ident = match &*expr.pat {
        syn::Pat::Ident(pi) => pi.ident.to_string(),
        syn::Pat::Reference(pr) => {
            if let syn::Pat::Ident(pi) = &*pr.pat {
                pi.ident.to_string()
            } else {
                return false;
            }
        }
        _ => return false,
    };

    let mut decoded_local_ident: Option<String> = None;
    let mut semantic_write_count = 0usize;

    for stmt in &expr.body.stmts {
        match stmt {
            syn::Stmt::Local(local) => {
                if decoded_local_ident.is_some() {
                    return false;
                }
                let mut bound_idents = BTreeSet::new();
                extract_pat_bindings(&local.pat, &mut bound_idents);
                if bound_idents.len() != 1 {
                    return false;
                }
                let Some(var_name) = bound_idents.into_iter().next() else {
                    return false;
                };
                let Some(init) = &local.init else {
                    return false;
                };
                if !is_exact_from_le_bytes_expr(&init.expr, &chunk_ident, chunk_len) {
                    return false;
                }
                decoded_local_ident = Some(var_name);
            }
            syn::Stmt::Expr(expr, _) => match expr {
                syn::Expr::MethodCall(mc) => {
                    let mname = mc.method.to_string();
                    if mname == "reserve" || mname == "reserve_exact" {
                        continue;
                    }
                    if mname == "push" {
                        if mc.args.len() != 1 {
                            return false;
                        }
                        let arg = &mc.args[0];
                        let is_valid_arg = if let Some(local_id) = &decoded_local_ident {
                            match arg {
                                syn::Expr::Path(p) => {
                                    p.path.get_ident().is_some_and(|id| id == local_id)
                                }
                                _ => is_exact_from_le_bytes_expr(arg, &chunk_ident, chunk_len),
                            }
                        } else {
                            is_exact_from_le_bytes_expr(arg, &chunk_ident, chunk_len)
                        };
                        if !is_valid_arg {
                            return false;
                        }
                        semantic_write_count += 1;
                    } else if mname == "extend" || mname == "extend_from_slice" {
                        if mc.args.len() != 1 {
                            return false;
                        }
                        let arg = &mc.args[0];
                        let is_valid_arg = match arg {
                            syn::Expr::Array(arr) => {
                                arr.elems.len() == 1 && {
                                    let elem = &arr.elems[0];
                                    if let Some(local_id) = &decoded_local_ident {
                                        match elem {
                                            syn::Expr::Path(p) => {
                                                p.path.get_ident().is_some_and(|id| id == local_id)
                                            }
                                            _ => is_exact_from_le_bytes_expr(
                                                elem,
                                                &chunk_ident,
                                                chunk_len,
                                            ),
                                        }
                                    } else {
                                        is_exact_from_le_bytes_expr(elem, &chunk_ident, chunk_len)
                                    }
                                }
                            }
                            _ => false,
                        };
                        if !is_valid_arg {
                            return false;
                        }
                        semantic_write_count += 1;
                    } else {
                        return false;
                    }
                }
                _ => return false,
            },
            _ => return false,
        }
    }

    semantic_write_count == 1
}
