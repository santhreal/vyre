//! Execution and input-binding scanners, dependency flow trackers, and method extractors.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::visit::Visit;

use super::host_oracle_elimination_ast::AstAnalysisVisitor;
use super::host_oracle_elimination_classify::is_byte_unpack_codec_expr;
use super::host_oracle_elimination_extract::{
    extract_pat_bindings, extract_read_idents_from_expr, extract_root_ident_from_expr,
    is_pure_decoder_loop, is_reduction_or_arithmetic_method,
};
use super::host_oracle_elimination_records::{
    base_module_path, extract_use_tree, normalize_qualified_path, ParamCalleeFlow,
};
use crate::gates::scan::attribute_is_test_only;

pub(super) struct SemanticOperationScanner<'a> {
    pub(super) visitor: &'a AstAnalysisVisitor,
    pub(super) has_semantic_op: bool,
}

impl<'ast, 'a> syn::visit::Visit<'ast> for SemanticOperationScanner<'a> {
    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        if !self
            .visitor
            .is_dispatch_execution_expr(&syn::Expr::Binary(expr.clone()))
        {
            if !is_byte_unpack_codec_expr(expr) {
                self.has_semantic_op = true;
            }
        }
        syn::visit::visit_expr_binary(self, expr);
    }

    fn visit_expr_unary(&mut self, expr: &'ast syn::ExprUnary) {
        if matches!(expr.op, syn::UnOp::Neg(_) | syn::UnOp::Not(_)) {
            self.has_semantic_op = true;
        }
        syn::visit::visit_expr_unary(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        let method_name = expr.method.to_string();
        if !self
            .visitor
            .is_dispatch_execution_expr(&syn::Expr::MethodCall(expr.clone()))
        {
            if is_reduction_or_arithmetic_method(&method_name) {
                self.has_semantic_op = true;
            }
        }
        syn::visit::visit_expr_method_call(self, expr);
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        if !self.visitor.for_loop_dispatches(expr) && !is_pure_decoder_loop(expr) {
            self.has_semantic_op = true;
        }
        syn::visit::visit_expr_for_loop(self, expr);
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        if !self.visitor.while_loop_dispatches(expr) {
            self.has_semantic_op = true;
        }
        syn::visit::visit_expr_while(self, expr);
    }

    fn visit_expr_loop(&mut self, expr: &'ast syn::ExprLoop) {
        if !self.visitor.loop_dispatches(expr) {
            self.has_semantic_op = true;
        }
        syn::visit::visit_expr_loop(self, expr);
    }

    fn visit_expr_closure(&mut self, expr: &'ast syn::ExprClosure) {
        let mut sub_scanner = SemanticOperationScanner {
            visitor: self.visitor,
            has_semantic_op: false,
        };
        sub_scanner.visit_expr(&expr.body);
        if sub_scanner.has_semantic_op {
            self.has_semantic_op = true;
        }
        syn::visit::visit_expr_closure(self, expr);
    }
}

pub(super) fn stmt_contains_semantic_operation(
    stmt: &syn::Stmt,
    visitor: &AstAnalysisVisitor,
) -> bool {
    let mut scanner = SemanticOperationScanner {
        visitor,
        has_semantic_op: false,
    };
    syn::visit::visit_stmt(&mut scanner, stmt);
    scanner.has_semantic_op
}

pub(super) struct InputBindingScanner<'a> {
    pub(super) visitor: &'a AstAnalysisVisitor,
    pub(super) dispatcher_params: &'a BTreeSet<String>,
    pub(super) semantic_taint: &'a BTreeSet<String>,
    pub(super) consumes_semantic_storage: bool,
}

impl<'ast> Visit<'ast> for InputBindingScanner<'_> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let method = call.method.to_string();
        if self.visitor.derived_input_binding_methods.contains(&method) {
            let receiver_idents = extract_read_idents_from_expr(&call.receiver);
            let receiver_is_dispatcher = receiver_idents
                .iter()
                .any(|ident| self.dispatcher_params.contains(ident));
            let payload_is_semantic = call.args.iter().any(|argument| {
                extract_read_idents_from_expr(argument)
                    .iter()
                    .any(|ident| self.semantic_taint.contains(ident))
            });
            if receiver_is_dispatcher && payload_is_semantic {
                self.consumes_semantic_storage = true;
                return;
            }
        }
        syn::visit::visit_expr_method_call(self, call);
    }

    /// Binding at this seam is a request constructor, not a device upload.
    ///
    /// The request type owns input binding, so a staging function binds bytes by
    /// calling that constructor and never needs an executor at all. Requiring a
    /// dispatcher receiver would leave every staging function at this seam
    /// unrecognized, which reads as unreachable host data processing.
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path_expr) = &*call.func {
            let segments = &path_expr.path.segments;
            let names_binding_constructor = segments.len() >= 2
                && segments.last().is_some_and(|segment| {
                    self.visitor
                        .derived_input_binding_methods
                        .contains(&segment.ident.to_string())
                })
                && ident_names_canonical_execution_request(&segments[segments.len() - 2].ident);
            let payload_is_semantic = call.args.iter().any(|argument| {
                extract_read_idents_from_expr(argument)
                    .iter()
                    .any(|ident| self.semantic_taint.contains(ident))
            });
            if names_binding_constructor && payload_is_semantic {
                self.consumes_semantic_storage = true;
                return;
            }
        }
        syn::visit::visit_expr_call(self, call);
    }
}

/// Whether an identifier names the canonical semantic execution request type.
pub(super) fn ident_names_canonical_execution_request(ident: &syn::Ident) -> bool {
    ident == "SemanticExecutionRequest"
}
pub(super) fn type_is_canonical_ir_program(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(tp) => {
            if let Some(seg) = tp.path.segments.last() {
                seg.ident == "Program"
            } else {
                false
            }
        }
        syn::Type::Reference(r) => type_is_canonical_ir_program(&r.elem),
        syn::Type::Slice(s) => type_is_canonical_ir_program(&s.elem),
        syn::Type::Array(a) => type_is_canonical_ir_program(&a.elem),
        syn::Type::Group(g) => type_is_canonical_ir_program(&g.elem),
        syn::Type::Paren(p) => type_is_canonical_ir_program(&p.elem),
        _ => false,
    }
}

pub(super) fn type_is_canonical_execution_request(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(tp) => tp
            .path
            .segments
            .last()
            .is_some_and(|seg| ident_names_canonical_execution_request(&seg.ident)),
        syn::Type::Reference(r) => type_is_canonical_execution_request(&r.elem),
        syn::Type::Slice(s) => type_is_canonical_execution_request(&s.elem),
        syn::Type::Array(a) => type_is_canonical_execution_request(&a.elem),
        syn::Type::Group(g) => type_is_canonical_execution_request(&g.elem),
        syn::Type::Paren(p) => type_is_canonical_execution_request(&p.elem),
        _ => false,
    }
}
pub(super) fn type_is_exact_generic_param(ty: &syn::Type, generic_name: &str) -> bool {
    match ty {
        syn::Type::Path(tp) => {
            if tp.qself.is_none() && tp.path.segments.len() == 1 {
                let seg = &tp.path.segments[0];
                seg.ident == generic_name && matches!(seg.arguments, syn::PathArguments::None)
            } else {
                false
            }
        }
        syn::Type::Reference(r) => type_is_exact_generic_param(&r.elem, generic_name),
        syn::Type::Group(g) => type_is_exact_generic_param(&g.elem, generic_name),
        syn::Type::Paren(p) => type_is_exact_generic_param(&p.elem, generic_name),
        _ => false,
    }
}

pub(super) fn type_contains_immutable_byte_slice(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Reference(reference) => {
            if reference.mutability.is_some() {
                return false;
            }
            matches!(
                &*reference.elem,
                syn::Type::Slice(slice)
                    if matches!(
                        &*slice.elem,
                        syn::Type::Path(path)
                            if path.path.is_ident("u8")
                    )
            ) || type_contains_immutable_byte_slice(&reference.elem)
        }
        syn::Type::Path(path) => path.path.segments.last().is_some_and(|segment| {
            if let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments {
                arguments.args.iter().any(|argument| {
                    matches!(
                        argument,
                        syn::GenericArgument::Type(inner)
                            if type_contains_immutable_byte_slice(inner)
                    )
                })
            } else {
                false
            }
        }),
        syn::Type::Slice(slice) => type_contains_immutable_byte_slice(&slice.elem),
        syn::Type::Array(array) => type_contains_immutable_byte_slice(&array.elem),
        syn::Type::Tuple(tuple) => tuple.elems.iter().any(type_contains_immutable_byte_slice),
        syn::Type::Group(group) => type_contains_immutable_byte_slice(&group.elem),
        syn::Type::Paren(paren) => type_contains_immutable_byte_slice(&paren.elem),
        _ => false,
    }
}

pub(super) fn is_request_input_binding_method(sig: &syn::Signature) -> bool {
    let has_immutable_byte_payload = sig.inputs.iter().any(|input| {
        matches!(
            input,
            syn::FnArg::Typed(pat_type)
                if type_contains_immutable_byte_slice(&pat_type.ty)
        )
    });
    let returns_result = matches!(
        &sig.output,
        syn::ReturnType::Type(_, ty)
            if matches!(
                &**ty,
                syn::Type::Path(path)
                    if path.path.segments.last().is_some_and(|segment| segment.ident == "Result")
            )
    );
    has_immutable_byte_payload && returns_result
}

pub(super) fn is_trait_method_dispatch_execution(sig: &syn::Signature) -> bool {
    let has_program_or_request_param = sig.inputs.iter().any(|input| {
        if let syn::FnArg::Typed(pat_type) = input {
            type_is_canonical_ir_program(&pat_type.ty)
                || type_is_canonical_execution_request(&pat_type.ty)
        } else {
            false
        }
    });

    let returns_result_or_data = match &sig.output {
        syn::ReturnType::Default => false,
        syn::ReturnType::Type(_, ty) => match &**ty {
            syn::Type::Path(tp) => {
                if let Some(seg) = tp.path.segments.last() {
                    seg.ident == "Result" || seg.ident == "Vec"
                } else {
                    false
                }
            }
            syn::Type::Tuple(tup) => tup.elems.is_empty(),
            _ => false,
        },
    };

    has_program_or_request_param && returns_result_or_data
}

/// Each `OperationRegistration` constructor, mapped to the argument position of
/// its `expected_output` parameter.
///
/// A registration's fixture and expected-output callbacks are passed by name, so
/// the walk has to know which argument is the expected-output one to separate a
/// reached callback from a host oracle executed while producing expected bytes.
/// The map is read out of the `impl OperationRegistration` block rather than
/// listed here: a hardcoded roster went stale when the constructors were renamed
/// with the `_unconstrained` suffix, and the whole registered-fixture class then
/// read as unreachable host data processing while an oracle inside
/// `expected_output` went unconvicted.
pub(super) fn derive_registration_expected_output_indices(
    file: &syn::File,
) -> BTreeMap<String, usize> {
    fn inspect_items(items: &[syn::Item], out: &mut BTreeMap<String, usize>) {
        for item in items {
            match item {
                syn::Item::Impl(item_impl)
                    if item_impl.trait_.is_none()
                        && matches!(
                            &*item_impl.self_ty,
                            syn::Type::Path(path)
                                if path.path.segments.last().is_some_and(|segment| {
                                    segment.ident == "OperationRegistration"
                                })
                        ) =>
                {
                    for impl_item in &item_impl.items {
                        let syn::ImplItem::Fn(method) = impl_item else {
                            continue;
                        };
                        let position = method.sig.inputs.iter().position(|input| {
                            matches!(
                                input,
                                syn::FnArg::Typed(pat_type)
                                    if matches!(
                                        &*pat_type.pat,
                                        syn::Pat::Ident(pat)
                                            if pat.ident == "expected_output"
                                    )
                            )
                        });
                        let takes_receiver = method
                            .sig
                            .inputs
                            .first()
                            .is_some_and(|input| matches!(input, syn::FnArg::Receiver(_)));
                        if let Some(position) = position {
                            if !takes_receiver {
                                out.insert(method.sig.ident.to_string(), position);
                            }
                        }
                    }
                }
                syn::Item::Mod(item_mod) => {
                    if item_mod.attrs.iter().any(attribute_is_test_only) {
                        continue;
                    }
                    if let Some((_, inner)) = &item_mod.content {
                        inspect_items(inner, out);
                    }
                }
                _ => {}
            }
        }
    }

    let mut out = BTreeMap::new();
    inspect_items(&file.items, &mut out);
    out
}

fn type_names_canonical_executor(ty: &syn::Type) -> bool {
    fn bound_names_executor(bound: &syn::TypeParamBound) -> bool {
        matches!(
            bound,
            syn::TypeParamBound::Trait(trait_bound)
                if trait_bound
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "SemanticExecutor")
        )
    }

    match ty {
        syn::Type::Path(path) => path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "SemanticExecutor"),
        syn::Type::TraitObject(object) => object.bounds.iter().any(bound_names_executor),
        syn::Type::ImplTrait(imp) => imp.bounds.iter().any(bound_names_executor),
        syn::Type::Reference(reference) => type_names_canonical_executor(&reference.elem),
        syn::Type::Group(group) => type_names_canonical_executor(&group.elem),
        syn::Type::Paren(paren) => type_names_canonical_executor(&paren.elem),
        _ => false,
    }
}

/// Bindings of a signature's parameters that carry the canonical executor.
///
/// Both spellings the seam admits are covered: a trait object parameter and a
/// generic parameter bounded by the trait, whether the bound sits inline or in a
/// where clause.
fn executor_param_bindings(sig: &syn::Signature) -> BTreeSet<String> {
    let mut executor_generics = BTreeSet::new();
    for param in &sig.generics.params {
        if let syn::GenericParam::Type(type_param) = param {
            if type_param.bounds.iter().any(|bound| {
                matches!(
                    bound,
                    syn::TypeParamBound::Trait(trait_bound)
                        if trait_bound
                            .path
                            .segments
                            .last()
                            .is_some_and(|segment| segment.ident == "SemanticExecutor")
                )
            }) {
                executor_generics.insert(type_param.ident.to_string());
            }
        }
    }
    if let Some(where_clause) = &sig.generics.where_clause {
        for predicate in &where_clause.predicates {
            if let syn::WherePredicate::Type(predicate) = predicate {
                let names_executor = predicate.bounds.iter().any(|bound| {
                    matches!(
                        bound,
                        syn::TypeParamBound::Trait(trait_bound)
                            if trait_bound
                                .path
                                .segments
                                .last()
                                .is_some_and(|segment| segment.ident == "SemanticExecutor")
                    )
                });
                if names_executor {
                    if let syn::Type::Path(path) = &predicate.bounded_ty {
                        if let Some(ident) = path.path.get_ident() {
                            executor_generics.insert(ident.to_string());
                        }
                    }
                }
            }
        }
    }

    let mut bindings = BTreeSet::new();
    for input in &sig.inputs {
        let syn::FnArg::Typed(pat_type) = input else {
            continue;
        };
        let carries_executor = type_names_canonical_executor(&pat_type.ty)
            || executor_generics
                .iter()
                .any(|generic| type_is_exact_generic_param(&pat_type.ty, generic));
        if carries_executor {
            extract_pat_bindings(&pat_type.pat, &mut bindings);
        }
    }
    bindings
}

struct ExecutorExecutionScanner<'a> {
    executor_params: &'a BTreeSet<String>,
    dispatch_methods: &'a BTreeSet<String>,
    executes: bool,
}

impl<'ast, 'a> Visit<'ast> for ExecutorExecutionScanner<'a> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if self.dispatch_methods.contains(&call.method.to_string())
            && extract_read_idents_from_expr(&call.receiver)
                .iter()
                .any(|ident| self.executor_params.contains(ident))
        {
            self.executes = true;
        }
        syn::visit::visit_expr_method_call(self, call);
    }
}

/// Derive the free execution helpers the canonical seam publishes.
///
/// A library caller reaches a device through one of these rather than through
/// the trait method, because request construction and canonical output ordering
/// live behind the helper. The seam crate is outside the scanned roots, so
/// without this set no call in a scanned crate resolves to device execution and
/// every wrapper reads as an unreachable host oracle. The names are read out of
/// the seam source, so a helper added beside the existing one is picked up
/// without a second edit.
pub(super) fn derive_canonical_execution_fns(
    file: &syn::File,
    dispatch_methods: &BTreeSet<String>,
) -> BTreeSet<String> {
    fn inspect_items(
        items: &[syn::Item],
        dispatch_methods: &BTreeSet<String>,
        found: &mut BTreeSet<String>,
    ) {
        for item in items {
            match item {
                syn::Item::Fn(item_fn) => {
                    let executor_params = executor_param_bindings(&item_fn.sig);
                    if executor_params.is_empty() {
                        continue;
                    }
                    let mut scanner = ExecutorExecutionScanner {
                        executor_params: &executor_params,
                        dispatch_methods,
                        executes: false,
                    };
                    scanner.visit_block(&item_fn.block);
                    if scanner.executes {
                        found.insert(item_fn.sig.ident.to_string());
                    }
                }
                syn::Item::Mod(item_mod) => {
                    if item_mod.attrs.iter().any(attribute_is_test_only) {
                        continue;
                    }
                    if let Some((_, inner)) = &item_mod.content {
                        inspect_items(inner, dispatch_methods, found);
                    }
                }
                _ => {}
            }
        }
    }

    let mut found = BTreeSet::new();
    inspect_items(&file.items, dispatch_methods, &mut found);
    found
}

/// Derive the execution and input-binding method names from the canonical seam.
///
/// The trait carries execution; the request type carries input binding, because
/// a caller at this seam cannot upload to a device and instead binds byte
/// payloads into the request it submits. Both sets are read out of the source
/// rather than listed here, so a method added to either one is picked up without
/// a second edit.
pub(super) fn derive_canonical_dispatcher_methods(
    file: &syn::File,
    dispatch_methods: &mut BTreeSet<String>,
    input_binding_methods: &mut BTreeSet<String>,
) {
    fn inspect_items(
        items: &[syn::Item],
        dispatch_methods: &mut BTreeSet<String>,
        input_binding_methods: &mut BTreeSet<String>,
    ) {
        for item in items {
            match item {
                syn::Item::Trait(item_trait) if item_trait.ident == "SemanticExecutor" => {
                    for trait_item in &item_trait.items {
                        if let syn::TraitItem::Fn(method) = trait_item {
                            if is_trait_method_dispatch_execution(&method.sig) {
                                dispatch_methods.insert(method.sig.ident.to_string());
                            }
                        }
                    }
                }
                syn::Item::Impl(item_impl)
                    if item_impl.trait_.is_none()
                        && type_is_canonical_execution_request(&item_impl.self_ty) =>
                {
                    for impl_item in &item_impl.items {
                        if let syn::ImplItem::Fn(method) = impl_item {
                            if is_request_input_binding_method(&method.sig) {
                                input_binding_methods.insert(method.sig.ident.to_string());
                            }
                        }
                    }
                }
                syn::Item::Mod(item_mod) => {
                    if let Some((_, inner_items)) = &item_mod.content {
                        inspect_items(inner_items, dispatch_methods, input_binding_methods);
                    }
                }
                _ => {}
            }
        }
    }

    inspect_items(&file.items, dispatch_methods, input_binding_methods);
}

pub(super) struct BlockDispatchScanner<'a> {
    pub(super) visitor: &'a AstAnalysisVisitor,
    pub(super) dispatcher_params: BTreeSet<String>,
    pub(super) has_direct_dispatch: bool,
    pub(super) dispatcher_callees: Vec<String>,
}

impl<'a> BlockDispatchScanner<'a> {
    pub(super) fn new(
        visitor: &'a AstAnalysisVisitor,
        dispatcher_params: BTreeSet<String>,
    ) -> Self {
        Self {
            visitor,
            dispatcher_params,
            has_direct_dispatch: false,
            dispatcher_callees: Vec::new(),
        }
    }
}

impl<'ast, 'a> Visit<'ast> for BlockDispatchScanner<'a> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let method_name = call.method.to_string();
        let receiver_idents = extract_read_idents_from_expr(&call.receiver);
        let receiver_carries_dispatcher = receiver_idents
            .iter()
            .any(|ident| self.dispatcher_params.contains(ident));

        if receiver_carries_dispatcher {
            if self.visitor.is_canonical_dispatch_exec_method(&method_name) {
                self.has_direct_dispatch = true;
            } else {
                self.dispatcher_callees.push(method_name.clone());
            }
        }

        for arg in &call.args {
            let arg_idents = extract_read_idents_from_expr(arg);
            if arg_idents
                .iter()
                .any(|ident| self.dispatcher_params.contains(ident))
            {
                self.dispatcher_callees.push(method_name.clone());
            }
        }
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let resolved_callee = if let syn::Expr::Path(path) = &*call.func {
            self.visitor.resolve_qualified_fn_path(&path.path)
        } else {
            None
        };
        for arg in &call.args {
            let arg_idents = extract_read_idents_from_expr(arg);
            if arg_idents
                .iter()
                .any(|ident| self.dispatcher_params.contains(ident))
            {
                if let Some(name) = &resolved_callee {
                    self.dispatcher_callees.push(name.clone());
                }
            }
        }
        syn::visit::visit_expr_call(self, call);
    }
}

pub(super) struct FnDispatchScan {
    pub(super) name: String,
    pub(super) has_direct_dispatch: bool,
    pub(super) dispatcher_callees: Vec<String>,
}

pub(super) fn scan_item_for_dispatch(
    item: &syn::Item,
    visitor: &mut AstAnalysisVisitor,
    scans: &mut Vec<FnDispatchScan>,
) {
    match item {
        syn::Item::Fn(item_fn) => {
            let dispatcher_params = visitor.extract_dispatcher_params(&item_fn.sig);
            if !dispatcher_params.is_empty() {
                let mut scanner = BlockDispatchScanner::new(visitor, dispatcher_params);
                scanner.visit_block(&item_fn.block);
                let qualified = visitor.qualified_local_type_name(&item_fn.sig.ident);
                scans.push(FnDispatchScan {
                    name: qualified,
                    has_direct_dispatch: scanner.has_direct_dispatch,
                    dispatcher_callees: scanner.dispatcher_callees,
                });
            }
        }
        syn::Item::Impl(item_impl) => {
            let self_has_dispatcher =
                visitor.type_contains_canonical_dispatcher(&item_impl.self_ty);
            let prev_self = visitor.current_impl_self_is_dispatcher;
            visitor.current_impl_self_is_dispatcher = self_has_dispatcher;
            for impl_item in &item_impl.items {
                if let syn::ImplItem::Fn(impl_fn) = impl_item {
                    let dispatcher_params = visitor.extract_dispatcher_params(&impl_fn.sig);
                    if !dispatcher_params.is_empty() {
                        let mut scanner = BlockDispatchScanner::new(visitor, dispatcher_params);
                        scanner.visit_block(&impl_fn.block);
                        let type_prefix = if let syn::Type::Path(tp) = &*item_impl.self_ty {
                            visitor
                                .resolve_qualified_type_path(&tp.path)
                                .unwrap_or_else(|| quote::quote!(#tp).to_string())
                        } else {
                            "Self".to_string()
                        };
                        let qualified = format!("{type_prefix}::{}", impl_fn.sig.ident);
                        scans.push(FnDispatchScan {
                            name: qualified,
                            has_direct_dispatch: scanner.has_direct_dispatch,
                            dispatcher_callees: scanner.dispatcher_callees,
                        });
                    }
                }
            }
            visitor.current_impl_self_is_dispatcher = prev_self;
        }
        syn::Item::Mod(item_mod) => {
            if let Some((_, items)) = &item_mod.content {
                visitor.current_module.push(item_mod.ident.to_string());
                visitor.scope_imports.push(BTreeMap::new());
                if let Some(inner_scope) = visitor.scope_imports.last_mut() {
                    for inner_item in items {
                        if let syn::Item::Use(item_use) = inner_item {
                            extract_use_tree("", &item_use.tree, inner_scope);
                        }
                    }
                }
                for inner_item in items {
                    scan_item_for_dispatch(inner_item, visitor, scans);
                }
                visitor.scope_imports.pop();
                visitor.current_module.pop();
            }
        }
        _ => {}
    }
}

pub(super) fn collect_wrapper_structs_item(item: &syn::Item, visitor: &mut AstAnalysisVisitor) {
    match item {
        syn::Item::Struct(item_struct) => {
            if visitor.struct_contains_canonical_dispatcher(item_struct) {
                let qualified = visitor.qualified_local_type_name(&item_struct.ident);
                visitor.struct_types_with_dispatcher.insert(qualified);
            }
        }
        syn::Item::Mod(item_mod) => {
            if let Some((_, inner_items)) = &item_mod.content {
                visitor.current_module.push(item_mod.ident.to_string());
                visitor.scope_imports.push(BTreeMap::new());
                if let Some(inner_scope) = visitor.scope_imports.last_mut() {
                    for inner_item in inner_items {
                        if let syn::Item::Use(item_use) = inner_item {
                            extract_use_tree("", &item_use.tree, inner_scope);
                        }
                    }
                }
                for inner_item in inner_items {
                    collect_wrapper_structs_item(inner_item, visitor);
                }
                visitor.scope_imports.pop();
                visitor.current_module.pop();
            }
        }
        _ => {}
    }
}

pub(super) fn compute_known_dispatch_exec_fns_multi(
    files: &[(&Path, &syn::File)],
    canonical_dispatch_methods: &BTreeSet<String>,
    canonical_input_binding_methods: &BTreeSet<String>,
    canonical_execution_fns: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut temp_visitor = AstAnalysisVisitor::new(
        PathBuf::from("<multi>"),
        false,
        0,
        canonical_dispatch_methods.clone(),
        canonical_input_binding_methods.clone(),
    );

    for (file_path, file) in files {
        temp_visitor.file = (*file_path).to_path_buf();
        temp_visitor.current_module = base_module_path(file_path);
        temp_visitor.scope_imports = vec![BTreeMap::new()];

        let mut collector = FileImportCollector {
            scope_imports: &mut temp_visitor.scope_imports,
        };
        collector.visit_file(file);

        loop {
            let previous_count = temp_visitor.struct_types_with_dispatcher.len();
            for item in &file.items {
                collect_wrapper_structs_item(item, &mut temp_visitor);
            }
            if temp_visitor.struct_types_with_dispatcher.len() == previous_count {
                break;
            }
        }
    }

    let mut scans = Vec::new();
    for (file_path, file) in files {
        temp_visitor.file = (*file_path).to_path_buf();
        temp_visitor.current_module = base_module_path(file_path);
        temp_visitor.scope_imports = vec![BTreeMap::new()];

        let mut collector = FileImportCollector {
            scope_imports: &mut temp_visitor.scope_imports,
        };
        collector.visit_file(file);

        for item in &file.items {
            scan_item_for_dispatch(item, &mut temp_visitor, &mut scans);
        }
    }

    let mut exec_set = canonical_execution_fns.clone();
    for scan in &scans {
        if scan.has_direct_dispatch {
            exec_set.insert(scan.name.clone());
            let norm = normalize_qualified_path(&scan.name);
            exec_set.insert(norm);
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for scan in &scans {
            if !exec_set.contains(&scan.name)
                && scan.dispatcher_callees.iter().any(|callee| {
                    exec_set.contains(callee)
                        || exec_set.contains(&normalize_qualified_path(callee))
                        || temp_visitor.is_canonical_dispatch_exec_method(callee)
                })
            {
                exec_set.insert(scan.name.clone());
                let norm = normalize_qualified_path(&scan.name);
                exec_set.insert(norm);
                changed = true;
            }
        }
    }
    exec_set
}

pub(super) fn compute_known_dispatch_exec_fns(
    file: &syn::File,
    visitor: &mut AstAnalysisVisitor,
) -> BTreeSet<String> {
    let file_tuple = [(visitor.file.as_path(), file)];
    let seed = visitor.known_dispatch_exec_fns.clone();
    compute_known_dispatch_exec_fns_multi(
        &file_tuple,
        &visitor.derived_trait_dispatch_exec_methods,
        &visitor.derived_input_binding_methods,
        &seed,
    )
}
pub(super) fn extract_expr_param_deps(
    expr: &syn::Expr,
    var_deps: &BTreeMap<String, BTreeSet<usize>>,
) -> BTreeSet<usize> {
    let mut deps = BTreeSet::new();
    match expr {
        syn::Expr::Path(p) => {
            if let Some(ident) = p.path.get_ident() {
                if let Some(d) = var_deps.get(&ident.to_string()) {
                    deps.extend(d);
                }
            } else if let Some(seg) = p.path.segments.first() {
                if let Some(d) = var_deps.get(&seg.ident.to_string()) {
                    deps.extend(d);
                }
            }
        }
        syn::Expr::Field(f) => {
            deps.extend(extract_expr_param_deps(&f.base, var_deps));
        }
        syn::Expr::Index(i) => {
            deps.extend(extract_expr_param_deps(&i.expr, var_deps));
        }
        syn::Expr::Reference(r) => {
            deps.extend(extract_expr_param_deps(&r.expr, var_deps));
        }
        syn::Expr::Unary(u) => {
            deps.extend(extract_expr_param_deps(&u.expr, var_deps));
        }
        syn::Expr::Binary(b) => {
            deps.extend(extract_expr_param_deps(&b.left, var_deps));
            deps.extend(extract_expr_param_deps(&b.right, var_deps));
        }
        syn::Expr::Cast(c) => {
            deps.extend(extract_expr_param_deps(&c.expr, var_deps));
        }
        syn::Expr::Try(t) => {
            deps.extend(extract_expr_param_deps(&t.expr, var_deps));
        }
        syn::Expr::Paren(p) => {
            deps.extend(extract_expr_param_deps(&p.expr, var_deps));
        }
        syn::Expr::Group(g) => {
            deps.extend(extract_expr_param_deps(&g.expr, var_deps));
        }
        syn::Expr::Array(a) => {
            for elem in &a.elems {
                deps.extend(extract_expr_param_deps(elem, var_deps));
            }
        }
        syn::Expr::Tuple(t) => {
            for elem in &t.elems {
                deps.extend(extract_expr_param_deps(elem, var_deps));
            }
        }
        syn::Expr::Struct(s) => {
            let is_resident_step = s
                .path
                .segments
                .last()
                .is_some_and(|seg| seg.ident == "SemanticExecutionRequest");
            let is_read_range = s
                .path
                .segments
                .last()
                .is_some_and(|seg| seg.ident == "SemanticExecutionOutput");
            if is_resident_step {
                for f in &s.fields {
                    if let syn::Member::Named(ident) = &f.member {
                        if ident == "handle_ids" {
                            deps.extend(extract_expr_param_deps(&f.expr, var_deps));
                        }
                    }
                }
            } else if is_read_range {
                for f in &s.fields {
                    if let syn::Member::Named(ident) = &f.member {
                        if ident == "handle_id" {
                            deps.extend(extract_expr_param_deps(&f.expr, var_deps));
                        }
                    }
                }
            } else {
                for f in &s.fields {
                    deps.extend(extract_expr_param_deps(&f.expr, var_deps));
                }
            }
        }
        syn::Expr::Call(c) => {
            for arg in &c.args {
                deps.extend(extract_expr_param_deps(arg, var_deps));
            }
        }
        syn::Expr::MethodCall(mc) => {
            deps.extend(extract_expr_param_deps(&mc.receiver, var_deps));
            for arg in &mc.args {
                deps.extend(extract_expr_param_deps(arg, var_deps));
            }
        }
        syn::Expr::If(i) => {
            deps.extend(extract_expr_param_deps(&i.cond, var_deps));
            for stmt in &i.then_branch.stmts {
                if let syn::Stmt::Expr(e, _) = stmt {
                    deps.extend(extract_expr_param_deps(e, var_deps));
                }
            }
            if let Some((_, else_expr)) = &i.else_branch {
                deps.extend(extract_expr_param_deps(else_expr, var_deps));
            }
        }
        syn::Expr::Match(m) => {
            deps.extend(extract_expr_param_deps(&m.expr, var_deps));
            for arm in &m.arms {
                deps.extend(extract_expr_param_deps(&arm.body, var_deps));
            }
        }
        syn::Expr::Block(b) => {
            for stmt in &b.block.stmts {
                if let syn::Stmt::Expr(e, _) = stmt {
                    deps.extend(extract_expr_param_deps(e, var_deps));
                }
            }
        }
        syn::Expr::Repeat(r) => {
            deps.extend(extract_expr_param_deps(&r.expr, var_deps));
        }
        _ => {}
    }
    deps
}

/// Methods that move a payload into the collection they are called on.
///
/// The semantic seam binds a byte payload to a graph value through a map, so a
/// parameter reaches submission by being inserted into a local collection that
/// the request constructor then consumes. Without these the dataflow stops at
/// the local and every staged carrier looks unconsumed.
const PAYLOAD_ACCUMULATING_METHODS: &[&str] = &[
    "push",
    "extend",
    "insert",
    "append",
    "push_str",
    "extend_from_slice",
];

pub(super) fn scan_block_for_param_dispatch_flow(
    block: &syn::Block,
    dispatcher_params: &BTreeSet<String>,
    canonical_dispatch_methods: &BTreeSet<String>,
    var_deps: &mut BTreeMap<String, BTreeSet<usize>>,
    direct_dispatched_params: &mut BTreeSet<usize>,
    param_callee_flows: &mut Vec<ParamCalleeFlow>,
) {
    for stmt in &block.stmts {
        match stmt {
            syn::Stmt::Local(local) => {
                let init_deps = if let Some(init) = &local.init {
                    scan_expr_for_param_dispatch_flow(
                        &init.expr,
                        dispatcher_params,
                        canonical_dispatch_methods,
                        var_deps,
                        direct_dispatched_params,
                        param_callee_flows,
                    );
                    extract_expr_param_deps(&init.expr, var_deps)
                } else {
                    BTreeSet::new()
                };
                let mut idents = BTreeSet::new();
                extract_pat_bindings(&local.pat, &mut idents);
                for ident in idents {
                    var_deps.insert(ident, init_deps.clone());
                }
            }
            syn::Stmt::Expr(expr, _) => {
                if let syn::Expr::Assign(assign) = expr {
                    scan_expr_for_param_dispatch_flow(
                        &assign.right,
                        dispatcher_params,
                        canonical_dispatch_methods,
                        var_deps,
                        direct_dispatched_params,
                        param_callee_flows,
                    );
                    let deps = extract_expr_param_deps(&assign.right, var_deps);
                    let mut idents = BTreeSet::new();
                    extract_root_ident_from_expr(&assign.left, &mut idents);
                    for ident in idents {
                        var_deps.entry(ident).or_default().extend(deps.clone());
                    }
                } else if let syn::Expr::MethodCall(mc) = expr {
                    let mname = mc.method.to_string();
                    if PAYLOAD_ACCUMULATING_METHODS.contains(&mname.as_str()) {
                        if let syn::Expr::Path(p) = &*mc.receiver {
                            if let Some(ident) = p.path.get_ident() {
                                for arg in &mc.args {
                                    let deps = extract_expr_param_deps(arg, var_deps);
                                    var_deps.entry(ident.to_string()).or_default().extend(deps);
                                }
                            }
                        }
                    }
                    scan_expr_for_param_dispatch_flow(
                        expr,
                        dispatcher_params,
                        canonical_dispatch_methods,
                        var_deps,
                        direct_dispatched_params,
                        param_callee_flows,
                    );
                } else {
                    scan_expr_for_param_dispatch_flow(
                        expr,
                        dispatcher_params,
                        canonical_dispatch_methods,
                        var_deps,
                        direct_dispatched_params,
                        param_callee_flows,
                    );
                }
            }
            syn::Stmt::Item(_) => {}
            syn::Stmt::Macro(_) => {}
        }
    }
}

pub(super) fn scan_expr_for_param_dispatch_flow(
    expr: &syn::Expr,
    dispatcher_params: &BTreeSet<String>,
    canonical_dispatch_methods: &BTreeSet<String>,
    var_deps: &BTreeMap<String, BTreeSet<usize>>,
    direct_dispatched_params: &mut BTreeSet<usize>,
    param_callee_flows: &mut Vec<ParamCalleeFlow>,
) {
    match expr {
        syn::Expr::MethodCall(mc) => {
            let method_name = mc.method.to_string();
            let is_disp_receiver = if let syn::Expr::Path(p) = &*mc.receiver {
                p.path
                    .get_ident()
                    .is_some_and(|id| dispatcher_params.contains(&id.to_string()))
            } else {
                false
            };

            if is_disp_receiver && canonical_dispatch_methods.contains(&method_name) {
                for arg in &mc.args {
                    let deps = extract_expr_param_deps(arg, var_deps);
                    direct_dispatched_params.extend(deps);
                }
            } else {
                let callee_name = method_name;
                let recv_deps = extract_expr_param_deps(&mc.receiver, var_deps);
                for p_idx in recv_deps {
                    param_callee_flows.push(ParamCalleeFlow {
                        param_idx: p_idx,
                        callee_name: callee_name.clone(),
                        callee_arg_idx: 0,
                    });
                }
                for (arg_idx, arg) in mc.args.iter().enumerate() {
                    let deps = extract_expr_param_deps(arg, var_deps);
                    for p_idx in deps {
                        param_callee_flows.push(ParamCalleeFlow {
                            param_idx: p_idx,
                            callee_name: callee_name.clone(),
                            callee_arg_idx: arg_idx + 1,
                        });
                    }
                }
            }

            scan_expr_for_param_dispatch_flow(
                &mc.receiver,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
            for arg in &mc.args {
                scan_expr_for_param_dispatch_flow(
                    arg,
                    dispatcher_params,
                    canonical_dispatch_methods,
                    var_deps,
                    direct_dispatched_params,
                    param_callee_flows,
                );
            }
        }
        syn::Expr::Call(c) => {
            let callee_name = if let syn::Expr::Path(p) = &*c.func {
                p.path.segments.last().map(|s| s.ident.to_string())
            } else {
                None
            };
            if let Some(cname) = callee_name {
                for (arg_idx, arg) in c.args.iter().enumerate() {
                    let deps = extract_expr_param_deps(arg, var_deps);
                    for p_idx in deps {
                        param_callee_flows.push(ParamCalleeFlow {
                            param_idx: p_idx,
                            callee_name: cname.clone(),
                            callee_arg_idx: arg_idx,
                        });
                    }
                }
            }
            for arg in &c.args {
                scan_expr_for_param_dispatch_flow(
                    arg,
                    dispatcher_params,
                    canonical_dispatch_methods,
                    var_deps,
                    direct_dispatched_params,
                    param_callee_flows,
                );
            }
        }
        syn::Expr::If(i) => {
            scan_expr_for_param_dispatch_flow(
                &i.cond,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
            let mut branch_deps = var_deps.clone();
            scan_block_for_param_dispatch_flow(
                &i.then_branch,
                dispatcher_params,
                canonical_dispatch_methods,
                &mut branch_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
            if let Some((_, else_expr)) = &i.else_branch {
                scan_expr_for_param_dispatch_flow(
                    else_expr,
                    dispatcher_params,
                    canonical_dispatch_methods,
                    var_deps,
                    direct_dispatched_params,
                    param_callee_flows,
                );
            }
        }
        syn::Expr::Match(m) => {
            scan_expr_for_param_dispatch_flow(
                &m.expr,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
            for arm in &m.arms {
                let mut arm_deps = var_deps.clone();
                let mut idents = BTreeSet::new();
                extract_pat_bindings(&arm.pat, &mut idents);
                let expr_deps = extract_expr_param_deps(&m.expr, var_deps);
                for ident in idents {
                    arm_deps.insert(ident, expr_deps.clone());
                }
                scan_expr_for_param_dispatch_flow(
                    &arm.body,
                    dispatcher_params,
                    canonical_dispatch_methods,
                    &arm_deps,
                    direct_dispatched_params,
                    param_callee_flows,
                );
            }
        }
        syn::Expr::Block(b) => {
            let mut block_deps = var_deps.clone();
            scan_block_for_param_dispatch_flow(
                &b.block,
                dispatcher_params,
                canonical_dispatch_methods,
                &mut block_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
        }
        syn::Expr::ForLoop(f) => {
            scan_expr_for_param_dispatch_flow(
                &f.expr,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
            let mut loop_deps = var_deps.clone();
            let mut idents = BTreeSet::new();
            extract_pat_bindings(&f.pat, &mut idents);
            let expr_deps = extract_expr_param_deps(&f.expr, var_deps);
            for ident in idents {
                loop_deps.insert(ident, expr_deps.clone());
            }
            scan_block_for_param_dispatch_flow(
                &f.body,
                dispatcher_params,
                canonical_dispatch_methods,
                &mut loop_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
        }
        syn::Expr::While(w) => {
            scan_expr_for_param_dispatch_flow(
                &w.cond,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
            let mut loop_deps = var_deps.clone();
            scan_block_for_param_dispatch_flow(
                &w.body,
                dispatcher_params,
                canonical_dispatch_methods,
                &mut loop_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
        }
        syn::Expr::Loop(l) => {
            let mut loop_deps = var_deps.clone();
            scan_block_for_param_dispatch_flow(
                &l.body,
                dispatcher_params,
                canonical_dispatch_methods,
                &mut loop_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
        }
        syn::Expr::Binary(b) => {
            scan_expr_for_param_dispatch_flow(
                &b.left,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
            scan_expr_for_param_dispatch_flow(
                &b.right,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
        }
        syn::Expr::Unary(u) => {
            scan_expr_for_param_dispatch_flow(
                &u.expr,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
        }
        syn::Expr::Cast(c) => {
            scan_expr_for_param_dispatch_flow(
                &c.expr,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
        }
        syn::Expr::Try(t) => {
            scan_expr_for_param_dispatch_flow(
                &t.expr,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
        }
        syn::Expr::Paren(p) => {
            scan_expr_for_param_dispatch_flow(
                &p.expr,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
        }
        syn::Expr::Group(g) => {
            scan_expr_for_param_dispatch_flow(
                &g.expr,
                dispatcher_params,
                canonical_dispatch_methods,
                var_deps,
                direct_dispatched_params,
                param_callee_flows,
            );
        }
        syn::Expr::Array(a) => {
            for elem in &a.elems {
                scan_expr_for_param_dispatch_flow(
                    elem,
                    dispatcher_params,
                    canonical_dispatch_methods,
                    var_deps,
                    direct_dispatched_params,
                    param_callee_flows,
                );
            }
        }
        syn::Expr::Tuple(t) => {
            for elem in &t.elems {
                scan_expr_for_param_dispatch_flow(
                    elem,
                    dispatcher_params,
                    canonical_dispatch_methods,
                    var_deps,
                    direct_dispatched_params,
                    param_callee_flows,
                );
            }
        }
        syn::Expr::Struct(s) => {
            for f in &s.fields {
                scan_expr_for_param_dispatch_flow(
                    &f.expr,
                    dispatcher_params,
                    canonical_dispatch_methods,
                    var_deps,
                    direct_dispatched_params,
                    param_callee_flows,
                );
            }
        }
        _ => {}
    }
}
pub(super) struct FileImportCollector<'a> {
    pub(super) scope_imports: &'a mut Vec<BTreeMap<String, String>>,
}

impl<'ast, 'a> Visit<'ast> for FileImportCollector<'a> {
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if let Some(current_scope) = self.scope_imports.last_mut() {
            extract_use_tree("", &item.tree, current_scope);
        }
    }

    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        self.scope_imports.push(BTreeMap::new());
        syn::visit::visit_item_mod(self, item);
        self.scope_imports.pop();
    }
}
