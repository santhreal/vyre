//! AST traversal visitor definition and method implementations for host oracle detection.

use crate::gate::Finding;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use syn::spanned::Spanned;
use syn::visit::Visit;

use super::host_oracle_elimination_classify::{
    is_byte_unpack_codec_expr, type_is_numeric_payload_input,
};
use super::host_oracle_elimination_extract::{
    extract_mutated_storage_from_stmt, extract_pat_bindings, extract_read_idents_from_expr,
    extract_read_idents_from_stmt,
};
use super::host_oracle_elimination_records::{
    base_module_path, compares_operands, computes_from_operands, normalize_qualified_path,
    CallSiteRecord, FunctionParamRecord, FunctionRecord, ParamCalleeFlow, StaticConstRecord,
    EXACT_CANONICAL_DISPATCHER_PATHS, EXACT_CANONICAL_IR_BUILDER_PATHS, FIX, SCALAR_TYPES,
};
use super::host_oracle_elimination_scanners::{
    scan_block_for_param_dispatch_flow, stmt_contains_semantic_operation,
    type_is_exact_generic_param, InputBindingScanner,
};

/// Where a recorded call sits, which decides the context flags its
/// [`CallSiteRecord`] carries.
///
/// The walk records a call from several places: the traversal itself, a
/// fixture string it parsed, an argument of an operation registration. Each
/// place answers the same six questions about the call, so the answers are
/// named here once instead of being restated at every push.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CallContext {
    /// The walk's own position answers every question.
    Walk,
    /// A call parsed out of an expected-output fixture string, which has no
    /// enclosing function of its own.
    ExpectedOutputFixture,
    /// A call parsed out of a fallback fixture string.
    FallbackFixture,
    /// An argument of an expected-output constructor, inside the function the
    /// walk is in.
    ExpectedOutputArgument,
    /// A path named by an operation registration, which is neither test nor
    /// dispatch context.
    OperationRegistration,
}

/// Context flags one [`CallSiteRecord`] carries.
struct CallSiteContext {
    caller_fn_idx: Option<usize>,
    is_in_test: bool,
    is_in_expected_output: bool,
    is_in_fallback: bool,
    is_in_post_dispatch: bool,
    is_in_op_reg: bool,
}

pub(super) struct AstAnalysisVisitor {
    pub(super) file: PathBuf,
    pub(super) current_module: Vec<String>,
    pub(super) fn_index_offset: usize,
    pub(super) current_fn_idx: Option<usize>,
    pub(super) test_mod_depth: usize,
    pub(super) test_impl_depth: usize,
    pub(super) test_trait_depth: usize,
    pub(super) fmt_impl_depth: usize,
    pub(super) item_test_depth: usize,
    pub(super) in_expected_output_depth: usize,
    pub(super) in_fallback_depth: usize,
    pub(super) in_op_reg_depth: usize,
    pub(super) in_synthetic_oracle_depth: usize,
    /// Depth of closures passed to a combinator whose receiver executed a
    /// dispatch. Such a closure runs on the host with the dispatch result in
    /// hand, so the post-dispatch rules keep applying inside it.
    pub(super) in_post_dispatch_combinator_depth: usize,
    pub(super) in_gpu_dispatch_root: bool,
    pub(super) post_dispatch_phase: bool,
    pub(super) dispatcher_params: BTreeSet<String>,
    pub(super) non_data_diagnostic_params: BTreeSet<String>,
    pub(super) known_dispatch_exec_fns: BTreeSet<String>,
    /// Each `OperationRegistration` constructor mapped to the argument position
    /// of its `expected_output` callback, read from the registration source.
    pub(super) registration_expected_output_indices: BTreeMap<String, usize>,
    pub(super) derived_trait_dispatch_exec_methods: BTreeSet<String>,
    pub(super) derived_input_binding_methods: BTreeSet<String>,
    pub(super) local_declared_types: BTreeSet<String>,
    pub(super) scope_imports: Vec<BTreeMap<String, String>>,
    pub(super) struct_types_with_dispatcher: BTreeSet<String>,
    pub(super) current_impl_self_is_dispatcher: bool,
    pub(super) dispatched_data_vars: BTreeSet<String>,
    pub(super) functions: Vec<FunctionRecord>,
    pub(super) calls: Vec<CallSiteRecord>,
    pub(super) static_consts: Vec<StaticConstRecord>,
    pub(super) direct_findings: Vec<Finding>,
    pub(super) types_with_public_fields: BTreeSet<String>,
    pub(super) trait_impl_depth: usize,
}

impl AstAnalysisVisitor {
    pub(super) fn new(
        file: PathBuf,
        file_is_test_scoped: bool,
        fn_index_offset: usize,
        derived_trait_dispatch_exec_methods: BTreeSet<String>,
        derived_input_binding_methods: BTreeSet<String>,
    ) -> Self {
        let base_mod = base_module_path(&file);
        Self {
            file,
            current_module: base_mod,
            fn_index_offset,
            current_fn_idx: None,
            test_mod_depth: if file_is_test_scoped { 1 } else { 0 },
            test_impl_depth: 0,
            test_trait_depth: 0,
            fmt_impl_depth: 0,
            trait_impl_depth: 0,
            item_test_depth: 0,
            in_expected_output_depth: 0,
            in_fallback_depth: 0,
            in_op_reg_depth: 0,
            in_synthetic_oracle_depth: 0,
            in_post_dispatch_combinator_depth: 0,
            in_gpu_dispatch_root: false,
            post_dispatch_phase: false,
            dispatcher_params: BTreeSet::new(),
            non_data_diagnostic_params: BTreeSet::new(),
            known_dispatch_exec_fns: BTreeSet::new(),
            registration_expected_output_indices: BTreeMap::new(),
            derived_trait_dispatch_exec_methods,
            derived_input_binding_methods,
            local_declared_types: BTreeSet::new(),
            scope_imports: vec![BTreeMap::new()],
            struct_types_with_dispatcher: BTreeSet::new(),
            current_impl_self_is_dispatcher: false,
            dispatched_data_vars: BTreeSet::new(),
            functions: Vec::new(),
            calls: Vec::new(),
            static_consts: Vec::new(),
            direct_findings: Vec::new(),
            types_with_public_fields: BTreeSet::new(),
        }
    }

    pub(super) fn in_test(&self) -> bool {
        self.test_mod_depth > 0
            || self.test_impl_depth > 0
            || self.test_trait_depth > 0
            || self.item_test_depth > 0
    }

    /// Name of the function the walk is in, or `<anonymous>` outside one.
    pub(super) fn current_fn_name(&self) -> String {
        self.current_fn_idx
            .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
            .unwrap_or_else(|| "<anonymous>".to_string())
    }

    /// Report a host loop that runs after a dispatch in the same function.
    ///
    /// A `for`, `while` and `loop` each reach this the same way: the walk is
    /// inside a dispatch function past its dispatch, the loop itself dispatches
    /// nothing, and the code is not a test. The three differ only in how they
    /// find their body statements, so the condition and the finding are stated
    /// here once.
    pub(super) fn report_post_dispatch_host_loop(&mut self, span: proc_macro2::Span) {
        if !self.in_gpu_dispatch_root || !self.post_dispatch_phase || self.in_test() {
            return;
        }
        let current_fn_name = self.current_fn_name();
        self.direct_findings.push(Finding::at(
            self.file.clone(),
            span.start().line as u32,
            format!(
                "GPU dispatch function `{current_fn_name}` contains post-dispatch host loop/accumulation; \
                 post-dispatch semantic reductions must be dispatched on GPU"
            ),
            FIX,
        ));
    }

    /// Whether any statement of `block` executes a dispatch.
    fn block_dispatches(&self, block: &syn::Block) -> bool {
        block
            .stmts
            .iter()
            .any(|stmt| self.is_dispatch_execution_stmt(stmt))
    }

    /// Whether a `for` loop dispatches, in the sequence it walks or its body.
    pub(super) fn for_loop_dispatches(&self, expr: &syn::ExprForLoop) -> bool {
        self.is_dispatch_execution_expr(&expr.expr) || self.block_dispatches(&expr.body)
    }

    /// Whether a `while` loop dispatches, in its condition or its body.
    pub(super) fn while_loop_dispatches(&self, expr: &syn::ExprWhile) -> bool {
        self.is_dispatch_execution_expr(&expr.cond) || self.block_dispatches(&expr.body)
    }

    /// Whether a `loop` dispatches in its body.
    pub(super) fn loop_dispatches(&self, expr: &syn::ExprLoop) -> bool {
        self.block_dispatches(&expr.body)
    }

    /// Report host arithmetic over a dispatch result in a dispatch function.
    ///
    /// The operator must read its operands as data, the expression must not be
    /// the dispatch call itself, it must not be a permitted byte-unpacking
    /// codec, and it must read a value the walk saw a dispatch produce.
    pub(super) fn report_post_dispatch_host_arithmetic(&mut self, expr: &syn::ExprBinary) {
        if !self.in_gpu_dispatch_root || !self.post_dispatch_phase || self.in_test() {
            return;
        }
        if !computes_from_operands(&expr.op) && !compares_operands(&expr.op) {
            return;
        }
        let binary = syn::Expr::Binary(expr.clone());
        if self.is_dispatch_execution_expr(&binary) || is_byte_unpack_codec_expr(expr) {
            return;
        }
        if self.dispatched_data_vars.is_empty() {
            return;
        }
        let operates_on_dispatched = extract_read_idents_from_expr(&binary)
            .iter()
            .any(|id| self.dispatched_data_vars.contains(id));
        if !operates_on_dispatched {
            return;
        }
        let current_fn_name = self.current_fn_name();
        self.direct_findings.push(Finding::at(
            self.file.clone(),
            expr.span().start().line as u32,
            format!(
                "GPU dispatch function `{current_fn_name}` executes post-dispatch host arithmetic / semantic derivation on GPU results; \
                 mathematical operations on dispatch output must be dispatched on GPU"
            ),
            FIX,
        ));
    }

    /// Record a call to `callee` on `line`, in `context`.
    pub(super) fn record_call(
        &mut self,
        callee: String,
        line: u32,
        is_method_call: bool,
        context: CallContext,
    ) {
        let context = self.call_site_context(context);
        self.calls.push(CallSiteRecord {
            callee,
            caller_file: self.file.clone(),
            caller_module: self.current_module.clone(),
            caller_fn_idx: context.caller_fn_idx,
            line,
            is_method_call,
            is_in_test: context.is_in_test,
            is_in_expected_output: context.is_in_expected_output,
            is_in_fallback: context.is_in_fallback,
            is_in_post_dispatch: context.is_in_post_dispatch,
            is_in_op_reg: context.is_in_op_reg,
        });
    }

    /// Answer the six context questions for a call recorded in `context`.
    fn call_site_context(&self, context: CallContext) -> CallSiteContext {
        let walk = CallSiteContext {
            caller_fn_idx: self.current_fn_idx,
            is_in_test: self.in_test(),
            is_in_expected_output: self.in_expected_output_depth > 0,
            is_in_fallback: self.in_fallback_depth > 0,
            is_in_post_dispatch: self.in_gpu_dispatch_root && self.post_dispatch_phase,
            is_in_op_reg: self.in_op_reg_depth > 0,
        };
        match context {
            CallContext::Walk => walk,
            CallContext::ExpectedOutputFixture => CallSiteContext {
                caller_fn_idx: None,
                is_in_test: false,
                is_in_expected_output: true,
                is_in_fallback: false,
                is_in_post_dispatch: false,
                ..walk
            },
            CallContext::FallbackFixture => CallSiteContext {
                caller_fn_idx: None,
                is_in_test: false,
                is_in_expected_output: false,
                is_in_fallback: true,
                is_in_post_dispatch: false,
                ..walk
            },
            CallContext::ExpectedOutputArgument => CallSiteContext {
                is_in_expected_output: true,
                is_in_fallback: false,
                is_in_post_dispatch: false,
                ..walk
            },
            CallContext::OperationRegistration => CallSiteContext {
                is_in_test: false,
                is_in_expected_output: false,
                is_in_fallback: false,
                is_in_post_dispatch: false,
                is_in_op_reg: true,
                ..walk
            },
        }
    }

    pub(super) fn clean_path_string(path: &syn::Path) -> String {
        path.segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::")
    }

    pub(super) fn strip_turbofish(s: &str) -> &str {
        if let Some(pos) = s.find("::<") {
            &s[..pos]
        } else if let Some(pos) = s.find('<') {
            &s[..pos]
        } else {
            s
        }
    }

    pub(super) fn resolve_path_str(&self, path: &syn::Path) -> String {
        let clean_raw = Self::clean_path_string(path);
        if path.segments.len() == 1 {
            let ident = path.segments[0].ident.to_string();
            for scope in self.scope_imports.iter().rev() {
                if let Some(imported) = scope.get(&ident) {
                    return imported.clone();
                }
            }
            // Unimported bare identifiers fail closed (not matching any exact canonical path)
            ident
        } else {
            let first = path.segments[0].ident.to_string();
            for scope in self.scope_imports.iter().rev() {
                if let Some(imported_prefix) = scope.get(&first) {
                    let rest: Vec<String> = path
                        .segments
                        .iter()
                        .skip(1)
                        .map(|s| s.ident.to_string())
                        .collect();
                    return format!("{imported_prefix}::{}", rest.join("::"));
                }
            }
            clean_raw
        }
    }
    pub(super) fn resolve_qualified_type_path(&self, path: &syn::Path) -> Option<String> {
        let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        if segments.is_empty() {
            return None;
        }

        // 1. Path explicitly starting with `crate::` or external/stdlib root (`vyre_*`, `std::`, `core::`)
        if segments[0] == "crate"
            || segments[0].starts_with("vyre_")
            || segments[0] == "std"
            || segments[0] == "core"
        {
            return Some(segments.join("::"));
        }
        // 2. Path starting with `super::`
        if segments[0] == "super" {
            let mut mod_parts = self.current_module.clone();
            let mut skip = 0;
            while skip < segments.len() && segments[skip] == "super" {
                mod_parts.pop();
                skip += 1;
            }
            let mut res = vec!["crate".to_string()];
            res.extend(mod_parts);
            res.extend(segments[skip..].to_vec());
            return Some(res.join("::"));
        }

        // 3. Path starting with `self::`
        if segments[0] == "self" {
            let mut res = vec!["crate".to_string()];
            res.extend(self.current_module.clone());
            res.extend(segments[1..].to_vec());
            return Some(res.join("::"));
        }

        // 4. Single segment identifier
        if segments.len() == 1 {
            let ident = &segments[0];
            for scope in self.scope_imports.iter().rev() {
                if let Some(imported) = scope.get(ident) {
                    if imported.starts_with("crate::") || imported.starts_with("vyre_") {
                        return Some(imported.clone());
                    } else if imported.starts_with("super::") {
                        let mut mod_parts = self.current_module.clone();
                        let sub_segs: Vec<&str> = imported.split("::").collect();
                        let mut skip = 0;
                        while skip < sub_segs.len() && sub_segs[skip] == "super" {
                            mod_parts.pop();
                            skip += 1;
                        }
                        let mut res = vec!["crate".to_string()];
                        res.extend(mod_parts);
                        res.extend(sub_segs[skip..].iter().map(|s| s.to_string()));
                        return Some(res.join("::"));
                    } else {
                        return Some(format!("crate::{imported}"));
                    }
                }
            }
            if self.local_declared_types.contains(ident) {
                let mut res = vec!["crate".to_string()];
                res.extend(self.current_module.clone());
                res.push(ident.clone());
                return Some(res.join("::"));
            }
            return None;
        }

        // 5. Multi-segment path
        let first = &segments[0];
        for scope in self.scope_imports.iter().rev() {
            if let Some(imported_prefix) = scope.get(first) {
                let rest = segments[1..].join("::");
                if imported_prefix.starts_with("crate::") || imported_prefix.starts_with("vyre_") {
                    return Some(format!("{imported_prefix}::{rest}"));
                } else {
                    return Some(format!("crate::{imported_prefix}::{rest}"));
                }
            }
        }

        let mut res = vec!["crate".to_string()];
        res.extend(self.current_module.clone());
        res.extend(segments);
        Some(res.join("::"))
    }
    pub(super) fn resolve_qualified_fn_path(&self, path: &syn::Path) -> Option<String> {
        let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        if segments.is_empty() {
            return None;
        }

        if segments[0] == "crate"
            || segments[0].starts_with("vyre_")
            || segments[0] == "std"
            || segments[0] == "core"
        {
            return Some(segments.join("::"));
        }
        if segments[0] == "super" {
            let mut mod_parts = self.current_module.clone();
            let mut skip = 0;
            while skip < segments.len() && segments[skip] == "super" {
                mod_parts.pop();
                skip += 1;
            }
            let mut res = vec!["crate".to_string()];
            res.extend(mod_parts);
            res.extend(segments[skip..].to_vec());
            return Some(res.join("::"));
        }

        if segments[0] == "self" {
            let mut res = vec!["crate".to_string()];
            res.extend(self.current_module.clone());
            res.extend(segments[1..].to_vec());
            return Some(res.join("::"));
        }

        if segments.len() == 1 {
            let ident = &segments[0];
            for scope in self.scope_imports.iter().rev() {
                if let Some(imported) = scope.get(ident) {
                    if imported.starts_with("crate::") || imported.starts_with("vyre_") {
                        return Some(imported.clone());
                    } else if imported.starts_with("super::") {
                        let mut mod_parts = self.current_module.clone();
                        let sub_segs: Vec<&str> = imported.split("::").collect();
                        let mut skip = 0;
                        while skip < sub_segs.len() && sub_segs[skip] == "super" {
                            mod_parts.pop();
                            skip += 1;
                        }
                        let mut res = vec!["crate".to_string()];
                        res.extend(mod_parts);
                        res.extend(sub_segs[skip..].iter().map(|s| s.to_string()));
                        return Some(res.join("::"));
                    } else {
                        return Some(format!("crate::{imported}"));
                    }
                }
            }
            let mut res = vec!["crate".to_string()];
            res.extend(self.current_module.clone());
            res.push(ident.clone());
            return Some(res.join("::"));
        }

        let first = &segments[0];
        for scope in self.scope_imports.iter().rev() {
            if let Some(imported_prefix) = scope.get(first) {
                let rest = segments[1..].join("::");
                if imported_prefix.starts_with("crate::") || imported_prefix.starts_with("vyre_") {
                    return Some(format!("{imported_prefix}::{rest}"));
                } else {
                    return Some(format!("crate::{imported_prefix}::{rest}"));
                }
            }
        }

        let mut res = vec!["crate".to_string()];
        res.extend(self.current_module.clone());
        res.extend(segments);
        Some(res.join("::"))
    }

    pub(super) fn is_known_dispatch_fn_identifier(&self, callee_name: &str) -> bool {
        if self.known_dispatch_exec_fns.contains(callee_name)
            || self.is_canonical_dispatch_exec_method(callee_name)
        {
            return true;
        }
        let normalized = normalize_qualified_path(callee_name);
        if self.known_dispatch_exec_fns.contains(&normalized) {
            return true;
        }
        false
    }

    pub(super) fn extract_qualified_custom_types(
        &self,
        ty: &syn::Type,
        out: &mut BTreeSet<String>,
    ) {
        match ty {
            syn::Type::Path(tp) => {
                if let Some(seg) = tp.path.segments.last() {
                    let ident = seg.ident.to_string();
                    if ident == "Result"
                        || ident == "Option"
                        || ident == "Arc"
                        || ident == "Box"
                        || ident == "Vec"
                        || ident == "Cell"
                        || ident == "RefCell"
                        || ident == "Mutex"
                        || ident == "RwLock"
                    {
                        if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                            for arg in &args.args {
                                if let syn::GenericArgument::Type(inner_ty) = arg {
                                    self.extract_qualified_custom_types(inner_ty, out);
                                }
                            }
                        }
                        return;
                    }
                }

                if let Some(resolved) = self.resolve_qualified_type_path(&tp.path) {
                    let last_ident = tp
                        .path
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default();
                    if !SCALAR_TYPES.contains(&last_ident.as_str())
                        && last_ident != "String"
                        && last_ident != "str"
                        && last_ident != "Self"
                        && last_ident != "SemanticExecutionError"
                        && last_ident != "SemanticExecutor"
                        && last_ident != "SemanticExecutionRequest"
                        && last_ident != "SemanticExecutionOutput"
                        && last_ident != "Program"
                        && resolved.starts_with("crate::")
                    {
                        out.insert(resolved);
                    }
                }
            }
            syn::Type::Reference(r) => self.extract_qualified_custom_types(&r.elem, out),
            syn::Type::Slice(s) => self.extract_qualified_custom_types(&s.elem, out),
            syn::Type::Array(a) => self.extract_qualified_custom_types(&a.elem, out),
            syn::Type::Tuple(t) => {
                for elem in &t.elems {
                    self.extract_qualified_custom_types(elem, out);
                }
            }
            syn::Type::Paren(p) => self.extract_qualified_custom_types(&p.elem, out),
            syn::Type::Group(g) => self.extract_qualified_custom_types(&g.elem, out),
            _ => {}
        }
    }

    pub(super) fn is_canonical_ir_builder_path(&self, resolved: &str) -> bool {
        EXACT_CANONICAL_IR_BUILDER_PATHS.contains(&resolved)
            || (resolved.starts_with("crate::")
                && (resolved.ends_with("Plan")
                    || resolved.ends_with("LaunchPlan")
                    || resolved.ends_with("DispatchPlan")
                    || resolved.ends_with("ProgramShape")
                    || resolved.ends_with("ProgramLayout")
                    || resolved.ends_with("GpuScratch")
                    || resolved.ends_with("StaticInputKey")
                    || resolved.ends_with("ProgramCacheKey")
                    || resolved.ends_with("Programs")
                    || resolved.ends_with("Materializer")
                    || resolved.ends_with("HardwareEntry")
                    || resolved.ends_with("Stages")
                    || resolved.ends_with("Evidence")
                    || resolved.ends_with("Table")
                    || resolved.ends_with("Planner")
                    || resolved.ends_with("Selector")
                    || resolved.ends_with("Geometry")
                    || resolved.ends_with("LaunchGeometry")
                    || resolved.ends_with("Spec")
                    || resolved.ends_with("MatmulTiledCore")
                    || resolved.ends_with("ProgramCache")
                    || resolved.ends_with("ProgramCacheEntry")
                    || resolved.ends_with("ActionTable")
                    || resolved.ends_with("GotoTable")
                    || resolved.ends_with("DispatchTable")
                    || resolved.ends_with("BytecodeDispatchTable")))
            || (self.local_declared_types.contains(resolved)
                && (resolved.ends_with("Plan")
                    || resolved.ends_with("LaunchPlan")
                    || resolved.ends_with("DispatchPlan")
                    || resolved.ends_with("ProgramShape")
                    || resolved.ends_with("ProgramLayout")
                    || resolved.ends_with("GpuScratch")
                    || resolved.ends_with("StaticInputKey")
                    || resolved.ends_with("ProgramCacheKey")
                    || resolved.ends_with("Programs")
                    || resolved.ends_with("Materializer")
                    || resolved.ends_with("Stages")
                    || resolved.ends_with("Evidence")
                    || resolved.ends_with("Planner")
                    || resolved.ends_with("Selector")
                    || resolved.ends_with("Geometry")
                    || resolved.ends_with("LaunchGeometry")
                    || resolved.ends_with("Spec")
                    || resolved.ends_with("MatmulTiledCore")
                    || resolved.ends_with("ProgramCache")
                    || resolved.ends_with("ProgramCacheEntry")
                    || resolved.ends_with("ActionTable")
                    || resolved.ends_with("GotoTable")
                    || resolved.ends_with("DispatchTable")
                    || resolved.ends_with("BytecodeDispatchTable")))
    }

    pub(super) fn is_canonical_dispatcher_path(&self, resolved: &str) -> bool {
        EXACT_CANONICAL_DISPATCHER_PATHS.contains(&resolved)
    }

    pub(super) fn is_ir_builder_sig(&self, sig: &syn::Signature) -> bool {
        let syn::ReturnType::Type(_, return_type) = &sig.output else {
            return false;
        };
        self.is_canonical_ir_builder_return_type(return_type)
    }

    pub(super) fn is_canonical_ir_builder_return_type(&self, ty: &syn::Type) -> bool {
        match ty {
            syn::Type::Path(type_path) => {
                let resolved = self.resolve_path_str(&type_path.path);
                if self.is_canonical_ir_builder_path(&resolved) {
                    return true;
                }

                // Handle Result<T, E>, Option<T>, Arc<T>, Box<T>, Vec<T>
                if let Some(seg) = type_path.path.segments.last() {
                    let ident = seg.ident.to_string();
                    if ident == "Result" {
                        if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                            if let Some(syn::GenericArgument::Type(ok_ty)) = args.args.first() {
                                return self.is_canonical_ir_builder_return_type(ok_ty);
                            }
                        }
                        return false;
                    }
                    if ident == "Option" || ident == "Arc" || ident == "Box" || ident == "Vec" {
                        if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                            if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                                return self.is_canonical_ir_builder_return_type(inner_ty);
                            }
                        }
                        return false;
                    }
                }
                false
            }
            syn::Type::Tuple(tuple) => {
                // A tuple is an IR root ONLY IF it is non-empty and EVERY element is a canonical AST owner
                !tuple.elems.is_empty()
                    && tuple
                        .elems
                        .iter()
                        .all(|elem| self.is_canonical_ir_builder_return_type(elem))
            }
            syn::Type::ImplTrait(impl_trait) => {
                for bound in &impl_trait.bounds {
                    if let syn::TypeParamBound::Trait(trait_bound) = bound {
                        if let Some(seg) = trait_bound.path.segments.last() {
                            if seg.ident == "Iterator" {
                                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                                    for arg in &args.args {
                                        if let syn::GenericArgument::AssocType(assoc) = arg {
                                            if assoc.ident == "Item" {
                                                return self.is_canonical_ir_builder_return_type(
                                                    &assoc.ty,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                false
            }
            syn::Type::Reference(r) => self.is_canonical_ir_builder_return_type(&r.elem),
            syn::Type::Slice(s) => self.is_canonical_ir_builder_return_type(&s.elem),
            syn::Type::Array(a) => self.is_canonical_ir_builder_return_type(&a.elem),
            _ => false,
        }
    }

    pub(super) fn has_canonical_ir_param(&self, sig: &syn::Signature) -> bool {
        sig.inputs.iter().any(|input| {
            if let syn::FnArg::Typed(pat_type) = input {
                self.type_contains_canonical_ir(&pat_type.ty)
            } else {
                false
            }
        })
    }

    pub(super) fn type_contains_canonical_ir(&self, ty: &syn::Type) -> bool {
        match ty {
            syn::Type::Path(p) => {
                let resolved = self.resolve_path_str(&p.path);
                if self.is_canonical_ir_builder_path(&resolved) {
                    return true;
                }
                if let Some(seg) = p.path.segments.last() {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        for arg in &args.args {
                            if let syn::GenericArgument::Type(inner) = arg {
                                if self.type_contains_canonical_ir(inner) {
                                    return true;
                                }
                            }
                        }
                    }
                }
                false
            }
            syn::Type::Reference(r) => self.type_contains_canonical_ir(&r.elem),
            syn::Type::Slice(s) => self.type_contains_canonical_ir(&s.elem),
            syn::Type::Array(a) => self.type_contains_canonical_ir(&a.elem),
            syn::Type::Tuple(t) => t
                .elems
                .iter()
                .any(|elem| self.type_contains_canonical_ir(elem)),
            _ => false,
        }
    }

    pub(super) fn type_contains_canonical_dispatcher(&self, ty: &syn::Type) -> bool {
        match ty {
            syn::Type::Path(p) => {
                let resolved = self.resolve_path_str(&p.path);
                if self.is_canonical_dispatcher_path(&resolved) {
                    return true;
                }

                let wrapper_path = self.resolve_qualified_type_path(&p.path).or_else(|| {
                    if p.path.segments.len() == 1 {
                        let mut parts = vec!["crate".to_string()];
                        parts.extend(self.current_module.clone());
                        parts.push(p.path.segments[0].ident.to_string());
                        Some(parts.join("::"))
                    } else {
                        None
                    }
                });
                if wrapper_path
                    .as_ref()
                    .is_some_and(|path| self.struct_types_with_dispatcher.contains(path))
                {
                    return true;
                }

                if let Some(seg) = p.path.segments.last() {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        for arg in &args.args {
                            if let syn::GenericArgument::Type(inner) = arg {
                                if self.type_contains_canonical_dispatcher(inner) {
                                    return true;
                                }
                            }
                        }
                    }
                }
                false
            }
            syn::Type::TraitObject(to) => {
                for bound in &to.bounds {
                    if let syn::TypeParamBound::Trait(tb) = bound {
                        let resolved = self.resolve_path_str(&tb.path);
                        if self.is_canonical_dispatcher_path(&resolved) {
                            return true;
                        }
                    }
                }
                false
            }
            syn::Type::ImplTrait(it) => {
                for bound in &it.bounds {
                    if let syn::TypeParamBound::Trait(tb) = bound {
                        let resolved = self.resolve_path_str(&tb.path);
                        if self.is_canonical_dispatcher_path(&resolved) {
                            return true;
                        }
                    }
                }
                false
            }
            syn::Type::Reference(r) => self.type_contains_canonical_dispatcher(&r.elem),
            _ => false,
        }
    }

    pub(super) fn qualified_local_type_name(&self, ident: &syn::Ident) -> String {
        let mut parts = vec!["crate".to_string()];
        parts.extend(self.current_module.clone());
        parts.push(ident.to_string());
        parts.join("::")
    }

    pub(super) fn struct_contains_canonical_dispatcher(&self, item: &syn::ItemStruct) -> bool {
        let mut dispatcher_generics = BTreeSet::new();
        for param in &item.generics.params {
            if let syn::GenericParam::Type(type_param) = param {
                if type_param.bounds.iter().any(|bound| {
                    matches!(
                        bound,
                        syn::TypeParamBound::Trait(trait_bound)
                            if self.is_canonical_dispatcher_path(
                                &self.resolve_path_str(&trait_bound.path)
                            )
                    )
                }) {
                    dispatcher_generics.insert(type_param.ident.to_string());
                }
            }
        }
        if let Some(where_clause) = &item.generics.where_clause {
            for predicate in &where_clause.predicates {
                let syn::WherePredicate::Type(type_predicate) = predicate else {
                    continue;
                };
                let syn::Type::Path(type_path) = &type_predicate.bounded_ty else {
                    continue;
                };
                let Some(generic_ident) = type_path.path.get_ident() else {
                    continue;
                };
                if type_predicate.bounds.iter().any(|bound| {
                    matches!(
                        bound,
                        syn::TypeParamBound::Trait(trait_bound)
                            if self.is_canonical_dispatcher_path(
                                &self.resolve_path_str(&trait_bound.path)
                            )
                    )
                }) {
                    dispatcher_generics.insert(generic_ident.to_string());
                }
            }
        }

        item.fields.iter().any(|field| {
            self.type_contains_canonical_dispatcher(&field.ty)
                || dispatcher_generics
                    .iter()
                    .any(|generic| type_is_exact_generic_param(&field.ty, generic))
        })
    }
    pub(super) fn returns_operation_metadata_type(&self, ret: &syn::ReturnType) -> bool {
        match ret {
            syn::ReturnType::Default => false,
            syn::ReturnType::Type(_, ty) => self.type_is_operation_metadata(ty),
        }
    }

    pub(super) fn type_is_operation_metadata(&self, ty: &syn::Type) -> bool {
        match ty {
            syn::Type::ImplTrait(impl_trait) => impl_trait.bounds.iter().any(|bound| {
                let syn::TypeParamBound::Trait(trait_bound) = bound else {
                    return false;
                };
                let Some(iterator) = trait_bound.path.segments.last() else {
                    return false;
                };
                if iterator.ident != "Iterator" {
                    return false;
                }
                let syn::PathArguments::AngleBracketed(arguments) = &iterator.arguments else {
                    return false;
                };
                arguments.args.iter().any(|argument| {
                    let syn::GenericArgument::AssocType(item) = argument else {
                        return false;
                    };
                    item.ident == "Item" && self.type_is_operation_metadata(&item.ty)
                })
            }),
            syn::Type::Path(type_path) => {
                self.resolve_path_str(&type_path.path)
                    == "vyre_foundation::operation::SemanticOperation"
            }
            syn::Type::Reference(reference) => self.type_is_operation_metadata(&reference.elem),
            syn::Type::Paren(paren) => self.type_is_operation_metadata(&paren.elem),
            _ => false,
        }
    }

    pub(super) fn is_canonical_dispatch_exec_method(&self, method: &str) -> bool {
        self.derived_trait_dispatch_exec_methods.contains(method)
    }

    pub(super) fn is_dispatch_execution_expr(&self, expr: &syn::Expr) -> bool {
        if self.dispatcher_params.is_empty() {
            return false;
        }
        match expr {
            syn::Expr::MethodCall(mc) => {
                let method_name = mc.method.to_string();
                // 1. Direct method call on a dispatcher param or wrapper field
                if self.is_canonical_dispatch_exec_method(&method_name) {
                    let idents = extract_read_idents_from_expr(&mc.receiver);
                    if idents.iter().any(|id| self.dispatcher_params.contains(id)) {
                        return true;
                    }
                }
                if self.is_dispatch_execution_expr(&mc.receiver) {
                    return true;
                }
                // 3. Dispatcher param passed as an argument to a method that executes dispatch
                for arg in &mc.args {
                    let arg_idents = extract_read_idents_from_expr(arg);
                    if arg_idents
                        .iter()
                        .any(|id| self.dispatcher_params.contains(id))
                    {
                        if self.known_dispatch_exec_fns.contains(&method_name)
                            || self.is_canonical_dispatch_exec_method(&method_name)
                        {
                            return true;
                        }
                    }
                    if self.is_dispatch_execution_expr(arg) {
                        return true;
                    }
                }
                false
            }
            syn::Expr::Call(c) => {
                let resolved_callee = if let syn::Expr::Path(p) = c.func.as_ref() {
                    self.resolve_qualified_fn_path(&p.path)
                } else {
                    None
                };
                for arg in &c.args {
                    let arg_idents = extract_read_idents_from_expr(arg);
                    if arg_idents
                        .iter()
                        .any(|id| self.dispatcher_params.contains(id))
                    {
                        if let Some(name) = &resolved_callee {
                            if self.is_known_dispatch_fn_identifier(name) {
                                return true;
                            }
                        }
                    }
                    if self.is_dispatch_execution_expr(arg) {
                        return true;
                    }
                }
                false
            }
            syn::Expr::Binary(b) => {
                self.is_dispatch_execution_expr(&b.left)
                    || self.is_dispatch_execution_expr(&b.right)
            }
            syn::Expr::Unary(u) => self.is_dispatch_execution_expr(&u.expr),
            syn::Expr::Cast(c) => self.is_dispatch_execution_expr(&c.expr),
            syn::Expr::Try(t) => self.is_dispatch_execution_expr(&t.expr),
            syn::Expr::Match(m) => {
                self.is_dispatch_execution_expr(&m.expr)
                    || m.arms
                        .iter()
                        .any(|a| self.is_dispatch_execution_expr(&a.body))
            }
            syn::Expr::If(i) => {
                self.is_dispatch_execution_expr(&i.cond)
                    || i.then_branch
                        .stmts
                        .iter()
                        .any(|s| self.is_dispatch_execution_stmt(s))
                    || i.else_branch
                        .as_ref()
                        .is_some_and(|(_, e)| self.is_dispatch_execution_expr(e))
            }
            syn::Expr::ForLoop(f) => {
                self.is_dispatch_execution_expr(&f.expr)
                    || f.body
                        .stmts
                        .iter()
                        .any(|s| self.is_dispatch_execution_stmt(s))
            }
            syn::Expr::While(w) => {
                self.is_dispatch_execution_expr(&w.cond)
                    || w.body
                        .stmts
                        .iter()
                        .any(|s| self.is_dispatch_execution_stmt(s))
            }
            syn::Expr::Loop(l) => l
                .body
                .stmts
                .iter()
                .any(|s| self.is_dispatch_execution_stmt(s)),
            syn::Expr::Block(b) => b
                .block
                .stmts
                .iter()
                .any(|s| self.is_dispatch_execution_stmt(s)),
            syn::Expr::Paren(p) => self.is_dispatch_execution_expr(&p.expr),
            syn::Expr::Reference(r) => self.is_dispatch_execution_expr(&r.expr),
            syn::Expr::Tuple(t) => t.elems.iter().any(|e| self.is_dispatch_execution_expr(e)),
            syn::Expr::Array(a) => a.elems.iter().any(|e| self.is_dispatch_execution_expr(e)),
            syn::Expr::Struct(s) => {
                s.fields
                    .iter()
                    .any(|f| self.is_dispatch_execution_expr(&f.expr))
                    || s.rest
                        .as_ref()
                        .is_some_and(|r| self.is_dispatch_execution_expr(r))
            }
            syn::Expr::Assign(a) => {
                self.is_dispatch_execution_expr(&a.left)
                    || self.is_dispatch_execution_expr(&a.right)
            }
            syn::Expr::Index(idx) => {
                self.is_dispatch_execution_expr(&idx.expr)
                    || self.is_dispatch_execution_expr(&idx.index)
            }
            syn::Expr::Field(f) => self.is_dispatch_execution_expr(&f.base),
            syn::Expr::Return(r) => r
                .expr
                .as_ref()
                .is_some_and(|e| self.is_dispatch_execution_expr(e)),
            syn::Expr::Let(l) => self.is_dispatch_execution_expr(&l.expr),
            syn::Expr::Macro(m) => {
                if let Ok(expr) = syn::parse2::<syn::Expr>(m.mac.tokens.clone()) {
                    self.is_dispatch_execution_expr(&expr)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub(super) fn is_dispatch_execution_stmt(&self, stmt: &syn::Stmt) -> bool {
        match stmt {
            syn::Stmt::Local(l) => {
                if let Some(init) = &l.init {
                    self.is_dispatch_execution_expr(&init.expr)
                } else {
                    false
                }
            }
            syn::Stmt::Expr(e, _) => self.is_dispatch_execution_expr(e),
            syn::Stmt::Macro(m) => {
                if let Ok(expr) = syn::parse2::<syn::Expr>(m.mac.tokens.clone()) {
                    self.is_dispatch_execution_expr(&expr)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub(super) fn extract_dispatcher_params(&self, sig: &syn::Signature) -> BTreeSet<String> {
        let mut dispatcher_generic_names = BTreeSet::new();
        for param in &sig.generics.params {
            if let syn::GenericParam::Type(type_param) = param {
                for bound in &type_param.bounds {
                    if let syn::TypeParamBound::Trait(trait_bound) = bound {
                        let resolved = self.resolve_path_str(&trait_bound.path);
                        if self.is_canonical_dispatcher_path(&resolved) {
                            dispatcher_generic_names.insert(type_param.ident.to_string());
                        }
                    }
                }
            }
        }
        if let Some(where_clause) = &sig.generics.where_clause {
            for predicate in &where_clause.predicates {
                if let syn::WherePredicate::Type(pred_type) = predicate {
                    for bound in &pred_type.bounds {
                        if let syn::TypeParamBound::Trait(trait_bound) = bound {
                            let resolved = self.resolve_path_str(&trait_bound.path);
                            if self.is_canonical_dispatcher_path(&resolved) {
                                if let syn::Type::Path(tp) = &pred_type.bounded_ty {
                                    if tp.qself.is_none() && tp.path.segments.len() == 1 {
                                        let seg = &tp.path.segments[0];
                                        if matches!(seg.arguments, syn::PathArguments::None) {
                                            dispatcher_generic_names.insert(seg.ident.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut dispatcher_params = BTreeSet::new();
        for input in &sig.inputs {
            match input {
                syn::FnArg::Receiver(_) => {
                    if self.current_impl_self_is_dispatcher {
                        dispatcher_params.insert("self".to_string());
                    }
                }
                syn::FnArg::Typed(pat_type) => {
                    let is_disp = self.type_contains_canonical_dispatcher(&pat_type.ty)
                        || dispatcher_generic_names
                            .iter()
                            .any(|g| type_is_exact_generic_param(&pat_type.ty, g));
                    if is_disp {
                        let mut idents = BTreeSet::new();
                        extract_pat_bindings(&pat_type.pat, &mut idents);
                        dispatcher_params.extend(idents);
                    }
                }
            }
        }
        dispatcher_params
    }

    pub(super) fn extract_non_data_diagnostic_params(
        &self,
        sig: &syn::Signature,
    ) -> BTreeSet<String> {
        let mut params = BTreeSet::new();
        for input in &sig.inputs {
            let syn::FnArg::Typed(pat_type) = input else {
                continue;
            };
            if !type_is_numeric_payload_input(&pat_type.ty)
                && !self.type_contains_canonical_dispatcher(&pat_type.ty)
                && !self.type_contains_canonical_ir(&pat_type.ty)
            {
                extract_pat_bindings(&pat_type.pat, &mut params);
            }
        }
        params
    }

    pub(super) fn is_gpu_dispatch_sig(&self, sig: &syn::Signature, block: &syn::Block) -> bool {
        let dispatcher_params = self.extract_dispatcher_params(sig);
        if dispatcher_params.is_empty() {
            return false;
        }

        let mut temp_visitor = AstAnalysisVisitor::new(
            self.file.clone(),
            false,
            0,
            self.derived_trait_dispatch_exec_methods.clone(),
            self.derived_input_binding_methods.clone(),
        );
        temp_visitor.dispatcher_params = dispatcher_params;
        temp_visitor.known_dispatch_exec_fns = self.known_dispatch_exec_fns.clone();
        temp_visitor.scope_imports = self.scope_imports.clone();
        temp_visitor.current_module = self.current_module.clone();
        temp_visitor.local_declared_types = self.local_declared_types.clone();
        temp_visitor.struct_types_with_dispatcher = self.struct_types_with_dispatcher.clone();
        block
            .stmts
            .iter()
            .any(|s| temp_visitor.is_dispatch_execution_stmt(s))
    }
    pub(super) fn extract_dispatch_inputs_from_stmt(&self, stmt: &syn::Stmt) -> BTreeSet<String> {
        let mut inputs = BTreeSet::new();
        match stmt {
            syn::Stmt::Local(l) => {
                if let Some(init) = &l.init {
                    inputs.extend(extract_read_idents_from_expr(&init.expr));
                }
            }
            syn::Stmt::Expr(e, _) => {
                inputs.extend(extract_read_idents_from_expr(e));
            }
            syn::Stmt::Macro(m) => {
                if let Ok(expr) = syn::parse2::<syn::Expr>(m.mac.tokens.clone()) {
                    inputs.extend(extract_read_idents_from_expr(&expr));
                }
            }
            _ => {}
        }
        inputs
    }

    pub(super) fn inspect_function_statements(&mut self, stmts: &[syn::Stmt], is_gpu_root: bool) {
        let prev_in_gpu = self.in_gpu_dispatch_root;
        let prev_post_dispatch = self.post_dispatch_phase;
        let prev_dispatched_vars = self.dispatched_data_vars.clone();

        self.in_gpu_dispatch_root = is_gpu_root;
        self.dispatched_data_vars.clear();

        if !is_gpu_root {
            self.post_dispatch_phase = false;
            for stmt in stmts {
                syn::visit::visit_stmt(self, stmt);
            }
            self.in_gpu_dispatch_root = prev_in_gpu;
            self.post_dispatch_phase = prev_post_dispatch;
            self.dispatched_data_vars = prev_dispatched_vars;
            return;
        }

        let dispatch_indices: Vec<usize> = stmts
            .iter()
            .enumerate()
            .filter_map(|(idx, stmt)| self.is_dispatch_execution_stmt(stmt).then_some(idx))
            .collect();
        let mut has_prior_dispatch = false;

        for (idx, stmt) in stmts.iter().enumerate() {
            let is_dispatch_exec = dispatch_indices.contains(&idx);
            let has_subsequent_dispatch = dispatch_indices
                .iter()
                .any(|&dispatch_idx| dispatch_idx > idx);

            let has_semantic_op = stmt_contains_semantic_operation(stmt, self);

            let semantic_output_feeds_dispatch = if has_prior_dispatch
                && has_subsequent_dispatch
                && !is_dispatch_exec
                && has_semantic_op
            {
                let mut tainted = extract_mutated_storage_from_stmt(stmt);
                if let syn::Stmt::Local(l) = stmt {
                    extract_pat_bindings(&l.pat, &mut tainted);
                }
                let mut feeds_dispatch = false;
                if !tainted.is_empty() {
                    for later_stmt in &stmts[idx + 1..] {
                        let reads = extract_read_idents_from_stmt(later_stmt);
                        if self.is_dispatch_execution_stmt(later_stmt) {
                            if reads.iter().any(|name| tainted.contains(name)) {
                                feeds_dispatch = true;
                                break;
                            }
                            continue;
                        }
                        if reads.iter().any(|name| tainted.contains(name)) {
                            tainted.extend(extract_mutated_storage_from_stmt(later_stmt));
                            if let syn::Stmt::Local(l) = later_stmt {
                                extract_pat_bindings(&l.pat, &mut tainted);
                            }
                        }
                    }
                }
                feeds_dispatch
            } else {
                false
            };

            self.post_dispatch_phase = has_prior_dispatch
                && !is_dispatch_exec
                && (!has_subsequent_dispatch || !semantic_output_feeds_dispatch);

            if is_dispatch_exec {
                self.dispatched_data_vars
                    .extend(extract_mutated_storage_from_stmt(stmt));
                if let syn::Stmt::Local(l) = stmt {
                    extract_pat_bindings(&l.pat, &mut self.dispatched_data_vars);
                }
            } else if has_prior_dispatch {
                let reads = extract_read_idents_from_stmt(stmt);
                if reads
                    .iter()
                    .any(|name| self.dispatched_data_vars.contains(name))
                {
                    self.dispatched_data_vars
                        .extend(extract_mutated_storage_from_stmt(stmt));
                    if let syn::Stmt::Local(l) = stmt {
                        extract_pat_bindings(&l.pat, &mut self.dispatched_data_vars);
                    }
                    if let syn::Stmt::Expr(syn::Expr::ForLoop(f), _) = stmt {
                        extract_pat_bindings(&f.pat, &mut self.dispatched_data_vars);
                    }
                }
            }

            syn::visit::visit_stmt(self, stmt);

            if is_dispatch_exec {
                has_prior_dispatch = true;
                self.post_dispatch_phase = !has_subsequent_dispatch;
            }
        }

        self.in_gpu_dispatch_root = prev_in_gpu;
        self.post_dispatch_phase = prev_post_dispatch;
        self.dispatched_data_vars = prev_dispatched_vars;
    }

    pub(super) fn stages_semantic_input_binding(&self, stmts: &[syn::Stmt]) -> bool {
        if self.derived_input_binding_methods.is_empty() {
            return false;
        }

        let mut semantic_taint = BTreeSet::new();
        for stmt in stmts {
            let reads = extract_read_idents_from_stmt(stmt);
            let extends_taint = stmt_contains_semantic_operation(stmt, self)
                || reads.iter().any(|ident| semantic_taint.contains(ident));
            if extends_taint {
                semantic_taint.extend(extract_mutated_storage_from_stmt(stmt));
            }

            let mut scanner = InputBindingScanner {
                visitor: self,
                dispatcher_params: &self.dispatcher_params,
                semantic_taint: &semantic_taint,
                consumes_semantic_storage: false,
            };
            scanner.visit_stmt(stmt);
            if scanner.consumes_semantic_storage {
                return true;
            }
        }
        false
    }

    pub(super) fn scan_macro_tokens_for_references(&mut self, tokens: &proc_macro2::TokenStream) {
        if self.in_expected_output_depth > 0
            || self.in_fallback_depth > 0
            || self.in_op_reg_depth > 0
        {
            for tt in tokens.clone() {
                match tt {
                    proc_macro2::TokenTree::Ident(ident) => {
                        let ident_str = ident.to_string();
                        if ident_str != "vec"
                            && ident_str != "Vec"
                            && ident_str != "new"
                            && ident_str != "as"
                            && ident_str != "u8"
                            && ident_str != "u32"
                            && ident_str != "Ok"
                            && ident_str != "Some"
                        {
                            let line = ident.span().start().line as u32;
                            self.record_call(ident_str, line, false, CallContext::Walk);
                        }
                    }
                    proc_macro2::TokenTree::Group(group) => {
                        self.scan_macro_tokens_for_references(&group.stream());
                    }
                    _ => {}
                }
            }
        }
    }
    pub(super) fn analyze_fn_params_and_dispatch_flow(
        &self,
        sig: &syn::Signature,
        block: Option<&syn::Block>,
    ) -> (
        Vec<FunctionParamRecord>,
        BTreeSet<usize>,
        Vec<ParamCalleeFlow>,
    ) {
        let mut params = Vec::new();
        let mut var_deps = BTreeMap::new();

        for (input_idx, input) in sig.inputs.iter().enumerate() {
            match input {
                syn::FnArg::Receiver(_) => {
                    params.push(FunctionParamRecord {
                        name: "self".to_string(),
                        qualified_custom_types: BTreeSet::new(),
                    });
                    var_deps.insert("self".to_string(), std::iter::once(input_idx).collect());
                }
                syn::FnArg::Typed(pat_type) => {
                    let mut idents = BTreeSet::new();
                    extract_pat_bindings(&pat_type.pat, &mut idents);
                    let mut custom_types = BTreeSet::new();
                    self.extract_qualified_custom_types(&pat_type.ty, &mut custom_types);
                    let primary_name = idents
                        .iter()
                        .next()
                        .cloned()
                        .unwrap_or_else(|| format!("arg{input_idx}"));
                    params.push(FunctionParamRecord {
                        name: primary_name,
                        qualified_custom_types: custom_types,
                    });
                    for ident in idents {
                        var_deps.insert(ident, std::iter::once(input_idx).collect());
                    }
                }
            }
        }

        let mut direct_dispatched_param_indices = BTreeSet::new();
        let mut param_callee_flows = Vec::new();

        if let Some(blk) = block {
            scan_block_for_param_dispatch_flow(
                blk,
                &self.dispatcher_params,
                &self.derived_trait_dispatch_exec_methods,
                &mut var_deps,
                &mut direct_dispatched_param_indices,
                &mut param_callee_flows,
            );
        }

        (params, direct_dispatched_param_indices, param_callee_flows)
    }
}
