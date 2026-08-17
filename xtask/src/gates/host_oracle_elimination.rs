//! `cargo xtask host-oracle-elimination`  -  zero production CPU oracles in shipping crates.
//!
//! A shipping library (`vyre-libs`, `vyre-primitives`) must not compile or execute
//! host mathematical oracles, reference simulations, or unisolated data-processing semantic
//! twins in production code. CPU reference implementations (`cpu_ref`, `cpu_reference`,
//! `vyre_reference` simulators, and generic-named host algorithms that only serve tests)
//! exist exclusively to provide independent semantic witnesses for test verification; they
//! must never be linked into production binaries or invoked at registration time for dynamic
//! expected-output evaluation.
//!
//! The classification is 100% source-derived and structural:
//! - Candidate detection is role-independent: body AST visitor (`BodyFeatureVisitor`) inspects
//!   `ExprBinary` arithmetic/bitwise/shifts, `ExprUnary` numeric/not, numeric methods (min/max/clamp/abs/sqrt/etc.),
//!   branch-on-data classifiers (ExprIf/ExprMatch), loops, iterators, and search/sort algorithms.
//! - Roles establish reachability through structural types, effects, and call graphs, never by
//!   erasing candidate status.
//! - Trusted roots require exact canonical qualified type provenance derived from actual workspace
//!   declarations and imports (`vyre_foundation::ir::*`, `vyre_foundation::operation::OperationRegistration`,
//!   `vyre_foundation::program_dispatch::*`); bare names, glob imports, `crate::bogus::*`,
//!   sibling-module imports, and local dummy traits/structs fail closed.
//! - IR builder roots strictly require returning AST/IR owner types (`Program`, `Node`, `Expr`,
//!   `OperationRegistration`), optionally wrapped in `Result<T, _>`, `Option<T>`,
//!   `Arc<T>`, `Box<T>`, `Vec<T>`, or homogenous AST owner tuples; metadata types (`DataType`, etc.) and
//!   mixed data-output tuples (`(Vec<u32>, DataType)`) or data results (`Result<Vec<u32>, FusionError>`)
//!   do NOT establish builder roots.
//! - Dispatch roots strictly require an exact canonical dispatcher capability parameter
//!   (`ProgramDispatcher`) AND device dispatch execution in the body, derived dynamically from grounded
//!   trait signatures taking `Program` or `ResidentDispatchStep` plan types and producing dispatch/readback
//!   effects (capability, metadata, allocation, upload-only, and free methods do not establish execution).
//!   Passing dispatcher to non-dispatching helpers does not root; helpers that execute dispatch establish execution
//!   transitively.
//! - Dispatch error / fallback paths (`Err(_)`, `unwrap_or_else`, `or_else`, `*.is_err()`, etc.) are
//!   forbidden from executing host candidates or inline semantic operations; fallback calls do NOT
//!   receive reachability edges and are convicted.
//! - Post-dispatch host reductions / aggregations (`.any()`, `.all()`, `.sum()`, `.count()`, `.fold()`,
//!   `.reduce()`, loops over output) are forbidden; reductions must be dispatched on GPU.
//!   Post-dispatch phase is expression-granular: nested match expressions (`match dispatcher.dispatch(..) { Ok(out) => ... }`),
//!   chained method calls (`dispatcher.dispatch(..).map(|out| ...)`), and conditional expressions flip into post-dispatch
//!   phase for their success continuations.
//! - OperationRegistration expected-output fixture producer contexts require exact byte literal constants
//!   (or allocations from constant byte arrays via `.to_vec()` / `vec![]`). Any dynamic helper function call,
//!   wire codec invocation (`pack_u32_slice`), local helper closure alias, loop, or arithmetic in `expected_output`
//!   convicts the registration; `test_inputs` generators and codecs remain permitted.
//! - Caller identity is tracked by exact definition index to prevent collapsing same-named methods
//!   across different impl blocks or traits.
//! - Macro contents (`ItemMacro`, `ExprMacro`, `StmtMacro`) such as `inventory::submit!` and `vec![]`
//!   are recursively parsed into AST nodes without double-counting traversals.
//! - Test scoping covers parent module graphs, `#[cfg(test)] impl`, and `#[cfg(test)] trait` items.
//! - Dynamic `expected_output` evaluations and computed static/const fixture initializers (resolved via
//!   path references regardless of token naming) are strictly forbidden from executing semantic candidates.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use syn::parse::Parser;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Target roots that must contain zero production host oracles.
const TARGET_ROOTS: &[&str] = &["vyre-libs/src", "vyre-primitives/src"];

/// Exact canonical qualified IR builder and operation types representing AST/IR owners.
const EXACT_CANONICAL_IR_BUILDER_PATHS: &[&str] = &[
    "vyre_foundation::ir::Program",
    "vyre_foundation::ir::Node",
    "vyre_foundation::ir::Expr",
    "vyre_foundation::operation::OperationRegistration",
];

/// Exact canonical qualified dispatcher capability traits and types derived from actual source imports.
const EXACT_CANONICAL_DISPATCHER_PATHS: &[&str] =
    &["vyre_foundation::program_dispatch::ProgramDispatcher"];

/// Exact terminal scalar types.
const SCALAR_TYPES: &[&str] = &[
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32",
    "f64", "bool",
];

/// Corrective action for a production host oracle finding.
const FIX: &str = "move the host reference implementation into a #[cfg(test)] module or vyre-reference, \
                   or replace dynamic registration evaluation with exact byte fixtures; production code \
                   in shipping crates must not execute host CPU mathematical oracles or unisolated semantic twins";

/// Zero-baseline gate that eliminates host CPU oracles and semantic twins from production library code.
pub struct HostOracleElimination;

impl crate::gate::GateBehavior for HostOracleElimination {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let sources = tree.rust(TARGET_ROOTS)?;
        report.cover_complete("production library sources", sources.len());

        let test_scoped_files = discover_test_scoped_files(&tree, &sources)?;
        let findings = analyze_sources(&tree, &sources, &test_scoped_files)?;
        for finding in findings {
            report.find(finding);
        }

        report.note(format!(
            "{} production library source file(s) analyzed",
            sources.len()
        ));
        Ok(report)
    }
}

/// Structural record of a function declared in a source file.
#[derive(Clone, Debug)]
struct FunctionParamRecord {
    name: String,
    qualified_custom_types: BTreeSet<String>,
}

#[derive(Clone, Debug)]
struct ParamCalleeFlow {
    param_idx: usize,
    callee_name: String,
    callee_arg_idx: usize,
}

#[derive(Clone, Debug)]
struct FunctionRecord {
    name: String,
    file: PathBuf,
    module_path: Vec<String>,
    line: u32,
    is_test_scoped: bool,
    is_ir_builder: bool,
    is_gpu_dispatch_root: bool,
    is_data_processing: bool,
    is_wire_codec: bool,
    returns_data_output: bool,
    is_explicit_oracle_name: bool,
    has_canonical_dispatcher_param: bool,
    param_custom_types: BTreeSet<String>,
    return_custom_types: BTreeSet<String>,
    params: Vec<FunctionParamRecord>,
    direct_dispatched_param_indices: BTreeSet<usize>,
    param_callee_flows: Vec<ParamCalleeFlow>,
    stages_semantic_resident_upload: bool,
}

/// Structural record of a function call or reference site.
#[derive(Clone, Debug)]
struct CallSiteRecord {
    callee: String,
    caller_file: PathBuf,
    caller_module: Vec<String>,
    caller_fn_idx: Option<usize>,
    line: u32,
    is_in_test: bool,
    is_in_expected_output: bool,
    is_in_fallback: bool,
    is_in_post_dispatch: bool,
    is_in_op_reg: bool,
}

/// Structural record of a static or const definition in source code.
#[derive(Clone, Debug)]
struct StaticConstRecord {
    name: String,
    file: PathBuf,
    module_path: Vec<String>,
    line: u32,
    is_test_scoped: bool,
    has_semantic_operation: bool,
}

/// Derive base module path from file path relative to crate `src/` directory.
fn base_module_path(path: &Path) -> Vec<String> {
    let path_str = path.to_string_lossy();
    let relative = if let Some(idx) = path_str.find("/src/") {
        &path_str[idx + 5..]
    } else if let Some(idx) = path_str.find("src/") {
        &path_str[idx + 4..]
    } else {
        path_str.as_ref()
    };

    let without_ext = relative.strip_suffix(".rs").unwrap_or(relative);
    let segments: Vec<&str> = without_ext.split('/').collect();

    let mut mod_path = Vec::new();
    for &seg in &segments {
        if seg == "mod" || seg == "lib" || seg == "main" {
            continue;
        }
        mod_path.push(seg.to_string());
    }
    mod_path
}

/// Whether an attribute puts its item strictly in the test harness and nowhere else.
fn is_test_only_attribute(attr: &syn::Attribute) -> bool {
    if attr.path().is_ident("test") {
        return true;
    }
    if !attr.path().is_ident("cfg") {
        return false;
    }
    let Ok(meta) = attr.parse_args::<syn::Meta>() else {
        return false;
    };
    is_test_only_meta(&meta)
}

/// Recursively check if a cfg meta predicate strictly requires test.
fn is_test_only_meta(meta: &syn::Meta) -> bool {
    match meta {
        syn::Meta::Path(path) => path.is_ident("test"),
        syn::Meta::List(list) => {
            if list.path.is_ident("all") {
                let Ok(nested) = list.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                ) else {
                    return false;
                };
                nested.iter().any(is_test_only_meta)
            } else {
                false
            }
        }
        syn::Meta::NameValue(_) => false,
    }
}

/// Discover files that are wholly test-scoped because a parent file includes them via `#[cfg(test)] mod foo;`.
fn discover_test_scoped_files(
    tree: &Tree,
    sources: &[PathBuf],
) -> Result<BTreeSet<PathBuf>, GateError> {
    let mut test_scoped_files = BTreeSet::new();
    let mut parent_to_children: BTreeMap<PathBuf, Vec<(String, bool)>> = BTreeMap::new();

    for path in sources {
        let text = tree.read(path)?;
        let file_ast = syn::parse_file(&text).map_err(|err| {
            GateError::new(
                format!("failed to parse `{}`: {err}", path.display()),
                "fix syntax defect so the file parses as valid Rust",
            )
        })?;

        for item in &file_ast.items {
            if let syn::Item::Mod(item_mod) = item {
                if item_mod.content.is_none() {
                    let mod_name = item_mod.ident.to_string();
                    let is_test = item_mod.attrs.iter().any(is_test_only_attribute);
                    parent_to_children
                        .entry(path.clone())
                        .or_default()
                        .push((mod_name, is_test));
                }
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for (parent_path, modules) in &parent_to_children {
            let parent_is_test = test_scoped_files.contains(parent_path);
            let parent_dir = parent_path.parent().unwrap_or_else(|| Path::new(""));

            for (mod_name, is_test_attr) in modules {
                if parent_is_test || *is_test_attr {
                    let candidate1 = parent_dir.join(format!("{mod_name}.rs"));
                    let candidate2 = parent_dir.join(format!("{mod_name}/mod.rs"));
                    let stem = parent_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");
                    let candidate3 = if stem != "mod" && stem != "lib" && stem != "main" {
                        parent_dir.join(stem).join(format!("{mod_name}.rs"))
                    } else {
                        parent_dir.join(format!("{mod_name}.rs"))
                    };

                    for cand in [candidate1, candidate2, candidate3] {
                        if sources.contains(&cand) && test_scoped_files.insert(cand) {
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    Ok(test_scoped_files)
}

/// Flatten a `syn::UseTree` into mapping of local name -> fully-qualified imported path.
fn extract_use_tree(prefix: &str, tree: &syn::UseTree, out: &mut BTreeMap<String, String>) {
    match tree {
        syn::UseTree::Path(p) => {
            let new_prefix = if prefix.is_empty() {
                p.ident.to_string()
            } else {
                format!("{prefix}::{}", p.ident)
            };
            extract_use_tree(&new_prefix, &p.tree, out);
        }
        syn::UseTree::Name(n) => {
            let full_path = if prefix.is_empty() {
                n.ident.to_string()
            } else {
                format!("{prefix}::{}", n.ident)
            };
            out.insert(n.ident.to_string(), full_path);
        }
        syn::UseTree::Rename(r) => {
            let full_path = if prefix.is_empty() {
                r.ident.to_string()
            } else {
                format!("{prefix}::{}", r.ident)
            };
            out.insert(r.rename.to_string(), full_path);
        }
        syn::UseTree::Group(g) => {
            for item in &g.items {
                extract_use_tree(prefix, item, out);
            }
        }
        syn::UseTree::Glob(_) => {
            // Globs do not provide explicit single-ident bindings and fail closed for root qualification
        }
    }
}

/// AST visitor collecting functions, bodies, attributes, statics, and call expressions.
struct AstAnalysisVisitor {
    file: PathBuf,
    current_module: Vec<String>,
    fn_index_offset: usize,
    current_fn_idx: Option<usize>,
    test_mod_depth: usize,
    test_impl_depth: usize,
    test_trait_depth: usize,
    fmt_impl_depth: usize,
    item_test_depth: usize,
    in_expected_output_depth: usize,
    in_fallback_depth: usize,
    in_op_reg_depth: usize,
    in_synthetic_oracle_depth: usize,
    in_gpu_dispatch_root: bool,
    post_dispatch_phase: bool,
    dispatcher_params: BTreeSet<String>,
    non_data_diagnostic_params: BTreeSet<String>,
    known_dispatch_exec_fns: BTreeSet<String>,
    derived_trait_dispatch_exec_methods: BTreeSet<String>,
    derived_trait_resident_upload_methods: BTreeSet<String>,
    local_declared_types: BTreeSet<String>,
    scope_imports: Vec<BTreeMap<String, String>>,
    struct_types_with_dispatcher: BTreeSet<String>,
    current_impl_self_is_dispatcher: bool,
    dispatched_data_vars: BTreeSet<String>,
    functions: Vec<FunctionRecord>,
    calls: Vec<CallSiteRecord>,
    static_consts: Vec<StaticConstRecord>,
    direct_findings: Vec<Finding>,
    types_with_public_fields: BTreeSet<String>,
}

impl AstAnalysisVisitor {
    fn new(
        file: PathBuf,
        file_is_test_scoped: bool,
        fn_index_offset: usize,
        derived_trait_dispatch_exec_methods: BTreeSet<String>,
        derived_trait_resident_upload_methods: BTreeSet<String>,
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
            item_test_depth: 0,
            in_expected_output_depth: 0,
            in_fallback_depth: 0,
            in_op_reg_depth: 0,
            in_synthetic_oracle_depth: 0,
            in_gpu_dispatch_root: false,
            post_dispatch_phase: false,
            dispatcher_params: BTreeSet::new(),
            non_data_diagnostic_params: BTreeSet::new(),
            known_dispatch_exec_fns: BTreeSet::new(),
            derived_trait_dispatch_exec_methods,
            derived_trait_resident_upload_methods,
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

    fn in_test(&self) -> bool {
        self.test_mod_depth > 0
            || self.test_impl_depth > 0
            || self.test_trait_depth > 0
            || self.item_test_depth > 0
    }

    fn clean_path_string(path: &syn::Path) -> String {
        path.segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::")
    }

    fn strip_turbofish(s: &str) -> &str {
        if let Some(pos) = s.find("::<") {
            &s[..pos]
        } else if let Some(pos) = s.find('<') {
            &s[..pos]
        } else {
            s
        }
    }

    fn resolve_path_str(&self, path: &syn::Path) -> String {
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
    fn resolve_qualified_type_path(&self, path: &syn::Path) -> Option<String> {
        let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        if segments.is_empty() {
            return None;
        }

        // 1. Path explicitly starting with `crate::`
        if segments[0] == "crate" {
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

    fn extract_qualified_custom_types(&self, ty: &syn::Type, out: &mut BTreeSet<String>) {
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
                        && last_ident != "DispatchError"
                        && last_ident != "ProgramDispatcher"
                        && last_ident != "ResidentDispatchStep"
                        && last_ident != "ResidentReadRange"
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

    fn is_canonical_ir_builder_path(&self, resolved: &str) -> bool {
        EXACT_CANONICAL_IR_BUILDER_PATHS.contains(&resolved)
    }

    fn is_canonical_dispatcher_path(&self, resolved: &str) -> bool {
        EXACT_CANONICAL_DISPATCHER_PATHS.contains(&resolved)
    }

    fn is_ir_builder_sig(&self, sig: &syn::Signature) -> bool {
        let syn::ReturnType::Type(_, return_type) = &sig.output else {
            return false;
        };
        self.is_canonical_ir_builder_return_type(return_type)
    }

    fn is_canonical_ir_builder_return_type(&self, ty: &syn::Type) -> bool {
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
            syn::Type::Reference(r) => self.is_canonical_ir_builder_return_type(&r.elem),
            syn::Type::Slice(s) => self.is_canonical_ir_builder_return_type(&s.elem),
            syn::Type::Array(a) => self.is_canonical_ir_builder_return_type(&a.elem),
            _ => false,
        }
    }

    fn has_canonical_ir_param(&self, sig: &syn::Signature) -> bool {
        sig.inputs.iter().any(|input| {
            if let syn::FnArg::Typed(pat_type) = input {
                self.type_contains_canonical_ir(&pat_type.ty)
            } else {
                false
            }
        })
    }

    fn type_contains_canonical_ir(&self, ty: &syn::Type) -> bool {
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

    fn type_contains_canonical_dispatcher(&self, ty: &syn::Type) -> bool {
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

    fn qualified_local_type_name(&self, ident: &syn::Ident) -> String {
        let mut parts = vec!["crate".to_string()];
        parts.extend(self.current_module.clone());
        parts.push(ident.to_string());
        parts.join("::")
    }

    fn struct_contains_canonical_dispatcher(&self, item: &syn::ItemStruct) -> bool {
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
    fn returns_operation_metadata_type(&self, ret: &syn::ReturnType) -> bool {
        match ret {
            syn::ReturnType::Default => false,
            syn::ReturnType::Type(_, ty) => self.type_is_operation_metadata(ty),
        }
    }

    fn type_is_operation_metadata(&self, ty: &syn::Type) -> bool {
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

    fn is_canonical_dispatch_exec_method(&self, method: &str) -> bool {
        self.derived_trait_dispatch_exec_methods.contains(method)
    }

    fn is_dispatch_execution_expr(&self, expr: &syn::Expr) -> bool {
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
                // 2. Chained receiver that executes dispatch
                if self.is_dispatch_execution_expr(&*mc.receiver) {
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
                let callee_name = if let syn::Expr::Path(p) = &*c.func {
                    p.path.segments.last().map(|s| s.ident.to_string())
                } else {
                    None
                };

                for arg in &c.args {
                    let arg_idents = extract_read_idents_from_expr(arg);
                    if arg_idents
                        .iter()
                        .any(|id| self.dispatcher_params.contains(id))
                    {
                        if let Some(name) = &callee_name {
                            if self.known_dispatch_exec_fns.contains(name)
                                || self.is_canonical_dispatch_exec_method(name)
                            {
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
                        .map_or(false, |(_, e)| self.is_dispatch_execution_expr(e))
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
                        .map_or(false, |r| self.is_dispatch_execution_expr(r))
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
                .map_or(false, |e| self.is_dispatch_execution_expr(e)),
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

    fn is_dispatch_execution_stmt(&self, stmt: &syn::Stmt) -> bool {
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

    fn extract_dispatcher_params(&self, sig: &syn::Signature) -> BTreeSet<String> {
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

    fn extract_non_data_diagnostic_params(&self, sig: &syn::Signature) -> BTreeSet<String> {
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

    fn is_gpu_dispatch_sig(&self, sig: &syn::Signature, block: &syn::Block) -> bool {
        let dispatcher_params = self.extract_dispatcher_params(sig);
        if dispatcher_params.is_empty() {
            return false;
        }

        let mut temp_visitor = AstAnalysisVisitor::new(
            self.file.clone(),
            false,
            0,
            self.derived_trait_dispatch_exec_methods.clone(),
            self.derived_trait_resident_upload_methods.clone(),
        );
        temp_visitor.dispatcher_params = dispatcher_params;
        temp_visitor.known_dispatch_exec_fns = self.known_dispatch_exec_fns.clone();
        block
            .stmts
            .iter()
            .any(|s| temp_visitor.is_dispatch_execution_stmt(s))
    }
    fn extract_dispatch_inputs_from_stmt(&self, stmt: &syn::Stmt) -> BTreeSet<String> {
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

    fn inspect_function_statements(&mut self, stmts: &[syn::Stmt], is_gpu_root: bool) {
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

            let semantic_output_feeds_dispatch = if has_prior_dispatch
                && has_subsequent_dispatch
                && !is_dispatch_exec
                && stmt_contains_semantic_operation(stmt, self)
            {
                let mut tainted = extract_mutated_storage_from_stmt(stmt);
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
            } else if has_prior_dispatch {
                let reads = extract_read_idents_from_stmt(stmt);
                if reads
                    .iter()
                    .any(|name| self.dispatched_data_vars.contains(name))
                {
                    self.dispatched_data_vars
                        .extend(extract_mutated_storage_from_stmt(stmt));
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

    fn stages_semantic_resident_upload(&self, stmts: &[syn::Stmt]) -> bool {
        if self.dispatcher_params.is_empty()
            || self.derived_trait_resident_upload_methods.is_empty()
        {
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

            let mut scanner = ResidentUploadScanner {
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

    fn scan_macro_tokens_for_references(&mut self, tokens: &proc_macro2::TokenStream) {
        if self.in_expected_output_depth > 0 || self.in_fallback_depth > 0 {
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
                            self.calls.push(CallSiteRecord {
                                callee: ident_str,
                                caller_file: self.file.clone(),
                                caller_module: self.current_module.clone(),
                                caller_fn_idx: self.current_fn_idx,
                                line,
                                is_in_test: self.in_test(),
                                is_in_expected_output: self.in_expected_output_depth > 0,
                                is_in_fallback: self.in_fallback_depth > 0,
                                is_in_post_dispatch: self.in_gpu_dispatch_root
                                    && self.post_dispatch_phase,
                                is_in_op_reg: self.in_op_reg_depth > 0,
                            });
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
    fn analyze_fn_params_and_dispatch_flow(
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
fn extract_pat_bindings(pat: &syn::Pat, out: &mut BTreeSet<String>) {
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

fn extract_root_ident_from_expr(expr: &syn::Expr, out: &mut BTreeSet<String>) {
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
        _ => {}
    }
}

#[derive(Default)]
struct IdentReadCollector {
    idents: BTreeSet<String>,
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

fn extract_read_idents_from_stmt(stmt: &syn::Stmt) -> BTreeSet<String> {
    let mut collector = IdentReadCollector::default();
    syn::visit::visit_stmt(&mut collector, stmt);
    collector.idents
}

fn extract_read_idents_from_expr(expr: &syn::Expr) -> BTreeSet<String> {
    let mut collector = IdentReadCollector::default();
    syn::visit::visit_expr(&mut collector, expr);
    collector.idents
}

#[derive(Default)]
struct MutatedStorageCollector {
    mutated: BTreeSet<String>,
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
        match expr.op {
            syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_) => {
                extract_root_ident_from_expr(&expr.left, &mut self.mutated);
            }
            _ => {}
        }
        syn::visit::visit_expr_binary(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
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
            }
        }
        syn::visit::visit_expr_call(self, expr);
    }
}

fn extract_mutated_storage_from_stmt(stmt: &syn::Stmt) -> BTreeSet<String> {
    let mut collector = MutatedStorageCollector::default();
    if let syn::Stmt::Expr(syn::Expr::MethodCall(call), _) = stmt {
        extract_root_ident_from_expr(&call.receiver, &mut collector.mutated);
    }
    syn::visit::visit_stmt(&mut collector, stmt);
    collector.mutated
}

fn is_reduction_or_arithmetic_method(method_name: &str) -> bool {
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
struct SemanticOperationScanner<'a> {
    visitor: &'a AstAnalysisVisitor,
    has_semantic_op: bool,
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
        let contains_dispatch = self.visitor.is_dispatch_execution_expr(&expr.expr)
            || expr
                .body
                .stmts
                .iter()
                .any(|s| self.visitor.is_dispatch_execution_stmt(s));
        if !contains_dispatch {
            self.has_semantic_op = true;
        }
        syn::visit::visit_expr_for_loop(self, expr);
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        let contains_dispatch = self.visitor.is_dispatch_execution_expr(&expr.cond)
            || expr
                .body
                .stmts
                .iter()
                .any(|s| self.visitor.is_dispatch_execution_stmt(s));
        if !contains_dispatch {
            self.has_semantic_op = true;
        }
        syn::visit::visit_expr_while(self, expr);
    }

    fn visit_expr_loop(&mut self, expr: &'ast syn::ExprLoop) {
        let contains_dispatch = expr
            .body
            .stmts
            .iter()
            .any(|s| self.visitor.is_dispatch_execution_stmt(s));
        if !contains_dispatch {
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

fn stmt_contains_semantic_operation(stmt: &syn::Stmt, visitor: &AstAnalysisVisitor) -> bool {
    let mut scanner = SemanticOperationScanner {
        visitor,
        has_semantic_op: false,
    };
    syn::visit::visit_stmt(&mut scanner, stmt);
    scanner.has_semantic_op
}

struct ResidentUploadScanner<'a> {
    visitor: &'a AstAnalysisVisitor,
    dispatcher_params: &'a BTreeSet<String>,
    semantic_taint: &'a BTreeSet<String>,
    consumes_semantic_storage: bool,
}

impl<'ast> Visit<'ast> for ResidentUploadScanner<'_> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let method = call.method.to_string();
        if self
            .visitor
            .derived_trait_resident_upload_methods
            .contains(&method)
        {
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
}
fn type_is_canonical_ir_program(ty: &syn::Type) -> bool {
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

fn type_is_canonical_resident_step(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(tp) => {
            if let Some(seg) = tp.path.segments.last() {
                seg.ident == "ResidentDispatchStep"
            } else {
                false
            }
        }
        syn::Type::Reference(r) => type_is_canonical_resident_step(&r.elem),
        syn::Type::Slice(s) => type_is_canonical_resident_step(&s.elem),
        syn::Type::Array(a) => type_is_canonical_resident_step(&a.elem),
        syn::Type::Group(g) => type_is_canonical_resident_step(&g.elem),
        syn::Type::Paren(p) => type_is_canonical_resident_step(&p.elem),
        _ => false,
    }
}
fn type_is_exact_generic_param(ty: &syn::Type, generic_name: &str) -> bool {
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

fn type_contains_immutable_byte_slice(ty: &syn::Type) -> bool {
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

fn is_trait_method_resident_upload(sig: &syn::Signature) -> bool {
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

fn is_trait_method_dispatch_execution(sig: &syn::Signature) -> bool {
    let has_program_or_step_param = sig.inputs.iter().any(|input| {
        if let syn::FnArg::Typed(pat_type) = input {
            type_is_canonical_ir_program(&pat_type.ty)
                || type_is_canonical_resident_step(&pat_type.ty)
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

    has_program_or_step_param && returns_result_or_data
}

fn derive_canonical_dispatcher_methods(
    file: &syn::File,
    dispatch_methods: &mut BTreeSet<String>,
    resident_upload_methods: &mut BTreeSet<String>,
) {
    fn inspect_items(
        items: &[syn::Item],
        dispatch_methods: &mut BTreeSet<String>,
        resident_upload_methods: &mut BTreeSet<String>,
    ) {
        for item in items {
            match item {
                syn::Item::Trait(item_trait) if item_trait.ident == "ProgramDispatcher" => {
                    for trait_item in &item_trait.items {
                        if let syn::TraitItem::Fn(method) = trait_item {
                            if is_trait_method_dispatch_execution(&method.sig) {
                                dispatch_methods.insert(method.sig.ident.to_string());
                            }
                            if is_trait_method_resident_upload(&method.sig) {
                                resident_upload_methods.insert(method.sig.ident.to_string());
                            }
                        }
                    }
                }
                syn::Item::Mod(item_mod) => {
                    if let Some((_, inner_items)) = &item_mod.content {
                        inspect_items(inner_items, dispatch_methods, resident_upload_methods);
                    }
                }
                _ => {}
            }
        }
    }

    inspect_items(&file.items, dispatch_methods, resident_upload_methods);
}

struct BlockDispatchScanner<'a> {
    visitor: &'a AstAnalysisVisitor,
    dispatcher_params: BTreeSet<String>,
    has_direct_dispatch: bool,
    dispatcher_callees: Vec<String>,
}

impl<'a> BlockDispatchScanner<'a> {
    fn new(visitor: &'a AstAnalysisVisitor, dispatcher_params: BTreeSet<String>) -> Self {
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
        let callee_name = if let syn::Expr::Path(path) = &*call.func {
            path.path
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
        } else {
            None
        };
        for arg in &call.args {
            let arg_idents = extract_read_idents_from_expr(arg);
            if arg_idents
                .iter()
                .any(|ident| self.dispatcher_params.contains(ident))
            {
                if let Some(name) = &callee_name {
                    self.dispatcher_callees.push(name.clone());
                }
            }
        }
        syn::visit::visit_expr_call(self, call);
    }
}

struct FnDispatchScan {
    name: String,
    has_direct_dispatch: bool,
    dispatcher_callees: Vec<String>,
}

fn scan_item_for_dispatch(
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
                scans.push(FnDispatchScan {
                    name: item_fn.sig.ident.to_string(),
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
                        scans.push(FnDispatchScan {
                            name: impl_fn.sig.ident.to_string(),
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

fn compute_known_dispatch_exec_fns(
    file: &syn::File,
    visitor: &mut AstAnalysisVisitor,
) -> BTreeSet<String> {
    fn collect_wrapper_structs(items: &[syn::Item], visitor: &mut AstAnalysisVisitor) {
        for item in items {
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
                        collect_wrapper_structs(inner_items, visitor);
                        visitor.scope_imports.pop();
                        visitor.current_module.pop();
                    }
                }
                _ => {}
            }
        }
    }

    loop {
        let previous_count = visitor.struct_types_with_dispatcher.len();
        collect_wrapper_structs(&file.items, visitor);
        if visitor.struct_types_with_dispatcher.len() == previous_count {
            break;
        }
    }

    let mut scans = Vec::new();
    for item in &file.items {
        scan_item_for_dispatch(item, visitor, &mut scans);
    }

    let mut exec_set = BTreeSet::new();
    for scan in &scans {
        if scan.has_direct_dispatch {
            exec_set.insert(scan.name.clone());
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for scan in &scans {
            if !exec_set.contains(&scan.name)
                && scan.dispatcher_callees.iter().any(|callee| {
                    exec_set.contains(callee) || visitor.is_canonical_dispatch_exec_method(callee)
                })
            {
                exec_set.insert(scan.name.clone());
                changed = true;
            }
        }
    }
    exec_set
}
fn extract_expr_param_deps(
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
                .map_or(false, |seg| seg.ident == "ResidentDispatchStep");
            let is_read_range = s
                .path
                .segments
                .last()
                .map_or(false, |seg| seg.ident == "ResidentReadRange");
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

fn scan_block_for_param_dispatch_flow(
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
                    if mname == "push" || mname == "extend" {
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

fn scan_expr_for_param_dispatch_flow(
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
                    .map_or(false, |id| dispatcher_params.contains(&id.to_string()))
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
                    &mut arm_deps,
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
struct FileImportCollector<'a> {
    scope_imports: &'a mut Vec<BTreeMap<String, String>>,
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

impl<'ast> Visit<'ast> for AstAnalysisVisitor {
    fn visit_file(&mut self, file: &'ast syn::File) {
        let mut collector = FileImportCollector {
            scope_imports: &mut self.scope_imports,
        };
        collector.visit_file(file);
        self.known_dispatch_exec_fns = compute_known_dispatch_exec_fns(file, self);
        syn::visit::visit_file(self, file);
    }
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if let Some(current_scope) = self.scope_imports.last_mut() {
            extract_use_tree("", &item.tree, current_scope);
        }
        syn::visit::visit_item_use(self, item);
    }

    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        let ident = item.ident.to_string();
        let mut qualified_parts = vec!["crate".to_string()];
        qualified_parts.extend(self.current_module.clone());
        qualified_parts.push(ident.clone());
        let qualified_path = qualified_parts.join("::");

        let has_pub_field = item
            .fields
            .iter()
            .any(|f| matches!(f.vis, syn::Visibility::Public(_)));
        if has_pub_field {
            self.types_with_public_fields.insert(qualified_path);
        }

        if self.struct_contains_canonical_dispatcher(item) {
            self.struct_types_with_dispatcher
                .insert(self.qualified_local_type_name(&item.ident));
        }

        syn::visit::visit_item_struct(self, item);
    }

    fn visit_item_union(&mut self, item: &'ast syn::ItemUnion) {
        let ident = item.ident.to_string();
        let mut qualified_parts = vec!["crate".to_string()];
        qualified_parts.extend(self.current_module.clone());
        qualified_parts.push(ident);
        let qualified_path = qualified_parts.join("::");

        let has_pub_field = item
            .fields
            .named
            .iter()
            .any(|f| matches!(f.vis, syn::Visibility::Public(_)));
        if has_pub_field {
            self.types_with_public_fields.insert(qualified_path);
        }

        syn::visit::visit_item_union(self, item);
    }

    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let mod_name = item.ident.to_string();
        let is_test_mod = item.attrs.iter().any(is_test_only_attribute);

        self.current_module.push(mod_name);
        self.scope_imports.push(BTreeMap::new());
        if is_test_mod {
            self.test_mod_depth += 1;
        }

        syn::visit::visit_item_mod(self, item);

        if is_test_mod {
            self.test_mod_depth -= 1;
        }
        self.scope_imports.pop();
        self.current_module.pop();
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        let is_test_impl = item.attrs.iter().any(is_test_only_attribute);
        if is_test_impl {
            self.test_impl_depth += 1;
        }

        let self_has_dispatcher = self.type_contains_canonical_dispatcher(&item.self_ty);
        let prev_self_is_dispatcher = self.current_impl_self_is_dispatcher;
        if self_has_dispatcher {
            self.current_impl_self_is_dispatcher = true;
        }
        let is_fmt_trait = if let Some((_, trait_path, _)) = &item.trait_ {
            let trait_str = self.resolve_path_str(trait_path);
            trait_str == "std::fmt::Display"
                || trait_str == "core::fmt::Display"
                || trait_str == "Display"
                || trait_str == "std::fmt::Debug"
                || trait_str == "core::fmt::Debug"
                || trait_str == "Debug"
                || trait_str == "std::error::Error"
                || trait_str == "core::error::Error"
                || trait_str == "Error"
        } else {
            false
        };

        if is_fmt_trait {
            self.fmt_impl_depth += 1;
        }

        syn::visit::visit_item_impl(self, item);

        if is_fmt_trait {
            self.fmt_impl_depth -= 1;
        }

        if is_test_impl {
            self.test_impl_depth -= 1;
        }
        self.current_impl_self_is_dispatcher = prev_self_is_dispatcher;
    }

    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        let is_test_trait = item.attrs.iter().any(is_test_only_attribute);
        if is_test_trait {
            self.test_trait_depth += 1;
        }

        syn::visit::visit_item_trait(self, item);

        if is_test_trait {
            self.test_trait_depth -= 1;
        }
    }

    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        let is_test = item.attrs.iter().any(is_test_only_attribute);
        if is_test {
            self.item_test_depth += 1;
        }
        let const_name = item.ident.to_string();
        let line = item.ident.span().start().line as u32;

        let mut feature_visitor = BodyFeatureVisitor::default();
        feature_visitor.visit_expr(&item.expr);
        let has_semantic = feature_visitor.has_semantic_operation();

        self.static_consts.push(StaticConstRecord {
            name: const_name,
            file: self.file.clone(),
            module_path: self.current_module.clone(),
            line,
            is_test_scoped: self.in_test(),
            has_semantic_operation: has_semantic,
        });

        syn::visit::visit_item_const(self, item);

        if is_test {
            self.item_test_depth -= 1;
        }
    }

    fn visit_item_static(&mut self, item: &'ast syn::ItemStatic) {
        let is_test = item.attrs.iter().any(is_test_only_attribute);
        if is_test {
            self.item_test_depth += 1;
        }
        let static_name = item.ident.to_string();
        let line = item.ident.span().start().line as u32;

        let mut feature_visitor = BodyFeatureVisitor::default();
        feature_visitor.visit_expr(&item.expr);
        let has_semantic = feature_visitor.has_semantic_operation();

        self.static_consts.push(StaticConstRecord {
            name: static_name,
            file: self.file.clone(),
            module_path: self.current_module.clone(),
            line,
            is_test_scoped: self.in_test(),
            has_semantic_operation: has_semantic,
        });

        syn::visit::visit_item_static(self, item);

        if is_test {
            self.item_test_depth -= 1;
        }
    }

    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        let mut parsed = false;
        if let Ok(file) = syn::parse2::<syn::File>(item.mac.tokens.clone()) {
            parsed = true;
            for it in &file.items {
                self.visit_item(it);
            }
        } else if let Ok(expr) = syn::parse2::<syn::Expr>(item.mac.tokens.clone()) {
            parsed = true;
            self.visit_expr(&expr);
        } else if let Ok(stmt) = syn::parse2::<syn::Stmt>(item.mac.tokens.clone()) {
            parsed = true;
            self.visit_stmt(&stmt);
        }
        if !parsed {
            self.scan_macro_tokens_for_references(&item.mac.tokens);
        }
    }

    fn visit_expr_macro(&mut self, expr: &'ast syn::ExprMacro) {
        let mut parsed = false;
        if let Ok(inner_expr) = syn::parse2::<syn::Expr>(expr.mac.tokens.clone()) {
            parsed = true;
            self.visit_expr(&inner_expr);
        } else if let Ok(punctuated) =
            syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
                .parse2(expr.mac.tokens.clone())
        {
            parsed = true;
            for inner in &punctuated {
                self.visit_expr(inner);
            }
        }
        if !parsed {
            self.scan_macro_tokens_for_references(&expr.mac.tokens);
        }
    }

    fn visit_stmt_macro(&mut self, stmt: &'ast syn::StmtMacro) {
        let mut parsed = false;
        if let Ok(inner_expr) = syn::parse2::<syn::Expr>(stmt.mac.tokens.clone()) {
            parsed = true;
            self.visit_expr(&inner_expr);
        } else if let Ok(punctuated) =
            syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
                .parse2(stmt.mac.tokens.clone())
        {
            parsed = true;
            for inner in &punctuated {
                self.visit_expr(inner);
            }
        }
        if !parsed {
            self.scan_macro_tokens_for_references(&stmt.mac.tokens);
        }
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        let fn_name = item.sig.ident.to_string();
        let line = item.sig.ident.span().start().line as u32;
        let is_fn_test_attr = item.attrs.iter().any(is_test_only_attribute);
        let is_test_scoped = self.test_mod_depth > 0
            || self.test_impl_depth > 0
            || self.test_trait_depth > 0
            || is_fn_test_attr;

        let prev_dispatcher_params = std::mem::take(&mut self.dispatcher_params);
        self.dispatcher_params = self.extract_dispatcher_params(&item.sig);
        let prev_non_data_diagnostic_params = std::mem::take(&mut self.non_data_diagnostic_params);
        self.non_data_diagnostic_params = self.extract_non_data_diagnostic_params(&item.sig);

        let is_ir = self.is_ir_builder_sig(&item.sig);
        let is_gpu = self.is_gpu_dispatch_sig(&item.sig, &item.block);
        let returns_data = has_data_output_ast(&item.sig);
        let is_explicit = fn_name == "cpu_ref"
            || fn_name == "cpu_reference"
            || fn_name.starts_with("cpu_ref_")
            || fn_name.ends_with("_cpu_ref");

        let is_fmt_method =
            self.fmt_impl_depth > 0 || (fn_name == "fmt" && is_fmt_signature(&item.sig));
        let is_pure_telemetry = returns_unit(&item.sig.output) && !has_mutable_params(&item.sig);
        let is_pure_validator =
            returns_result_unit(&item.sig.output) && !has_mutable_params(&item.sig);
        let is_ir_inspector =
            self.has_canonical_ir_param(&item.sig) && !has_mutable_data_output_param(&item.sig);
        let is_metadata_inspector = self.returns_operation_metadata_type(&item.sig.output);
        let is_wire_codec = is_wire_codec_ast(&item.sig, &item.block);

        let is_dp = is_explicit
            || (!is_fmt_method
                && !is_pure_telemetry
                && !is_pure_validator
                && !is_ir_inspector
                && !is_metadata_inspector
                && !is_wire_codec
                && !is_ir
                && is_data_processing_ast(&item.sig, &item.block));

        let has_canonical_dispatcher_param = !self.dispatcher_params.is_empty();
        let mut param_custom_types = BTreeSet::new();
        for input in &item.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                self.extract_qualified_custom_types(&pat_type.ty, &mut param_custom_types);
            }
        }
        let mut return_custom_types = BTreeSet::new();
        if let syn::ReturnType::Type(_, ret_ty) = &item.sig.output {
            self.extract_qualified_custom_types(ret_ty, &mut return_custom_types);
        }

        let (params, direct_dispatched_param_indices, param_callee_flows) =
            self.analyze_fn_params_and_dispatch_flow(&item.sig, Some(&item.block));
        let stages_semantic_resident_upload =
            self.stages_semantic_resident_upload(&item.block.stmts);

        let fn_idx = self.fn_index_offset + self.functions.len();
        self.functions.push(FunctionRecord {
            name: fn_name.clone(),
            file: self.file.clone(),
            module_path: self.current_module.clone(),
            line,
            is_test_scoped,
            is_ir_builder: is_ir,
            is_gpu_dispatch_root: is_gpu,
            is_data_processing: is_dp,
            is_wire_codec,
            returns_data_output: returns_data,
            is_explicit_oracle_name: is_explicit,
            has_canonical_dispatcher_param,
            param_custom_types,
            return_custom_types,
            params,
            direct_dispatched_param_indices,
            param_callee_flows,
            stages_semantic_resident_upload,
        });

        let prev_fn = self.current_fn_idx.replace(fn_idx);
        let is_expected_output = fn_name == "expected_output"
            || fn_name.starts_with("expected_output_")
            || fn_name.ends_with("_expected_output");
        if is_expected_output {
            self.in_expected_output_depth += 1;
        }
        if is_fn_test_attr {
            self.item_test_depth += 1;
        }

        self.inspect_function_statements(&item.block.stmts, is_gpu);

        if is_fn_test_attr {
            self.item_test_depth -= 1;
        }
        if is_expected_output {
            self.in_expected_output_depth -= 1;
        }
        self.current_fn_idx = prev_fn;
        self.dispatcher_params = prev_dispatcher_params;
        self.non_data_diagnostic_params = prev_non_data_diagnostic_params;
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        let fn_name = item.sig.ident.to_string();
        let line = item.sig.ident.span().start().line as u32;
        let is_fn_test_attr = item.attrs.iter().any(is_test_only_attribute);
        let is_test_scoped = self.test_mod_depth > 0
            || self.test_impl_depth > 0
            || self.test_trait_depth > 0
            || is_fn_test_attr;

        let prev_dispatcher_params = std::mem::take(&mut self.dispatcher_params);
        self.dispatcher_params = self.extract_dispatcher_params(&item.sig);
        let prev_non_data_diagnostic_params = std::mem::take(&mut self.non_data_diagnostic_params);
        self.non_data_diagnostic_params = self.extract_non_data_diagnostic_params(&item.sig);

        let is_ir = self.is_ir_builder_sig(&item.sig);
        let is_gpu = self.is_gpu_dispatch_sig(&item.sig, &item.block);
        let returns_data = has_data_output_ast(&item.sig);
        let is_explicit = fn_name == "cpu_ref" || fn_name == "cpu_reference";

        let is_fmt_method =
            self.fmt_impl_depth > 0 || (fn_name == "fmt" && is_fmt_signature(&item.sig));
        let is_pure_telemetry = returns_unit(&item.sig.output) && !has_mutable_params(&item.sig);
        let is_pure_validator =
            returns_result_unit(&item.sig.output) && !has_mutable_params(&item.sig);
        let is_ir_inspector =
            self.has_canonical_ir_param(&item.sig) && !has_mutable_data_output_param(&item.sig);
        let is_metadata_inspector = self.returns_operation_metadata_type(&item.sig.output);
        let is_wire_codec = is_wire_codec_ast(&item.sig, &item.block);

        let is_dp = is_explicit
            || (!is_fmt_method
                && !is_pure_telemetry
                && !is_pure_validator
                && !is_ir_inspector
                && !is_metadata_inspector
                && !is_wire_codec
                && !is_ir
                && is_data_processing_ast(&item.sig, &item.block));

        let has_canonical_dispatcher_param = !self.dispatcher_params.is_empty();
        let mut param_custom_types = BTreeSet::new();
        for input in &item.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                self.extract_qualified_custom_types(&pat_type.ty, &mut param_custom_types);
            }
        }
        let mut return_custom_types = BTreeSet::new();
        if let syn::ReturnType::Type(_, ret_ty) = &item.sig.output {
            self.extract_qualified_custom_types(ret_ty, &mut return_custom_types);
        }

        let (params, direct_dispatched_param_indices, param_callee_flows) =
            self.analyze_fn_params_and_dispatch_flow(&item.sig, Some(&item.block));
        let stages_semantic_resident_upload =
            self.stages_semantic_resident_upload(&item.block.stmts);

        let fn_idx = self.fn_index_offset + self.functions.len();
        self.functions.push(FunctionRecord {
            name: fn_name.clone(),
            file: self.file.clone(),
            module_path: self.current_module.clone(),
            line,
            is_test_scoped,
            is_ir_builder: is_ir,
            is_gpu_dispatch_root: is_gpu,
            is_data_processing: is_dp,
            is_wire_codec,
            returns_data_output: returns_data,
            is_explicit_oracle_name: is_explicit,
            has_canonical_dispatcher_param,
            param_custom_types,
            return_custom_types,
            params,
            direct_dispatched_param_indices,
            param_callee_flows,
            stages_semantic_resident_upload,
        });

        let prev_fn = self.current_fn_idx.replace(fn_idx);
        let is_expected_output = fn_name == "expected_output"
            || fn_name.starts_with("expected_output_")
            || fn_name.ends_with("_expected_output");
        if is_expected_output {
            self.in_expected_output_depth += 1;
        }
        if is_fn_test_attr {
            self.item_test_depth += 1;
        }

        self.inspect_function_statements(&item.block.stmts, is_gpu);

        if is_fn_test_attr {
            self.item_test_depth -= 1;
        }
        if is_expected_output {
            self.in_expected_output_depth -= 1;
        }
        self.current_fn_idx = prev_fn;
        self.dispatcher_params = prev_dispatcher_params;
        self.non_data_diagnostic_params = prev_non_data_diagnostic_params;
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        let fn_name = item.sig.ident.to_string();
        let line = item.sig.ident.span().start().line as u32;
        let is_fn_test_attr = item.attrs.iter().any(is_test_only_attribute);
        let is_test_scoped = self.test_mod_depth > 0
            || self.test_impl_depth > 0
            || self.test_trait_depth > 0
            || is_fn_test_attr;

        let prev_dispatcher_params = std::mem::take(&mut self.dispatcher_params);
        self.dispatcher_params = self.extract_dispatcher_params(&item.sig);
        let prev_non_data_diagnostic_params = std::mem::take(&mut self.non_data_diagnostic_params);
        self.non_data_diagnostic_params = self.extract_non_data_diagnostic_params(&item.sig);

        let is_ir = self.is_ir_builder_sig(&item.sig);
        let is_gpu = if let Some(block) = &item.default {
            self.is_gpu_dispatch_sig(&item.sig, block)
        } else {
            false
        };
        let returns_data = has_data_output_ast(&item.sig);
        let is_explicit = fn_name == "cpu_ref" || fn_name == "cpu_reference";

        let is_fmt_method =
            self.fmt_impl_depth > 0 || (fn_name == "fmt" && is_fmt_signature(&item.sig));
        let is_pure_telemetry = returns_unit(&item.sig.output) && !has_mutable_params(&item.sig);
        let is_pure_validator =
            returns_result_unit(&item.sig.output) && !has_mutable_params(&item.sig);
        let is_ir_inspector =
            self.has_canonical_ir_param(&item.sig) && !has_mutable_data_output_param(&item.sig);
        let is_metadata_inspector = self.returns_operation_metadata_type(&item.sig.output);
        let is_wire_codec = item
            .default
            .as_ref()
            .is_some_and(|block| is_wire_codec_ast(&item.sig, block));

        let is_dp = is_explicit
            || (if let Some(block) = &item.default {
                !is_fmt_method
                    && !is_pure_telemetry
                    && !is_pure_validator
                    && !is_ir_inspector
                    && !is_metadata_inspector
                    && !is_wire_codec
                    && !is_ir
                    && is_data_processing_ast(&item.sig, block)
            } else {
                false
            });

        let has_canonical_dispatcher_param = !self.dispatcher_params.is_empty();
        let mut param_custom_types = BTreeSet::new();
        for input in &item.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                self.extract_qualified_custom_types(&pat_type.ty, &mut param_custom_types);
            }
        }
        let mut return_custom_types = BTreeSet::new();
        if let syn::ReturnType::Type(_, ret_ty) = &item.sig.output {
            self.extract_qualified_custom_types(ret_ty, &mut return_custom_types);
        }

        let (params, direct_dispatched_param_indices, param_callee_flows) =
            self.analyze_fn_params_and_dispatch_flow(&item.sig, item.default.as_ref());
        let stages_semantic_resident_upload = item
            .default
            .as_ref()
            .is_some_and(|block| self.stages_semantic_resident_upload(&block.stmts));

        let fn_idx = self.fn_index_offset + self.functions.len();
        self.functions.push(FunctionRecord {
            name: fn_name.clone(),
            file: self.file.clone(),
            module_path: self.current_module.clone(),
            line,
            is_test_scoped,
            is_ir_builder: is_ir,
            is_gpu_dispatch_root: is_gpu,
            is_data_processing: is_dp,
            is_wire_codec,
            returns_data_output: returns_data,
            is_explicit_oracle_name: is_explicit,
            has_canonical_dispatcher_param,
            param_custom_types,
            return_custom_types,
            params,
            direct_dispatched_param_indices,
            param_callee_flows,
            stages_semantic_resident_upload,
        });

        if let Some(block) = &item.default {
            let prev_fn = self.current_fn_idx.replace(fn_idx);
            let is_expected_output = fn_name == "expected_output"
                || fn_name.starts_with("expected_output_")
                || fn_name.ends_with("_expected_output");
            if is_expected_output {
                self.in_expected_output_depth += 1;
            }
            if is_fn_test_attr {
                self.item_test_depth += 1;
            }
            self.inspect_function_statements(&block.stmts, is_gpu);
            if is_fn_test_attr {
                self.item_test_depth -= 1;
            }
            if is_expected_output {
                self.in_expected_output_depth -= 1;
            }
            self.current_fn_idx = prev_fn;
        }
        self.dispatcher_params = prev_dispatcher_params;
        self.non_data_diagnostic_params = prev_non_data_diagnostic_params;
    }

    fn visit_expr_match(&mut self, expr: &'ast syn::ExprMatch) {
        let expr_is_dispatch =
            self.in_gpu_dispatch_root && self.is_dispatch_execution_expr(&expr.expr);
        syn::visit::visit_expr(self, &expr.expr);
        if expr_is_dispatch {
            self.post_dispatch_phase = true;
        }

        for arm in &expr.arms {
            let pat_str = quote::quote!(#arm.pat).to_string().replace(' ', "");
            let is_fallback_arm = self.in_gpu_dispatch_root
                && (pat_str.starts_with("Err(")
                    || pat_str.starts_with("Result::Err(")
                    || pat_str == "None"
                    || pat_str == "Option::None"
                    || pat_str == "false");

            if is_fallback_arm {
                self.in_fallback_depth += 1;
            }
            if let Some((_, guard)) = &arm.guard {
                syn::visit::visit_expr(self, guard);
            }
            syn::visit::visit_expr(self, &arm.body);
            if is_fallback_arm {
                self.in_fallback_depth -= 1;
            }
        }
    }

    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        let cond_str = quote::quote!(#expr.cond).to_string().replace(' ', "");
        let is_fallback_cond = self.in_gpu_dispatch_root
            && (cond_str.contains(".is_err()")
                || cond_str.contains("!*.is_ok()")
                || cond_str.contains(".is_none()")
                || cond_str.starts_with("letErr(")
                || cond_str.starts_with("letResult::Err(")
                || cond_str.starts_with("letNone="));

        let cond_is_dispatch =
            self.in_gpu_dispatch_root && self.is_dispatch_execution_expr(&expr.cond);
        syn::visit::visit_expr(self, &expr.cond);
        if cond_is_dispatch {
            self.post_dispatch_phase = true;
        }

        if is_fallback_cond {
            self.in_fallback_depth += 1;
        }
        for stmt in &expr.then_branch.stmts {
            syn::visit::visit_stmt(self, stmt);
        }
        if is_fallback_cond {
            self.in_fallback_depth -= 1;
        }

        if let Some((_, else_branch)) = &expr.else_branch {
            syn::visit::visit_expr(self, else_branch);
        }
    }

    fn visit_expr_closure(&mut self, expr: &'ast syn::ExprClosure) {
        if self.in_expected_output_depth > 0
            && !self.in_test()
            && self.in_synthetic_oracle_depth > 0
        {
            let line = expr.span().start().line as u32;
            self.direct_findings.push(Finding::at(
                self.file.clone(),
                line,
                "OperationRegistration `expected_output` defines dynamic helper closure alias; \
                 expected outputs must be exact byte literal constants"
                    .to_string(),
                FIX,
            ));
        }

        if self.in_expected_output_depth > 0
            && !self.in_test()
            && self.in_synthetic_oracle_depth == 0
        {
            let mut feature_visitor = BodyFeatureVisitor::default();
            feature_visitor.visit_expr(&expr.body);
            if feature_visitor.has_semantic_operation() {
                let line = expr.span().start().line as u32;
                let name = format!("<inline expected_output closure at line {line}>");
                let fn_idx = self.fn_index_offset + self.functions.len();
                self.functions.push(FunctionRecord {
                    name: name.clone(),
                    file: self.file.clone(),
                    module_path: self.current_module.clone(),
                    line,
                    is_test_scoped: false,
                    is_ir_builder: false,
                    is_gpu_dispatch_root: false,
                    is_data_processing: true,
                    is_wire_codec: false,
                    returns_data_output: true,
                    is_explicit_oracle_name: false,
                    has_canonical_dispatcher_param: false,
                    param_custom_types: BTreeSet::new(),
                    return_custom_types: BTreeSet::new(),
                    params: Vec::new(),
                    direct_dispatched_param_indices: BTreeSet::new(),
                    param_callee_flows: Vec::new(),
                    stages_semantic_resident_upload: false,
                });
                self.calls.push(CallSiteRecord {
                    callee: name,
                    caller_file: self.file.clone(),
                    caller_module: self.current_module.clone(),
                    caller_fn_idx: None,
                    line,
                    is_in_test: false,
                    is_in_expected_output: true,
                    is_in_fallback: false,
                    is_in_post_dispatch: false,
                    is_in_op_reg: self.in_op_reg_depth > 0,
                });
            }
        }

        if self.in_fallback_depth > 0 && !self.in_test() && self.in_synthetic_oracle_depth == 0 {
            let mut feature_visitor = BodyFeatureVisitor::default();
            feature_visitor.visit_expr(&expr.body);
            if feature_visitor.has_semantic_operation() {
                let line = expr.span().start().line as u32;
                let name = format!("<inline fallback closure at line {line}>");
                let fn_idx = self.fn_index_offset + self.functions.len();
                self.functions.push(FunctionRecord {
                    name: name.clone(),
                    file: self.file.clone(),
                    module_path: self.current_module.clone(),
                    line,
                    is_test_scoped: false,
                    is_ir_builder: false,
                    is_gpu_dispatch_root: false,
                    is_data_processing: true,
                    is_wire_codec: false,
                    returns_data_output: true,
                    is_explicit_oracle_name: false,
                    has_canonical_dispatcher_param: false,
                    param_custom_types: BTreeSet::new(),
                    return_custom_types: BTreeSet::new(),
                    params: Vec::new(),
                    direct_dispatched_param_indices: BTreeSet::new(),
                    param_callee_flows: Vec::new(),
                    stages_semantic_resident_upload: false,
                });
                self.calls.push(CallSiteRecord {
                    callee: name,
                    caller_file: self.file.clone(),
                    caller_module: self.current_module.clone(),
                    caller_fn_idx: None,
                    line,
                    is_in_test: false,
                    is_in_expected_output: false,
                    is_in_fallback: true,
                    is_in_post_dispatch: false,
                    is_in_op_reg: self.in_op_reg_depth > 0,
                });
            }
        }

        self.in_synthetic_oracle_depth += 1;
        syn::visit::visit_expr_closure(self, expr);
        self.in_synthetic_oracle_depth -= 1;
    }

    fn visit_expr_block(&mut self, expr: &'ast syn::ExprBlock) {
        if self.in_expected_output_depth > 0
            && !self.in_test()
            && self.in_synthetic_oracle_depth == 0
        {
            let mut feature_visitor = BodyFeatureVisitor::default();
            for stmt in &expr.block.stmts {
                feature_visitor.visit_stmt(stmt);
            }
            if feature_visitor.has_semantic_operation() {
                let line = expr.span().start().line as u32;
                let name = format!("<inline expected_output block at line {line}>");
                let fn_idx = self.fn_index_offset + self.functions.len();
                self.functions.push(FunctionRecord {
                    name: name.clone(),
                    file: self.file.clone(),
                    module_path: self.current_module.clone(),
                    line,
                    is_test_scoped: false,
                    is_ir_builder: false,
                    is_gpu_dispatch_root: false,
                    is_data_processing: true,
                    is_wire_codec: false,
                    returns_data_output: true,
                    is_explicit_oracle_name: false,
                    has_canonical_dispatcher_param: false,
                    param_custom_types: BTreeSet::new(),
                    return_custom_types: BTreeSet::new(),
                    params: Vec::new(),
                    direct_dispatched_param_indices: BTreeSet::new(),
                    param_callee_flows: Vec::new(),
                    stages_semantic_resident_upload: false,
                });
                self.calls.push(CallSiteRecord {
                    callee: name,
                    caller_file: self.file.clone(),
                    caller_module: self.current_module.clone(),
                    caller_fn_idx: None,
                    line,
                    is_in_test: false,
                    is_in_expected_output: true,
                    is_in_fallback: false,
                    is_in_post_dispatch: false,
                    is_in_op_reg: self.in_op_reg_depth > 0,
                });
            }
        }

        if self.in_fallback_depth > 0 && !self.in_test() && self.in_synthetic_oracle_depth == 0 {
            let mut feature_visitor = BodyFeatureVisitor::default();
            for stmt in &expr.block.stmts {
                feature_visitor.visit_stmt(stmt);
            }
            if feature_visitor.has_semantic_operation() {
                let line = expr.span().start().line as u32;
                let name = format!("<inline fallback block at line {line}>");
                let fn_idx = self.fn_index_offset + self.functions.len();
                self.functions.push(FunctionRecord {
                    name: name.clone(),
                    file: self.file.clone(),
                    module_path: self.current_module.clone(),
                    line,
                    is_test_scoped: false,
                    is_ir_builder: false,
                    is_gpu_dispatch_root: false,
                    is_data_processing: true,
                    is_wire_codec: false,
                    returns_data_output: true,
                    is_explicit_oracle_name: false,
                    has_canonical_dispatcher_param: false,
                    param_custom_types: BTreeSet::new(),
                    return_custom_types: BTreeSet::new(),
                    params: Vec::new(),
                    direct_dispatched_param_indices: BTreeSet::new(),
                    param_callee_flows: Vec::new(),
                    stages_semantic_resident_upload: false,
                });
                self.calls.push(CallSiteRecord {
                    callee: name,
                    caller_file: self.file.clone(),
                    caller_module: self.current_module.clone(),
                    caller_fn_idx: None,
                    line,
                    is_in_test: false,
                    is_in_expected_output: false,
                    is_in_fallback: true,
                    is_in_post_dispatch: false,
                    is_in_op_reg: self.in_op_reg_depth > 0,
                });
            }
        }

        self.in_synthetic_oracle_depth += 1;
        syn::visit::visit_expr_block(self, expr);
        self.in_synthetic_oracle_depth -= 1;
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        let contains_dispatch = self.is_dispatch_execution_expr(&expr.expr)
            || expr
                .body
                .stmts
                .iter()
                .any(|s| self.is_dispatch_execution_stmt(s));
        if self.in_gpu_dispatch_root
            && self.post_dispatch_phase
            && !contains_dispatch
            && !self.in_test()
        {
            let line = expr.span().start().line as u32;
            let current_fn_name = self
                .current_fn_idx
                .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
                .unwrap_or_else(|| "<anonymous>".to_string());
            self.direct_findings.push(Finding::at(
                self.file.clone(),
                line,
                format!(
                    "GPU dispatch function `{current_fn_name}` contains post-dispatch host loop/accumulation; \
                     post-dispatch semantic reductions must be dispatched on GPU"
                ),
                FIX,
            ));
        }
        syn::visit::visit_expr_for_loop(self, expr);
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        let contains_dispatch = self.is_dispatch_execution_expr(&expr.cond)
            || expr
                .body
                .stmts
                .iter()
                .any(|s| self.is_dispatch_execution_stmt(s));
        if self.in_gpu_dispatch_root
            && self.post_dispatch_phase
            && !contains_dispatch
            && !self.in_test()
        {
            let line = expr.span().start().line as u32;
            let current_fn_name = self
                .current_fn_idx
                .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
                .unwrap_or_else(|| "<anonymous>".to_string());
            self.direct_findings.push(Finding::at(
                self.file.clone(),
                line,
                format!(
                    "GPU dispatch function `{current_fn_name}` contains post-dispatch host loop/accumulation; \
                     post-dispatch semantic reductions must be dispatched on GPU"
                ),
                FIX,
            ));
        }
        syn::visit::visit_expr_while(self, expr);
    }

    fn visit_expr_loop(&mut self, expr: &'ast syn::ExprLoop) {
        let contains_dispatch = expr
            .body
            .stmts
            .iter()
            .any(|s| self.is_dispatch_execution_stmt(s));
        if self.in_gpu_dispatch_root
            && self.post_dispatch_phase
            && !contains_dispatch
            && !self.in_test()
        {
            let line = expr.span().start().line as u32;
            let current_fn_name = self
                .current_fn_idx
                .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
                .unwrap_or_else(|| "<anonymous>".to_string());
            self.direct_findings.push(Finding::at(
                self.file.clone(),
                line,
                format!(
                    "GPU dispatch function `{current_fn_name}` contains post-dispatch host loop/accumulation; \
                     post-dispatch semantic reductions must be dispatched on GPU"
                ),
                FIX,
            ));
        }
        syn::visit::visit_expr_loop(self, expr);
    }
    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        if self.in_gpu_dispatch_root && self.post_dispatch_phase && !self.in_test() {
            let line = expr.span().start().line as u32;
            let is_dispatcher_call =
                self.is_dispatch_execution_expr(&syn::Expr::Binary(expr.clone()));
            if !is_dispatcher_call {
                let is_permitted_codec = is_byte_unpack_codec_expr(expr);
                if !is_permitted_codec {
                    let operates_on_dispatched = if !self.dispatched_data_vars.is_empty() {
                        let reads = extract_read_idents_from_expr(&syn::Expr::Binary(expr.clone()));
                        reads
                            .iter()
                            .any(|id| self.dispatched_data_vars.contains(id))
                    } else {
                        true
                    };
                    if operates_on_dispatched {
                        let current_fn_name = self
                            .current_fn_idx
                            .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
                            .unwrap_or_else(|| "<anonymous>".to_string());
                        self.direct_findings.push(Finding::at(
                            self.file.clone(),
                            line,
                            format!(
                                "GPU dispatch function `{current_fn_name}` executes post-dispatch host arithmetic / semantic derivation on GPU results; \
                                 mathematical operations on dispatch output must be dispatched on GPU"
                            ),
                            FIX,
                        ));
                    }
                }
            }
        }
        syn::visit::visit_expr_binary(self, expr);
    }

    fn visit_expr_call(&mut self, expr: &'ast syn::ExprCall) {
        let (full_callee, resolved_callee) = if let syn::Expr::Path(path_expr) = &*expr.func {
            let raw_callee = quote::quote!(#path_expr).to_string().replace(' ', "");
            let clean_callee = Self::clean_path_string(&path_expr.path);
            let resolved = self.resolve_path_str(&path_expr.path);
            let line = expr.func.span().start().line as u32;
            self.calls.push(CallSiteRecord {
                callee: clean_callee,
                caller_file: self.file.clone(),
                caller_module: self.current_module.clone(),
                caller_fn_idx: self.current_fn_idx,
                line,
                is_in_test: self.in_test(),
                is_in_expected_output: self.in_expected_output_depth > 0,
                is_in_fallback: self.in_fallback_depth > 0,
                is_in_post_dispatch: self.in_gpu_dispatch_root && self.post_dispatch_phase,
                is_in_op_reg: self.in_op_reg_depth > 0,
            });
            (raw_callee, resolved)
        } else {
            (String::new(), String::new())
        };

        // Structurally inspect OperationRegistration constructor calls to mark expected_output fixture contexts
        // Requires exact canonical origin vyre_foundation::operation::OperationRegistration
        let expected_idx = match resolved_callee.as_str() {
            "vyre_foundation::operation::OperationRegistration::library" => Some(3),
            "vyre_foundation::operation::OperationRegistration::primitive" => Some(3),
            "vyre_foundation::operation::OperationRegistration::intrinsic" => Some(4),
            "vyre_foundation::operation::OperationRegistration::new" => Some(4),
            _ => None,
        };

        if let Some(target_idx) = expected_idx {
            self.in_op_reg_depth += 1;
            syn::visit::visit_expr(self, &*expr.func);
            for (idx, arg) in expr.args.iter().enumerate() {
                if idx == target_idx {
                    self.in_expected_output_depth += 1;
                    syn::visit::visit_expr(self, arg);
                    self.in_expected_output_depth -= 1;
                } else {
                    syn::visit::visit_expr(self, arg);
                }
            }
            self.in_op_reg_depth -= 1;
            return;
        }

        syn::visit::visit_expr_call(self, expr);
        if self.in_expected_output_depth > 0 && !self.in_test() {
            let callee_ident = if let syn::Expr::Path(path_expr) = &*expr.func {
                path_expr.path.segments.last().map(|s| s.ident.to_string())
            } else {
                None
            };
            let is_permitted_constructor = match callee_ident.as_deref() {
                Some("vec") | Some("Vec") | Some("Some") | Some("None") | Some("Ok") => true,
                _ => false,
            };
            if !is_permitted_constructor {
                let line = expr.func.span().start().line as u32;
                self.direct_findings.push(Finding::at(
                    self.file.clone(),
                    line,
                    format!(
                        "OperationRegistration `expected_output` invokes dynamic helper/codec `{full_callee}`; \
                         expected outputs must be exact byte literal constants"
                    ),
                    FIX,
                ));
            }
        }
    }

    fn visit_expr_struct(&mut self, expr: &'ast syn::ExprStruct) {
        let resolved_struct = self.resolve_path_str(&expr.path);
        let is_op_reg = resolved_struct == "vyre_foundation::operation::OperationRegistration";

        if is_op_reg {
            self.in_op_reg_depth += 1;
        }
        for field in &expr.fields {
            let is_expected = is_op_reg
                && match &field.member {
                    syn::Member::Named(ident) => ident == "expected_output",
                    _ => false,
                };

            if is_expected {
                self.in_expected_output_depth += 1;
                syn::visit::visit_expr(self, &field.expr);
                self.in_expected_output_depth -= 1;
            } else {
                syn::visit::visit_expr(self, &field.expr);
            }
        }
        if let Some(rest) = &expr.rest {
            syn::visit::visit_expr(self, rest);
        }
        if is_op_reg {
            self.in_op_reg_depth -= 1;
        }
    }

    fn visit_expr_path(&mut self, expr: &'ast syn::ExprPath) {
        let path_str = quote::quote!(#expr).to_string().replace(' ', "");
        let line = expr.span().start().line as u32;

        if self.in_op_reg_depth > 0 && self.in_expected_output_depth == 0 && !self.in_test() {
            // Function passed by name to OperationRegistration (e.g. Some(generate_deterministic_inputs), add_program)
            self.calls.push(CallSiteRecord {
                callee: path_str.clone(),
                caller_file: self.file.clone(),
                caller_module: self.current_module.clone(),
                caller_fn_idx: self.current_fn_idx,
                line,
                is_in_test: false,
                is_in_expected_output: false,
                is_in_fallback: false,
                is_in_post_dispatch: false,
                is_in_op_reg: true,
            });
        }

        syn::visit::visit_expr_path(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        let method_name = expr.method.to_string();
        let line = expr.method.span().start().line as u32;

        self.calls.push(CallSiteRecord {
            callee: method_name.clone(),
            caller_file: self.file.clone(),
            caller_module: self.current_module.clone(),
            caller_fn_idx: self.current_fn_idx,
            line,
            is_in_test: self.in_test(),
            is_in_expected_output: self.in_expected_output_depth > 0,
            is_in_fallback: self.in_fallback_depth > 0,
            is_in_post_dispatch: self.in_gpu_dispatch_root && self.post_dispatch_phase,
            is_in_op_reg: self.in_op_reg_depth > 0,
        });

        if self.in_expected_output_depth > 0 && !self.in_test() {
            let is_permitted_alloc = method_name == "to_vec" || method_name == "clone";
            if !is_permitted_alloc {
                self.direct_findings.push(Finding::at(
                    self.file.clone(),
                    line,
                    format!(
                        "OperationRegistration `expected_output` invokes dynamic method `.{method_name}`; \
                         expected outputs must be exact byte literal constants"
                    ),
                    FIX,
                ));
            }
        }

        // Check if receiver executes dispatch (e.g. dispatcher.dispatch(..).map(..))
        let receiver_is_dispatch =
            self.in_gpu_dispatch_root && self.is_dispatch_execution_expr(&*expr.receiver);

        syn::visit::visit_expr(self, &*expr.receiver);

        if receiver_is_dispatch {
            self.post_dispatch_phase = true;
        }

        // Convict post-dispatch host reduction / semantic methods in shipping dispatcher functions
        if self.in_gpu_dispatch_root && self.post_dispatch_phase && !self.in_test() {
            let is_dispatcher_call =
                self.is_dispatch_execution_expr(&syn::Expr::MethodCall(expr.clone()));
            if !is_dispatcher_call {
                match method_name.as_str() {
                    "any" | "all" | "sum" | "fold" | "rfold" | "reduce" | "count"
                    | "count_ones" | "count_zeros" | "leading_zeros" | "trailing_zeros"
                    | "rotate_left" | "rotate_right" | "reverse_bits" | "wrapping_add"
                    | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "checked_add" => {
                        let receiver_idents = extract_read_idents_from_expr(&expr.receiver);
                        let receiver_is_diagnostic = receiver_idents
                            .iter()
                            .any(|id| self.non_data_diagnostic_params.contains(id));
                        let operates_on_dispatched = receiver_is_dispatch
                            || receiver_idents
                                .iter()
                                .any(|id| self.dispatched_data_vars.contains(id))
                            || !receiver_is_diagnostic;
                        if operates_on_dispatched {
                            let current_fn_name = self
                                .current_fn_idx
                                .and_then(|idx| self.functions.get(idx).map(|f| f.name.clone()))
                                .unwrap_or_else(|| "<anonymous>".to_string());
                            self.direct_findings.push(Finding::at(
                                self.file.clone(),
                                line,
                                format!(
                                    "GPU dispatch function `{current_fn_name}` contains post-dispatch host reduction/aggregation `.{method_name}`; \
                                     post-dispatch semantic reductions must be dispatched on GPU"
                                ),
                                FIX,
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }

        let is_fallback_combinator = method_name == "unwrap_or_else"
            || method_name == "or_else"
            || method_name == "unwrap_or"
            || method_name == "unwrap_or_default";

        for arg in &expr.args {
            if is_fallback_combinator {
                self.in_fallback_depth += 1;
                syn::visit::visit_expr(self, arg);
                self.in_fallback_depth -= 1;
            } else if method_name == "with_expected_output" {
                self.in_expected_output_depth += 1;
                syn::visit::visit_expr(self, arg);
                self.in_expected_output_depth -= 1;
            } else {
                syn::visit::visit_expr(self, arg);
            }
        }
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        let path_str = quote::quote!(#path).to_string().replace(' ', "");
        let line = path.span().start().line as u32;

        if path_str.contains("vyre_reference") || path_str.contains("SubgroupSimulator") {
            self.calls.push(CallSiteRecord {
                callee: "vyre_reference".to_string(),
                caller_file: self.file.clone(),
                caller_module: self.current_module.clone(),
                caller_fn_idx: self.current_fn_idx,
                line,
                is_in_test: self.in_test(),
                is_in_expected_output: self.in_expected_output_depth > 0,
                is_in_fallback: self.in_fallback_depth > 0,
                is_in_post_dispatch: self.in_gpu_dispatch_root && self.post_dispatch_phase,
                is_in_op_reg: self.in_op_reg_depth > 0,
            });
        }

        if self.in_expected_output_depth > 0 {
            self.calls.push(CallSiteRecord {
                callee: path_str,
                caller_file: self.file.clone(),
                caller_module: self.current_module.clone(),
                caller_fn_idx: self.current_fn_idx,
                line,
                is_in_test: self.in_test(),
                is_in_expected_output: true,
                is_in_fallback: false,
                is_in_post_dispatch: false,
                is_in_op_reg: self.in_op_reg_depth > 0,
            });
        }

        syn::visit::visit_path(self, path);
    }
}

/// Analyze all source files across the workspace roots and return defect findings.
fn analyze_sources(
    tree: &Tree,
    sources: &[PathBuf],
    test_scoped_files: &BTreeSet<PathBuf>,
) -> Result<Vec<Finding>, GateError> {
    let canonical_path = PathBuf::from("vyre-foundation/src/program_dispatch/mod.rs");
    let text = tree.read(&canonical_path).map_err(|err| {
        GateError::new(
            format!(
                "failed to read canonical ProgramDispatcher source `{}`: {err}",
                canonical_path.display()
            ),
            "ensure canonical `vyre-foundation/src/program_dispatch/mod.rs` exists and is readable",
        )
    })?;

    let file_ast = syn::parse_file(&text).map_err(|err| {
        GateError::new(
            format!(
                "failed to parse canonical ProgramDispatcher source `{}`: {err}",
                canonical_path.display()
            ),
            "fix syntax defect in ProgramDispatcher trait definition",
        )
    })?;

    let mut canonical_trait_methods = BTreeSet::new();
    let mut canonical_resident_upload_methods = BTreeSet::new();
    derive_canonical_dispatcher_methods(
        &file_ast,
        &mut canonical_trait_methods,
        &mut canonical_resident_upload_methods,
    );

    if canonical_trait_methods.is_empty() || canonical_resident_upload_methods.is_empty() {
        return Err(GateError::new(
            format!(
                "canonical ProgramDispatcher in `{}` yielded zero dispatch execution or resident upload methods",
                canonical_path.display()
            ),
            "ensure ProgramDispatcher trait defines canonical execution methods (e.g. `dispatch`, `dispatch_resident`) taking Program/ResidentDispatchStep parameters and returning Result",
        ));
    }

    let mut all_functions = Vec::new();
    let mut all_calls = Vec::new();
    let mut all_static_consts = Vec::new();
    let mut all_findings = Vec::new();
    let mut all_types_with_public_fields = BTreeSet::new();
    for path in sources {
        let text = tree.read(path)?;
        let file_ast = syn::parse_file(&text).map_err(|err| {
            GateError::new(
                format!("failed to parse `{}`: {err}", path.display()),
                "fix syntax defect so the file parses as valid Rust",
            )
        })?;

        let is_test_scoped = test_scoped_files.contains(path);
        let fn_offset = all_functions.len();
        let mut visitor = AstAnalysisVisitor::new(
            path.clone(),
            is_test_scoped,
            fn_offset,
            canonical_trait_methods.clone(),
            canonical_resident_upload_methods.clone(),
        );
        // Pre-discover all types declared locally in this file
        for item in &file_ast.items {
            match item {
                syn::Item::Struct(s) => {
                    visitor.local_declared_types.insert(s.ident.to_string());
                }
                syn::Item::Enum(e) => {
                    visitor.local_declared_types.insert(e.ident.to_string());
                }
                syn::Item::Type(t) => {
                    visitor.local_declared_types.insert(t.ident.to_string());
                }
                syn::Item::Trait(tr) => {
                    visitor.local_declared_types.insert(tr.ident.to_string());
                }
                syn::Item::Union(u) => {
                    visitor.local_declared_types.insert(u.ident.to_string());
                }
                _ => {}
            }
        }

        visitor.visit_file(&file_ast);

        all_functions.extend(visitor.functions);
        all_calls.extend(visitor.calls);
        all_static_consts.extend(visitor.static_consts);
        all_findings.extend(visitor.direct_findings);
        all_types_with_public_fields.extend(visitor.types_with_public_fields);
    }

    let evaluated = evaluate_rules(
        &all_functions,
        &all_calls,
        &all_static_consts,
        &all_types_with_public_fields,
    );

    // Deduplicate findings by (file, line, message)
    let mut deduped_findings = Vec::new();
    let mut seen_findings = BTreeSet::new();
    for finding in all_findings {
        let key = (finding.file.clone(), finding.line, finding.message.clone());
        if seen_findings.insert(key) {
            deduped_findings.push(finding);
        }
    }

    Ok(deduped_findings)
}

/// Evaluate zero-baseline host oracle and transitive reachability rules.
fn evaluate_rules(
    functions: &[FunctionRecord],
    calls: &[CallSiteRecord],
    static_consts: &[StaticConstRecord],
    types_with_public_fields: &BTreeSet<String>,
) -> Vec<Finding> {
    let mut findings = Vec::new();

    // Check direct vyre_reference calls in production code
    for call in calls {
        if !call.is_in_test && call.callee == "vyre_reference" {
            findings.push(Finding::at(
                call.caller_file.clone(),
                call.line,
                "production dependency or invocation of host simulator `vyre_reference`",
                FIX,
            ));
        }
    }

    // Build map from target name -> Vec<idx> for static_consts
    let mut static_const_indices_by_target: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, sc) in static_consts.iter().enumerate() {
        if !sc.is_test_scoped {
            static_const_indices_by_target
                .entry(sc.name.clone())
                .or_default()
                .push(idx);
        }
    }

    // Build map from callee target name -> Vec<idx> for functions
    let mut func_indices_by_target: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, func) in functions.iter().enumerate() {
        if !func.is_test_scoped {
            func_indices_by_target
                .entry(func.name.clone())
                .or_default()
                .push(idx);
        }
    }

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); functions.len()];
    let mut dynamic_expected_output_calls: BTreeSet<usize> = BTreeSet::new();
    let mut dynamic_fallback_calls: BTreeSet<usize> = BTreeSet::new();
    let mut dynamic_expected_output_static_consts: BTreeSet<usize> = BTreeSet::new();
    let mut reachable_from_roots = vec![false; functions.len()];
    let mut queue = VecDeque::new();

    for call in calls {
        if call.is_in_test {
            continue;
        }

        // Fail-closed function call resolution with turbofish and CFG-alternative support
        let mut matched_targets: Vec<usize> = Vec::new();
        let clean_callee = AstAnalysisVisitor::strip_turbofish(&call.callee);

        // 1. Check exact intra-module match (matches all compatible CFG definitions)
        for (target_idx, target_fn) in functions.iter().enumerate() {
            if target_fn.is_test_scoped {
                continue;
            }

            let is_exact_intra_module = call.caller_file == target_fn.file
                && call.caller_module == target_fn.module_path
                && (clean_callee == target_fn.name
                    || clean_callee.ends_with(&format!("::{}", target_fn.name)));

            if is_exact_intra_module {
                matched_targets.push(target_idx);
            }
        }

        // 2. If no exact intra-module match, check qualified path matches
        if matched_targets.is_empty() {
            for (target_idx, target_fn) in functions.iter().enumerate() {
                if target_fn.is_test_scoped {
                    continue;
                }

                let mod_path = target_fn.module_path.join("::");
                if !mod_path.is_empty()
                    && (clean_callee == format!("{mod_path}::{}", target_fn.name)
                        || clean_callee.ends_with(&format!("::{mod_path}::{}", target_fn.name))
                        || clean_callee.contains(&format!("{mod_path}::{}", target_fn.name)))
                {
                    matched_targets.push(target_idx);
                }
            }
        }

        // 3. If still unmatched, search by bare name across non-test targets
        if matched_targets.is_empty() {
            let bare_name = clean_callee.rsplit("::").next().unwrap_or(clean_callee);
            if let Some(target_indices) = func_indices_by_target.get(bare_name) {
                matched_targets.extend(target_indices);
            } else {
                for (target_idx, target_fn) in functions.iter().enumerate() {
                    if !target_fn.is_test_scoped
                        && clean_callee.ends_with(&format!("::{}", target_fn.name))
                    {
                        matched_targets.push(target_idx);
                    }
                }
            }
        }
        for &target_idx in &matched_targets {
            if call.is_in_expected_output {
                dynamic_expected_output_calls.insert(target_idx);
            } else if call.is_in_fallback {
                dynamic_fallback_calls.insert(target_idx);
            } else if call.is_in_op_reg {
                // Top-level OperationRegistration arguments (build, test_inputs) are roots
                reachable_from_roots[target_idx] = true;
                queue.push_back(target_idx);
            } else if let Some(caller_idx) = call.caller_fn_idx {
                if caller_idx != target_idx {
                    adjacency[caller_idx].push(target_idx);
                }
            }
        }

        // Also resolve static/const references inside expected_output
        if call.is_in_expected_output {
            let mut matched_sc: Vec<usize> = Vec::new();

            for (sc_idx, sc) in static_consts.iter().enumerate() {
                if sc.is_test_scoped {
                    continue;
                }
                let is_exact = call.caller_file == sc.file
                    && call.caller_module == sc.module_path
                    && (call.callee == sc.name || call.callee.ends_with(&format!("::{}", sc.name)));
                if is_exact {
                    matched_sc.push(sc_idx);
                }
            }

            if matched_sc.is_empty() {
                for (sc_idx, sc) in static_consts.iter().enumerate() {
                    if sc.is_test_scoped {
                        continue;
                    }
                    let mod_path = sc.module_path.join("::");
                    if !mod_path.is_empty()
                        && call.callee.contains(&format!("{mod_path}::{}", sc.name))
                    {
                        matched_sc.push(sc_idx);
                    }
                }
            }

            if matched_sc.is_empty() {
                if let Some(sc_indices) = static_const_indices_by_target.get(&call.callee) {
                    matched_sc.extend(sc_indices);
                } else {
                    for (sc_idx, sc) in static_consts.iter().enumerate() {
                        if !sc.is_test_scoped && call.callee.ends_with(&format!("::{}", sc.name)) {
                            matched_sc.push(sc_idx);
                        }
                    }
                }
            }

            for &sc_idx in &matched_sc {
                dynamic_expected_output_static_consts.insert(sc_idx);
            }
        }
    }

    // Transitive propagation of expected_output callers to callees
    let mut dynamic_expected_output_queue: VecDeque<usize> =
        dynamic_expected_output_calls.iter().copied().collect();
    while let Some(curr) = dynamic_expected_output_queue.pop_front() {
        for &callee in &adjacency[curr] {
            if dynamic_expected_output_calls.insert(callee) {
                dynamic_expected_output_queue.push_back(callee);
            }
        }
    }

    // Flag referenced production static/const initializers with dynamic semantic operations
    for &sc_idx in &dynamic_expected_output_static_consts {
        let sc = &static_consts[sc_idx];
        if sc.has_semantic_operation {
            findings.push(Finding::at(
                sc.file.clone(),
                sc.line,
                format!(
                    "production operation registration `expected_output` references computed static/const `{}` containing dynamic semantic execution; \
                     operation registrations must use exact byte constants",
                    sc.name
                ),
                FIX,
            ));
        }
    }

    // Fixed-point propagation of GPU dispatch execution across function call graph
    let mut is_gpu_dispatch_exec: Vec<bool> = functions
        .iter()
        .map(|f| !f.is_test_scoped && f.is_gpu_dispatch_root)
        .collect();
    let mut dispatch_propagation_changed = true;
    while dispatch_propagation_changed {
        dispatch_propagation_changed = false;
        for (caller_idx, func) in functions.iter().enumerate() {
            if func.is_test_scoped
                || !func.has_canonical_dispatcher_param
                || is_gpu_dispatch_exec[caller_idx]
            {
                continue;
            }
            for &callee_idx in &adjacency[caller_idx] {
                if is_gpu_dispatch_exec[callee_idx] {
                    is_gpu_dispatch_exec[caller_idx] = true;
                    dispatch_propagation_changed = true;
                    break;
                }
            }
        }
    }

    // Seed production roots established strictly by canonical foundation types and device dispatch calls
    for (idx, func) in functions.iter().enumerate() {
        if func.is_test_scoped {
            continue;
        }

        let is_root = func.is_ir_builder || is_gpu_dispatch_exec[idx];
        if is_root {
            reachable_from_roots[idx] = true;
            queue.push_back(idx);
        }
    }

    // Map each exact qualified return type to the list of non-test producer function indices
    let mut non_test_producers_by_type: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, func) in functions.iter().enumerate() {
        if func.is_test_scoped {
            continue;
        }
        for ret_ty in &func.return_custom_types {
            non_test_producers_by_type
                .entry(ret_ty.clone())
                .or_default()
                .push(idx);
        }
    }

    // Map function name to indices for resolving callee flows
    let mut fn_indices_by_name: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, func) in functions.iter().enumerate() {
        fn_indices_by_name
            .entry(func.name.clone())
            .or_default()
            .push(idx);
    }

    // Inter-procedural fixed-point propagation of parameter dispatch flow
    let mut param_dispatched: Vec<Vec<bool>> = functions
        .iter()
        .map(|f| {
            let mut disp = vec![false; f.params.len()];
            for &p_idx in &f.direct_dispatched_param_indices {
                if p_idx < disp.len() {
                    disp[p_idx] = true;
                }
            }
            disp
        })
        .collect();

    let mut param_flow_changed = true;
    while param_flow_changed {
        param_flow_changed = false;
        for (caller_idx, func) in functions.iter().enumerate() {
            for flow in &func.param_callee_flows {
                if flow.param_idx < param_dispatched[caller_idx].len()
                    && !param_dispatched[caller_idx][flow.param_idx]
                {
                    if let Some(callee_indices) = fn_indices_by_name.get(&flow.callee_name) {
                        for &callee_idx in callee_indices {
                            if flow.callee_arg_idx < param_dispatched[callee_idx].len()
                                && param_dispatched[callee_idx][flow.callee_arg_idx]
                            {
                                param_dispatched[caller_idx][flow.param_idx] = true;
                                param_flow_changed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Map each function to the set of qualified custom types of parameters that flow into dispatch
    let func_dispatched_custom_types: Vec<BTreeSet<String>> = functions
        .iter()
        .enumerate()
        .map(|(fn_idx, func)| {
            let mut types = BTreeSet::new();
            for (p_idx, p_rec) in func.params.iter().enumerate() {
                if p_idx < param_dispatched[fn_idx].len() && param_dispatched[fn_idx][p_idx] {
                    types.extend(p_rec.qualified_custom_types.clone());
                }
            }
            types
        })
        .collect();

    // Fail-closed nominal bridge: a qualified custom type may root a producer only when:
    // (a) an actual call/dataflow path already connects it (handled by call graph BFS), OR
    // (b) it is the unique non-test producer of that exact qualified type AND a canonical
    //     dispatch-executing consumer accepts that exact qualified type AND structural
    //     dataflow proves the parameter feeds into dispatch execution handle_ids (direct
    //     or transitive) AND the type has no externally public fields that bypass the unique producer.
    for (idx, func) in functions.iter().enumerate() {
        if func.is_test_scoped || !is_gpu_dispatch_exec[idx] {
            continue;
        }
        for param_ty in &func_dispatched_custom_types[idx] {
            if types_with_public_fields.contains(param_ty) {
                continue;
            }
            if let Some(producer_indices) = non_test_producers_by_type.get(param_ty) {
                if producer_indices.len() == 1 {
                    let p_idx = producer_indices[0];
                    if !reachable_from_roots[p_idx] {
                        reachable_from_roots[p_idx] = true;
                        queue.push_back(p_idx);
                    }
                }
            }
        }
    }

    // Fixed-point candidate propagation: functions returning scalar/collection output
    // that call a candidate also become candidates.
    let mut is_candidate: Vec<bool> = functions.iter().map(|f| f.is_data_processing).collect();
    let mut candidate_propagation_changed = true;
    while candidate_propagation_changed {
        candidate_propagation_changed = false;
        for (caller_idx, caller_fn) in functions.iter().enumerate() {
            if caller_fn.is_test_scoped
                || caller_fn.is_ir_builder
                || caller_fn.is_wire_codec
                || is_candidate[caller_idx]
                || !caller_fn.returns_data_output
            {
                continue;
            }

            for &callee_idx in &adjacency[caller_idx] {
                if is_candidate[callee_idx] {
                    is_candidate[caller_idx] = true;
                    candidate_propagation_changed = true;
                    break;
                }
            }
        }
    }

    // Transitive BFS traversal from production roots
    while let Some(curr) = queue.pop_front() {
        for &next in &adjacency[curr] {
            if !reachable_from_roots[next] {
                reachable_from_roots[next] = true;
                queue.push_back(next);
            }
        }
    }

    // Judge all declared non-test functions
    for (idx, func) in functions.iter().enumerate() {
        if func.is_test_scoped {
            continue;
        }

        // Rule 1: Explicit cpu_ref names in production code
        if func.is_explicit_oracle_name {
            findings.push(Finding::at(
                func.file.clone(),
                func.line,
                format!(
                    "production host reference oracle function definition `{}`",
                    func.name
                ),
                FIX,
            ));
            continue;
        }

        // Rule 2: Expected output dynamic oracle call (unconditional on candidacy)
        if dynamic_expected_output_calls.contains(&idx) && is_candidate[idx] {
            findings.push(Finding::at(
                func.file.clone(),
                func.line,
                format!(
                    "production operation registration `expected_output` dynamically executes host oracle `{}`; \
                     operation registrations must use exact byte constants",
                    func.name
                ),
                FIX,
            ));
            continue;
        }

        // Rule 3: Dispatch error / fallback oracle call
        if dynamic_fallback_calls.contains(&idx) && is_candidate[idx] {
            findings.push(Finding::at(
                func.file.clone(),
                func.line,
                format!(
                    "GPU dispatch function contains host CPU reference fallback `{}` on dispatch error/failure; \
                     silent host fallback is forbidden",
                    func.name
                ),
                FIX,
            ));
            continue;
        }

        // Rule 4: Data-processing candidates must be transitively reachable from a production root
        if is_candidate[idx] && !reachable_from_roots[idx] {
            findings.push(Finding::at(
                func.file.clone(),
                func.line,
                format!(
                    "unisolated host data-processing semantic twin `{}` is not reachable from any production root; \
                     host semantic execution must live in tests or vyre-reference",
                    func.name
                ),
                FIX,
            ));
        }

        // Rule 5: GPU dispatch functions must not invoke host data-processing semantic helpers
        if is_gpu_dispatch_exec[idx] {
            for &callee_idx in &adjacency[idx] {
                if is_candidate[callee_idx] {
                    let callee = &functions[callee_idx];
                    let is_proven_resident_staging = callee.stages_semantic_resident_upload
                        && callee.return_custom_types.iter().any(|return_type| {
                            func_dispatched_custom_types.iter().enumerate().any(
                                |(consumer_idx, dispatched_types)| {
                                    is_gpu_dispatch_exec[consumer_idx]
                                        && dispatched_types.contains(return_type)
                                },
                            )
                        });
                    if is_proven_resident_staging {
                        continue;
                    }
                    findings.push(Finding::at(
                        func.file.clone(),
                        func.line,
                        format!(
                            "GPU dispatch function `{}` invokes host data-processing semantic helper `{}`; \
                             host mathematical calculations must be executed on GPU",
                            func.name, callee.name
                        ),
                        FIX,
                    ));
                }
            }
        }
    }

    findings
}

/// Recursively check if type is a scalar or data collection.
fn type_is_data_output(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(type_path) => {
            if let Some(seg) = type_path.path.segments.last() {
                let ident = seg.ident.to_string();
                if SCALAR_TYPES.contains(&ident.as_str()) {
                    return true;
                }
                if ident == "Vec"
                    || ident == "BTreeSet"
                    || ident == "HashSet"
                    || ident == "BTreeMap"
                    || ident == "HashMap"
                {
                    return true;
                }
                if ident == "Result"
                    || ident == "Option"
                    || ident == "Arc"
                    || ident == "Box"
                    || ident == "Rc"
                {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            return type_is_data_output(inner_ty);
                        }
                    }
                    return false;
                }
            }
            false
        }
        syn::Type::Tuple(tuple) => tuple.elems.iter().any(type_is_data_output),
        syn::Type::Array(_) | syn::Type::Slice(_) => true,
        syn::Type::Reference(r) => type_is_data_output(&r.elem),
        _ => false,
    }
}

/// Check if return type is a heap/collection data container.
fn has_data_container_output(sig: &syn::Signature) -> bool {
    match &sig.output {
        syn::ReturnType::Default => false,
        syn::ReturnType::Type(_, ty) => type_is_data_container(ty),
    }
}

fn type_is_data_container(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(type_path) => {
            if let Some(seg) = type_path.path.segments.last() {
                let ident = seg.ident.to_string();
                if ident == "Vec"
                    || ident == "BTreeSet"
                    || ident == "HashSet"
                    || ident == "BTreeMap"
                    || ident == "HashMap"
                {
                    return true;
                }
                if ident == "Result"
                    || ident == "Option"
                    || ident == "Arc"
                    || ident == "Box"
                    || ident == "Rc"
                {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            return type_is_data_container(inner_ty);
                        }
                    }
                }
            }
            false
        }
        syn::Type::Tuple(tuple) => tuple.elems.iter().any(type_is_data_container),
        syn::Type::Array(_) | syn::Type::Slice(_) => true,
        syn::Type::Reference(r) => type_is_data_container(&r.elem),
        _ => false,
    }
}

/// Check if the signature accepts primitive numeric payload data.
fn has_data_input_ast(sig: &syn::Signature) -> bool {
    sig.inputs.iter().any(|arg| match arg {
        syn::FnArg::Typed(pat_type) => type_is_numeric_payload_input(&pat_type.ty),
        syn::FnArg::Receiver(_) => false,
    })
}

fn type_is_numeric_payload_input(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Reference(reference) => type_is_numeric_payload_input(&reference.elem),
        syn::Type::Slice(slice) => type_is_numeric_payload_input(&slice.elem),
        syn::Type::Array(array) => type_is_numeric_payload_input(&array.elem),
        syn::Type::Tuple(tuple) => tuple.elems.iter().any(type_is_numeric_payload_input),
        syn::Type::Group(group) => type_is_numeric_payload_input(&group.elem),
        syn::Type::Paren(paren) => type_is_numeric_payload_input(&paren.elem),
        syn::Type::Path(path) => {
            let Some(segment) = path.path.segments.last() else {
                return false;
            };
            let ident = segment.ident.to_string();
            if SCALAR_TYPES.contains(&ident.as_str()) {
                return true;
            }
            if matches!(
                ident.as_str(),
                "Vec" | "BTreeSet" | "HashSet" | "Option" | "Result" | "Arc" | "Box" | "Rc"
            ) {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    return args.args.iter().any(|arg| {
                        matches!(
                            arg,
                            syn::GenericArgument::Type(inner)
                                if type_is_numeric_payload_input(inner)
                        )
                    });
                }
            }
            false
        }
        _ => false,
    }
}

/// Check if signature returns a non-unit value.
fn has_data_output_ast(sig: &syn::Signature) -> bool {
    match &sig.output {
        syn::ReturnType::Default => false,
        syn::ReturnType::Type(_, ty) => match &**ty {
            syn::Type::Tuple(t) if t.elems.is_empty() => false,
            _ => true,
        },
    }
}

/// Check if return type is unit `()`.
fn returns_unit(ret: &syn::ReturnType) -> bool {
    match ret {
        syn::ReturnType::Default => true,
        syn::ReturnType::Type(_, ty) => match &**ty {
            syn::Type::Tuple(t) => t.elems.is_empty(),
            _ => false,
        },
    }
}

/// Check if return type is `Result<(), E>`.
fn returns_result_unit(ret: &syn::ReturnType) -> bool {
    match ret {
        syn::ReturnType::Type(_, ty) => is_result_unit(ty),
        _ => false,
    }
}

fn is_result_unit(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(seg) = type_path.path.segments.last() {
            if seg.ident == "Result" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(ok_ty)) = args.args.first() {
                        return match ok_ty {
                            syn::Type::Tuple(t) => t.elems.is_empty(),
                            _ => false,
                        };
                    }
                }
            }
        }
    }
    false
}

/// Check if signature has mutable parameters.
fn has_mutable_params(sig: &syn::Signature) -> bool {
    sig.inputs.iter().any(|arg| match arg {
        syn::FnArg::Receiver(r) => r.mutability.is_some(),
        syn::FnArg::Typed(pat_type) => match &*pat_type.ty {
            syn::Type::Reference(r) => r.mutability.is_some(),
            _ => false,
        },
    })
}

/// Check if signature has mutable data output parameters.
fn has_mutable_data_output_param(sig: &syn::Signature) -> bool {
    sig.inputs.iter().any(|arg| match arg {
        syn::FnArg::Receiver(r) => r.mutability.is_some(),
        syn::FnArg::Typed(pat_type) => match &*pat_type.ty {
            syn::Type::Reference(r) => r.mutability.is_some() && type_is_data_output(&r.elem),
            _ => false,
        },
    })
}

/// Check if signature is a Formatter formatting method `fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result`.
fn is_fmt_signature(sig: &syn::Signature) -> bool {
    let is_fmt_ret = match &sig.output {
        syn::ReturnType::Type(_, ty) => {
            if let syn::Type::Path(p) = &**ty {
                p.path
                    .segments
                    .last()
                    .map_or(false, |s| s.ident == "Result")
            } else {
                false
            }
        }
        _ => false,
    };
    let has_formatter_param = sig.inputs.iter().any(|input| {
        if let syn::FnArg::Typed(pat_type) = input {
            let ty_str = quote::quote!(#pat_type.ty).to_string().replace(' ', "");
            ty_str.contains("Formatter")
        } else {
            false
        }
    });
    is_fmt_ret && has_formatter_param
}

fn is_byte_unpack_codec_expr(expr: &syn::ExprBinary) -> bool {
    match expr.op {
        syn::BinOp::Shl(_)
        | syn::BinOp::ShlAssign(_)
        | syn::BinOp::Shr(_)
        | syn::BinOp::ShrAssign(_) => {
            if let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(lit_int),
                ..
            }) = &*expr.right
            {
                let val = lit_int.base10_parse::<u32>().unwrap_or(0);
                val == 8
                    || val == 16
                    || val == 24
                    || val == 32
                    || val == 40
                    || val == 48
                    || val == 56
            } else {
                false
            }
        }
        syn::BinOp::BitOr(_)
        | syn::BinOp::BitOrAssign(_)
        | syn::BinOp::BitAnd(_)
        | syn::BinOp::BitAndAssign(_)
        | syn::BinOp::BitXor(_)
        | syn::BinOp::BitXorAssign(_) => true,
        _ => false,
    }
}

fn has_byte_buffer_param(sig: &syn::Signature) -> bool {
    sig.inputs.iter().any(|input| {
        if let syn::FnArg::Typed(pat_type) = input {
            type_is_byte_buffer(&pat_type.ty)
        } else {
            false
        }
    })
}

fn type_is_byte_buffer(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Reference(r) => match &*r.elem {
            syn::Type::Slice(s) => match &*s.elem {
                syn::Type::Path(p) => p.path.is_ident("u8"),
                _ => false,
            },
            syn::Type::Path(p) => {
                if let Some(seg) = p.path.segments.last() {
                    if seg.ident == "Vec" {
                        if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                            if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                                if let syn::Type::Path(inner_p) = inner_ty {
                                    return inner_p.path.is_ident("u8");
                                }
                            }
                        }
                    }
                }
                false
            }
            _ => false,
        },
        syn::Type::Path(p) => {
            if let Some(seg) = p.path.segments.last() {
                if seg.ident == "Vec" {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            if let syn::Type::Path(inner_p) = inner_ty {
                                return inner_p.path.is_ident("u8");
                            }
                        }
                    }
                }
            }
            false
        }
        _ => false,
    }
}

fn type_is_direct_numeric_scalar(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Reference(reference) => type_is_direct_numeric_scalar(&reference.elem),
        syn::Type::Group(group) => type_is_direct_numeric_scalar(&group.elem),
        syn::Type::Paren(paren) => type_is_direct_numeric_scalar(&paren.elem),
        syn::Type::Path(path) => path
            .path
            .segments
            .last()
            .is_some_and(|segment| SCALAR_TYPES.contains(&segment.ident.to_string().as_str())),
        _ => false,
    }
}

struct WireCodecSemanticVisitor {
    semantic_idents: BTreeSet<String>,
    forbidden: bool,
}

impl<'ast> Visit<'ast> for WireCodecSemanticVisitor {
    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        let is_arithmetic = matches!(
            expr.op,
            syn::BinOp::Add(_)
                | syn::BinOp::Sub(_)
                | syn::BinOp::Mul(_)
                | syn::BinOp::Div(_)
                | syn::BinOp::Rem(_)
                | syn::BinOp::AddAssign(_)
                | syn::BinOp::SubAssign(_)
                | syn::BinOp::MulAssign(_)
                | syn::BinOp::DivAssign(_)
                | syn::BinOp::RemAssign(_)
        );
        if is_arithmetic && !is_byte_unpack_codec_expr(expr) {
            let mut reads = IdentReadCollector::default();
            reads.visit_expr_binary(expr);
            if reads
                .idents
                .iter()
                .any(|ident| self.semantic_idents.contains(ident))
            {
                self.forbidden = true;
            }
        }
        syn::visit::visit_expr_binary(self, expr);
    }

    fn visit_expr_unary(&mut self, expr: &'ast syn::ExprUnary) {
        let mut reads = IdentReadCollector::default();
        reads.visit_expr_unary(expr);
        if reads
            .idents
            .iter()
            .any(|ident| self.semantic_idents.contains(ident))
        {
            self.forbidden = true;
        }
        syn::visit::visit_expr_unary(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        let method = expr.method.to_string();
        if matches!(
            method.as_str(),
            "sum"
                | "product"
                | "fold"
                | "rfold"
                | "reduce"
                | "count"
                | "any"
                | "all"
                | "filter"
                | "filter_map"
                | "map"
                | "flat_map"
                | "sort"
                | "sort_by"
                | "sort_unstable"
                | "dedup"
                | "wrapping_add"
                | "wrapping_sub"
                | "wrapping_mul"
                | "wrapping_div"
                | "wrapping_rem"
                | "overflowing_add"
                | "overflowing_sub"
                | "overflowing_mul"
                | "count_ones"
                | "count_zeros"
                | "rotate_left"
                | "rotate_right"
                | "reverse_bits"
                | "abs"
                | "sqrt"
                | "powf"
                | "powi"
                | "exp"
                | "ln"
                | "sin"
                | "cos"
                | "tan"
        ) {
            self.forbidden = true;
        }
        syn::visit::visit_expr_method_call(self, expr);
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        let is_index_range = matches!(&*expr.expr, syn::Expr::Range(_));
        if is_index_range {
            syn::visit::visit_expr_for_loop(self, expr);
            return;
        }
        let mut loop_bindings = BTreeSet::new();
        extract_pat_bindings(&expr.pat, &mut loop_bindings);
        let added = loop_bindings
            .iter()
            .filter(|binding| self.semantic_idents.insert((*binding).clone()))
            .cloned()
            .collect::<Vec<_>>();
        syn::visit::visit_expr(&mut *self, &expr.expr);
        self.visit_block(&expr.body);
        for binding in added {
            self.semantic_idents.remove(&binding);
        }
    }
}

fn is_wire_codec_ast(sig: &syn::Signature, block: &syn::Block) -> bool {
    if !has_byte_buffer_param(sig) {
        return false;
    }

    let mut semantic_idents = BTreeSet::new();
    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            if type_is_direct_numeric_scalar(&pat_type.ty) {
                extract_pat_bindings(&pat_type.pat, &mut semantic_idents);
            }
        }
    }
    let mut visitor = WireCodecSemanticVisitor {
        semantic_idents,
        forbidden: false,
    };
    visitor.visit_block(block);
    !visitor.forbidden
}

/// AST visitor extracting structural mathematical, arithmetic, loop, and algorithmic features from function bodies.
#[derive(Default)]
struct BodyFeatureVisitor {
    has_payload_arithmetic: bool,
    has_comparison: bool,
    has_unary_op: bool,
    has_numeric_method: bool,
    has_branch_on_data: bool,
    has_loops: bool,
    has_iterators_or_transforms: bool,
    has_algorithms: bool,
}

impl BodyFeatureVisitor {
    fn has_semantic_operation(&self) -> bool {
        self.has_payload_arithmetic
            || self.has_comparison
            || self.has_unary_op
            || self.has_numeric_method
            || self.has_loops
            || self.has_iterators_or_transforms
            || self.has_algorithms
            || self.has_branch_on_data
    }
}

impl<'ast> Visit<'ast> for BodyFeatureVisitor {
    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        match expr.op {
            syn::BinOp::Add(_)
            | syn::BinOp::Sub(_)
            | syn::BinOp::Mul(_)
            | syn::BinOp::Div(_)
            | syn::BinOp::Rem(_)
            | syn::BinOp::BitAnd(_)
            | syn::BinOp::BitOr(_)
            | syn::BinOp::BitXor(_)
            | syn::BinOp::Shl(_)
            | syn::BinOp::Shr(_)
            | syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_) => {
                self.has_payload_arithmetic = true;
            }
            syn::BinOp::Eq(_)
            | syn::BinOp::Ne(_)
            | syn::BinOp::Lt(_)
            | syn::BinOp::Le(_)
            | syn::BinOp::Gt(_)
            | syn::BinOp::Ge(_) => {
                self.has_comparison = true;
            }
            _ => {}
        }
        syn::visit::visit_expr_binary(self, expr);
    }

    fn visit_expr_unary(&mut self, expr: &'ast syn::ExprUnary) {
        match expr.op {
            syn::UnOp::Neg(_) | syn::UnOp::Not(_) => {
                self.has_unary_op = true;
            }
            _ => {}
        }
        syn::visit::visit_expr_unary(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        let name = expr.method.to_string();
        match name.as_str() {
            "min" | "max" | "clamp" | "abs" | "round" | "floor" | "ceil" | "trunc" | "fract"
            | "signum" | "recip" | "sqrt" | "cbrt" | "powf" | "powi" | "exp" | "exp2" | "ln"
            | "log" | "log2" | "log10" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "atan2" | "sinh" | "cosh" | "tanh" | "count_ones" | "count_zeros"
            | "leading_zeros" | "trailing_zeros" | "leading_ones" | "trailing_ones"
            | "rotate_left" | "rotate_right" | "reverse_bits" | "swap_bytes" | "wrapping_add"
            | "wrapping_sub" | "wrapping_mul" | "wrapping_div" | "wrapping_rem"
            | "wrapping_neg" | "wrapping_shl" | "wrapping_shr" | "saturating_add"
            | "saturating_sub" | "saturating_mul" | "saturating_div" | "saturating_pow"
            | "overflowing_add" | "overflowing_sub" | "overflowing_mul" | "checked_add"
            | "checked_sub" | "checked_mul" | "checked_div" | "sum" | "product" | "count"
            | "any" | "all" | "mul_add" | "copysign" => {
                self.has_numeric_method = true;
            }
            "map" | "filter" | "filter_map" | "fold" | "rfold" | "reduce" | "for_each" | "zip"
            | "enumerate" | "step_by" | "chain" | "flatten" | "flat_map" => {
                self.has_iterators_or_transforms = true;
            }
            "binary_search"
            | "binary_search_by"
            | "binary_search_by_key"
            | "partition_point"
            | "sort"
            | "sort_unstable"
            | "sort_by"
            | "sort_unstable_by"
            | "sort_by_key"
            | "sort_unstable_by_key"
            | "dedup"
            | "dedup_by"
            | "dedup_by_key"
            | "select_nth_unstable"
            | "select_nth_unstable_by" => {
                self.has_algorithms = true;
            }
            _ => {}
        }
        syn::visit::visit_expr_method_call(self, expr);
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        self.has_loops = true;
        syn::visit::visit_expr_for_loop(self, expr);
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        self.has_loops = true;
        syn::visit::visit_expr_while(self, expr);
    }

    fn visit_expr_loop(&mut self, expr: &'ast syn::ExprLoop) {
        self.has_loops = true;
        syn::visit::visit_expr_loop(self, expr);
    }

    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        self.has_branch_on_data = true;
        syn::visit::visit_expr_if(self, expr);
    }

    fn visit_expr_match(&mut self, expr: &'ast syn::ExprMatch) {
        self.has_branch_on_data = true;
        syn::visit::visit_expr_match(self, expr);
    }
}

/// Check if signature inputs/outputs and body indicate host data-processing / simulation work.
fn is_data_processing_ast(sig: &syn::Signature, block: &syn::Block) -> bool {
    let has_inputs = has_data_input_ast(sig);

    let has_typed_inputs = sig
        .inputs
        .iter()
        .any(|arg| matches!(arg, syn::FnArg::Typed(_)));
    let has_receiver = sig
        .inputs
        .iter()
        .any(|arg| matches!(arg, syn::FnArg::Receiver(_)));
    if (has_typed_inputs && !has_inputs) || (!has_typed_inputs && has_receiver) {
        return false;
    }

    let mut visitor = BodyFeatureVisitor::default();
    visitor.visit_block(block);

    // If function is a byte codec operating over byte buffers &[u8] / &mut [u8]
    // and only performs bounds checks, slicing, and byte copying without semantic arithmetic/numeric transforms:
    if has_byte_buffer_param(sig) {
        if !visitor.has_payload_arithmetic
            && !visitor.has_numeric_method
            && !visitor.has_iterators_or_transforms
            && !visitor.has_algorithms
        {
            return false;
        }
    }

    visitor.has_semantic_operation()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyze_files(files: &[(&str, &str)]) -> Vec<Finding> {
        let canonical_source = include_str!("../../../vyre-foundation/src/program_dispatch/mod.rs");
        let canonical_parsed =
            syn::parse_file(canonical_source).expect("canonical trait must parse as Rust");
        let mut canonical_trait_methods = BTreeSet::new();
        let mut canonical_resident_upload_methods = BTreeSet::new();
        derive_canonical_dispatcher_methods(
            &canonical_parsed,
            &mut canonical_trait_methods,
            &mut canonical_resident_upload_methods,
        );
        assert!(
            !canonical_trait_methods.is_empty(),
            "canonical ProgramDispatcher must yield non-empty execution methods"
        );
        assert!(
            !canonical_resident_upload_methods.is_empty(),
            "canonical ProgramDispatcher must yield non-empty resident upload methods"
        );

        let mut all_functions = Vec::new();
        let mut all_calls = Vec::new();
        let mut all_static_consts = Vec::new();
        let mut all_findings = Vec::new();
        let mut all_types_with_public_fields = BTreeSet::new();
        for &(path, code) in files {
            let parsed = syn::parse_file(code).expect("test code must parse as Rust");
            let fn_offset = all_functions.len();
            let mut visitor = AstAnalysisVisitor::new(
                PathBuf::from(path),
                false,
                fn_offset,
                canonical_trait_methods.clone(),
                canonical_resident_upload_methods.clone(),
            );

            for item in &parsed.items {
                match item {
                    syn::Item::Struct(s) => {
                        visitor.local_declared_types.insert(s.ident.to_string());
                    }
                    syn::Item::Enum(e) => {
                        visitor.local_declared_types.insert(e.ident.to_string());
                    }
                    syn::Item::Type(t) => {
                        visitor.local_declared_types.insert(t.ident.to_string());
                    }
                    syn::Item::Trait(tr) => {
                        visitor.local_declared_types.insert(tr.ident.to_string());
                    }
                    syn::Item::Union(u) => {
                        visitor.local_declared_types.insert(u.ident.to_string());
                    }
                    _ => {}
                }
            }

            visitor.visit_file(&parsed);
            all_functions.extend(visitor.functions);
            all_calls.extend(visitor.calls);
            all_static_consts.extend(visitor.static_consts);
            all_findings.extend(visitor.direct_findings);
            all_types_with_public_fields.extend(visitor.types_with_public_fields);
        }

        let evaluated = evaluate_rules(
            &all_functions,
            &all_calls,
            &all_static_consts,
            &all_types_with_public_fields,
        );
        all_findings.extend(evaluated);

        let mut deduped_findings = Vec::new();
        let mut seen_findings = BTreeSet::new();
        for finding in all_findings {
            let key = (finding.file.clone(), finding.line, finding.message.clone());
            if seen_findings.insert(key) {
                deduped_findings.push(finding);
            }
        }
        deduped_findings
    }

    #[test]
    fn canonical_program_dispatcher_exact_path_derives_execution_methods() {
        let canonical_source = include_str!("../../../vyre-foundation/src/program_dispatch/mod.rs");
        let canonical_parsed =
            syn::parse_file(canonical_source).expect("canonical trait must parse as Rust");
        let mut canonical_trait_methods = BTreeSet::new();
        let mut canonical_resident_upload_methods = BTreeSet::new();
        derive_canonical_dispatcher_methods(
            &canonical_parsed,
            &mut canonical_trait_methods,
            &mut canonical_resident_upload_methods,
        );
        assert!(
            canonical_trait_methods.contains("dispatch"),
            "must contain direct dispatch method"
        );
        assert!(
            canonical_trait_methods.contains("dispatch_resident"),
            "must contain dispatch_resident"
        );
        assert!(
            canonical_trait_methods.contains("dispatch_resident_sequence"),
            "must contain dispatch_resident_sequence"
        );
        assert!(
            canonical_trait_methods.contains("dispatch_resident_sequence_read_many"),
            "must contain dispatch_resident_sequence_read_many"
        );
        assert!(
            canonical_trait_methods.contains("dispatch_resident_sequence_read_ranges"),
            "must contain dispatch_resident_sequence_read_ranges"
        );
        assert!(
            !canonical_trait_methods.contains("supports_persistent"),
            "metadata methods must not be execution methods"
        );
        assert!(
            !canonical_trait_methods.contains("alloc_resident"),
            "allocation methods must not be execution methods"
        );
        assert!(
            canonical_resident_upload_methods.contains("upload_resident"),
            "must derive resident upload methods from immutable byte payload parameters"
        );
        assert!(
            !canonical_resident_upload_methods.contains("read_resident_ranges_into"),
            "mutable readback outputs must not masquerade as upload methods"
        );
    }

    #[test]
    fn clean_production_code_produces_no_findings() {
        let code = r#"
use vyre_foundation::ir::Program;

pub fn add_u32(input: &str, out: &str, n: u32) -> Result<Program, String> {
    Ok(Program::new())
}

const EXPECTED_OUTPUT: [u8; 4] = [0, 1, 2, 3];

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![EXPECTED_OUTPUT.to_vec()]]
}
"#;
        let findings = analyze_files(&[("vyre-primitives/src/add.rs", code)]);
        assert!(
            findings.is_empty(),
            "expected clean production code to pass without findings, got: {findings:?}"
        );
    }

    #[test]
    fn expected_output_with_wire_pack_and_literal_helper_is_convicted() {
        let code = r#"
use vyre_foundation::ir::Program;

const EXPECTED: [u32; 2] = [42, 99];

fn expected_bytes() -> Vec<u8> {
    vec![1, 2, 3, 4]
}

pub fn add_u32(input: &str, out: &str, n: u32) -> Result<Program, String> {
    Ok(Program::new())
}

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![
        crate::wire::pack_u32_slice(&EXPECTED),
        expected_bytes(),
    ]]
}
"#;
        let findings = analyze_files(&[("vyre-primitives/src/add.rs", code)]);
        assert!(
            !findings.is_empty(),
            "expected_output with wire pack and literal helper must be convicted"
        );
    }

    #[test]
    fn gpu_dispatcher_boundary_and_validation_functions_are_permitted() {
        let code = r#"
use vyre_foundation::program_dispatch::ProgramDispatcher;

pub fn validate_circuit(n: u32) -> Result<(), String> {
    if n == 0 {
        return Err("n must be non-zero".to_string());
    }
    Ok(())
}

pub fn predict_runtime_fixed_via(
    dispatcher: &impl ProgramDispatcher,
    weights: &[u32],
) -> Result<(u32, u32), String> {
    validate_circuit(weights.len() as u32)?;
    let _ = dispatcher.dispatch(1, 2);
    Ok((10, 20))
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/analysis/cost_model.rs", code)]);
        assert!(
            findings.is_empty(),
            "gpu dispatcher boundary and validation functions must be permitted, got: {findings:?}"
        );
    }

    #[test]
    fn compiler_planner_reachable_from_builder_is_permitted() {
        let code = r#"
use vyre_foundation::ir::Program;

pub struct DiagnosticAggregationPlan {
    pub items: u32,
}

pub fn plan_compact_diagnostic_readback(n: u32) -> Result<DiagnosticAggregationPlan, String> {
    let _table = binary_byte_lut();
    Ok(DiagnosticAggregationPlan { items: n })
}

pub fn compile_pipeline(n: u32) -> Program {
    let _plan = plan_compact_diagnostic_readback(n).unwrap();
    Program::new()
}

fn binary_byte_lut() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0..256 {
        table[i] = (i as u32).wrapping_mul(3);
    }
    table
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/analysis/planner.rs", code)]);
        assert!(
            findings.is_empty(),
            "compiler planner called by IR builder must be permitted, got: {findings:?}"
        );
    }

    #[test]
    fn cfg_test_cpu_ref_is_permitted_and_not_flagged() {
        let code = r#"
use vyre_foundation::ir::Program;

pub fn fma_f32(a: &str, b: &str, c: &str, out: &str, n: u32) -> Program {
    Program::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cpu_ref(a: &[f32], b: &[f32], c: &[f32]) -> Vec<u8> {
        let sim = vyre_reference::subgroup::SubgroupSimulator::default();
        vec![]
    }

    #[test]
    fn test_fma_correctness() {
        let out = test_cpu_ref(&[], &[], &[]);
        assert_eq!(out, vec![]);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/math/fma.rs", code)]);
        assert!(
            findings.is_empty(),
            "expected test-scoped cpu_ref helper to be permitted, got: {findings:?}"
        );
    }

    #[test]
    fn cfg_test_item_level_function_body_vyre_reference_is_not_misclassified() {
        let code = r#"
#[test]
fn test_sim_in_standalone_function() {
    let _sim = vyre_reference::subgroup::SubgroupSimulator::default();
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/test_item.rs", code)]);
        assert!(
            findings.is_empty(),
            "item-level test functions must not report production simulator findings, got: {findings:?}"
        );
    }

    #[test]
    fn cfg_test_impl_block_is_scoped_as_test() {
        let code = r#"
pub struct TestHelper;

#[cfg(test)]
impl TestHelper {
    pub fn compute_oracle(&self, input: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &x in input {
            out.extend_from_slice(&x.wrapping_mul(3).to_le_bytes());
        }
        out
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/scoped_impl.rs", code)]);
        assert!(
            findings.is_empty(),
            "#[cfg(test)] impl methods must be scoped as test-only and produce zero findings, got: {findings:?}"
        );
    }

    #[test]
    fn cfg_test_trait_block_is_scoped_as_test() {
        let code = r#"
#[cfg(test)]
pub trait TestOracleTrait {
    fn default_sim(&self, input: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &x in input {
            out.extend_from_slice(&x.wrapping_add(1).to_le_bytes());
        }
        out
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/scoped_trait.rs", code)]);
        assert!(
            findings.is_empty(),
            "#[cfg(test)] trait with default body must be scoped as test-only, got: {findings:?}"
        );
    }

    #[test]
    fn production_trait_default_method_uncalled_oracle_is_flagged() {
        let code = r#"
pub trait ProductionTrait {
    fn default_sim(&self, input: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &x in input {
            out.extend_from_slice(&x.wrapping_add(1).to_le_bytes());
        }
        out
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/trait_oracle.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "production trait with unreached default oracle body must be flagged"
        );
        assert!(findings[0].message.contains("`default_sim`"));
    }

    #[test]
    fn mutation_oracle_detection_catches_production_cpu_ref_fn() {
        let code = r#"
use vyre_foundation::ir::Program;

pub fn popcount(input: &str, out: &str, n: u32) -> Program {
    Program::new()
}

fn cpu_ref(input: &[u32]) -> Vec<u8> {
    input.iter().map(|x| x.count_ones()).collect()
}
"#;
        let findings = analyze_files(&[("vyre-primitives/src/hardware/popcount.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "expected finding for production cpu_ref definition"
        );
        assert!(
            findings[0]
                .message
                .contains("production host reference oracle function definition"),
            "finding message should name the oracle function defect: {}",
            findings[0].message
        );
        assert_eq!(findings[0].line, Some(8));
    }

    #[test]
    fn mutation_oracle_detection_catches_production_vyre_reference_usage() {
        let code = r#"
pub fn simulate_runtime() {
    let _sim = vyre_reference::subgroup::SubgroupSimulator::default();
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/runtime.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "expected finding for production vyre_reference simulator usage"
        );
        assert_eq!(findings[0].line, Some(3));
    }

    #[test]
    fn mutation_catches_local_dummy_program_masquerade() {
        let code = r#"
pub struct Program;

pub fn fake_builder(x: f32) -> Program {
    let _ = x * x;
    Program
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/fake_program.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "locally declared Program masquerade must be flagged"
        );
        assert!(findings[0].message.contains("`fake_builder`"));
        assert_eq!(findings[0].line, Some(4));
    }

    #[test]
    fn mutation_catches_crate_bogus_program_masquerade() {
        let code = r#"
use crate::bogus::Program;

pub fn fake_builder(x: f32) -> Program {
    let _ = x * x;
    Program
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/bogus_program.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "crate::bogus::Program masquerade must be flagged"
        );
        assert!(findings[0].message.contains("`fake_builder`"));
        assert_eq!(findings[0].line, Some(4));
    }

    #[test]
    fn mutation_catches_glob_import_program_masquerade() {
        let code = r#"
mod fake {
    pub struct Program;
}
use fake::*;

pub fn fake_builder(x: f32) -> Program {
    let _ = x * x;
    Program
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/glob_program.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "glob imported Program masquerade must be flagged"
        );
        assert!(findings[0].message.contains("`fake_builder`"));
        assert_eq!(findings[0].line, Some(7));
    }

    #[test]
    fn mutation_catches_sibling_imported_fake_program_masquerade() {
        let code = r#"
mod sibling {
    pub struct Program;
}
use sibling::Program;

pub fn fake_builder(x: f32) -> Program {
    let _ = x * x;
    Program
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/sibling_program.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "sibling imported Program masquerade must be flagged"
        );
        assert!(findings[0].message.contains("`fake_builder`"));
        assert_eq!(findings[0].line, Some(7));
    }

    #[test]
    fn mutation_catches_sibling_imported_fake_dispatcher_trait_masquerade() {
        let code = r#"
mod sibling {
    pub trait ProgramDispatcher {
        fn dispatch(&self, a: u32, b: u32);
    }
}
use sibling::ProgramDispatcher;

pub fn fake_dispatch(d: &impl ProgramDispatcher, x: f32) -> f32 {
    d.dispatch(1, 2);
    x + 1.0
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/sibling_dispatcher.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "sibling imported ProgramDispatcher masquerade must be flagged"
        );
        assert!(findings[0].message.contains("`fake_dispatch`"));
        assert_eq!(findings[0].line, Some(9));
    }

    #[test]
    fn mutation_catches_fake_dispatch_error_without_canonical_dispatcher() {
        let code = r#"
pub struct FakeDispatchError;

pub struct LocalDevice;
impl LocalDevice {
    pub fn dispatch(&self, _a: u32, _b: u32) {}
}

pub fn fake_dispatch(x: f32) -> Result<f32, FakeDispatchError> {
    let obj = LocalDevice;
    obj.dispatch(1, 2);
    Ok(x + 1.0)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/fake_device.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "dispatch call without canonical dispatcher parameter must be flagged"
        );
        assert!(findings[0].message.contains("`fake_dispatch`"));
        assert_eq!(findings[0].line, Some(9));
    }

    #[test]
    fn mutation_catches_dispatch_error_and_resident_read_range_param_masquerade() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ResidentReadRange};

pub struct LocalDevice;
impl LocalDevice {
    pub fn dispatch(&self, _a: u32, _b: u32) {}
}

pub fn fake_dispatch_with_error_param(
    _err: &DispatchError,
    _range: &ResidentReadRange,
    x: f32,
) -> f32 {
    let obj = LocalDevice;
    obj.dispatch(1, 2);
    x + 1.0
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/fake_dispatch_params.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "DispatchError/ResidentReadRange parameters must not establish dispatch roots"
        );
        assert!(findings[0]
            .message
            .contains("`fake_dispatch_with_error_param`"));
        assert_eq!(findings[0].line, Some(9));
    }

    #[test]
    fn mutation_catches_mixed_tuple_data_type_masquerade() {
        let code = r#"
use vyre_foundation::ir::DataType;

pub fn oracle_with_mixed_tuple(data: &[u32]) -> (Vec<u32>, DataType) {
    let mut out = Vec::new();
    for &x in data {
        out.push(x * 2);
    }
    (out, DataType::U32)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/mixed_tuple.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "mixed tuple with DataType metadata must not establish an IR builder root"
        );
        assert!(findings[0].message.contains("`oracle_with_mixed_tuple`"));
        assert_eq!(findings[0].line, Some(4));
    }

    #[test]
    fn mutation_catches_result_vec_with_fusion_error_masquerade() {
        let code = r#"
use vyre_foundation::execution_plan::fusion::FusionError;

pub fn fake_fusion_oracle(data: &[u32]) -> Result<Vec<u32>, FusionError> {
    let mut out = Vec::new();
    for &x in data {
        out.push(x.wrapping_add(1));
    }
    Ok(out)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/fake_fusion.rs", code)]);
        assert_eq!(findings.len(), 1, "Result<Vec<u32>, FusionError> where success type is data must not establish an IR builder root");
        assert!(findings[0].message.contains("`fake_fusion_oracle`"));
        assert_eq!(findings[0].line, Some(4));
    }

    #[test]
    fn mutation_operation_registration_allows_test_inputs_generator_and_catches_expected_output_oracle(
    ) {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program(a: &str, b: &str, out: &str, n: u32) -> Program {
    Program::new()
}

pub fn generate_deterministic_inputs() -> Vec<Vec<Vec<u8>>> {
    let mut words = Vec::new();
    for i in 0..10 {
        words.push(i.wrapping_mul(7));
    }
    vec![vec![crate::wire::pack_u32_slice(&words)]]
}

pub fn dynamic_math_oracle(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &w in words {
        out.extend_from_slice(&w.wrapping_mul(17).to_le_bytes());
    }
    out
}

inventory::submit! {
    OperationRegistration::library(
        "test::op",
        || add_program("a", "b", "out", 2),
        Some(generate_deterministic_inputs),
        Some(|| {
            let fixture = dynamic_math_oracle(&[1, 2]);
            vec![vec![fixture]]
        }),
    )
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/op_test_inputs.rs", code)]);
        assert!(
            !findings.is_empty(),
            "test_inputs must be permitted while dynamic expected_output oracle is convicted"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("`dynamic_math_oracle`")
                || f.message.contains("dynamic_math_oracle")));
    }

    #[test]
    fn mutation_operation_registration_struct_literal_catches_expected_output_oracle() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::{OperationRegistration, OperationTier};

pub fn add_program() -> Program {
    Program::new()
}

pub fn struct_literal_oracle(words: &[u32]) -> Vec<u8> {
    words.iter().map(|w| (w.wrapping_mul(3)) as u8).collect()
}

pub static REG: OperationRegistration = OperationRegistration {
    id: "test::literal",
    semantic_version: 1,
    signature: None,
    tier: OperationTier::Library,
    category: None,
    build: Some(add_program),
    test_inputs: None,
    expected_output: Some(|| vec![vec![struct_literal_oracle(&[1, 2])]]),
    laws: &[],
    tolerance: vyre_foundation::operation::TolerancePolicy::EXACT,
    geometry_requirements: None,
    source_file: "test.rs",
    explicit_effects: None,
    explicit_capabilities: None,
};
"#;
        let findings = analyze_files(&[("vyre-libs/src/op_struct_literal.rs", code)]);
        assert!(
            !findings.is_empty(),
            "struct literal expected_output oracle must be caught"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("`struct_literal_oracle`")
                || f.message.contains("struct_literal_oracle")));
    }

    #[test]
    fn mutation_operation_registration_aliased_as_or_catches_expected_output_oracle() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration as OR;

pub fn add_program() -> Program {
    Program::new()
}

pub fn aliased_oracle(words: &[u32]) -> Vec<u8> {
    words.iter().map(|w| (w.wrapping_mul(2)) as u8).collect()
}

inventory::submit! {
    OR::library(
        "test::aliased",
        add_program,
        None,
        Some(|| vec![vec![aliased_oracle(&[1, 2])]]),
    )
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/op_aliased.rs", code)]);
        assert!(
            !findings.is_empty(),
            "aliased OR::library expected_output oracle must be caught"
        );
        assert!(findings.iter().any(
            |f| f.message.contains("`aliased_oracle`") || f.message.contains("aliased_oracle")
        ));
    }

    #[test]
    fn control_bogus_local_operation_registration_does_not_false_positive() {
        let code = r#"
pub struct BogusOperationRegistration {
    pub expected_output: fn() -> Vec<u8>,
}

impl BogusOperationRegistration {
    pub fn library(_id: &str, _build: fn(), _inputs: Option<fn()>, _expected: Option<fn()>) {}
}

pub fn setup_mock() {
    BogusOperationRegistration::library("mock", || {}, None, None);
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/op_bogus_control.rs", code)]);
        assert!(
            findings.is_empty(),
            "bogus local OperationRegistration struct must not false positive, got: {findings:?}"
        );
    }

    #[test]
    fn mutation_operation_registration_catches_inline_closure_loop_and_math() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program() -> Program {
    Program::new()
}

inventory::submit! {
    OperationRegistration::library(
        "test::inline_math",
        add_program,
        None,
        Some(|| {
            let mut out = Vec::new();
            for i in 0..10 {
                out.push((i * 2) as u8);
            }
            vec![vec![out]]
        }),
    )
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/op_inline_math.rs", code)]);
        assert!(
            !findings.is_empty(),
            "inline closure loop and math must be convicted"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("inline expected_output closure")
                || f.message.contains("expected_output")));
    }

    #[test]
    fn clean_operation_registration_allows_literal_and_const_byte_array() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program() -> Program {
    Program::new()
}

const EXPECTED_BYTES: [u8; 4] = [100, 0, 200, 0];

inventory::submit! {
    OperationRegistration::library(
        "test::clean_literal",
        add_program,
        None,
        Some(|| {
            vec![vec![EXPECTED_BYTES.to_vec()]]
        }),
    )
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/op_clean_literal.rs", code)]);
        assert!(
            findings.is_empty(),
            "clean literal array and to_vec must produce zero findings, got: {findings:?}"
        );
    }

    #[test]
    fn mutation_operation_registration_catches_wire_pack_in_expected_output() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program() -> Program {
    Program::new()
}

const EXPECTED_BYTES: [u32; 2] = [100, 200];

inventory::submit! {
    OperationRegistration::library(
        "test::clean_pack",
        add_program,
        None,
        Some(|| {
            vec![vec![crate::wire::pack_u32_slice(&EXPECTED_BYTES)]]
        }),
    )
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/op_clean_pack.rs", code)]);
        assert!(
            !findings.is_empty(),
            "wire pack in expected_output must be convicted"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("expected_output")));
    }

    #[test]
    fn mutation_operation_registration_catches_helper_function_in_expected_output() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program() -> Program {
    Program::new()
}

fn expected_bytes() -> Vec<u8> {
    vec![1, 2, 3, 4]
}

inventory::submit! {
    OperationRegistration::library(
        "test::helper_expected",
        add_program,
        None,
        Some(|| {
            vec![vec![expected_bytes()]]
        }),
    )
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/op_helper_expected.rs", code)]);
        assert!(
            !findings.is_empty(),
            "helper function call in expected_output must be convicted"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("expected_output")));
    }

    #[test]
    fn mutation_operation_registration_catches_local_closure_alias_in_expected_output() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn reduce_count(bitset: &str, out: &str, words: u32) -> Program {
    Program::default()
}

inventory::submit! {
    OperationRegistration::library(
        "vyre-libs::reduce::count",
        || reduce_count("bitset", "out", 2),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0b1111, 0xFFFF_FFFF]), to_bytes(&[0])]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[36])]]
        }),
    )
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/reduce/count.rs", code)]);
        assert!(
            !findings.is_empty(),
            "expected_output local closure/codec execution must be convicted"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("expected_output")));
    }

    #[test]
    fn mutation_catches_dispatcher_err_fallback_branch_oracle() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn host_oracle_fallback(words: &[u32]) -> u32 {
    let mut sum = 0u32;
    for &w in words {
        sum = sum.wrapping_add(w);
    }
    sum
}

pub fn dispatch_with_cpu_fallback(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<u32, DispatchError> {
    match dispatcher.dispatch(1, 2) {
        Ok(_) => Ok(42),
        Err(_) => Ok(host_oracle_fallback(words)),
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/dispatch_fallback.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "dispatcher fallback branch oracle must be convicted"
        );
        assert!(findings[0].message.contains("host CPU reference fallback"));
        assert!(findings[0].message.contains("`host_oracle_fallback`"));
        assert_eq!(findings[0].line, Some(4));
    }

    #[test]
    fn mutation_catches_dispatcher_unwrap_or_else_fallback_oracle() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn host_oracle_fallback(words: &[u32]) -> u32 {
    let mut sum = 0u32;
    for &w in words {
        sum = sum.wrapping_add(w);
    }
    sum
}

pub fn dispatch_with_unwrap_fallback(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<u32, DispatchError> {
    dispatcher
        .dispatch(1, 2)
        .map(|_| 42)
        .unwrap_or_else(|_| Ok(host_oracle_fallback(words)))
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/dispatch_unwrap_fallback.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "dispatcher unwrap_or_else fallback oracle must be convicted"
        );
        assert!(findings[0].message.contains("host CPU reference fallback"));
        assert!(findings[0].message.contains("`host_oracle_fallback`"));
        assert_eq!(findings[0].line, Some(4));
    }

    #[test]
    fn mutation_catches_dispatcher_post_dispatch_iter_any_reduction() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn motif_matches_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<bool, DispatchError> {
    let _ = dispatcher.dispatch(1, 2)?;
    let output = [1u32, 0, 0];
    let any_match = output.iter().any(|&x| x == 1);
    Ok(any_match)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/post_dispatch_any.rs", code)]);
        assert!(
            !findings.is_empty(),
            "post-dispatch .iter().any reduction must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host reduction/aggregation `.any`")));
    }
    fn mutation_catches_dispatcher_match_ok_post_dispatch_iter_any() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn motif_matches_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<bool, DispatchError> {
    match dispatcher.dispatch(1, 2) {
        Ok(output) => {
            let any_match = output.iter().any(|&x| x == 1);
            Ok(any_match)
        }
        Err(e) => Err(e),
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/post_dispatch_match_any.rs", code)]);
        assert!(
            !findings.is_empty(),
            "match Ok arm post-dispatch .iter().any reduction must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host reduction/aggregation `.any`")));
    }
    fn mutation_catches_dispatcher_map_closure_post_dispatch_reduction() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn motif_count_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<usize, DispatchError> {
    dispatcher
        .dispatch(1, 2)
        .map(|output| output.iter().count())
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/post_dispatch_map_count.rs", code)]);
        assert!(
            !findings.is_empty(),
            "chained .map closure post-dispatch count reduction must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host reduction/aggregation `.count`")));
    }
    fn mutation_catches_dispatcher_post_dispatch_loop_accumulation() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn motif_participation_count_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<u32, DispatchError> {
    let _ = dispatcher.dispatch(1, 2)?;
    let output = [1u32, 2, 0];
    let mut count = 0u32;
    for &x in &output {
        if x != 0 {
            count += 1;
        }
    }
    Ok(count)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/post_dispatch_loop.rs", code)]);
        assert!(
            !findings.is_empty(),
            "post-dispatch loop accumulation must be convicted"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("post-dispatch host loop/accumulation")));
    }
    fn mutation_catches_dispatcher_post_dispatch_filter_count_reduction() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn count_matches_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<usize, DispatchError> {
    let _ = dispatcher.dispatch(1, 2)?;
    let output = [1u32, 2, 3, 0];
    let count = output.iter().filter(|&&x| x > 0).count();
    Ok(count)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/post_dispatch_count.rs", code)]);
        assert!(
            !findings.is_empty(),
            "post-dispatch filter count reduction must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host reduction/aggregation `.count`")));
    }
    #[test]
    fn clean_dispatcher_allows_inter_dispatch_staging_loop() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn multi_stage_dispatch_via(dispatcher: &dyn ProgramDispatcher, input: &[u32]) -> Result<Vec<u8>, DispatchError> {
    let prog1 = Program::default();
    let out1 = dispatcher.dispatch(&prog1, &[vec![]], None)?;
    let mut staged = vec![0u32; input.len()];
    for i in 0..input.len() {
        staged[i] = input[i] ^ (out1[0][0] as u32);
    }
    let prog2 = Program::default();
    let out2 = dispatcher.dispatch(&prog2, &[staged], None)?;
    Ok(out2[0].clone())
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/staging_loop.rs", code)]);
        assert!(
            findings.is_empty(),
            "inter-dispatch staging loop preceding subsequent dispatch must be permitted: {findings:?}"
        );
    }

    fn clean_dispatcher_with_pre_validation_and_typed_unpacking_passes() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct CircuitSummary {
    pub a: u32,
    pub b: u32,
}

pub fn predict_summary_via(
    dispatcher: &impl ProgramDispatcher,
    weights: &[u32],
) -> Result<CircuitSummary, DispatchError> {
    if weights.is_empty() {
        return Err(DispatchError::BadInputs("empty weights".to_string()));
    }
    let raw = dispatcher.dispatch(1, 2)?;
    Ok(CircuitSummary {
        a: raw.get(0).copied().unwrap_or(0),
        b: raw.get(1).copied().unwrap_or(0),
    })
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/clean_dispatch.rs", code)]);
        assert!(
            findings.is_empty(),
            "clean pre-validation and post-dispatch struct unpacking must pass, got: {findings:?}"
        );
    }

    #[test]
    fn clean_dispatcher_with_dispatch_map_unpack_passes() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct CircuitSummary {
    pub a: u32,
    pub b: u32,
}

fn unpack_only(raw: Vec<u32>) -> CircuitSummary {
    CircuitSummary {
        a: raw.get(0).copied().unwrap_or(0),
        b: raw.get(1).copied().unwrap_or(0),
    }
}

pub fn predict_summary_via(
    dispatcher: &impl ProgramDispatcher,
    weights: &[u32],
) -> Result<CircuitSummary, DispatchError> {
    dispatcher
        .dispatch(1, 2)
        .map(unpack_only)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/clean_dispatch_map.rs", code)]);
        assert!(
            findings.is_empty(),
            "clean dispatch.map(unpack_only) must pass with zero findings, got: {findings:?}"
        );
    }

    #[test]
    fn clean_dispatcher_with_gpu_reduction_chain_passes() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn reduce_any_via(
    dispatcher: &impl ProgramDispatcher,
    data: &[u32],
) -> Result<bool, DispatchError> {
    let _ = dispatcher.dispatch(1, 2)?;
    Ok(true)
}

pub fn motif_matches_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<bool, DispatchError> {
    let _ = dispatcher.dispatch(1, 2)?;
    let witness = [1u32, 2];
    reduce_any_via(dispatcher, &witness)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/clean_gpu_reduction.rs", code)]);
        assert!(
            findings.is_empty(),
            "clean GPU reduction dispatch chain must pass, got: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_local_dummy_dispatcher_masquerade() {
        let code = r#"
pub struct Dispatcher;

pub fn fake_dispatch(d: &Dispatcher, x: f32) -> f32 {
    x + 1.0
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/fake_dispatcher.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "locally declared Dispatcher masquerade must be flagged"
        );
        assert!(findings[0].message.contains("`fake_dispatch`"));
        assert_eq!(findings[0].line, Some(4));
    }

    #[test]
    fn mutation_oracle_detection_catches_scalar_square_semantic_twin() {
        let code = r#"
pub fn square(x: f32) -> f32 {
    x * x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq() {
        assert_eq!(square(3.0), 9.0);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/math/scalar.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "scalar square semantic twin must be caught"
        );
        assert!(findings[0].message.contains("`square`"));
        assert_eq!(findings[0].line, Some(2));
    }

    #[test]
    fn mutation_oracle_detection_catches_branch_classifier_returning_custom_enum() {
        let code = r#"
pub enum RegionClass {
    Small,
    Medium,
    Large,
}

pub fn classify_region(size: usize) -> RegionClass {
    if size < 10 {
        RegionClass::Small
    } else if size < 100 {
        RegionClass::Medium
    } else {
        RegionClass::Large
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class() {
        let _ = classify_region(5);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/pattern/classifier.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "branch classifier returning custom enum must be caught"
        );
        assert!(findings[0].message.contains("`classify_region`"));
        assert_eq!(findings[0].line, Some(8));
    }

    #[test]
    fn mutation_oracle_detection_catches_scalar_bitwise_transform() {
        let code = r#"
pub fn compute_mask(tag: u32, shift: u32) -> u32 {
    (tag ^ 0xAA) << shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask() {
        assert_eq!(compute_mask(1, 2), 680);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/bitset/mask.rs", code)]);
        assert_eq!(findings.len(), 1, "scalar bitwise transform must be caught");
        assert!(findings[0].message.contains("`compute_mask`"));
        assert_eq!(findings[0].line, Some(2));
    }

    #[test]
    fn mutation_oracle_detection_catches_numeric_methods_clamp_abs() {
        let code = r#"
pub fn normalize_weight(w: f32) -> f32 {
    w.abs().clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm() {
        assert_eq!(normalize_weight(-0.5), 0.5);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/math/norm.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "numeric method semantic twin must be caught"
        );
        assert!(findings[0].message.contains("`normalize_weight`"));
        assert_eq!(findings[0].line, Some(2));
    }

    #[test]
    fn mutation_oracle_detection_catches_generic_named_host_twin_encode_payload() {
        let code = r#"
pub fn encode_payload(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &w in words {
        out.extend_from_slice(&w.wrapping_mul(31).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_payload() {
        let bytes = encode_payload(&[1, 2, 3]);
        assert_eq!(bytes.len(), 12);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/bitset/stochastic_compute.rs", code)]);
        assert!(
            !findings.is_empty(),
            "encode_payload unreached from production roots must be flagged"
        );
        assert!(
            findings.iter().any(|f| f
                .message
                .contains("unisolated host data-processing semantic twin `encode_payload`")),
            "finding message should name the unisolated semantic twin: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_renamed_oracle_in_wire_path() {
        let code = r#"
pub fn pack_oracle(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &w in words {
        out.extend_from_slice(&w.wrapping_mul(31).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire() {
        let _ = pack_oracle(&[1, 2]);
    }
}
"#;
        let findings = analyze_files(&[("vyre-primitives/src/wire/adversarial.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "wire-named unreached oracle must be flagged"
        );
        assert!(findings[0].message.contains("`pack_oracle`"));
        assert_eq!(findings[0].line, Some(2));
    }

    #[test]
    fn mutation_catches_renamed_oracle_with_parse_witness_report_name() {
        let code = r#"
pub fn parse_witness_report(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &w in words {
        out.extend_from_slice(&w.wrapping_add(1).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let _ = parse_witness_report(&[1, 2]);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/security/witness.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "parse/witness/report named unreached oracle must be flagged"
        );
        assert!(findings[0].message.contains("`parse_witness_report`"));
        assert_eq!(findings[0].line, Some(2));
    }

    #[test]
    fn mutation_catches_validator_like_result_bool_oracle() {
        let code = r#"
pub fn check_valid_witness(words: &[u32]) -> Result<bool, String> {
    let mut hash = 0u32;
    for &w in words {
        hash = hash.wrapping_mul(37).wrapping_add(w);
    }
    Ok(hash == 42)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check() {
        let _ = check_valid_witness(&[1, 2, 3]);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/validation/adversarial.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "Result<bool> computational oracle must be flagged"
        );
        assert!(findings[0].message.contains("`check_valid_witness`"));
        assert_eq!(findings[0].line, Some(2));
    }

    #[test]
    fn mutation_catches_table_like_name_oracle() {
        let code = r#"
pub fn generate_table_oracle(words: &[u32]) -> Vec<u32> {
    words.iter().map(|w| w.wrapping_add(1)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen() {
        let _ = generate_table_oracle(&[1, 2]);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/tables/adversarial.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "table-named uncalled oracle must be flagged"
        );
        assert!(findings[0].message.contains("`generate_table_oracle`"));
        assert_eq!(findings[0].line, Some(2));
    }

    #[test]
    fn mutation_catches_two_function_cycle_evasion() {
        let code = r#"
pub fn encode_step_a(words: &[u32]) -> Vec<u8> {
    encode_step_b(words)
}

pub fn encode_step_b(words: &[u32]) -> Vec<u8> {
    if words.is_empty() {
        return Vec::new();
    }
    encode_step_a(&words[1..])
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/cycle.rs", code)]);
        assert_eq!(
            findings.len(),
            2,
            "mutual recursion cycle without production roots must flag both functions"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("`encode_step_a`")));
        assert!(findings
            .iter()
            .any(|f| f.message.contains("`encode_step_b`")));
    }

    #[test]
    fn mutation_catches_ungated_tests_module() {
        let code = r#"
mod tests {
    pub fn host_oracle_helper(input: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &x in input {
            out.extend_from_slice(&x.wrapping_add(1).to_le_bytes());
        }
        out
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/ungated.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "ungated mod tests must be judged as production code and flag uncalled oracle"
        );
        assert!(findings[0].message.contains("`host_oracle_helper`"));
    }

    #[test]
    fn mutation_catches_oracle_under_cfg_any_test_and_cpu_parity() {
        let code = r#"
#[cfg(any(test, feature = "cpu-parity"))]
pub fn stochastic_decode(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/bitset/stochastic_compute.rs", code)]);
        assert!(
            !findings.is_empty(),
            "cfg(any(test, feature = ...)) is not test-only and must be judged as production code"
        );
        assert!(
            findings.iter().any(|f| f
                .message
                .contains("unisolated host data-processing semantic twin `stochastic_decode`")),
            "finding message should report unisolated semantic twin under cfg(any): {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_name_collision_uncalled_oracle() {
        let code_a = r#"
use vyre_foundation::ir::Program;

pub fn process_stream(input: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &x in input {
        out.extend_from_slice(&x.wrapping_add(1).to_le_bytes());
    }
    out
}

pub fn compile_domain_a() -> Program {
    let _ = process_stream(&[]);
    Program::new()
}
"#;
        let code_b = r#"
pub fn process_stream(input: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &x in input {
        out.extend_from_slice(&x.wrapping_mul(7).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b() {
        let _ = process_stream(&[1, 2]);
    }
}
"#;
        let findings = analyze_files(&[
            ("vyre-libs/src/domain_a/mod.rs", code_a),
            ("vyre-libs/src/domain_b/mod.rs", code_b),
        ]);
        assert_eq!(
            findings.len(),
            1,
            "name collision must not hide uncalled oracle in domain_b"
        );
        assert_eq!(
            findings[0].file,
            Some(PathBuf::from("vyre-libs/src/domain_b/mod.rs"))
        );
        assert_eq!(findings[0].line, Some(2));
    }

    #[test]
    fn mutation_distinguishes_same_named_methods_in_different_impl_blocks() {
        let code = r#"
use vyre_foundation::ir::Program;

pub struct ProductionPipeline;
pub struct UnusedOracleType;

impl ProductionPipeline {
    pub fn process(&self) -> Program {
        Program::new()
    }
}

impl UnusedOracleType {
    pub fn process(&self, words: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &w in words {
            out.extend_from_slice(&w.wrapping_mul(13).to_le_bytes());
        }
        out
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/impl_collision.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "UnusedOracleType::process must be flagged as unreached semantic twin"
        );
        assert!(findings[0].message.contains("`process`"));
        assert_eq!(findings[0].line, Some(14));
    }

    #[test]
    fn mutation_catches_unnamed_computed_const_referenced_by_expected_output() {
        let code = r#"
const ORACLE_SCALAR: u32 = 7 * 9;

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![vec![ORACLE_SCALAR as u8]]]
}
"#;
        let findings = analyze_files(&[("vyre-primitives/src/unnamed_const.rs", code)]);
        assert_eq!(
            findings.len(),
            1,
            "computed const without EXPECTED/OUTPUT name token must be flagged via path resolution"
        );
        assert!(findings[0].message.contains("`ORACLE_SCALAR`"));
        assert_eq!(findings[0].line, Some(2));
    }

    #[test]
    fn mutation_catches_static_facade_referenced_by_expected_output() {
        let code = r#"
static COMPUTED_DATA: [u32; 2] = [10 + 2, 20 * 3];

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![crate::wire::pack_u32_slice(&COMPUTED_DATA)]]
}
"#;
        let findings = analyze_files(&[("vyre-primitives/src/static_facade.rs", code)]);
        assert!(
            !findings.is_empty(),
            "computed static referenced by expected_output must be flagged"
        );
        assert!(findings.iter().any(
            |f| f.message.contains("`COMPUTED_DATA`") || f.message.contains("expected_output")
        ));
    }

    #[test]
    fn mutation_oracle_detection_catches_dynamic_expected_output_oracle_invocation() {
        let code = r#"
pub fn compute_twin_fixture(input: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &x in input {
        out.extend_from_slice(&x.wrapping_add(1).to_le_bytes());
    }
    out
}

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![compute_twin_fixture(&[1, 2, 3, 4])]]
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/op.rs", code)]);
        assert!(
            !findings.is_empty(),
            "expected finding for expected_output dynamic oracle invocation"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("compute_twin_fixture")
                    || f.message.contains("expected_output")),
            "finding message should report dynamic execution in expected_output: {findings:?}"
        );
    }

    #[test]
    fn production_compiler_analysis_with_callers_is_classified_as_reachable() {
        let code = r#"
use vyre_foundation::ir::Program;

pub fn analyze_cost_graph(nodes: &[u32]) -> u32 {
    let mut total = 0u32;
    for &n in nodes {
        total = total.wrapping_add(n);
    }
    total
}

pub fn compile_pipeline(nodes: &[u32]) -> Program {
    let _cost = analyze_cost_graph(nodes);
    Program::new()
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/analysis/cost_model.rs", code)]);
        assert!(
            findings.is_empty(),
            "compiler analysis reachable from IR builder root must not be flagged, got: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_zero_arg_oracle_behind_facade() {
        let code = r#"
pub fn zero_arg_oracle() -> u32 {
    let mut total = 0u32;
    for i in 0..10 {
        total = total.wrapping_add(i);
    }
    total
}

pub fn facade() -> u32 {
    zero_arg_oracle()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facade() {
        assert_eq!(facade(), 45);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/facade.rs", code)]);
        assert!(
            findings.len() >= 1,
            "zero-arg oracle behind facade must be flagged"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("`zero_arg_oracle`") || f.message.contains("`facade`")));
    }

    #[test]
    fn mutation_catches_owned_vec_input_oracle() {
        let code = r#"
pub fn process_owned_words(words: Vec<u32>) -> Vec<u8> {
    let mut out = Vec::new();
    for w in words {
        out.extend_from_slice(&w.wrapping_mul(17).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owned() {
        let _ = process_owned_words(vec![1, 2, 3]);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/owned.rs", code)]);
        assert!(
            !findings.is_empty(),
            "owned Vec input arithmetic candidate must be flagged"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("`process_owned_words`")));
    }

    #[test]
    fn mutation_catches_expected_output_target_also_called_by_builder() {
        let code = r#"
use vyre_foundation::ir::Program;

pub fn compute_twin_fixture(input: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &x in input {
        out.extend_from_slice(&x.wrapping_add(1).to_le_bytes());
    }
    out
}

pub fn compile_pipeline() -> Program {
    let _ = compute_twin_fixture(&[1, 2, 3]);
    Program::new()
}

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![compute_twin_fixture(&[1, 2, 3, 4])]]
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/op.rs", code)]);
        assert!(
            !findings.is_empty(),
            "dynamic expected_output call must be flagged even if target is called by a builder"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("compute_twin_fixture")
                    || f.message.contains("expected_output")),
            "finding message should report dynamic execution in expected_output: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_unreached_binary_search_or_partition_oracle() {
        let code = r#"
pub fn region_lookup(pos: u32, boundaries: &[u32]) -> usize {
    match boundaries.binary_search(&pos) {
        Ok(idx) => idx,
        Err(idx) => idx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup() {
        assert_eq!(region_lookup(5, &[0, 10, 20]), 1);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/pattern/lookup.rs", code)]);
        assert!(
            !findings.is_empty(),
            "unreached binary_search candidate must be flagged"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("`region_lookup`")));
    }
    #[test]
    fn mutation_permits_side_effect_telemetry_unit_function() {
        let code = r#"
pub fn record_fixpoint_telemetry(step: usize, active_nodes: usize) {
    if step > 0 {
        let _ = active_nodes + step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_telemetry() {
        record_fixpoint_telemetry(1, 10);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/telemetry.rs", code)]);
        assert!(
            findings.is_empty(),
            "side-effect telemetry returning () must be permitted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_unreached_semantic_popcount_reduction() {
        let code = r#"
pub fn checked_frontier_popcount(frontier: &[u64]) -> usize {
    let mut count = 0;
    for &word in frontier {
        count += word.count_ones() as usize;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_popcount() {
        assert_eq!(checked_frontier_popcount(&[0b111]), 3);
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/bitset/frontier.rs", code)]);
        assert!(
            !findings.is_empty(),
            "unreached host bitset popcount reduction must be convicted"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("`checked_frontier_popcount`")));
    }

    #[test]
    fn mutation_permits_pure_validator_returning_result_unit() {
        let code = r#"
pub fn check_signature_invariants(inputs: usize, outputs: usize) -> Result<(), String> {
    if inputs == 0 || outputs == 0 {
        return Err("invalid shape".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_validator() {
        assert!(check_signature_invariants(2, 2).is_ok());
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/validator.rs", code)]);
        assert!(
            findings.is_empty(),
            "pure validator returning Result<(), E> must be permitted: {findings:?}"
        );
    }

    #[test]
    fn mutation_permits_display_debug_error_impl_formatters() {
        let code = r#"
use std::fmt;

pub enum ErrorReason {
    InvalidInput(u32),
}

impl fmt::Display for ErrorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(code) => write!(f, "error: {code}"),
        }
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/display.rs", code)]);
        assert!(
            findings.is_empty(),
            "Display/Debug formatters must be permitted: {findings:?}"
        );
    }

    #[test]
    fn mutation_permits_wire_byte_codec_without_arithmetic() {
        let code = r#"
pub fn encode_wire_header(dst: &mut [u8], magic: u32, version: u16) -> Result<usize, String> {
    if dst.len() < 6 {
        return Err("buffer too short".to_string());
    }
    dst[0..4].copy_from_slice(&magic.to_le_bytes());
    dst[4..6].copy_from_slice(&version.to_le_bytes());
    Ok(6)
}

pub fn decode_wire_header(src: &[u8]) -> Result<(u32, u16), String> {
    if src.len() < 6 {
        return Err("buffer too short".to_string());
    }
    let magic = u32::from_le_bytes(src[0..4].try_into().map_err(|_| "slice error")?);
    let version = u16::from_le_bytes(src[4..6].try_into().map_err(|_| "slice error")?);
    Ok((magic, version))
}
"#;
        let findings = analyze_files(&[("vyre-primitives/src/wire.rs", code)]);
        assert!(
            findings.is_empty(),
            "wire byte codecs without arithmetic transforms must be permitted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_wire_codec_with_semantic_data_math() {
        let code = r#"
pub fn encode_transformed_payload(dst: &mut [u8], data: &[u32]) -> Result<usize, String> {
    let mut offset = 0;
    for &elem in data {
        let transformed = elem.wrapping_mul(31).wrapping_add(7);
        dst[offset..offset + 4].copy_from_slice(&transformed.to_le_bytes());
        offset += 4;
    }
    Ok(offset)
}
"#;
        let findings = analyze_files(&[("vyre-primitives/src/wire_math.rs", code)]);
        assert!(
            !findings.is_empty(),
            "codecs performing semantic data math must be convicted"
        );
        assert!(findings
            .iter()
            .any(|f| f.message.contains("`encode_transformed_payload`")));
    }

    #[test]
    fn mutation_catches_post_dispatch_float_arithmetic_derivation() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn total_set_bits_via(dispatcher: &dyn ProgramDispatcher, _input: &[u32]) -> Result<Vec<u8>, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    Ok(out[0].clone())
}

pub fn saturation_ratio_via(dispatcher: &dyn ProgramDispatcher, input: &[u32]) -> Result<f64, DispatchError> {
    if input.is_empty() {
        return Ok(0.0);
    }
    let capacity = (input.len() as u64) * 32;
    let set_bytes = total_set_bits_via(dispatcher, input)?;
    let set = u64::from(set_bytes[0]);
    Ok((set as f64) / (capacity as f64))
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/bitset_summary.rs", code)]);
        assert!(
            !findings.is_empty(),
            "post-dispatch float division metric must be convicted: {findings:?}"
        );
        assert!(
            findings.iter().any(|f| f.message.contains("post-dispatch host arithmetic / semantic derivation")),
            "post-dispatch float division must generate specific arithmetic derivation finding: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_imported_alias_transitive_post_dispatch_derivation() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher as CustomDispatcher};

pub fn leaf_dispatch(d: &dyn CustomDispatcher, _input: &[u32]) -> Result<Vec<u8>, DispatchError> {
    let prog = Program::default();
    let out = d.dispatch(&prog, &[vec![]], None)?;
    Ok(out[0].clone())
}

pub fn caller_with_arithmetic(d: &dyn CustomDispatcher, input: &[u32]) -> Result<u64, DispatchError> {
    let bytes = leaf_dispatch(d, input)?;
    let val = u64::from(bytes[0]);
    Ok(val * 42)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/custom_alias.rs", code)]);
        assert!(
            !findings.is_empty(),
            "transitive post-dispatch arithmetic with imported alias must be convicted: {findings:?}"
        );
        assert!(
            findings.iter().any(|f| f.message.contains("post-dispatch host arithmetic / semantic derivation")),
            "imported alias transitive caller must generate specific arithmetic derivation finding: {findings:?}"
        );
    }

    #[test]
    fn mutation_permits_post_dispatch_byte_unpacking_and_indexing() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn popcount_via(dispatcher: &dyn ProgramDispatcher, _input: &[u32]) -> Result<Vec<u32>, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    let raw_bytes = &out[0];
    unpack_words(raw_bytes)
}

fn unpack_words(raw_bytes: &[u8]) -> Result<Vec<u32>, DispatchError> {
    let mut words = Vec::with_capacity(raw_bytes.len() / 4);
    for i in 0..(raw_bytes.len() / 4) {
        let word = u32::from_le_bytes(raw_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        words.push(word);
    }
    Ok(words)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/popcount_clean.rs", code)]);
        assert!(
            findings.is_empty(),
            "post-dispatch byte slice indexing and unpacking must be permitted: {findings:?}"
        );
    }

    #[test]
    fn mutation_permits_ir_inspector_returning_bool_or_analysis_plan() {
        let code = r#"
use vyre_foundation::ir::Program;

pub fn is_bitset_equal_program(program: &Program) -> bool {
    program.is_valid()
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/bitset/equal.rs", code)]);
        assert!(
            findings.is_empty(),
            "IR inspector returning bool must be permitted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_generic_program_dispatcher_bound_host_reduction() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn pass_conflicts_via<D: ProgramDispatcher>(dispatcher: &D, _input: &[u32]) -> Result<bool, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    Ok(out[0].iter().any(|&b| b != 0))
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/generic_dispatch.rs", code)]);
        assert!(!findings.is_empty(), "generic ProgramDispatcher bounded function with post-dispatch reduction must be convicted");
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host reduction/aggregation `.any`")));
    }
    #[test]
    fn mutation_catches_post_dispatch_integer_addition_derivation() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn popcount_plus_one_via(dispatcher: &dyn ProgramDispatcher, _input: &[u32]) -> Result<u64, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    let decoded_u32 = out[0][0] as u64;
    Ok(decoded_u32 + 1)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/plus_one.rs", code)]);
        assert!(
            !findings.is_empty(),
            "post-dispatch integer addition derivation must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host arithmetic / semantic derivation")));
    }

    #[test]
    fn mutation_catches_post_dispatch_count_ones_reduction() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn popcount_count_ones_via(dispatcher: &dyn ProgramDispatcher, _input: &[u32]) -> Result<u32, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    let word = u32::from_le_bytes(out[0][0..4].try_into().unwrap());
    Ok(word.count_ones())
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/count_ones.rs", code)]);
        assert!(
            !findings.is_empty(),
            "post-dispatch count_ones reduction must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host reduction/aggregation `.count_ones`")));
    }

    #[test]
    fn mutation_catches_post_dispatch_integer_division_derivation() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn average_via(dispatcher: &dyn ProgramDispatcher, input: &[u32]) -> Result<u64, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    let total = u64::from(out[0][0]);
    let count = input.len() as u64;
    Ok(total / count)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/average.rs", code)]);
        assert!(
            !findings.is_empty(),
            "post-dispatch integer division derivation must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host arithmetic / semantic derivation")));
    }
    #[test]
    fn mutation_catches_metadata_only_dispatcher_call_not_establishing_execution() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn fake_oracle_capabilities_only(dispatcher: &dyn ProgramDispatcher, input: &[u32]) -> u32 {
    let _caps = dispatcher.capabilities();
    let mut sum = 0u32;
    for &x in input {
        sum = sum.wrapping_add(x);
    }
    sum
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/fake_oracle.rs", code)]);
        assert!(
            !findings.is_empty(),
            "metadata-only dispatcher caller must be convicted as unisolated host algorithm"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("unisolated host data-processing semantic twin")));
    }

    #[test]
    fn mutation_catches_unrelated_field_receiver_masquerading_as_dispatch() {
        let code = r#"
use vyre_foundation::program_dispatch::ProgramDispatcher;

struct LocalDevice;
impl LocalDevice {
    fn dispatch(&self, _left: u32, _right: u32) {}
}

struct LocalContext {
    device: LocalDevice,
}

pub fn fake_field_dispatch(
    _dispatcher: &dyn ProgramDispatcher,
    input: &[u32],
) -> u32 {
    let context = LocalContext { device: LocalDevice };
    context.device.dispatch(1, 2);
    input.iter().copied().sum()
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/fake_field_dispatch.rs", code)]);
        assert!(
            findings.iter().any(|finding| {
                finding
                    .message
                    .contains("unisolated host data-processing semantic twin")
            }),
            "a field receiver unrelated to the canonical dispatcher parameter must not establish a GPU execution root: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_non_dispatching_helper_not_establishing_execution() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

fn record_dispatcher(_dispatcher: &dyn ProgramDispatcher) {
    // telemetry/record only, does not dispatch
}

pub fn fake_oracle_with_record(dispatcher: &dyn ProgramDispatcher, input: &[u32]) -> u32 {
    record_dispatcher(dispatcher);
    let mut sum = 0u32;
    for &x in input {
        sum = sum.wrapping_add(x);
    }
    sum
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/fake_record.rs", code)]);
        assert!(
            !findings.is_empty(),
            "non-dispatching helper caller must be convicted as unisolated host algorithm"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("unisolated host data-processing semantic twin")));
    }

    #[test]
    fn mutation_permits_transitive_dispatch_helper_execution() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

fn helper_dispatch(dispatcher: &dyn ProgramDispatcher, _input: &[u32]) -> Result<Vec<Vec<u8>>, DispatchError> {
    let prog = Program::default();
    dispatcher.dispatch(&prog, &[vec![]], None)
}

pub fn wrapper_dispatch_via(dispatcher: &dyn ProgramDispatcher, input: &[u32]) -> Result<Vec<u8>, DispatchError> {
    let out = helper_dispatch(dispatcher, input)?;
    Ok(out[0].clone())
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/wrapper_dispatch.rs", code)]);
        assert!(findings.is_empty(), "transitive dispatch helper must be recognized as valid GPU dispatch root: {findings:?}");
    }
    #[test]
    fn mutation_catches_generic_masquerade_with_similar_ident() {
        let code = r#"
pub struct NotD;

pub fn fake_oracle_with_not_d<D: vyre_foundation::program_dispatch::ProgramDispatcher>(
    not_d: &NotD,
    input: &[u32],
) -> u32 {
    let mut sum = 0u32;
    for &x in input {
        sum = sum.wrapping_add(x);
    }
    sum
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/encoding/not_d.rs", code)]);
        assert!(
            !findings.is_empty(),
            "function taking NotD where D is bounded by ProgramDispatcher must be convicted as unisolated host algorithm"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("unisolated host data-processing semantic twin")));
    }
    #[test]
    fn mutation_permits_legitimate_transpose_input_staging() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn forward_backward_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    n: usize,
) -> Result<(Vec<u8>, Vec<u8>), DispatchError> {
    let prog1 = Program::default();
    let fwd = dispatcher.dispatch(&prog1, &[vec![]], None)?;
    let mut transpose = vec![0u32; n * n];
    for i in 0..n {
        for j in 0..n {
            transpose[j * n + i] = adj[i * n + j];
        }
    }
    let prog2 = Program::default();
    let bwd = dispatcher.dispatch(&prog2, &[transpose], None)?;
    Ok((fwd[0].clone(), bwd[0].clone()))
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/staging/transpose.rs", code)]);
        assert!(
            findings.is_empty(),
            "legitimate inter-dispatch input matrix transpose staging must be permitted with zero findings: {findings:?}"
        );
    }

    #[test]
    fn mutation_permits_gpu_result_transform_feeding_later_dispatch() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn chained_transform_dispatch_via(
    dispatcher: &dyn ProgramDispatcher,
    input: &[u32],
) -> Result<Vec<u8>, DispatchError> {
    let prog1 = Program::default();
    let out1 = dispatcher.dispatch(&prog1, &[vec![]], None)?;
    let mut staged = vec![0u32; input.len()];
    for i in 0..input.len() {
        staged[i] = input[i] ^ (out1[0][0] as u32);
    }
    let prog2 = Program::default();
    let out2 = dispatcher.dispatch(&prog2, &[staged], None)?;
    Ok(out2[0].clone())
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/staging/gpu_transform.rs", code)]);
        assert!(
            findings.is_empty(),
            "GPU-result intermediate transform feeding subsequent dispatch must be permitted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_unrelated_sum_between_dispatches_returned_afterward() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn evasion_unrelated_sum_via(
    dispatcher: &dyn ProgramDispatcher,
    _input: &[u32],
) -> Result<(u32, Vec<u8>), DispatchError> {
    let prog1 = Program::default();
    let out1 = dispatcher.dispatch(&prog1, &[vec![]], None)?;
    let host_sum = out1[0].iter().map(|&x| x as u32).sum::<u32>();
    let prog2 = Program::default();
    let out2 = dispatcher.dispatch(&prog2, &[vec![]], None)?;
    Ok((host_sum, out2[0].clone()))
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/evasion/unrelated_sum.rs", code)]);
        assert!(
            !findings.is_empty(),
            "unrelated sum between dispatches returned afterward must be convicted"
        );
        assert!(
            findings.iter().any(|f| f
                .message
                .contains("post-dispatch host reduction/aggregation `.sum`")),
            "must convict with reduction finding: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_unrelated_semantic_side_effect_between_dispatches() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn evasion_side_effect_via(
    dispatcher: &dyn ProgramDispatcher,
    acc: &mut Vec<u32>,
) -> Result<Vec<u8>, DispatchError> {
    let prog1 = Program::default();
    let out1 = dispatcher.dispatch(&prog1, &[vec![]], None)?;
    for &b in &out1[0] {
        acc.push((b as u32) * 2);
    }
    let prog2 = Program::default();
    let out2 = dispatcher.dispatch(&prog2, &[vec![]], None)?;
    Ok(out2[0].clone())
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/evasion/side_effect.rs", code)]);
        assert!(
            !findings.is_empty(),
            "unrelated semantic side effect between dispatches must be convicted"
        );
        assert!(
            findings.iter().any(
                |f| f.message.contains("post-dispatch host loop/accumulation")
                    || f.message
                        .contains("post-dispatch host arithmetic / semantic derivation")
            ),
            "must convict with loop/arithmetic finding: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_terminal_post_dispatch_math() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn evasion_terminal_math_via(
    dispatcher: &dyn ProgramDispatcher,
    _input: &[u32],
) -> Result<u64, DispatchError> {
    let prog1 = Program::default();
    let _out1 = dispatcher.dispatch(&prog1, &[vec![]], None)?;
    let prog2 = Program::default();
    let out2 = dispatcher.dispatch(&prog2, &[vec![]], None)?;
    let total = (out2[0][0] as u64) + 100;
    Ok(total)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/evasion/terminal_math.rs", code)]);
        assert!(
            !findings.is_empty(),
            "terminal post-dispatch math must be convicted"
        );
        assert!(
            findings.iter().any(|f| f
                .message
                .contains("post-dispatch host arithmetic / semantic derivation on GPU results")),
            "must convict with arithmetic finding: {findings:?}"
        );
    }
    #[test]
    fn mutation_permits_arbitrary_index_names_in_inter_dispatch_staging() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn staged_arbitrary_names_into(
    dispatcher: &dyn ProgramDispatcher,
    matrix: &[u32],
    dim_rows: usize,
    dim_cols: usize,
) -> Result<Vec<Vec<u8>>, DispatchError> {
    let mut staged_transpose = vec![0u8; dim_rows * dim_cols * 4];
    for row_arbitrary_alpha in 0..dim_rows {
        for col_arbitrary_beta in 0..dim_cols {
            let src_k = row_arbitrary_alpha * dim_cols + col_arbitrary_beta;
            let dst_m = col_arbitrary_beta * dim_rows + row_arbitrary_alpha;
            let val = matrix[src_k];
            staged_transpose[dst_m * 4] = (val & 0xFF) as u8;
        }
    }
    let prog = Program::default();
    dispatcher.dispatch(&prog, &[staged_transpose], None)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/clean_staging_names.rs", code)]);
        assert!(
            findings.is_empty(),
            "arbitrarily named index staging feeding dispatch must be permitted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_semantic_scalar_named_index_or_idx() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn evasion_semantic_scalar_named_idx(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<u64, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    let idx = (out[0][0] as u64) + 10;
    let index = idx * 2;
    Ok(index)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/evasion_named_idx.rs", code)]);
        assert!(
            !findings.is_empty(),
            "semantic derivation named index/idx must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host arithmetic / semantic derivation")));
    }

    #[test]
    fn mutation_catches_post_dispatch_decoded_value_comparison() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn evasion_post_dispatch_comparison(
    dispatcher: &dyn ProgramDispatcher,
    threshold: u8,
) -> Result<bool, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    let decoded_byte = out[0][0];
    if decoded_byte > threshold {
        Ok(true)
    } else {
        Ok(false)
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/evasion_comparison.rs", code)]);
        assert!(
            !findings.is_empty(),
            "post-dispatch decoded value comparison must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host arithmetic / semantic derivation")));
    }

    #[test]
    fn mutation_catches_post_dispatch_reconstruction_loop_with_scalar_math() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn evasion_reconstruction_with_math_into(
    dispatcher: &dyn ProgramDispatcher,
    outputs: &mut Vec<Vec<u8>>,
) -> Result<(), DispatchError> {
    let readbacks = dispatcher.dispatch(&vec![], &[vec![]], None)?;
    for (output, readback) in outputs.iter_mut().zip(&readbacks) {
        output.clear();
        output.extend_from_slice(readback);
        output.push(readback[0] * 2);
    }
    Ok(())
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/evasion_copy_math.rs", code)]);
        assert!(
            !findings.is_empty(),
            "reconstruction loop with scalar arithmetic must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host loop/accumulation")
            || f.message.contains("post-dispatch host arithmetic")));
    }

    #[test]
    fn mutation_catches_post_dispatch_decoder_loop_with_accumulation() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

fn decode_u32_output_exact(
    _readback: &[u8],
    _expected_words: usize,
    _context: &str,
    _out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    Ok(())
}

pub fn evasion_decode_batch_into(
    dispatcher: &dyn ProgramDispatcher,
    mut outs: Vec<(usize, &'static str, &mut Vec<u32>)>,
) -> Result<u32, DispatchError> {
    let readbacks = dispatcher.dispatch(&vec![], &[vec![]], None)?;
    let mut total_words = 0u32;
    for (index, (expected_words, context, out)) in outs.into_iter().enumerate() {
        decode_u32_output_exact(&readbacks[index], expected_words, context, out)?;
        total_words += expected_words as u32;
    }
    Ok(total_words)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/evasion_decoder_accum.rs", code)]);
        assert!(
            !findings.is_empty(),
            "decoder loop with host accumulation must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host loop/accumulation")
            || f.message.contains("post-dispatch host arithmetic")));
    }

    #[test]
    fn mutation_catches_post_dispatch_output_base_arithmetic_derivation() {
        let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn evasion_output_base_scalar_math_via(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<u64, DispatchError> {
    let outputs = dispatcher.dispatch(&vec![], &[vec![]], None)?;
    let output_base = outputs[0][0] as u64;
    let computed = output_base + 42;
    Ok(computed)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/evasion_output_base_math.rs", code)]);
        assert!(
            !findings.is_empty(),
            "arithmetic on output_base derived from output byte must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("post-dispatch host arithmetic / semantic derivation")));
    }
    #[test]
    fn mutation_permits_turbofish_generic_calls_and_cfg_alternative_definitions() {
        let wire_code = r#"
#[cfg(target_endian = "little")]
pub fn fill_custom_words_into<T: Copy>(src: &[u8], count: usize, out: &mut Vec<T>) {
    let _ = (src, count, out);
}

#[cfg(target_endian = "big")]
pub fn fill_custom_words_into<T: Copy>(src: &[u8], count: usize, out: &mut Vec<T>) {
    let _ = (src, count, out);
}

pub fn unpack_custom_u32_slice_into(src: &[u8], count: usize, out: &mut Vec<u32>) {
    fill_custom_words_into::<u32>(src, count, out);
}
"#;
        let caller_code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn resident_caller_using_unpack(
    dispatcher: &dyn ProgramDispatcher,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let readbacks = dispatcher.dispatch(&vec![], &[vec![]], None)?;
    crate::wire::unpack_custom_u32_slice_into(&readbacks[0], 10, out);
    Ok(())
}
"#;
        let findings = analyze_files(&[
            ("vyre-primitives/src/wire.rs", wire_code),
            ("vyre-libs/src/caller.rs", caller_code),
        ]);
        assert!(findings.is_empty(), "turbofish generic caller reaching CFG-alternative definitions must be clean: {findings:?}");
    }

    #[test]
    fn mutation_catches_unreachable_generic_cfg_alternative_definition() {
        let wire_code = r#"
#[cfg(target_endian = "little")]
pub fn uncalled_twin_words_into<T: Copy>(src: &[u8], count: usize, out: &mut Vec<T>) {
    let mut sum = 0usize;
    for &b in src {
        sum += b as usize;
    }
    let _ = (sum, count, out);
}

#[cfg(target_endian = "big")]
pub fn uncalled_twin_words_into<T: Copy>(src: &[u8], count: usize, out: &mut Vec<T>) {
    let mut sum = 0usize;
    for &b in src {
        sum += b as usize;
    }
    let _ = (sum, count, out);
}
"#;
        let findings = analyze_files(&[("vyre-primitives/src/wire.rs", wire_code)]);
        assert!(
            !findings.is_empty(),
            "uncalled CFG alternative twin must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("unisolated host data-processing semantic twin")));
    }

    #[test]
    fn mutation_permits_operation_metadata_iterator_with_arbitrary_name() {
        let code = r#"
use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};

pub fn arbitrary_catalog_query_into() -> impl Iterator<Item = SemanticOperation> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Library)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/catalog.rs", code)]);
        assert!(
            findings.is_empty(),
            "operation metadata iterator must be permitted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_adversarial_numeric_iterator_twin_with_same_shape() {
        let code = r#"
pub fn arbitrary_catalog_query_into(data: &[u32]) -> impl Iterator<Item = u32> + '_ {
    data.iter().map(|&x| x + 42)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/catalog_adversarial.rs", code)]);
        assert!(
            !findings.is_empty(),
            "unisolated numeric iterator twin must be convicted"
        );
        assert!(findings.iter().any(|f| f
            .message
            .contains("unisolated host data-processing semantic twin")));
    }
    #[test]
    fn mutation_catches_local_type_masquerading_as_operation_metadata() {
        let code = r#"
pub struct SemanticOperation(u32);

pub fn arbitrary_catalog_query_into(
    data: &[u32],
) -> impl Iterator<Item = SemanticOperation> + '_ {
    data.iter().map(|&value| SemanticOperation(value + 42))
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/catalog_masquerade.rs", code)]);
        assert!(
            findings.iter().any(|finding| finding
                .message
                .contains("unisolated host data-processing semantic twin")),
            "a local same-named type must not receive canonical metadata treatment: {findings:?}"
        );
    }

    #[test]
    fn mutation_permits_genuine_resident_staging_consumed_by_canonical_dispatch() {
        let staging_code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct ResidentDemoGraph {
    pub handles: [u64; 2],
}

pub fn upload_resident_demo_graph(
    dispatcher: &impl ProgramDispatcher,
    node_count: u32,
    edges: &[u32],
) -> Result<ResidentDemoGraph, DispatchError> {
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {
        let encoded = (edge ^ 0x5A5A_5A5A).wrapping_mul(node_count | 1);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }
    let h0 = dispatcher.alloc_resident(packed.len())?;
    dispatcher.upload_resident(h0, &packed)?;
    let h1 = dispatcher.alloc_resident(16)?;
    Ok(ResidentDemoGraph { handles: [h0, h1] })
}
"#;
        let dispatch_code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher, ResidentDispatchStep};
use crate::staging::{upload_resident_demo_graph, ResidentDemoGraph};

pub fn execute_demo_traversal(
    dispatcher: &impl ProgramDispatcher,
    graph: &ResidentDemoGraph,
) -> Result<(), DispatchError> {
    let prog = Program::default();
    let step = ResidentDispatchStep {
        program: &prog,
        handle_ids: &graph.handles,
        grid_override: Some([1, 1, 1]),
    };
    dispatcher.dispatch_resident_sequence(&[step])
}

pub fn run_demo_traversal(
    dispatcher: &impl ProgramDispatcher,
    node_count: u32,
    edges: &[u32],
) -> Result<(), DispatchError> {
    let graph = upload_resident_demo_graph(dispatcher, node_count, edges)?;
    execute_demo_traversal(dispatcher, &graph)
}
"#;
        let findings = analyze_files(&[
            ("vyre-libs/src/staging.rs", staging_code),
            ("vyre-libs/src/dispatch.rs", dispatch_code),
        ]);
        assert!(
            findings.is_empty(),
            "resident staging consumed by genuine canonical dispatch must not be convicted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_same_basename_in_different_modules_rejected() {
        let staging_a = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct ResidentDemoGraph {
    pub handles: [u64; 2],
}

pub fn upload_resident_demo_graph(
    dispatcher: &impl ProgramDispatcher,
    node_count: u32,
    edges: &[u32],
) -> Result<ResidentDemoGraph, DispatchError> {
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {
        let encoded = (edge ^ 0x5A5A_5A5A).wrapping_mul(node_count | 1);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }
    let h0 = dispatcher.alloc_resident(packed.len())?;
    dispatcher.upload_resident(h0, &packed)?;
    let h1 = dispatcher.alloc_resident(16)?;
    Ok(ResidentDemoGraph { handles: [h0, h1] })
}
"#;
        let dispatch_b = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher, ResidentDispatchStep};

pub struct ResidentDemoGraph {
    pub handles: [u64; 2],
}

pub fn execute_demo_traversal(
    dispatcher: &impl ProgramDispatcher,
    graph: &ResidentDemoGraph,
) -> Result<(), DispatchError> {
    let prog = Program::default();
    let step = ResidentDispatchStep {
        program: &prog,
        handle_ids: &graph.handles,
        grid_override: Some([1, 1, 1]),
    };
    dispatcher.dispatch_resident_sequence(&[step])
}

pub fn run_b(
    dispatcher: &impl ProgramDispatcher,
    graph: &ResidentDemoGraph,
) -> Result<(), DispatchError> {
    execute_demo_traversal(dispatcher, graph)
}
"#;
        let findings = analyze_files(&[
            ("vyre-libs/src/staging_a.rs", staging_a),
            ("vyre-libs/src/dispatch_b.rs", dispatch_b),
        ]);
        assert!(
            !findings.is_empty(),
            "same basename in different modules must be rejected and convicted"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("upload_resident_demo_graph")),
            "upload_resident_demo_graph in staging_a must be convicted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_unused_alternative_producer_rejected() {
        let staging_code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct ResidentDemoGraph {
    pub handles: [u64; 2],
}

pub fn upload_resident_demo_graph(
    dispatcher: &impl ProgramDispatcher,
    node_count: u32,
    edges: &[u32],
) -> Result<ResidentDemoGraph, DispatchError> {
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {
        let encoded = (edge ^ 0x5A5A_5A5A).wrapping_mul(node_count | 1);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }
    let h0 = dispatcher.alloc_resident(packed.len())?;
    dispatcher.upload_resident(h0, &packed)?;
    let h1 = dispatcher.alloc_resident(16)?;
    Ok(ResidentDemoGraph { handles: [h0, h1] })
}

pub fn upload_unused_alt_graph(
    dispatcher: &impl ProgramDispatcher,
    node_count: u32,
    edges: &[u32],
) -> Result<ResidentDemoGraph, DispatchError> {
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {
        let encoded = (edge ^ 0x1234_5678).wrapping_mul(node_count | 1);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }
    let h0 = dispatcher.alloc_resident(packed.len())?;
    dispatcher.upload_resident(h0, &packed)?;
    let h1 = dispatcher.alloc_resident(16)?;
    Ok(ResidentDemoGraph { handles: [h0, h1] })
}
"#;
        let dispatch_code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher, ResidentDispatchStep};
use crate::staging::{upload_resident_demo_graph, ResidentDemoGraph};

pub fn execute_demo_traversal(
    dispatcher: &impl ProgramDispatcher,
    graph: &ResidentDemoGraph,
) -> Result<(), DispatchError> {
    let prog = Program::default();
    let step = ResidentDispatchStep {
        program: &prog,
        handle_ids: &graph.handles,
        grid_override: Some([1, 1, 1]),
    };
    dispatcher.dispatch_resident_sequence(&[step])
}

pub fn run_demo_traversal(
    dispatcher: &impl ProgramDispatcher,
    node_count: u32,
    edges: &[u32],
) -> Result<(), DispatchError> {
    let graph = upload_resident_demo_graph(dispatcher, node_count, edges)?;
    execute_demo_traversal(dispatcher, &graph)
}
"#;
        let findings = analyze_files(&[
            ("vyre-libs/src/staging.rs", staging_code),
            ("vyre-libs/src/dispatch.rs", dispatch_code),
        ]);
        assert!(
            !findings.is_empty(),
            "unused alternative producer returning same nominal type must be convicted"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("upload_unused_alt_graph")),
            "upload_unused_alt_graph must be convicted: {findings:?}"
        );
        assert!(
            !findings
                .iter()
                .any(|f| f.message.contains("upload_resident_demo_graph")),
            "used producer upload_resident_demo_graph must NOT be convicted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_unrelated_upload_only_host_math_not_consumed_by_dispatch() {
        let upload_only_code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct UnusedResidentGraph {
    pub handle: u64,
}

pub fn upload_unused_graph_with_math(
    dispatcher: &impl ProgramDispatcher,
    edges: &[u32],
) -> Result<UnusedResidentGraph, DispatchError> {
    let mut sum = 0u32;
    for &e in edges {
        sum = sum.wrapping_add(e ^ 0x1234_5678);
    }
    let handle = dispatcher.alloc_resident(sum as usize)?;
    Ok(UnusedResidentGraph { handle })
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/upload_only.rs", upload_only_code)]);
        assert!(
            !findings.is_empty(),
            "upload-only host math with no downstream dispatch root must be convicted"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("upload_unused_graph_with_math")),
            "upload_unused_graph_with_math must be flagged: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_resident_staging_values_not_uploaded_before_dispatch() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher, ResidentDispatchStep};

struct ResidentGraph {
    handles: [u64; 1],
}

fn prepare_graph_without_upload(
    dispatcher: &impl ProgramDispatcher,
    edges: &[u32],
) -> Result<ResidentGraph, DispatchError> {
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {
        let encoded = edge.wrapping_mul(3);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }
    let handle = dispatcher.alloc_resident(packed.len())?;
    Ok(ResidentGraph { handles: [handle] })
}

fn dispatch_graph(
    dispatcher: &impl ProgramDispatcher,
    graph: &ResidentGraph,
) -> Result<(), DispatchError> {
    let program = Program::default();
    let step = ResidentDispatchStep {
        program: &program,
        handle_ids: &graph.handles,
        grid_override: None,
    };
    dispatcher.dispatch_resident_sequence(&[step])
}

pub fn run_graph(
    dispatcher: &impl ProgramDispatcher,
    edges: &[u32],
) -> Result<(), DispatchError> {
    let graph = prepare_graph_without_upload(dispatcher, edges)?;
    dispatch_graph(dispatcher, &graph)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/unconsumed_staging.rs", code)]);
        assert!(
            findings
                .iter()
                .any(|finding| finding.message.contains("prepare_graph_without_upload")),
            "resident semantic staging that never feeds a canonical upload must be convicted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_staging_consumed_only_by_fake_local_dispatcher_masquerade() {
        let fake_code = r#"
pub struct FakeResidentGraph {
    pub handle: u64,
}

pub trait ProgramDispatcher {
    fn alloc_resident(&self, bytes: usize) -> Result<u64, String>;
    fn dispatch(&self, prog: u32, handles: &[u64]) -> Result<(), String>;
}

pub fn upload_fake_graph(
    dispatcher: &impl ProgramDispatcher,
    edges: &[u32],
) -> Result<FakeResidentGraph, String> {
    let mut sum = 0u32;
    for &e in edges {
        sum = sum.wrapping_add(e ^ 0xA5A5);
    }
    let handle = dispatcher.alloc_resident(sum as usize)?;
    Ok(FakeResidentGraph { handle })
}

pub fn fake_dispatch(
    dispatcher: &impl ProgramDispatcher,
    graph: &FakeResidentGraph,
) -> Result<(), String> {
    dispatcher.dispatch(1, &[graph.handle])
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/fake_staging.rs", fake_code)]);
        assert!(
            !findings.is_empty(),
            "staging with fake local dispatcher masquerade must be convicted"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("upload_fake_graph")),
            "upload_fake_graph must be flagged: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_parse_ambiguity_fails_closed() {
        let code = r#"
mod fake_ambiguous {
    pub struct AmbiguousGraph;
}

pub fn upload_ambiguous_graph(
    edges: &[u32],
) -> Result<fake_ambiguous::AmbiguousGraph, String> {
    let mut sum = 0u32;
    for &e in edges {
        sum = sum.wrapping_add(e ^ 0x3333);
    }
    let _ = sum;
    Ok(fake_ambiguous::AmbiguousGraph)
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/ambiguous.rs", code)]);
        assert!(
            !findings.is_empty(),
            "parse ambiguity or unresolvable type path must fail closed and convict host math"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("upload_ambiguous_graph")),
            "upload_ambiguous_graph must be flagged: {findings:?}"
        );
    }

    #[test]
    fn mutation_permits_genuine_resident_staging_separate_apis_unique_producer() {
        let staging_code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct ResidentDemoGraph {
    pub(crate) handles: [u64; 2],
}

pub fn upload_resident_demo_graph(
    dispatcher: &impl ProgramDispatcher,
    node_count: u32,
    edges: &[u32],
) -> Result<ResidentDemoGraph, DispatchError> {
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {
        let encoded = (edge ^ 0x5A5A_5A5A).wrapping_mul(node_count | 1);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }
    let h0 = dispatcher.alloc_resident(packed.len())?;
    dispatcher.upload_resident(h0, &packed)?;
    let h1 = dispatcher.alloc_resident(16)?;
    Ok(ResidentDemoGraph { handles: [h0, h1] })
}
"#;
        let dispatch_code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher, ResidentDispatchStep};
use crate::staging::ResidentDemoGraph;

pub fn execute_demo_traversal(
    dispatcher: &impl ProgramDispatcher,
    graph: &ResidentDemoGraph,
) -> Result<(), DispatchError> {
    let prog = Program::default();
    let step = ResidentDispatchStep {
        program: &prog,
        handle_ids: &graph.handles,
        grid_override: Some([1, 1, 1]),
    };
    dispatcher.dispatch_resident_sequence(&[step])
}
"#;
        let findings = analyze_files(&[
            ("vyre-libs/src/staging.rs", staging_code),
            ("vyre-libs/src/dispatch.rs", dispatch_code),
        ]);
        assert!(
            findings.is_empty(),
            "genuine staging with separate upload and dispatch APIs and unique producer must not be convicted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_staging_type_with_pub_fields_without_call_path() {
        let staging_code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct ResidentDemoGraphWithPubFields {
    pub handles: [u64; 2],
}

pub fn upload_resident_demo_graph(
    dispatcher: &impl ProgramDispatcher,
    node_count: u32,
    edges: &[u32],
) -> Result<ResidentDemoGraphWithPubFields, DispatchError> {
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {
        let encoded = (edge ^ 0x5A5A_5A5A).wrapping_mul(node_count | 1);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }
    let h0 = dispatcher.alloc_resident(packed.len())?;
    dispatcher.upload_resident(h0, &packed)?;
    let h1 = dispatcher.alloc_resident(16)?;
    Ok(ResidentDemoGraphWithPubFields { handles: [h0, h1] })
}
"#;
        let dispatch_code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher, ResidentDispatchStep};
use crate::staging::ResidentDemoGraphWithPubFields;

pub fn execute_demo_traversal(
    dispatcher: &impl ProgramDispatcher,
    graph: &ResidentDemoGraphWithPubFields,
) -> Result<(), DispatchError> {
    let prog = Program::default();
    let step = ResidentDispatchStep {
        program: &prog,
        handle_ids: &graph.handles,
        grid_override: Some([1, 1, 1]),
    };
    dispatcher.dispatch_resident_sequence(&[step])
}
"#;
        let findings = analyze_files(&[
            ("vyre-libs/src/staging.rs", staging_code),
            ("vyre-libs/src/dispatch.rs", dispatch_code),
        ]);
        assert!(
            !findings.is_empty(),
            "staging type with pub fields without call path must not gain nominal rooting and must be convicted"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("upload_resident_demo_graph")),
            "upload_resident_demo_graph must be flagged: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_staging_with_ignored_metadata_parameter_rejected() {
        let staging_code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct ResidentDemoGraph {
    pub(crate) handles: [u64; 2],
}

pub struct UnusedConfig {
    pub(crate) threshold: u32,
}

pub fn upload_resident_demo_graph(
    dispatcher: &impl ProgramDispatcher,
    node_count: u32,
    edges: &[u32],
) -> Result<ResidentDemoGraph, DispatchError> {
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {
        let encoded = (edge ^ 0x5A5A_5A5A).wrapping_mul(node_count | 1);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }
    let h0 = dispatcher.alloc_resident(packed.len())?;
    dispatcher.upload_resident(h0, &packed)?;
    let h1 = dispatcher.alloc_resident(16)?;
    Ok(ResidentDemoGraph { handles: [h0, h1] })
}

pub fn upload_unused_config_with_math(
    edges: &[u32],
) -> Result<UnusedConfig, DispatchError> {
    let mut sum = 0u32;
    for &e in edges {
        sum = sum.wrapping_add(e ^ 0x7777);
    }
    Ok(UnusedConfig { threshold: sum })
}
"#;
        let dispatch_code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher, ResidentDispatchStep};
use crate::staging::{ResidentDemoGraph, UnusedConfig};

pub fn execute_demo_traversal_with_config(
    dispatcher: &impl ProgramDispatcher,
    graph: &ResidentDemoGraph,
    config: &UnusedConfig,
) -> Result<(), DispatchError> {
    let _ = config;
    let prog = Program::default();
    let step = ResidentDispatchStep {
        program: &prog,
        handle_ids: &graph.handles,
        grid_override: Some([1, 1, 1]),
    };
    dispatcher.dispatch_resident_sequence(&[step])
}
"#;
        let findings = analyze_files(&[
            ("vyre-libs/src/staging.rs", staging_code),
            ("vyre-libs/src/dispatch.rs", dispatch_code),
        ]);
        assert!(
            !findings.is_empty(),
            "unused config parameter with host-math producer must be convicted"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("upload_unused_config_with_math")),
            "upload_unused_config_with_math must be flagged: {findings:?}"
        );
        assert!(
            !findings
                .iter()
                .any(|f| f.message.contains("upload_resident_demo_graph")),
            "upload_resident_demo_graph feeding handle_ids must NOT be flagged: {findings:?}"
        );
    }

    #[test]
    fn mutation_permits_genuine_resident_staging_transitive_helper_dispatch() {
        let staging_code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct ResidentDemoGraph {
    pub(crate) handles: [u64; 2],
}

pub fn upload_resident_demo_graph(
    dispatcher: &impl ProgramDispatcher,
    node_count: u32,
    edges: &[u32],
) -> Result<ResidentDemoGraph, DispatchError> {
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {
        let encoded = (edge ^ 0x5A5A_5A5A).wrapping_mul(node_count | 1);
        packed.extend_from_slice(&encoded.to_le_bytes());
    }
    let h0 = dispatcher.alloc_resident(packed.len())?;
    dispatcher.upload_resident(h0, &packed)?;
    let h1 = dispatcher.alloc_resident(16)?;
    Ok(ResidentDemoGraph { handles: [h0, h1] })
}
"#;
        let dispatch_code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher, ResidentDispatchStep};
use crate::staging::ResidentDemoGraph;

fn helper_dispatch(
    dispatcher: &impl ProgramDispatcher,
    handles: &[u64; 2],
) -> Result<(), DispatchError> {
    let prog = Program::default();
    let step = ResidentDispatchStep {
        program: &prog,
        handle_ids: handles,
        grid_override: Some([1, 1, 1]),
    };
    dispatcher.dispatch_resident_sequence(&[step])
}

pub fn execute_demo_traversal(
    dispatcher: &impl ProgramDispatcher,
    graph: &ResidentDemoGraph,
) -> Result<(), DispatchError> {
    helper_dispatch(dispatcher, &graph.handles)
}
"#;
        let findings = analyze_files(&[
            ("vyre-libs/src/staging.rs", staging_code),
            ("vyre-libs/src/dispatch.rs", dispatch_code),
        ]);
        assert!(
            findings.is_empty(),
            "genuine staging with transitive helper dispatch flow must not be convicted: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_pre_dispatch_host_math_helper_in_gpu_dispatch_fn() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn pre_calc_weights(input: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for &x in input {
        out.push(x * 2.5 + 1.0);
    }
    out
}

pub fn execute_with_pre_calc(
    dispatcher: &impl ProgramDispatcher,
    input: &[f32],
) -> Result<(), DispatchError> {
    let weights = pre_calc_weights(input);
    let prog = Program::default();
    dispatcher.dispatch(&prog, &[&weights], &mut [])
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/pre_calc.rs", code)]);
        assert!(
            !findings.is_empty(),
            "GPU dispatch function invoking pre-dispatch host math helper must be convicted: {findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("pre_calc_weights")
                    || f.message.contains("execute_with_pre_calc")),
            "expected pre-calc helper conviction, got: {findings:?}"
        );
    }

    #[test]
    fn mutation_recognizes_dispatcher_in_wrapper_struct_and_catches_post_dispatch_math() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct DispatchContext<'a, D: ProgramDispatcher> {
    pub dispatcher: &'a D,
}

impl<'a, D: ProgramDispatcher> DispatchContext<'a, D> {
    pub fn run_pipeline(&self, prog: &Program, out: &mut [u8]) -> Result<u32, DispatchError> {
        self.dispatcher.dispatch(prog, &[], out)?;
        let mut sum = 0u32;
        for &b in out.iter() {
            sum = sum.wrapping_add(b as u32);
        }
        Ok(sum)
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/wrapper.rs", code)]);
        assert!(
            !findings.is_empty(),
            "dispatcher in wrapper struct with post-dispatch host reduction must be convicted: {findings:?}"
        );
        assert!(
            findings.iter().any(|f| f.message.contains("run_pipeline") || f.message.contains("wrapping_add")),
            "expected post-dispatch reduction conviction in wrapper struct method, got: {findings:?}"
        );
    }

    #[test]
    fn mutation_catches_struct_literal_operation_registration_dynamic_expected_output() {
        let code = r#"
use vyre_foundation::operation::OperationRegistration;

pub fn dynamic_oracle_fixture(input: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &x in input {
        out.extend_from_slice(&x.wrapping_mul(3).to_le_bytes());
    }
    out
}

pub fn register_op() -> OperationRegistration {
    OperationRegistration {
        id: 42,
        expected_output: vec![dynamic_oracle_fixture(&[1, 2, 3])],
    }
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/op_struct.rs", code)]);
        assert!(
            !findings.is_empty(),
            "OperationRegistration struct literal with dynamic expected_output must be convicted: {findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("dynamic_oracle_fixture")
                    || f.message.contains("expected_output")),
            "expected struct literal dynamic expected_output conviction, got: {findings:?}"
        );
    }

    #[test]
    fn mutation_permits_post_dispatch_non_data_diagnostic_telemetry_methods() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct DispatchMetrics {
    pub total_ops: u32,
}

pub fn execute_with_metrics(
    dispatcher: &impl ProgramDispatcher,
    prog: &Program,
    metrics: &mut DispatchMetrics,
) -> Result<(), DispatchError> {
    dispatcher.dispatch(prog, &[], &mut [])?;
    metrics.total_ops = metrics.total_ops.wrapping_add(1);
    Ok(())
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/telemetry_dispatch.rs", code)]);
        assert!(
            findings.is_empty(),
            "non-data diagnostic telemetry in post-dispatch phase must be permitted: {findings:?}"
        );
    }

    #[test]
    fn mutation_permits_inter_dispatch_staging_buffer_operations() {
        let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn execute_two_stage_pipeline(
    dispatcher: &impl ProgramDispatcher,
    stage1_prog: &Program,
    stage2_prog: &Program,
    intermediate_scratch: &mut Vec<u8>,
) -> Result<(), DispatchError> {
    dispatcher.dispatch(stage1_prog, &[], intermediate_scratch.as_mut_slice())?;
    intermediate_scratch.clear();
    intermediate_scratch.resize(64, 0);
    dispatcher.dispatch(stage2_prog, &[], intermediate_scratch.as_mut_slice())
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/multi_stage.rs", code)]);
        assert!(
            findings.is_empty(),
            "intermediate buffer operations between sequential dispatches must be permitted: {findings:?}"
        );
    }
}
