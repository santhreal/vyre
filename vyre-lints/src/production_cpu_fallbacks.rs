//! Production CPU fallback guard.
//!
//! CPU/reference execution is valid only as an explicit oracle in tests,
//! conformance harnesses, or reference backend crates. It is forbidden in
//! production GPU dispatch paths because a hidden reference path can turn a
//! GPU regression into a green release.

use crate::{scan, Violation, ViolationKind};
use anyhow::{Context, Result};
use proc_macro2::{Span, TokenTree};
use std::path::Path;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Attribute, Expr, ItemFn, ItemImpl, ItemMacro, ItemMod, Lit, Meta, Token, UseTree};

const FORBIDDEN_SYMBOLS: &[&str] = &[
    "reference_c11_build_vast_nodes",
    "reference_c11_annotate_typedef_names",
    "reference_c11_classify_vast_node_kinds",
    "reference_ast_to_pg_nodes",
    "reference_c_keyword_types",
    "run_cpu_fixpoint_to_convergence",
    "cpu_vs_backend",
    "reference_semiring_gemm",
    "reference_sinkhorn_clustering",
    "reference_scc_components_via_substrate",
    "cpu_ref",
    "cpu_op",
    "cpu_references",
];

const APPROVED_PARITY_PATHS: &[&str] = &[
    "/tests/",
    "/benches/",
    "/examples/",
    "/fixtures/",
    "/vyre-reference/",
    "/vyre-driver-reference/",
    "/conform/",
    "/vyre-conform/",
];

/// Scan a source tree for production calls into CPU reference execution.
pub fn scan_tree(root: &Path) -> Result<Vec<Violation>> {
    scan::collect_violations(
        root,
        scan::RUST_SOURCE,
        |workspace_rel| {
            !is_approved_parity_path(workspace_rel) && !is_approved_parity_file(workspace_rel)
        },
        scan_file,
    )
}

fn is_approved_parity_path(workspace_rel: &str) -> bool {
    let wrapped = format!("/{workspace_rel}");
    APPROVED_PARITY_PATHS
        .iter()
        .any(|approved| wrapped.contains(approved))
}

fn is_approved_parity_file(workspace_rel: &str) -> bool {
    let file_name = workspace_rel.rsplit('/').next().unwrap_or(workspace_rel);
    file_name == "tests.rs"
        || file_name == "test.rs"
        || file_name == "reference.rs"
        || file_name == "oracle.rs"
        || file_name.ends_with("cpu_oracle.rs")
        || file_name == "cpu_fallback_reachability.rs"
        || file_name == "witness.rs"
        || file_name.starts_with("ref_")
        || file_name.ends_with("_tests.rs")
}

fn scan_file(path: &Path, workspace_rel: &str) -> Result<Vec<Violation>> {
    let source = crate::read_source_bounded(path)?;
    let file = syn::parse_file(&source).with_context(|| format!("parse {}", path.display()))?;
    if attrs_require_parity(&file.attrs) {
        return Ok(Vec::new());
    }

    let mut visitor = CpuFallbackVisitor {
        file: workspace_rel,
        approved_depth: 0,
        violations: Vec::new(),
    };
    visitor.visit_file(&file);
    Ok(visitor.violations)
}

fn attrs_require_parity(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if attr.path().is_ident("test") {
            return true;
        }
        if !attr.path().is_ident("cfg") {
            return false;
        }
        attr.parse_args::<Meta>()
            .is_ok_and(|meta| cfg_requires_parity(&meta))
    })
}

fn cfg_requires_parity(meta: &Meta) -> bool {
    match meta {
        Meta::Path(path) => path.is_ident("test"),
        Meta::NameValue(name_value) if name_value.path.is_ident("feature") => {
            matches!(&name_value.value, Expr::Lit(lit) if matches!(&lit.lit, Lit::Str(value) if value.value() == "cpu-parity"))
        }
        Meta::NameValue(_) => false,
        Meta::List(list) if list.path.is_ident("all") || list.path.is_ident("any") => {
            let Ok(items) = list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            else {
                return false;
            };
            if list.path.is_ident("all") {
                items.iter().any(cfg_requires_parity)
            } else {
                !items.is_empty() && items.iter().all(cfg_requires_parity)
            }
        }
        Meta::List(_) => false,
    }
}

fn forbidden_path(path: &syn::Path) -> Option<String> {
    let segments = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>();
    let rendered = segments.join("::");
    if rendered.contains("vyre_driver_reference")
        || rendered.ends_with("vyre_reference::reference_eval")
        || segments
            .iter()
            .any(|segment| FORBIDDEN_SYMBOLS.contains(&segment.as_str()))
    {
        Some(rendered)
    } else {
        None
    }
}

fn cpu_helper_name(name: &str) -> bool {
    name.starts_with("cpu_") || name.contains("_cpu")
}

struct CpuFallbackVisitor<'a> {
    file: &'a str,
    approved_depth: usize,
    violations: Vec<Violation>,
}

impl CpuFallbackVisitor<'_> {
    fn record_fallback(&mut self, span: Span, path: &str) {
        if self.approved_depth > 0 {
            return;
        }
        let start = span.start();
        self.violations.push(Violation {
            file: self.file.to_string(),
            line: start.line as u32,
            column: start.column as u32,
            kind: ViolationKind::ProductionCpuFallback,
            message: format!(
                "production CPU/reference fallback `{path}` outside approved parity surface"
            ),
        });
    }

    fn record_helper(&mut self, span: Span, name: &str) {
        if self.approved_depth > 0 {
            return;
        }
        let start = span.start();
        self.violations.push(Violation {
            file: self.file.to_string(),
            line: start.line as u32,
            column: start.column as u32,
            kind: ViolationKind::ProductionCpuFallback,
            message: format!(
                "production CPU/reference helper definition `{name}` outside approved parity surface"
            ),
        });
    }

    fn visit_scoped(&mut self, attrs: &[Attribute], visit: impl FnOnce(&mut Self)) {
        let approved = attrs_require_parity(attrs);
        self.approved_depth += usize::from(approved);
        visit(self);
        self.approved_depth -= usize::from(approved);
    }

    fn inspect_use_tree(&mut self, tree: &UseTree, prefix: &mut Vec<String>) {
        match tree {
            UseTree::Path(path) => {
                prefix.push(path.ident.to_string());
                self.inspect_use_tree(&path.tree, prefix);
                prefix.pop();
            }
            UseTree::Name(name) => {
                prefix.push(name.ident.to_string());
                self.inspect_segments(name.ident.span(), prefix);
                prefix.pop();
            }
            UseTree::Rename(rename) => {
                prefix.push(rename.ident.to_string());
                self.inspect_segments(rename.ident.span(), prefix);
                prefix.pop();
            }
            UseTree::Group(group) => {
                for item in &group.items {
                    self.inspect_use_tree(item, prefix);
                }
            }
            UseTree::Glob(glob) => self.inspect_segments(glob.star_token.span(), prefix),
        }
    }

    fn inspect_segments(&mut self, span: Span, segments: &[String]) {
        let rendered = segments.join("::");
        if rendered.contains("vyre_driver_reference")
            || rendered.ends_with("vyre_reference::reference_eval")
            || segments
                .iter()
                .any(|segment| FORBIDDEN_SYMBOLS.contains(&segment.as_str()))
        {
            self.record_fallback(span, &rendered);
        }
    }

    fn inspect_macro_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        for token in tokens {
            match token {
                TokenTree::Group(group) => self.inspect_macro_tokens(group.stream()),
                TokenTree::Ident(ident) => {
                    let name = ident.to_string();
                    if name == "vyre_driver_reference"
                        || name == "reference_eval"
                        || FORBIDDEN_SYMBOLS.contains(&name.as_str())
                    {
                        self.record_fallback(ident.span(), &name);
                    }
                }
                TokenTree::Literal(_) | TokenTree::Punct(_) => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for CpuFallbackVisitor<'_> {
    fn visit_item_mod(&mut self, node: &'ast ItemMod) {
        self.visit_scoped(&node.attrs, |visitor| {
            if cpu_helper_name(&node.ident.to_string()) {
                visitor.record_helper(node.ident.span(), &node.ident.to_string());
            }
            syn::visit::visit_item_mod(visitor, node);
        });
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.visit_scoped(&node.attrs, |visitor| {
            if cpu_helper_name(&node.sig.ident.to_string()) {
                visitor.record_helper(node.sig.ident.span(), &node.sig.ident.to_string());
            }
            syn::visit::visit_item_fn(visitor, node);
        });
    }

    fn visit_item_impl(&mut self, node: &'ast ItemImpl) {
        self.visit_scoped(&node.attrs, |visitor| {
            syn::visit::visit_item_impl(visitor, node)
        });
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.visit_scoped(&node.attrs, |visitor| {
            if cpu_helper_name(&node.sig.ident.to_string()) {
                visitor.record_helper(node.sig.ident.span(), &node.sig.ident.to_string());
            }
            syn::visit::visit_impl_item_fn(visitor, node);
        });
    }

    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        self.visit_scoped(&node.attrs, |visitor| {
            if cpu_helper_name(&node.sig.ident.to_string()) {
                visitor.record_helper(node.sig.ident.span(), &node.sig.ident.to_string());
            }
            syn::visit::visit_trait_item_fn(visitor, node);
        });
    }

    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        self.visit_scoped(&node.attrs, |visitor| {
            visitor.inspect_use_tree(&node.tree, &mut Vec::new());
        });
    }

    fn visit_item_extern_crate(&mut self, node: &'ast syn::ItemExternCrate) {
        self.visit_scoped(&node.attrs, |visitor| {
            if node.ident == "vyre_driver_reference" || node.ident == "vyre_reference" {
                visitor.record_fallback(node.ident.span(), &node.ident.to_string());
            }
        });
    }

    fn visit_item_macro(&mut self, node: &'ast ItemMacro) {
        // A declarative macro is a definition, not an executed fallback path.
        // Macro invocation token streams are inspected at their call sites.
        if node.ident.is_some() || node.mac.path.is_ident("macro_rules") {
            return;
        }
        self.visit_scoped(&node.attrs, |visitor| {
            visitor.inspect_macro_tokens(node.mac.tokens.clone());
        });
    }

    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        if let Some(path) = forbidden_path(&node.path) {
            self.record_fallback(node.path.span(), &path);
        }
        syn::visit::visit_expr_path(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let method = node.method.to_string();
        if FORBIDDEN_SYMBOLS.contains(&method.as_str()) {
            self.record_fallback(node.method.span(), &method);
        }
        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        self.inspect_macro_tokens(node.tokens.clone());
    }
}
