//! `raw_ir_in_libs`: forbid raw `Node::*` / `Expr::*` construction in
//! `vyre-libs/src/**`. Construction sites  -  struct literals, tuple
//! constructors, and associated-function calls  -  are flagged. Pattern
//! matching against the same enum variants is allowed (it's read, not
//! construct). Only explicitly test-gated items are allowed (`#[cfg(test)]`).

use crate::allowlist::Allowlist;
use crate::{scan, Violation, ViolationKind};
use anyhow::{Context, Result};
use proc_macro2::TokenTree;
use std::collections::HashMap;
use std::path::Path;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Meta, Token, UseTree};

const FORBIDDEN_TYPES: &[&str] = &["Node", "Expr"];

/// Scan a library source tree for disallowed direct IR construction.
pub fn scan_tree(root: &Path, allow: &Allowlist) -> Result<Vec<Violation>> {
    scan::collect_violations(
        root,
        scan::RUST_SOURCE,
        |workspace_rel| !allow.contains(workspace_rel),
        scan_file,
    )
}

fn scan_file(path: &Path, workspace_rel: &str) -> Result<Vec<Violation>> {
    let source = crate::read_source_bounded(path)?;
    let file = syn::parse_file(&source).with_context(|| format!("parse {}", path.display()))?;

    let mut visitor = LegoBlockVisitor {
        file: workspace_rel.to_string(),
        in_test_depth: 0,
        violations: Vec::new(),
        aliases: HashMap::from([("Node".to_string(), "Node"), ("Expr".to_string(), "Expr")]),
    };
    visitor.visit_file(&file);
    Ok(visitor.violations)
}

struct LegoBlockVisitor {
    file: String,
    in_test_depth: usize,
    aliases: HashMap<String, &'static str>,
    violations: Vec<Violation>,
}

impl LegoBlockVisitor {
    fn record(&mut self, span: proc_macro2::Span, ty: &str, what: &str) {
        if self.in_test_depth > 0 {
            return;
        }
        let kind = match ty {
            "Node" => ViolationKind::RawNodeConstruction,
            "Expr" => ViolationKind::RawExprConstruction,
            _ => return,
        };
        let start = span.start();
        self.violations.push(Violation {
            file: self.file.clone(),
            line: start.line as u32,
            column: start.column as u32,
            kind,
            message: format!("raw {ty}::{what} construction in vyre-libs"),
        });
    }

    fn forbidden_path(&self, path: &syn::Path) -> Option<(&'static str, String)> {
        if path.segments.len() < 2 {
            return None;
        }
        let what = path.segments.last()?.ident.to_string();
        let owner = path.segments[path.segments.len() - 2].ident.to_string();
        self.aliases.get(&owner).copied().map(|ty| (ty, what))
    }

    fn register_use_aliases(&mut self, tree: &UseTree) {
        fn walk(
            tree: &UseTree,
            prefix: &mut Vec<String>,
            aliases: &mut HashMap<String, &'static str>,
        ) {
            match tree {
                UseTree::Path(path) => {
                    prefix.push(path.ident.to_string());
                    walk(&path.tree, prefix, aliases);
                    prefix.pop();
                }
                UseTree::Name(name) => {
                    let canonical = name.ident.to_string();
                    if let Some(ty) = FORBIDDEN_TYPES.iter().copied().find(|ty| *ty == canonical) {
                        aliases.insert(canonical, ty);
                    }
                }
                UseTree::Rename(rename) => {
                    let canonical = rename.ident.to_string();
                    if let Some(ty) = FORBIDDEN_TYPES.iter().copied().find(|ty| *ty == canonical) {
                        aliases.insert(rename.rename.to_string(), ty);
                    }
                }
                UseTree::Group(group) => {
                    for item in &group.items {
                        walk(item, prefix, aliases);
                    }
                }
                UseTree::Glob(_) => {}
            }
        }
        walk(tree, &mut Vec::new(), &mut self.aliases);
    }

    fn inspect_macro_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        let tokens = tokens.into_iter().collect::<Vec<_>>();
        for (index, token) in tokens.iter().enumerate() {
            if let TokenTree::Group(group) = token {
                self.inspect_macro_tokens(group.stream());
            }
            let TokenTree::Ident(owner) = token else {
                continue;
            };
            let Some(ty) = self.aliases.get(&owner.to_string()).copied() else {
                continue;
            };
            let Some(TokenTree::Punct(first_colon)) = tokens.get(index + 1) else {
                continue;
            };
            let Some(TokenTree::Punct(second_colon)) = tokens.get(index + 2) else {
                continue;
            };
            let Some(TokenTree::Ident(what)) = tokens.get(index + 3) else {
                continue;
            };
            if first_colon.as_char() == ':' && second_colon.as_char() == ':' {
                self.record(owner.span(), ty, &what.to_string());
            }
        }
    }

    fn inspect_macro_definition(&mut self, tokens: proc_macro2::TokenStream) {
        let tokens = tokens.into_iter().collect::<Vec<_>>();
        for (index, token) in tokens.iter().enumerate() {
            if let TokenTree::Group(group) = token {
                self.inspect_macro_definition(group.stream());
            }
            let Some(TokenTree::Punct(equals)) = tokens.get(index) else {
                continue;
            };
            let Some(TokenTree::Punct(arrow)) = tokens.get(index + 1) else {
                continue;
            };
            if equals.as_char() != '=' || arrow.as_char() != '>' {
                continue;
            }
            if let Some(TokenTree::Group(expansion)) = tokens.get(index + 2) {
                self.inspect_macro_tokens(expansion.stream());
            }
        }
    }

    fn register_type_alias(&mut self, alias: &syn::Ident, ty: &syn::Type) {
        let syn::Type::Path(path) = ty else {
            return;
        };
        let Some(last) = path.path.segments.last() else {
            return;
        };
        let canonical = last.ident.to_string();
        if let Some(ir_type) = self.aliases.get(&canonical).copied() {
            self.aliases.insert(alias.to_string(), ir_type);
        }
    }
}

fn cfg_requires_test(meta: &Meta) -> bool {
    match meta {
        Meta::Path(path) => path.is_ident("test"),
        Meta::NameValue(_) => false,
        Meta::List(list) if list.path.is_ident("all") || list.path.is_ident("any") => {
            let Ok(items) = list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            else {
                return false;
            };
            if list.path.is_ident("all") {
                items.iter().any(cfg_requires_test)
            } else {
                !items.is_empty() && items.iter().all(cfg_requires_test)
            }
        }
        Meta::List(_) => false,
    }
}

fn is_test_attr(attr: &syn::Attribute) -> bool {
    attr.path().is_ident("test")
        || (attr.path().is_ident("cfg")
            && attr
                .parse_args::<Meta>()
                .is_ok_and(|meta| cfg_requires_test(&meta)))
}

fn module_is_test(item_mod: &syn::ItemMod) -> bool {
    item_mod.attrs.iter().any(is_test_attr)
}

impl<'ast> Visit<'ast> for LegoBlockVisitor {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        let aliases = self.aliases.clone();
        let test_mod = module_is_test(node);
        if test_mod {
            self.in_test_depth += 1;
        }
        syn::visit::visit_item_mod(self, node);
        if test_mod {
            self.in_test_depth -= 1;
        }
        self.aliases = aliases;
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let aliases = self.aliases.clone();
        let is_test = node.attrs.iter().any(is_test_attr);
        if is_test {
            self.in_test_depth += 1;
        }
        syn::visit::visit_item_fn(self, node);
        if is_test {
            self.in_test_depth -= 1;
        }
        self.aliases = aliases;
    }

    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        if self.in_test_depth == 0 {
            self.register_use_aliases(&node.tree);
        }
    }

    fn visit_item_type(&mut self, node: &'ast syn::ItemType) {
        if self.in_test_depth == 0 {
            self.register_type_alias(&node.ident, &node.ty);
        }
        syn::visit::visit_item_type(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast syn::ItemMacro) {
        if node.ident.is_some() || node.mac.path.is_ident("macro_rules") {
            self.inspect_macro_definition(node.mac.tokens.clone());
        } else if !node.mac.path.is_ident("matches") {
            self.inspect_macro_tokens(node.mac.tokens.clone());
        }
    }

    fn visit_expr_struct(&mut self, node: &'ast syn::ExprStruct) {
        if let Some((ty, variant)) = self.forbidden_path(&node.path) {
            self.record(node.path.span(), ty, &variant);
        }
        syn::visit::visit_expr_struct(self, node);
    }

    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        if let Some((ty, what)) = self.forbidden_path(&node.path) {
            self.record(node.path.span(), ty, &what);
        }
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if !node.path.is_ident("matches") {
            self.inspect_macro_tokens(node.tokens.clone());
        }
    }
}
