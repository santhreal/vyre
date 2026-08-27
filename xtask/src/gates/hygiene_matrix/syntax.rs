use quote::ToTokens;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use syn::visit::Visit;

use super::records::{HygieneFinding, STRUCTURAL_GATE_SOURCE};

/// Markers for a function that resolves the checkout this test run reads.
///
/// A source-inspecting gate has to name the tree it judges, and the workspace
/// has three resolvers plus the compiled-in manifest directory. A function that
/// names none of them reads whatever root its caller handed it.
const CHECKOUT_RESOLVERS: &[&str] = &[
    "checkout_root",
    "workspace_root",
    "vyre_crate_directory",
    "CARGO_MANIFEST_DIR",
];

/// Markers for a temporary root a test creates for itself.
const TEMPORARY_ROOTS: &[&str] = &["temp_dir", "tempdir", "TempDir", "tempfile"];

/// Markers for a function that writes the tree it later reads.
const TREE_WRITERS: &[&str] = &["fs::write", "create_dir_all"];

/// Calls that inspect source text without a string method.
///
/// Handing the text to a parser is the same inspection as searching it: the
/// parsers here exist for no other purpose, and a test that reads a `.rs` file
/// and passes it to one used to escape classification entirely because it never
/// called `contains` itself.
const SOURCE_TEXT_PARSERS: &[&str] = &[
    "braced_body(",
    "top_level_variant_names(",
    "declared_level_variants(",
    "parse_file(",
    "parse_str(",
];

/// Whether `tokens` reads Rust source out of a root the checkout resolved.
///
/// A resolver and a Rust path anywhere in one function is not enough. A test
/// that writes a fixture workspace and copies one tool out of the checkout into
/// it names both, and every Rust path it names is a file it wrote. The statement
/// naming the Rust path has to be the statement naming a checkout-rooted value,
/// which is either the resolver call or a binding taken from one.
///
/// `tokens` is a whitespace-free token string, so statements are split on `;`.
/// A `for` header and the first statement of its body land in one chunk, which
/// widens the answer rather than narrowing it: this decides whether a finding is
/// kept, so an uncertain read stays a finding.
fn checkout_rooted_rust_read(tokens: &str) -> bool {
    let mut rooted: Vec<String> = Vec::new();
    for statement in tokens.split(';') {
        let names_root = CHECKOUT_RESOLVERS
            .iter()
            .any(|resolver| statement.contains(resolver))
            || rooted
                .iter()
                .any(|name| mentions_identifier(statement, name));
        if !names_root {
            continue;
        }
        if statement.contains("\"rs\"")
            || statement.contains(".rs\"")
            || (statement.contains("extension()") && statement.contains("==\"rs\""))
        {
            return true;
        }
        if let Some(bound) = bound_identifier(statement) {
            rooted.push(bound);
        }
    }
    false
}

/// The name a `let` statement binds, from a whitespace-free token string.
///
/// Reads `letname=`, `letmutname=`, and `letname:Type=`. A pattern that binds
/// several names reports none, so the value it destructures is not tracked and
/// the finding is kept.
fn bound_identifier(statement: &str) -> Option<String> {
    let rest = statement.trim_start_matches(['{', '}']);
    let rest = rest.strip_prefix("let")?;
    let rest = rest.strip_prefix("mut").unwrap_or(rest);
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        return None;
    }
    let tail = &rest[name.len()..];
    if tail.starts_with('=') || tail.starts_with(':') {
        Some(name)
    } else {
        None
    }
}

/// Whether `name` appears in `haystack` as a whole identifier.
fn mentions_identifier(haystack: &str, name: &str) -> bool {
    haystack.match_indices(name).any(|(at, _)| {
        let before = haystack[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
        let after = haystack[at + name.len()..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
        before && after
    })
}

#[derive(Default)]
pub(crate) struct RustSourceFactsVisitor {
    pub(crate) reads_rust_source: bool,
    pub(crate) calls_read_to_string: bool,
    pub(crate) mentions_rust_path: bool,
    pub(crate) inspects_text: bool,
    /// Names a temporary root, its own or one a helper made.
    pub(crate) creates_temporary_root: bool,
    /// Writes files into whatever root it was handed.
    pub(crate) writes_source_tree: bool,
    /// Reads Rust source out of the checkout under test, rather than out of a
    /// root its caller supplied. Both halves have to appear in one function: a
    /// fixture builder that reads a generated JSON artifact from the checkout is
    /// not reading this tree's source.
    pub(crate) reads_checkout_rust_source: bool,
    pub(crate) callees: BTreeSet<String>,
    pub(crate) aliases: BTreeMap<String, String>,
}

impl RustSourceFactsVisitor {
    pub(crate) fn callee_name(expression: &syn::Expr) -> Option<String> {
        match expression {
            syn::Expr::Path(path) => path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string()),
            syn::Expr::Paren(paren) => Self::callee_name(&paren.expr),
            _ => None,
        }
    }

    pub(crate) fn resolved_callee(&self, name: String) -> String {
        self.aliases.get(&name).cloned().unwrap_or(name)
    }
}

/// Collect the identifiers a macro body names, recursing into its groups.
///
/// A macro body is opaque to `syn`'s typed visitors, so a call inside
/// `assert!(...)` is only reachable through its raw tokens. Rendering those
/// tokens to a string and splitting on non-identifier characters also splits
/// the CONTENTS of every string literal: `assert!(text.contains("vyre-scan"))`
/// then claims a call to `scan`, and the transitive walk enters whatever
/// unrelated local function carries that name. Walking the token trees keeps
/// the real call names and drops literals, punctuation, and lifetimes.
pub(crate) fn collect_macro_identifiers(
    tokens: proc_macro2::TokenStream,
    callees: &mut BTreeSet<String>,
) {
    for tree in tokens {
        match tree {
            proc_macro2::TokenTree::Ident(ident) => {
                callees.insert(ident.to_string());
            }
            proc_macro2::TokenTree::Group(group) => {
                collect_macro_identifiers(group.stream(), callees);
            }
            proc_macro2::TokenTree::Literal(_) | proc_macro2::TokenTree::Punct(_) => {}
        }
    }
}

impl<'ast> Visit<'ast> for RustSourceFactsVisitor {
    fn visit_macro(&mut self, expression: &'ast syn::Macro) {
        if expression.path.is_ident("include_str")
            && syn::parse2::<syn::LitStr>(expression.tokens.clone())
                .is_ok_and(|path| path.value().ends_with(".rs"))
        {
            self.reads_rust_source = true;
        }
        collect_macro_identifiers(expression.tokens.clone(), &mut self.callees);
        syn::visit::visit_macro(self, expression);
    }

    fn visit_expr_call(&mut self, expression: &'ast syn::ExprCall) {
        if let Some(name) = Self::callee_name(&expression.func) {
            let name = self.resolved_callee(name);
            if name == "read_to_string" {
                self.calls_read_to_string = true;
                let arguments = expression.args.to_token_stream().to_string();
                self.reads_rust_source |= arguments.contains(".rs");
            }
            self.callees.insert(name);
        }
        syn::visit::visit_expr_call(self, expression);
    }

    fn visit_expr_path(&mut self, expression: &'ast syn::ExprPath) {
        if let Some(segment) = expression.path.segments.last() {
            self.callees.insert(segment.ident.to_string());
            if segment.ident == "read_to_string" {
                self.calls_read_to_string = true;
            }
        }
        syn::visit::visit_expr_path(self, expression);
    }

    fn visit_expr_method_call(&mut self, expression: &'ast syn::ExprMethodCall) {
        let method = expression.method.to_string();
        if method == "read_to_string" {
            self.calls_read_to_string = true;
        }
        if matches!(
            method.as_str(),
            "contains" | "split" | "matches" | "starts_with" | "ends_with"
        ) {
            self.inspects_text = true;
        }
        self.callees.insert(method);
        if let syn::Expr::Path(receiver) = expression.receiver.as_ref() {
            if let Some(segment) = receiver.path.segments.last() {
                self.callees.insert(segment.ident.to_string());
            }
        }
        syn::visit::visit_expr_method_call(self, expression);
    }

    fn visit_expr_lit(&mut self, expression: &'ast syn::ExprLit) {
        if let syn::Lit::Str(value) = &expression.lit {
            let value = value.value();
            if value == "rs" || value.ends_with(".rs") {
                self.mentions_rust_path = true;
            }
        }
        syn::visit::visit_expr_lit(self, expression);
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let (syn::Pat::Ident(alias), Some(initializer)) = (&local.pat, &local.init) {
            if let Some(target) = Self::callee_name(&initializer.expr) {
                self.aliases.insert(alias.ident.to_string(), target);
            }
        }
        syn::visit::visit_local(self, local);
    }
}

pub(crate) fn attrs_are_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attribute| {
        attribute
            .path()
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "test")
    })
}

pub(crate) struct SourceInspectionFunction {
    pub(crate) line: usize,
    pub(crate) name: String,
    pub(crate) is_test: bool,
    pub(crate) facts: RustSourceFactsVisitor,
}

#[derive(Default)]
pub(crate) struct SourceInspectionFunctionCollector {
    pub(crate) cfg_test_depth: usize,
    pub(crate) functions: Vec<SourceInspectionFunction>,
}

impl SourceInspectionFunctionCollector {
    pub(crate) fn push_function(
        &mut self,
        name: String,
        line: usize,
        is_test: bool,
        block: &syn::Block,
    ) {
        let mut facts = RustSourceFactsVisitor::default();
        facts.visit_block(block);
        let mut tokens = block.to_token_stream().to_string();
        tokens.retain(|character| !character.is_whitespace());
        if !is_test {
            facts.calls_read_to_string |= tokens.contains("read_to_string");
            facts.mentions_rust_path |= tokens.contains("\"rs\"")
                || tokens.contains(".rs\"")
                || (tokens.contains("extension()") && tokens.contains("==\"rs\""));
            if tokens.contains("read_to_string") {
                facts.calls_read_to_string = true;
                facts.callees.insert("read_to_string".to_string());
            }
            facts.reads_rust_source |= facts.calls_read_to_string && facts.mentions_rust_path;
        }
        facts.inspects_text |= [
            ".contains(",
            ".split(",
            ".matches(",
            ".starts_with(",
            ".ends_with(",
        ]
        .iter()
        .chain(SOURCE_TEXT_PARSERS.iter())
        .any(|needle| tokens.contains(needle));
        if !is_test && facts.calls_read_to_string && facts.mentions_rust_path {
            facts.reads_rust_source = true;
        }
        let fixture_checkout = tokens.contains("fixture_checkout");
        facts.creates_temporary_root |=
            fixture_checkout || TEMPORARY_ROOTS.iter().any(|root| tokens.contains(root));
        facts.writes_source_tree |=
            fixture_checkout || TREE_WRITERS.iter().any(|writer| tokens.contains(writer));
        facts.reads_checkout_rust_source |= checkout_rooted_rust_read(&tokens);
        self.functions.push(SourceInspectionFunction {
            line,
            name,
            is_test,
            facts,
        });
    }
}

impl<'ast> Visit<'ast> for SourceInspectionFunctionCollector {
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.push_function(
            item.sig.ident.to_string(),
            item.sig.ident.span().start().line,
            self.cfg_test_depth != 0 || attrs_are_test(&item.attrs),
            &item.block,
        );
        for statement in &item.block.stmts {
            syn::visit::visit_stmt(self, statement);
        }
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.push_function(
            item.sig.ident.to_string(),
            item.sig.ident.span().start().line,
            self.cfg_test_depth != 0 || attrs_are_test(&item.attrs),
            &item.block,
        );
        for statement in &item.block.stmts {
            syn::visit::visit_stmt(self, statement);
        }
    }
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let is_test_module = item.attrs.iter().any(|attribute| {
            attribute.path().is_ident("cfg")
                && attribute
                    .meta
                    .to_token_stream()
                    .to_string()
                    .contains("test")
        });
        if is_test_module {
            self.cfg_test_depth += 1;
        }
        if let Some((_, items)) = &item.content {
            for nested in items {
                self.visit_item(nested);
            }
        }
        if is_test_module {
            self.cfg_test_depth -= 1;
        }
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        for implementation_item in &item.items {
            if let syn::ImplItem::Fn(function) = implementation_item {
                self.visit_impl_item_fn(function);
            }
        }
    }
}

pub(crate) fn source_inspection_test_findings(file: &syn::File) -> Vec<(usize, String)> {
    let mut collector = SourceInspectionFunctionCollector::default();
    collector.visit_file(file);
    let functions = &collector.functions;
    let mut by_name = BTreeMap::<&str, Vec<usize>>::new();
    for (index, function) in functions.iter().enumerate() {
        by_name.entry(&function.name).or_default().push(index);
    }
    let mut findings = Vec::new();

    for (test_index, test) in functions
        .iter()
        .enumerate()
        .filter(|(_, function)| function.is_test)
    {
        let mut stack = vec![test_index];
        let mut visited = BTreeSet::new();
        let mut reads_rust_source = false;
        let mut calls_read_to_string = false;
        let mut mentions_rust_path = false;
        let mut inspects_text = false;
        let mut creates_temporary_root = false;
        let mut writes_source_tree = false;
        let mut reads_checkout_rust_source = false;
        while let Some(index) = stack.pop() {
            if !visited.insert(index) {
                continue;
            }
            let facts = &functions[index].facts;
            reads_rust_source |= facts.reads_rust_source;
            calls_read_to_string |= facts.calls_read_to_string;
            mentions_rust_path |= facts.mentions_rust_path;
            inspects_text |= facts.inspects_text;
            creates_temporary_root |= facts.creates_temporary_root;
            writes_source_tree |= facts.writes_source_tree;
            reads_checkout_rust_source |= facts.reads_checkout_rust_source;
            for callee in &facts.callees {
                if let Some(indices) = by_name.get(callee.as_str()) {
                    stack.extend(indices);
                }
            }
        }
        reads_rust_source |= calls_read_to_string && mentions_rust_path;
        // A test that wrote the tree it reads is exercising the analyzer against
        // a fixture, which is behavior, and the analyzer's subject happens to be
        // source text. The root and the writes may sit in different helpers, so
        // the two facts are collected separately and combined here. A test that
        // authors a fixture and ALSO reads Rust source out of the resolved
        // checkout keeps the finding: the fixture does not excuse that walk.
        let authors_source_tree = creates_temporary_root && writes_source_tree;
        let inspects_this_checkout = !authors_source_tree || reads_checkout_rust_source;
        if reads_rust_source && inspects_text && inspects_this_checkout {
            findings.push((test.line, test.name.clone()));
        }
    }
    findings.sort();
    findings.dedup();
    findings
}

pub(crate) fn scan_source_inspection_tests(
    path: &Path,
    text: &str,
    findings: &mut Vec<HygieneFinding>,
) {
    if path.file_name().and_then(|name| name.to_str()) == Some("hygiene_matrix.rs") {
        return;
    }
    let Ok(file) = syn::parse_file(text) else {
        return;
    };
    for (line, test) in source_inspection_test_findings(&file) {
        findings.push(HygieneFinding {
            path: path.display().to_string(),
            line,
            pattern: "source_inspection_test",
            text: format!(
                "test `{test}` inspects Rust source text. Fix: assert behavior, lifecycle ownership, generated registry ownership, or emitted artifacts instead, or declare the property as unobservable in {STRUCTURAL_GATE_SOURCE}."
            ),
            test: Some(test),
        });
    }
}
