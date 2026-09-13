//! The oracle reads the specification and never runs a production transform.
//!
//! `vyre-reference` is the only trusted semantic oracle in this workspace. A
//! conformance failure is read as a backend defect because the oracle is
//! assumed to be arrived at independently. That assumption held only by
//! convention: `reference_eval` once called foundation's collectives lowering
//! and composite inliner before interpreting anything, so a defect in either
//! transform moved the implementation and its alleged oracle the same distance
//! and the comparison agreed on a wrong answer.
//!
//! Removing those two calls fixed one incident. This rule closes the class.
//!
//! `vyre-foundation` carries both halves of the contract. `ir`, `operation`,
//! `validate` and their neighbours state what a program means, and the oracle
//! must read them or it would be interpreting a second IR. `optimizer`,
//! `lower`, `transform`, `schedule`, `execution_plan` and their neighbours
//! decide how a program runs, and every one of them is a production decision
//! the oracle exists to check rather than to inherit.
//!
//! The surface is derived from `vyre-foundation/src/lib.rs` at run time: its
//! `pub mod` declarations and its `#[macro_export]` macros are what a dependent
//! crate can name. `SPECIFICATION_SURFACE` records the half the oracle may
//! read, and everything else on that derived surface is forbidden. Recording
//! the permitted half rather than the forbidden half is what makes a new
//! foundation module fail closed: the day one is added, the oracle cannot name
//! it until someone decides it states semantics, and nobody has to remember
//! this file exists.
//!
//! A recorded entry that no longer exists on the surface is also a finding. A
//! renamed module would otherwise leave a permission behind that matches
//! nothing, and a permission list nobody can see is wrong about is how the
//! forbidden half gets back in under an old name.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use syn::visit::Visit;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// The crate whose public surface the oracle is held to.
const FOUNDATION: &str = "vyre-foundation";

/// The file that declares that surface.
const FOUNDATION_LIB: &str = "vyre-foundation/src/lib.rs";

/// The interpreter's sources.
const ORACLE_SRC: &str = "vyre-reference/src";

/// The crate path the oracle names foundation by.
const FOUNDATION_PATH: &str = "vyre_foundation";

/// The `vyre-foundation` items the oracle may name.
///
/// Each states what a program means rather than how one runs. `ir` is the
/// representation itself and `operation`, `types`, `numeric` and `logical` are
/// the records that give its nodes meaning. `validate` and `verifier` reject a
/// program the semantics do not admit, which an oracle must do before
/// interpreting one. `match_result`, `failure_domain`, `diagnostics` and the
/// `diagnostic_conversions` macro are how a refusal is reported. `allocation`
/// reserves memory and decides nothing about the program. `cpu_op`, `fp_parity`
/// and `fp_expansion` state the arithmetic the semantics require, not a
/// schedule for performing it. `dialect`, `dialect_lookup`, `extension`,
/// `config_schema` and `program_caps` describe what a program declares about
/// itself. `visit` and `graph_view` read the graph without rewriting it.
///
/// Absent on purpose, and forbidden by being absent: `optimizer`, `transform`,
/// `lower`, `schedule`, `execution_plan`, `pass_math`, `algebraic_reordering`,
/// `algebraic_law_registry`, `region_ssa`, `vast`, `loop_bounds`, `substrate`,
/// `platform`, `perf`, `hashing`, `serial`, `canonical_codec`, `source_digest`,
/// `opaque_payload`, `composition`, `causal`, `security`.
const SPECIFICATION_SURFACE: &[&str] = &[
    "allocation",
    "config_schema",
    "cpu_op",
    "diagnostic_conversions",
    "diagnostics",
    "dialect",
    "dialect_lookup",
    "extension",
    "failure_domain",
    "fp_expansion",
    "fp_parity",
    "graph_view",
    "ir",
    "logical",
    "match_result",
    "numeric",
    "operation",
    "program_caps",
    "types",
    "validate",
    "verifier",
    "visit",
];

/// One place the oracle names a `vyre-foundation` item.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Reference {
    /// The oracle source that names it, repository-relative.
    file: PathBuf,
    /// The line it is named on, one-based.
    line: u32,
    /// The first path segment after `vyre_foundation`.
    item: String,
}

/// The oracle never reaches a production transform.
pub struct OracleIndependence;

impl crate::gate::GateBehavior for OracleIndependence {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        if let Some(note) = tree.absence_note() {
            report.note(note);
        }

        let surface = foundation_surface(&tree)?;
        let sources = tree.rust(&[ORACLE_SRC])?;
        let mut references = Vec::new();
        for path in &sources {
            references.extend(references_in(&tree.read(path)?, path)?);
        }

        report.cover_complete("oracle sources", sources.len());
        report.cover_complete(
            "foundation items the oracle names",
            references
                .iter()
                .map(|reference| reference.item.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
        );

        let permitted: BTreeSet<&str> = SPECIFICATION_SURFACE.iter().copied().collect();
        for recorded in &permitted {
            if !surface.contains(*recorded) {
                report.find(Finding::in_file(
                    FOUNDATION_LIB,
                    format!(
                        "`{recorded}` is recorded as part of the semantic surface the oracle may \
                         read, and `{FOUNDATION}` publishes no such item"
                    ),
                    "drop the stale entry from SPECIFICATION_SURFACE, or restore the item under \
                     the name the oracle was permitted",
                ));
            }
        }

        for reference in &references {
            if permitted.contains(reference.item.as_str()) {
                continue;
            }
            let known = if surface.contains(&reference.item) {
                "decides how a program runs"
            } else {
                "is not published by the crate"
            };
            report.find(Finding::at(
                reference.file.clone(),
                reference.line,
                format!(
                    "the oracle names `{FOUNDATION_PATH}::{}`, which {known}: an interpreter that \
                     runs a production transform cannot detect a defect in it",
                    reference.item
                ),
                "interpret the semantics the specification records instead, or record the item in \
                 SPECIFICATION_SURFACE in the commit that establishes it states semantics",
            ));
        }

        report.note(format!(
            "{} item(s) on the `{FOUNDATION}` surface, {} readable by the oracle, {} reference(s) \
             across {} oracle source(s)",
            surface.len(),
            permitted.len(),
            references.len(),
            sources.len()
        ));
        Ok(report)
    }
}

/// Every item a dependent crate can name through `vyre_foundation::`.
///
/// The `pub mod` declarations and the exported macros, read from the crate that
/// publishes them. A list written here would state a surface the crate no
/// longer has the first time a module is added.
fn foundation_surface(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    let text = tree.read(FOUNDATION_LIB)?;
    let file = parse(&text, Path::new(FOUNDATION_LIB))?;
    let mut surface = BTreeSet::new();
    for item in &file.items {
        match item {
            syn::Item::Mod(module) if matches!(module.vis, syn::Visibility::Public(_)) => {
                surface.insert(module.ident.to_string());
            }
            syn::Item::Macro(entry) => {
                let exported = entry
                    .attrs
                    .iter()
                    .any(|attr| attr.path().is_ident("macro_export"));
                if let (true, Some(name)) = (exported, entry.ident.as_ref()) {
                    surface.insert(name.to_string());
                }
            }
            syn::Item::Use(entry) if matches!(entry.vis, syn::Visibility::Public(_)) => {
                collect_reexports(&entry.tree, &mut surface);
            }
            _ => {}
        }
    }
    if surface.is_empty() {
        return Err(GateError::new(
            format!("`{FOUNDATION_LIB}` publishes no module, macro or re-export"),
            "repoint this gate at the file that declares the crate's surface; a rule derived from \
             an empty surface permits everything",
        ));
    }
    surface.extend(exported_macros(tree)?);
    Ok(surface)
}

/// Every `#[macro_export]` macro the crate defines, wherever it is defined.
///
/// An exported macro is nameable as `vyre_foundation::<name>` from any module
/// of the crate, so reading only the root would leave one unnameable and report
/// the oracle's use of it as an item the crate does not publish.
fn exported_macros(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    let mut macros = BTreeSet::new();
    for path in tree.rust(&[&format!("{FOUNDATION}/src")])? {
        let text = tree.read(&path)?;
        if !text.contains("macro_export") {
            continue;
        }
        let file = parse(&text, &path)?;
        let mut visitor = MacroScan {
            names: BTreeSet::new(),
        };
        visitor.visit_file(&file);
        macros.extend(visitor.names);
    }
    Ok(macros)
}

/// The names the root re-exports directly under the crate path.
///
/// `pub use error::{IrError, IrResult}` publishes two type names and not the
/// `error` module, which is `pub(crate)` and unnameable from outside. Only the
/// leaf of a re-export path is published, so an intermediate segment is walked
/// through rather than recorded. A glob publishes a set this file cannot
/// enumerate, so it contributes nothing and an item reached through one reads
/// as unpublished until it is exported by name.
fn collect_reexports(tree: &syn::UseTree, surface: &mut BTreeSet<String>) {
    match tree {
        syn::UseTree::Path(path) => collect_reexports(&path.tree, surface),
        syn::UseTree::Name(name) => {
            surface.insert(name.ident.to_string());
        }
        syn::UseTree::Rename(rename) => {
            surface.insert(rename.rename.to_string());
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_reexports(item, surface);
            }
        }
        syn::UseTree::Glob(_) => {}
    }
}

/// Collects `#[macro_export]` macro names anywhere in a file.
struct MacroScan {
    names: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for MacroScan {
    fn visit_item_macro(&mut self, entry: &'ast syn::ItemMacro) {
        let exported = entry
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("macro_export"));
        if let (true, Some(name)) = (exported, entry.ident.as_ref()) {
            self.names.insert(name.to_string());
        }
        syn::visit::visit_item_macro(self, entry);
    }
}

/// Every `vyre_foundation::<item>` the source names, in code.
///
/// The file is parsed rather than scanned, so a path inside a doc comment or a
/// string is not a reference and a `use vyre_foundation::{a, b::c}` contributes
/// both of its items instead of neither. `lib.rs` names
/// `vyre_foundation::transform::grid_sync_split` in prose describing the
/// transform the oracle deliberately does not run, which a line scan would
/// report as the very defect the prose records as fixed.
fn references_in(text: &str, path: &Path) -> Result<Vec<Reference>, GateError> {
    let file = parse(text, path)?;
    let mut visitor = ReferenceScan {
        file: path.to_path_buf(),
        found: Vec::new(),
    };
    visitor.visit_file(&file);
    visitor.found.sort();
    visitor.found.dedup();
    Ok(visitor.found)
}

/// Collects the first segment after `vyre_foundation` in every path form.
struct ReferenceScan {
    file: PathBuf,
    found: Vec<Reference>,
}

impl ReferenceScan {
    /// Record the item a `vyre_foundation`-rooted path names.
    fn record(&mut self, item: &syn::Ident) {
        self.found.push(Reference {
            file: self.file.clone(),
            line: line_of(item),
            item: item.to_string(),
        });
    }

    /// Record every item a `use vyre_foundation::...` tree names.
    fn use_items(&mut self, tree: &syn::UseTree) {
        match tree {
            syn::UseTree::Path(path) => self.record(&path.ident),
            syn::UseTree::Name(name) => self.record(&name.ident),
            syn::UseTree::Rename(rename) => self.record(&rename.ident),
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.use_items(item);
                }
            }
            syn::UseTree::Glob(_) => {}
        }
    }
}

impl<'ast> Visit<'ast> for ReferenceScan {
    fn visit_item_use(&mut self, entry: &'ast syn::ItemUse) {
        if let syn::UseTree::Path(path) = &entry.tree {
            if path.ident == FOUNDATION_PATH {
                self.use_items(&path.tree);
            }
        }
        syn::visit::visit_item_use(self, entry);
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        let mut segments = path.segments.iter();
        if let (Some(first), Some(second)) = (segments.next(), segments.next()) {
            if first.ident == FOUNDATION_PATH {
                self.record(&second.ident);
            }
        }
        syn::visit::visit_path(self, path);
    }

    fn visit_macro(&mut self, entry: &'ast syn::Macro) {
        self.visit_path(&entry.path);
        syn::visit::visit_macro(self, entry);
    }
}

/// The one-based line a token begins on.
fn line_of(ident: &syn::Ident) -> u32 {
    u32::try_from(ident.span().start().line).unwrap_or(u32::MAX)
}

/// Parse one Rust source, naming the file that failed.
fn parse(text: &str, path: &Path) -> Result<syn::File, GateError> {
    syn::parse_file(text).map_err(|error| {
        GateError::new(
            format!("cannot parse `{}`: {error}", path.display()),
            "repair the source so it parses; a rule that skips a file it cannot read covers \
             nothing in it",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path expression, for driving the scanner without a whole crate.
    fn references(source: &str) -> Vec<String> {
        references_in(source, Path::new("probe.rs"))
            .expect("Fix: the probe source must parse.")
            .into_iter()
            .map(|reference| reference.item)
            .collect()
    }

    #[test]
    fn a_use_statement_contributes_every_item_it_names() {
        assert_eq!(
            references("use vyre_foundation::{ir::Program, validate, optimizer::Pass};"),
            vec!["ir".to_string(), "optimizer".into(), "validate".into()]
        );
    }

    #[test]
    fn an_inline_path_contributes_its_item() {
        assert_eq!(
            references("fn f() { vyre_foundation::transform::grid_sync_split(&mut p); }"),
            vec!["transform".to_string()]
        );
    }

    #[test]
    fn a_macro_invocation_contributes_its_item() {
        assert_eq!(
            references("vyre_foundation::diagnostic_conversions!(ReferenceError);"),
            vec!["diagnostic_conversions".to_string()]
        );
    }

    #[test]
    fn a_type_position_path_contributes_its_item() {
        assert_eq!(
            references("struct S { e: Option<vyre_foundation::lower::Plan> }"),
            vec!["lower".to_string()]
        );
    }

    #[test]
    fn prose_and_string_text_naming_a_transform_is_not_a_reference() {
        assert!(references(
            "//! calls vyre_foundation::transform::grid_sync_split\n\
             fn f() -> &'static str { \"vyre_foundation::optimizer\" }"
        )
        .is_empty());
    }

    #[test]
    fn a_renamed_import_contributes_the_source_name() {
        assert_eq!(
            references("use vyre_foundation::optimizer as opt;"),
            vec!["optimizer".to_string()]
        );
    }

    #[test]
    fn another_crates_module_of_the_same_name_is_not_a_reference() {
        assert!(references("use vyre_primitives::optimizer::Pass;").is_empty());
    }

    #[test]
    fn the_line_reported_is_the_line_the_item_is_named_on() {
        let found = references_in(
            "fn a() {}\nfn b() { vyre_foundation::optimizer::run(); }",
            Path::new("probe.rs"),
        )
        .expect("Fix: the probe source must parse.");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].line, 2);
    }

    #[test]
    fn every_recorded_item_is_published_by_the_crate() {
        let tree = Tree::open(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("Fix: the xtask package must sit one level under the workspace root."),
        )
        .expect("Fix: run this test inside a checkout.");
        let surface = foundation_surface(&tree).expect("Fix: the foundation surface must derive.");
        let stale: Vec<&&str> = SPECIFICATION_SURFACE
            .iter()
            .filter(|item| !surface.contains(**item))
            .collect();
        assert!(
            stale.is_empty(),
            "Fix: these recorded items are no longer published by {FOUNDATION}: {stale:?}"
        );
    }

    #[test]
    fn the_forbidden_half_of_the_surface_is_not_empty() {
        let tree = Tree::open(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("Fix: the xtask package must sit one level under the workspace root."),
        )
        .expect("Fix: run this test inside a checkout.");
        let surface = foundation_surface(&tree).expect("Fix: the foundation surface must derive.");
        let permitted: BTreeSet<&str> = SPECIFICATION_SURFACE.iter().copied().collect();
        for transform in [
            "optimizer",
            "transform",
            "lower",
            "schedule",
            "execution_plan",
        ] {
            assert!(
                surface.contains(transform),
                "Fix: `{transform}` must remain on the derived surface, or this rule stopped \
                 covering the transforms it exists for"
            );
            assert!(
                !permitted.contains(transform),
                "Fix: `{transform}` decides how a program runs and must not be readable by the \
                 oracle"
            );
        }
    }
}
