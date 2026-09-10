//! One crate, one definition per type name.
//!
//! A crate that declares `pub struct Foo` in two of its modules has two
//! definitions competing for one name. Every reader who resolves `Foo` from an
//! import list, a review diff or a search result gets whichever one they found
//! first, and a caller wired to the wrong one compiles. `vyre-foundation`
//! carried five such concepts at once: `ExprArena`, `ScalarLiteral`,
//! `CatalogBundle`, `AtomicOrdering`, and the `NodeId`/`RegionId`/`ExprId`
//! newtypes, each second definition holding a subset of the first.
//!
//! A second public *path* to one definition is a different thing and is not
//! reported. `pub type StreamShardError = stream_shard::StreamShardError;` in
//! `vyre-driver-wgpu` re-exports one definition under a second name, so the
//! keywords read here are [`TYPE_KEYWORDS`] and never `type`.
//!
//! Only a declaration written at column zero of a crate's `src/` tree counts. A
//! name declared inside an inline `mod` block is namespaced by that block in
//! the file the reader already has open, which is how the two `pub trait
//! Sealed` declarations in `vyre-driver` stay legible.
//!
//! The name space is read from the tree on every run, so a concept that grows a
//! second definition tomorrow is reported without an edit here.

use std::collections::BTreeMap;
use std::path::Path;

use crate::backend_vocabulary::is_test_source;
use crate::cfg_test::{cfg_test_line_mask, test_gated_module_files};
use crate::module_layout::CrateRoot;
use crate::source_scan::{
    is_word_byte, mask_comments_and_strings, rust_sources_with_text, SourceText,
};

/// The item keywords that introduce a type definition.
///
/// `type` is absent on purpose: an alias names a definition that already
/// exists, so a second alias is a second path rather than a second definition.
pub const TYPE_KEYWORDS: [&str; 4] = ["struct", "enum", "union", "trait"];

/// One public type declaration, as the grouping rule reads it.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Declaration {
    /// Identifier of the crate that compiles the declaration.
    pub owner: String,
    /// The declared type name.
    pub name: String,
    /// Checkout-relative file and one-based line.
    pub site: String,
}

/// Every type name a single crate declares publicly more than once.
///
/// A source under a crate's `src/` tree that could not be read is reported
/// rather than skipped: a name declared in a file nothing judged is a name this
/// rule would call unique.
#[must_use]
pub fn duplicate_public_type_name_failures(root: &Path, crate_roots: &[CrateRoot]) -> Vec<String> {
    let mut failures = Vec::new();
    let mut found = Vec::new();
    let test_gated = test_gated_module_files(root);

    for source in rust_sources_with_text(root) {
        let Some(owner) = owning_crate(source.path(), crate_roots) else {
            continue;
        };
        if is_test_source(source.path()) || test_gated.contains(source.path()) {
            continue;
        }
        match source {
            SourceText::Read { path, text } => found.extend(declarations(owner, &path, &text)),
            SourceText::Unread { path, reason } => failures.push(format!(
                "{path} was not read: {reason}; the duplicate type name gate cannot report an \
                 unread source as clean"
            )),
        }
    }

    failures.extend(duplicate_reports(&found));
    failures
}

/// Every public type declaration `text` writes at column zero.
///
/// Test text is excluded twice over, because a fixture holds names a production
/// build never compiles: lines inside a `#[cfg(test)]` item are dropped through
/// [`cfg_test_line_mask`], and comments and string literals are blanked through
/// [`mask_comments_and_strings`], which is what keeps the `pub struct Program`
/// written inside an `xtask` gate fixture string out of the count.
///
/// Line numbers survive both, so a report names the line the reader opens.
#[must_use]
pub fn declarations(owner: &str, path: &str, text: &str) -> Vec<Declaration> {
    let gated = cfg_test_line_mask(text);
    mask_comments_and_strings(text)
        .lines()
        .enumerate()
        .filter(|(index, _)| !gated.get(*index).copied().unwrap_or(false))
        .filter_map(|(index, line)| {
            declared_public_type_name(line).map(|name| Declaration {
                owner: owner.to_string(),
                name: name.to_string(),
                site: format!("{path}:{}", index + 1),
            })
        })
        .collect()
}

/// One report per crate-and-name pair that `found` declares more than once.
///
/// Grouped per crate rather than per workspace: two crates may both define a
/// `Config`, and a reader resolves those through the crate name imported from.
#[must_use]
pub fn duplicate_reports(found: &[Declaration]) -> Vec<String> {
    let mut sites: BTreeMap<(&str, &str), Vec<&str>> = BTreeMap::new();
    for declaration in found {
        sites
            .entry((&declaration.owner, &declaration.name))
            .or_default()
            .push(&declaration.site);
    }
    sites
        .into_iter()
        .filter(|(_, sites)| sites.len() > 1)
        .map(|((owner, name), sites)| {
            format!(
                "{owner} declares the public type `{name}` at {}; one crate resolves one name to \
                 one definition, so merge them or rename each to state what it is",
                sites.join(", ")
            )
        })
        .collect()
}

/// Identifier of the crate whose `src/` tree holds `path`.
///
/// The longest matching crate directory wins, so a crate nested under another
/// one is credited with its own files rather than its parent's.
fn owning_crate<'a>(path: &str, crate_roots: &'a [CrateRoot]) -> Option<&'a str> {
    crate_roots
        .iter()
        .filter(|root| {
            path.strip_prefix(root.directory.as_str())
                .is_some_and(|rest| rest.starts_with("/src/"))
        })
        .max_by_key(|root| root.directory.len())
        .map(|root| root.ident.as_str())
}

/// Name of the type `line` declares publicly at column zero.
fn declared_public_type_name(line: &str) -> Option<&str> {
    let rest = after_word(line, "pub")?;
    let rest = after_word(rest, "unsafe").unwrap_or(rest);
    let rest = TYPE_KEYWORDS
        .into_iter()
        .find_map(|keyword| after_word(rest, keyword))?;
    let end = rest
        .bytes()
        .position(|byte| !is_word_byte(byte))
        .unwrap_or(rest.len());
    (end > 0).then(|| &rest[..end])
}

/// `text` past a leading `word`, when whitespace follows it.
///
/// The boundary check is what keeps `pub` from matching `publish` and `struct`
/// from matching `structure`.
fn after_word<'a>(text: &'a str, word: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(word)?;
    rest.starts_with(char::is_whitespace)
        .then(|| rest.trim_start())
}

#[cfg(test)]
mod tests {
    use super::{declarations, duplicate_reports};

    #[test]
    fn two_files_of_one_crate_declaring_one_name_is_reported() {
        let mut found = declarations("crate-a", "crate-a/src/first.rs", "pub struct Planted;\n");
        found.extend(declarations(
            "crate-a",
            "crate-a/src/second.rs",
            "pub enum Planted {}\n",
        ));

        let reports = duplicate_reports(&found);

        assert_eq!(reports.len(), 1, "{reports:?}");
        assert!(
            reports[0].contains("crate-a/src/first.rs:1, crate-a/src/second.rs:1"),
            "{reports:?}"
        );
    }

    #[test]
    fn two_crates_declaring_one_name_is_not_reported() {
        let mut found = declarations("crate-a", "crate-a/src/lib.rs", "pub struct Config;\n");
        found.extend(declarations(
            "crate-b",
            "crate-b/src/lib.rs",
            "pub struct Config;\n",
        ));

        assert_eq!(duplicate_reports(&found), Vec::<String>::new());
    }

    #[test]
    fn every_type_keyword_is_read_and_type_alias_is_not() {
        let text = concat!(
            "pub struct One;\n",
            "pub enum Two {}\n",
            "pub union Three { a: u8 }\n",
            "pub unsafe trait Four {}\n",
            "pub type Five = One;\n",
        );

        let found = declarations("c", "c/src/lib.rs", text);
        let names: Vec<&str> = found
            .iter()
            .map(|declaration| declaration.name.as_str())
            .collect();

        assert_eq!(names, ["One", "Two", "Three", "Four"]);
    }

    #[test]
    fn an_indented_declaration_is_namespaced_by_its_block() {
        let text = "pub mod sealed {\n    pub trait Sealed {}\n}\n";

        assert_eq!(declarations("c", "c/src/lib.rs", text), Vec::new());
    }

    #[test]
    fn test_text_declares_nothing() {
        let text = concat!(
            "pub const FIXTURE: &str = \"pub struct Quoted;\";\n",
            "/// pub struct Documented;\n",
            "#[cfg(test)]\n",
            "mod tests {\n",
            "pub struct Gated;\n",
            "}\n",
        );

        assert_eq!(declarations("c", "c/src/lib.rs", text), Vec::new());
    }
}
