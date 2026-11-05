//! A declared variant list cannot fall behind its enum.
//!
//! An enum that publishes `const ALL: [Self; N]` states that the array is every
//! variant. Nothing in the language holds that open. Adding a variant breaks
//! each exhaustive `match` on the type, so the author is walked to every arm
//! that needs a decision, and the array is not one of them: it keeps compiling
//! with its old contents at its old length. Every sweep driven by that array
//! then skips the new variant in silence, which is the same failure as having
//! no sweep, arriving without a red run to announce it.
//!
//! The gate compares the array against the variants of the enum it belongs to
//! and reports the difference in either direction. A variant absent from the
//! array is the staleness above. A name in the array that the enum does not
//! declare cannot compile, so the case that survives here is a duplicate entry,
//! which is how a list stays the right length while dropping a member: bump the
//! count, copy a line, and the sweep is one variant short with nothing to see.
//!
//! Judgment needs both halves in view, so a list is judged against an enum
//! declared in the same file. A list separated from its enum is reported rather
//! than skipped: the alternative resolves a bare type name across a workspace
//! where two crates may spell one, and a gate that answers from the wrong enum
//! is worse than one that asks for the two to be kept together.
//!
//! A list whose entries are not bare variant paths is not this contract. An
//! enum whose variants carry fields cannot be enumerated by path at all, and
//! the array is holding constructed values rather than a variant roster.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use syn::spanned::Spanned;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Reports a `const ALL` array that is not exactly its enum's variants.
pub struct VariantListClosure;

impl crate::gate::GateBehavior for VariantListClosure {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let sources = tree.all_rust();
        let mut report = Report::clean();
        let mut declared = 0usize;
        for path in &sources {
            let text = tree.read(path)?;
            if !text.contains("const ALL") {
                continue;
            }
            let Ok(file) = syn::parse_file(&text) else {
                continue;
            };
            let mut found = FileItems::default();
            found.walk(&file.items);
            declared += found.lists.len();
            for finding in closure_findings(path, &found) {
                report.find(finding);
            }
        }
        report.cover_complete("declared variant lists", declared);
        Ok(report)
    }
}

/// One `const ALL: [Self; N]` array and the variant names it holds.
struct DeclaredList {
    /// The type the `impl` block names.
    owner: String,
    /// The line the constant starts on.
    line: u32,
    /// The variant names the array entries spell, in array order.
    entries: Vec<String>,
}

/// Enums and variant lists collected from one file, including nested modules.
#[derive(Default)]
struct FileItems {
    /// Enum name to its variants in declaration order.
    enums: BTreeMap<String, Vec<String>>,
    /// Every `const ALL` array found in the file.
    lists: Vec<DeclaredList>,
}

impl FileItems {
    /// Collects enums and lists from an item list and every module inside it.
    fn walk(&mut self, items: &[syn::Item]) {
        for item in items {
            match item {
                syn::Item::Enum(declaration) => {
                    self.enums.insert(
                        declaration.ident.to_string(),
                        declaration
                            .variants
                            .iter()
                            .map(|variant| variant.ident.to_string())
                            .collect(),
                    );
                }
                syn::Item::Impl(block) => self.collect_impl(block),
                syn::Item::Mod(module) => {
                    if let Some((_, nested)) = &module.content {
                        self.walk(nested);
                    }
                }
                _ => {}
            }
        }
    }

    /// Records the `const ALL` of one inherent `impl`, when it is a variant
    /// roster. A trait implementation states no roster of its own, and a
    /// generic or reference self type has no single enum to compare against.
    fn collect_impl(&mut self, block: &syn::ItemImpl) {
        if block.trait_.is_some() {
            return;
        }
        let Some(owner) = plain_type_name(&block.self_ty) else {
            return;
        };
        for item in &block.items {
            let syn::ImplItem::Const(constant) = item else {
                continue;
            };
            if constant.ident != "ALL" {
                continue;
            }
            let syn::Expr::Array(array) = &constant.expr else {
                continue;
            };
            let entries: Option<Vec<String>> = array
                .elems
                .iter()
                .map(|element| variant_entry(element, &owner))
                .collect();
            // An entry that is not a bare variant path leaves this constant
            // unjudged, and leaves the rest of the block still to read.
            let Some(entries) = entries else {
                continue;
            };
            self.lists.push(DeclaredList {
                owner: owner.clone(),
                line: u32::try_from(constant.span().start().line).unwrap_or(u32::MAX),
                entries,
            });
        }
    }
}

/// The name of a type written as a bare path, such as `ScalarFormat`.
fn plain_type_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let last = path.path.segments.last()?;
    if !matches!(last.arguments, syn::PathArguments::None) {
        return None;
    }
    Some(last.ident.to_string())
}

/// The variant an array entry names, when the entry is `Self::V` or `Owner::V`.
fn variant_entry(expr: &syn::Expr, owner: &str) -> Option<String> {
    let syn::Expr::Path(path) = expr else {
        return None;
    };
    let segments: Vec<String> = path
        .path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    let [qualifier, variant] = segments.as_slice() else {
        return None;
    };
    if qualifier.as_str() != "Self" && qualifier.as_str() != owner {
        return None;
    }
    Some(variant.clone())
}

/// The differences between each list and the enum it belongs to.
fn closure_findings(path: &Path, found: &FileItems) -> Vec<Finding> {
    let mut findings = Vec::new();
    for list in &found.lists {
        let Some(variants) = found.enums.get(&list.owner) else {
            findings.push(Finding::at(
                PathBuf::from(path),
                list.line,
                format!(
                    "`{}::ALL` names variants of an enum this file does not declare",
                    list.owner
                ),
                "keep a variant list in the file that declares its enum, so adding a variant and \
                 extending the list are the same edit and one gate can compare them",
            ));
            continue;
        };
        let listed: BTreeSet<&String> = list.entries.iter().collect();
        let missing: Vec<&String> = variants
            .iter()
            .filter(|variant| !listed.contains(variant))
            .collect();
        if !missing.is_empty() {
            findings.push(Finding::at(
                PathBuf::from(path),
                list.line,
                format!(
                    "`{}::ALL` omits {}",
                    list.owner,
                    join_names(missing.iter().map(|name| name.as_str()))
                ),
                "add the variant to the list; every sweep driven by this array skips a variant it \
                 does not name, and the compiler does not report the omission",
            ));
        }
        let mut seen = BTreeSet::new();
        let duplicates: Vec<&str> = list
            .entries
            .iter()
            .filter(|entry| !seen.insert(*entry))
            .map(String::as_str)
            .collect();
        if !duplicates.is_empty() {
            findings.push(Finding::at(
                PathBuf::from(path),
                list.line,
                format!(
                    "`{}::ALL` repeats {}",
                    list.owner,
                    join_names(duplicates.iter().copied())
                ),
                "give each entry its own variant; a repeated entry holds the array length while \
                 dropping a member, which is how a list passes its length check and still sweeps \
                 one variant short",
            ));
        }
    }
    findings
}

/// Names joined for a message, in backticks, comma separated.
fn join_names<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn findings(source: &str) -> Vec<String> {
        let file = syn::parse_file(source).expect("fixture must parse as Rust");
        let mut found = FileItems::default();
        found.walk(&file.items);
        closure_findings(Path::new("fixture.rs"), &found)
            .into_iter()
            .map(|finding| finding.message)
            .collect()
    }

    #[test]
    fn a_complete_list_is_clean() {
        assert!(findings(
            "enum Kind { A, B }
             impl Kind { pub const ALL: [Self; 2] = [Self::A, Self::B]; }"
        )
        .is_empty());
    }

    #[test]
    fn a_variant_the_list_omits_is_a_finding() {
        let reported = findings(
            "enum Kind { A, B, C }
             impl Kind { pub const ALL: [Self; 2] = [Self::A, Self::B]; }",
        );
        assert_eq!(reported, vec!["`Kind::ALL` omits `C`"]);
    }

    #[test]
    fn a_repeated_entry_is_a_finding_even_at_the_right_length() {
        let reported = findings(
            "enum Kind { A, B, C }
             impl Kind { pub const ALL: [Self; 3] = [Self::A, Self::B, Self::B]; }",
        );
        assert_eq!(
            reported,
            vec!["`Kind::ALL` omits `C`", "`Kind::ALL` repeats `B`"]
        );
    }

    #[test]
    fn the_enum_name_may_qualify_the_entries() {
        assert!(findings(
            "enum Kind { A, B }
             impl Kind { pub const ALL: [Self; 2] = [Kind::A, Kind::B]; }"
        )
        .is_empty());
    }

    #[test]
    fn a_list_separated_from_its_enum_is_reported() {
        let reported = findings("impl Kind { pub const ALL: [Self; 1] = [Self::A]; }");
        assert_eq!(
            reported,
            vec!["`Kind::ALL` names variants of an enum this file does not declare"]
        );
    }

    #[test]
    fn a_nested_module_is_judged_like_the_file_root() {
        let reported = findings(
            "mod inner {
                 enum Kind { A, B }
                 impl Kind { pub const ALL: [Self; 1] = [Self::A]; }
             }",
        );
        assert_eq!(reported, vec!["`Kind::ALL` omits `B`"]);
    }

    #[test]
    fn a_trait_implementation_declares_no_roster() {
        assert!(findings(
            "enum Kind { A, B }
             impl Listed for Kind { const ALL: [Self; 1] = [Self::A]; }"
        )
        .is_empty());
    }

    #[test]
    fn an_array_of_constructed_values_is_not_a_roster() {
        assert!(findings(
            "enum Kind { A(u8), B(u8) }
             impl Kind { pub const ALL: [Self; 1] = [Self::A(0)]; }"
        )
        .is_empty());
    }

    #[test]
    fn an_unrelated_const_all_is_not_a_roster() {
        assert!(findings(
            "enum Kind { A, B }
             impl Kind { pub const ALL: [Self; 2] = [Self::A, Other::B]; }"
        )
        .is_empty());
    }
}
