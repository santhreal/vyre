//! Shared Rust `use`-tree parsing for repository architecture audits.

use syn::spanned::Spanned;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UsePath {
    pub(crate) segments: Vec<String>,
    pub(crate) line: usize,
    pub(crate) is_public: bool,
}

impl UsePath {
    pub(crate) fn imports_dialect(&self, other_name: &str) -> bool {
        matches!(
            self.segments.as_slice(),
            [first, second, ..]
                if (first == "crate" || first == "vyre_libs") && second == other_name
        )
    }
}

pub(crate) fn collect_use_paths(file: &syn::File) -> Vec<UsePath> {
    let mut collector = UsePathCollector::default();
    syn::visit::visit_file(&mut collector, file);
    collector.paths
}
pub(crate) fn is_test_source_path(path: &std::path::Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == "tests")
        || path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .is_some_and(|stem| {
                stem == "test_support"
                    || stem == "test_helpers"
                    || stem.starts_with("test_")
                    || stem.ends_with("_test")
            })
}

fn is_test_only(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attribute| {
        attribute.path().is_ident("cfg")
            && attribute.meta.require_list().is_ok_and(|list| {
                list.tokens.to_string().split_whitespace().any(|token| {
                    token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric()) == "test"
                })
            })
    })
}

#[derive(Default)]
struct UsePathCollector {
    paths: Vec<UsePath>,
}

impl<'ast> syn::visit::Visit<'ast> for UsePathCollector {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if !is_test_only(&item.attrs) {
            syn::visit::visit_item_mod(self, item);
        }
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if !is_test_only(&item.attrs) {
            collect_use_tree(
                &item.tree,
                &mut Vec::new(),
                !matches!(item.vis, syn::Visibility::Inherited),
                &mut self.paths,
            );
        }
    }
}

fn collect_use_tree(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    is_public: bool,
    out: &mut Vec<UsePath>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_use_tree(&path.tree, prefix, is_public, out);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            push_use_path(prefix, &name.ident, name.span(), is_public, out);
        }
        syn::UseTree::Rename(rename) => {
            push_use_path(prefix, &rename.ident, rename.span(), is_public, out);
        }
        syn::UseTree::Glob(glob) => {
            let mut segments = prefix.clone();
            segments.push("*".to_string());
            out.push(UsePath {
                segments,
                line: glob.span().start().line,
                is_public,
            });
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_use_tree(item, prefix, is_public, out);
            }
        }
    }
}

fn push_use_path(
    prefix: &[String],
    ident: &syn::Ident,
    span: proc_macro2::Span,
    is_public: bool,
    out: &mut Vec<UsePath>,
) {
    let mut segments = prefix.to_vec();
    segments.push(ident.to_string());
    out.push(UsePath {
        segments,
        line: span.start().line,
        is_public,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This regression test keeps test fixtures from appearing as production cross-dialect dependencies.
    #[test]
    fn test_only_modules_do_not_contribute_use_paths() {
        let file = syn::parse_file(
            "#[cfg(test)] mod tests { use crate::other_dialect::fixture; }\nuse crate::own::runtime;",
        )
        .expect("Fix: architecture audit fixture must parse");
        let paths = collect_use_paths(&file);
        assert_eq!(
            paths,
            vec![UsePath {
                segments: vec!["crate".into(), "own".into(), "runtime".into()],
                line: 2,
                is_public: false,
            }]
        );
    }

    /// This boundary test keeps an individually gated import out of production architecture findings.
    #[test]
    fn test_only_use_items_are_ignored() {
        let file = syn::parse_file(
            "#[cfg(any(test, feature = \"fixtures\"))] use crate::other::fixture;\nuse crate::own::runtime;",
        )
        .expect("Fix: architecture audit fixture must parse");
        let paths = collect_use_paths(&file);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].segments, ["crate", "own", "runtime"]);
        assert!(!paths[0].is_public);
    }

    /// Public facades remain visible to callers but do not represent private cross-dialect reach-through.
    #[test]
    fn public_visibility_is_preserved_for_architecture_audits() {
        let file = syn::parse_file("pub use crate::scan::{Pipeline, ScanResult};")
            .expect("Fix: architecture audit fixture must parse");
        let paths = collect_use_paths(&file);
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|path| path.is_public));
    }

    /// This path-classification test keeps detached test helper files out of production dependency audits.
    #[test]
    fn test_source_paths_are_excluded_without_hiding_production_modules() {
        assert!(is_test_source_path(std::path::Path::new(
            "vyre-libs/src/scan/test_helpers.rs"
        )));
        assert!(is_test_source_path(std::path::Path::new(
            "vyre-libs/src/parsing/tests/contracts.rs"
        )));
        assert!(!is_test_source_path(std::path::Path::new(
            "vyre-libs/src/parsing/lexer.rs"
        )));
    }
}
