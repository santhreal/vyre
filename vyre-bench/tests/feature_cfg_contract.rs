//! Cargo feature declarations must agree with source-level feature guards.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::{Expr, Lit, Meta, Token};

fn feature_names_in_meta(meta: &Meta, names: &mut BTreeSet<String>) {
    match meta {
        Meta::NameValue(name_value) if name_value.path.is_ident("feature") => {
            if let Expr::Lit(expr) = &name_value.value {
                if let Lit::Str(feature) = &expr.lit {
                    names.insert(feature.value());
                }
            }
        }
        Meta::List(list) => {
            if let Ok(nested) =
                Punctuated::<Meta, Token![,]>::parse_terminated.parse2(list.tokens.clone())
            {
                for meta in nested {
                    feature_names_in_meta(&meta, names);
                }
            }
        }
        Meta::Path(_) | Meta::NameValue(_) => {}
    }
}

#[derive(Default)]
struct FeatureVisitor {
    names: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for FeatureVisitor {
    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        feature_names_in_meta(&attribute.meta, &mut self.names);
        syn::visit::visit_attribute(self, attribute);
    }
}

fn feature_names_in_source(source: &str) -> Result<BTreeSet<String>, syn::Error> {
    let file = syn::parse_file(source)?;
    let mut visitor = FeatureVisitor::default();
    visitor.visit_file(&file);
    Ok(visitor.names)
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path).expect("read source directory") {
            let path = entry.expect("read source entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// Prevents an unreachable benchmark module from hiding behind an undeclared Cargo feature.
#[test]
fn every_benchmark_feature_guard_is_declared() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest_source =
        fs::read_to_string(root.join("Cargo.toml")).expect("read vyre-bench manifest");
    let manifest: toml::Value =
        toml::from_str(&manifest_source).expect("parse vyre-bench manifest");
    let declared = manifest
        .get("features")
        .and_then(toml::Value::as_table)
        .expect("vyre-bench manifest has a features table")
        .keys()
        .filter(|name| name.as_str() != "default")
        .cloned()
        .collect::<BTreeSet<_>>();

    let mut used = BTreeSet::new();
    for path in rust_files(&root.join("src")) {
        let source = fs::read_to_string(&path).expect("read Rust source");
        used.extend(feature_names_in_source(&source).expect("parse Rust source"));
    }
    for target in manifest
        .get("bin")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
    {
        used.extend(
            target
                .get("required-features")
                .and_then(toml::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(toml::Value::as_str)
                .map(str::to_owned),
        );
    }

    assert_eq!(
        used, declared,
        "source guards and Cargo features diverged"
    );
}

/// Proves nested `cfg` and `cfg_attr` predicates cannot evade the feature inventory.
#[test]
fn nested_feature_guards_are_discovered() {
    let source = r#"
        #![cfg_attr(feature = "crate-docs", doc = "enabled")]
        #[cfg(any(feature = "first", all(unix, feature = "second")))]
        fn guarded() {}
    "#;
    assert_eq!(
        feature_names_in_source(source).unwrap(),
        BTreeSet::from([
            "crate-docs".to_string(),
            "first".to_string(),
            "second".to_string(),
        ])
    );
}

/// Prevents comments and string literals from manufacturing fake Cargo feature requirements.
#[test]
fn non_attribute_feature_text_is_ignored() {
    let source = r#"
        // #[cfg(feature = "commented-out")]
        const TEXT: &str = "cfg(feature = \"string-literal\")";
    "#;
    assert_eq!(feature_names_in_source(source).unwrap(), BTreeSet::new());
}
