//! `xtask lego-quick`  -  the fast pre-commit composition-boundary gate.
//!
//! One rule, over the staged diff by default: a Tier-3 dialect under
//! `vyre-libs/src/<dialect>/` does not import from a sibling dialect. Shared
//! plumbing two dialects both need lives above both of them, in
//! `crate::builder`, `crate::descriptor` or a primitive, so a sibling import
//! names a shared unit that was never hoisted.
//!
//! Two checks used to run here and no longer do.
//!
//! The large-file advisory was a third owner of one measurement. `file-size`
//! enforces the hard ceiling and `lego-audit` reports the 500-line review
//! prompt as a note. Reported here as a finding, it put 406 advisory rows into
//! a pinned defect count, which is how a pin starts ratcheting on the size of
//! the tree instead of on its defects.
//!
//! The raw-IR check forbade `Node::*` and `Expr::*` construction anywhere under
//! `vyre-libs/src`. The workspace composition policy defines a Category A
//! operation as a composition built over existing `Expr` and `Node` variants,
//! and places every Category A operation in `vyre-libs`, so the rule forbade
//! the thing the crate exists to do: 13616 sites across 166 files, held off by
//! a 203-row per-file exemption list that no file split or rename could
//! follow. What the policy forbids is reinventing a shape a registered
//! operation already emits, and that is measured by `lego-audit` checks 1, 2,
//! 6, 7 and 10, by `whats-similar` and by `dup-scan`. `vyre-lints` still ships
//! the strict lint for a consumer that wants it.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::use_paths::collect_use_paths;

const MAX_LEGO_QUICK_SOURCE_BYTES: u64 = 2_097_152;

/// Runs the fast composition boundary checks over the Rust sources of the tree.
pub struct LegoQuick;

impl Gate for LegoQuick {
    fn name(&self) -> &'static str {
        "lego-quick"
    }

    fn help(&self) -> &'static str {
        "Hold every dialect-to-dialect import in vyre-libs to an edge the manifest declares; --staged narrows to the staged set"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let root = workspace_root().ok_or_else(|| {
            GateError::new(
                "lego-quick found no git checkout of the vyre workspace to scan",
                "run it inside a checkout of the workspace",
            )
        })?;
        let files = if ctx.has("--staged") {
            staged_rust_files(&root).map_err(|error| {
                GateError::new(
                    format!("`git diff --cached --name-only` failed: {error}"),
                    "repair the git checkout, or run the gate without --staged to scan the whole tree",
                )
            })?
        } else {
            all_rust_files(&root)
        };
        if files.is_empty() {
            return Err(GateError::new(
                "lego-quick found no Rust source to scan, so it judged nothing",
                "run it without --staged to scan the whole tree",
            ));
        }

        let mut hits: Vec<Hit> = check_cross_dialect(&root, &files);
        hits.sort_by_key(|hit| (hit.file.clone(), hit.line, hit.category.clone()));

        let mut report = Report::clean();
        report.note(format!(
            "{} Rust file(s) scanned for sibling-dialect imports",
            files.len()
        ));
        for hit in &hits {
            report.find(Finding::at(
                &hit.file,
                hit.line,
                format!("{}: {}", hit.category, hit.message),
                hit.fix.clone(),
            ));
        }
        Ok(report)
    }
}

#[derive(Debug)]
struct Hit {
    file: String,
    line: u32,
    category: String,
    message: String,
    fix: String,
}

fn workspace_root() -> Option<PathBuf> {
    Some(crate::checkout::checkout_root())
}

fn staged_rust_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let output = Command::new("git")
        .args(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string();
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.ends_with(".rs") {
            continue;
        }
        let path = root.join(trimmed);
        if path.is_file() {
            out.push(path);
        }
    }
    Ok(out)
}

fn all_rust_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                ".git" | "target" | "target-codex" | "target-fusion-fix"
            )
        })
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path.to_path_buf());
        }
    }
    out
}

fn workspace_relative(path: &str, marker: &str) -> String {
    match path.find(marker) {
        Some(idx) => path[idx..].to_string(),
        None => path.to_string(),
    }
}

/// A dialect under `vyre-libs/src/<X>/` imports a sibling dialect `<Y>` only
/// where the manifest declares the edge.
///
/// Composing another dialect's operation is what the composition policy asks
/// for: attention composes matmul, and matmul is a `math` operation while
/// attention is an `nn` one. A blanket ban on sibling imports forbade that and
/// pointed the writer at `vyre-primitives`, which holds hardware contracts and
/// not shared plumbing. What is left to forbid is coupling nobody declared, so
/// the question is whether some feature that gates `X` enables some feature
/// that gates `Y`. Both sides are read at run time, from `lib.rs` and from the
/// manifest, so a new dialect or a new feature edge needs no edit here.
///
/// The importing side is the route to the file, not the gate on its dialect
/// module. `encoding/mod.rs` declares one file behind the four neural-network
/// features, so that file compiles only where `nn` does; judged by the
/// `encoding` gate alone, 13 of its imports read as coupling no feature enables.
/// A file on no route is not compiled in a production build, which is how a
/// `#[cfg(test)]` module file leaves this rule rather than through a name test.
fn check_cross_dialect(root: &Path, files: &[PathBuf]) -> Vec<Hit> {
    let libs = root.join("vyre-libs");
    let dialects = dialect_features(&libs.join("src"));
    if dialects.is_empty() {
        return Vec::new();
    }
    let closure = feature_closure(&libs.join("Cargo.toml"));
    let routes: BTreeMap<PathBuf, BTreeSet<String>> =
        structure_gate::source_scan::module_routes(&libs.join("src"))
            .into_iter()
            .map(|route| (route.path, route.features.into_iter().collect()))
            .collect();

    let mut out = Vec::new();
    for path in files {
        let path_str = path.to_string_lossy();
        let Some(idx) = path_str.find("vyre-libs/src/") else {
            continue;
        };
        let after = &path_str[idx + "vyre-libs/src/".len()..];
        let Some(this_dialect) = after.split('/').next() else {
            continue;
        };
        if !dialects.contains_key(this_dialect) {
            continue;
        }
        let Some(this_features) = routes.get(path.as_path()) else {
            continue;
        };
        let Ok(text) = read_text_bounded(path) else {
            continue;
        };
        let Ok(file) = syn::parse_file(&text) else {
            out.push(Hit {
                file: workspace_relative(&path_str, "vyre-libs/"),
                line: 0,
                category: "parse".to_string(),
                message: "failed to parse Rust source".to_string(),
                fix: "make the file syntactically valid before committing".to_string(),
            });
            continue;
        };
        for use_path in collect_use_paths(&file) {
            if use_path.is_public {
                continue;
            }
            for (other, other_features) in &dialects {
                if other == this_dialect {
                    continue;
                }
                if !use_path.imports_dialect(other) {
                    continue;
                }
                if edge_is_declared(this_features, other_features, &closure) {
                    continue;
                }
                out.push(Hit {
                    file: workspace_relative(&path_str, "vyre-libs/"),
                    line: use_path.line as u32,
                    category: "cross-dialect".to_string(),
                    message: format!(
                        "imports `{}` from dialect `{other}`, which no feature gating `{this_dialect}` enables",
                        use_path.segments.join("::"),
                    ),
                    fix: format!(
                        "declare the edge in vyre-libs/Cargo.toml, so a feature gating `{this_dialect}` enables one gating `{other}`, or hoist the shared unit above both dialects into `crate::builder` or `crate::descriptor`"
                    ),
                });
            }
        }
    }
    out
}

/// Whether some feature gating the importing dialect enables one gating the
/// imported dialect.
fn edge_is_declared(
    from: &BTreeSet<String>,
    to: &BTreeSet<String>,
    closure: &BTreeMap<String, BTreeSet<String>>,
) -> bool {
    from.iter().any(|feature| {
        closure
            .get(feature)
            .is_some_and(|reached| reached.iter().any(|name| to.contains(name)))
    })
}

/// Each dialect module under `vyre-libs/src/`, and the features that gate it.
///
/// The set is the tree: a module `lib.rs` declares behind a `#[cfg(feature =
/// ...)]` is a dialect, and one it declares unconditionally is shared plumbing
/// every dialect may reach. A hand-kept exclusion list stood here before and
/// named five modules, which is a second declaration of what `lib.rs` says.
fn dialect_features(libs_src: &Path) -> BTreeMap<String, BTreeSet<String>> {
    let Ok(text) = read_text_bounded(&libs_src.join("lib.rs")) else {
        return BTreeMap::new();
    };
    let Ok(file) = syn::parse_file(&text) else {
        return BTreeMap::new();
    };
    let mut out = BTreeMap::new();
    for item in &file.items {
        let syn::Item::Mod(item_mod) = item else {
            continue;
        };
        let features = cfg_features(&item_mod.attrs);
        if features.is_empty() {
            continue;
        }
        out.insert(item_mod.ident.to_string(), features);
    }
    out
}

/// Every `feature = "..."` name a `#[cfg]` attribute mentions.
///
/// `any`, `all` and `not` are flattened to the names they mention, because the
/// question is which features can reach the module, not the predicate that
/// admits it.
fn cfg_features(attrs: &[syn::Attribute]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for attr in attrs {
        if !attr.path().is_ident("cfg") {
            continue;
        }
        if let Ok(meta) = attr.parse_args::<syn::Meta>() {
            collect_cfg_features(&meta, &mut out);
        }
    }
    out
}

fn collect_cfg_features(meta: &syn::Meta, out: &mut BTreeSet<String>) {
    match meta {
        syn::Meta::NameValue(pair) if pair.path.is_ident("feature") => {
            if let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(name),
                ..
            }) = &pair.value
            {
                out.insert(name.value());
            }
        }
        syn::Meta::List(list) => {
            if let Ok(nested) = list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) {
                for inner in &nested {
                    collect_cfg_features(inner, out);
                }
            }
        }
        syn::Meta::Path(_) | syn::Meta::NameValue(_) => {}
    }
}

/// Transitive closure of the `vyre-libs` feature graph, one entry per feature.
///
/// An edge naming another crate (`vyre-primitives/math`) or an optional
/// dependency (`dep:regex`) leaves this crate and cannot gate a module in it,
/// so the walk stops there.
fn feature_closure(manifest: &Path) -> BTreeMap<String, BTreeSet<String>> {
    #[derive(serde::Deserialize)]
    struct FeatureManifest {
        #[serde(default)]
        features: BTreeMap<String, Vec<String>>,
    }

    let Ok(text) = read_text_bounded(manifest) else {
        return BTreeMap::new();
    };
    let Ok(parsed) = toml::from_str::<FeatureManifest>(&text) else {
        return BTreeMap::new();
    };
    let direct: BTreeMap<&str, Vec<&str>> = parsed
        .features
        .iter()
        .map(|(name, enabled)| {
            (
                name.as_str(),
                enabled.iter().map(String::as_str).collect::<Vec<&str>>(),
            )
        })
        .collect();

    direct
        .keys()
        .map(|start| {
            let mut reached: BTreeSet<String> = BTreeSet::new();
            let mut pending = vec![*start];
            while let Some(current) = pending.pop() {
                if !reached.insert(current.to_string()) {
                    continue;
                }
                for next in direct.get(current).into_iter().flatten() {
                    if next.contains('/') || next.starts_with("dep:") {
                        continue;
                    }
                    pending.push(next);
                }
            }
            ((*start).to_string(), reached)
        })
        .collect()
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    crate::output_arg::read_text_bounded(path, MAX_LEGO_QUICK_SOURCE_BYTES, "lego quick")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write(dir: &Path, rel: &str, body: &str) -> PathBuf {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    /// A crate whose `lib.rs` gates `math` behind `math-scan` and `parsing`
    /// behind `parsing`, with `edge` declaring the one edge from math to
    /// parsing.
    fn crate_fixture(dir: &Path) {
        write(
            dir,
            "vyre-libs/src/lib.rs",
            "#[cfg(feature = \"math-scan\")]\npub mod math;\n\
             #[cfg(feature = \"parsing\")]\npub mod parsing;\n\
             pub mod builder;\n",
        );
        write(
            dir,
            "vyre-libs/Cargo.toml",
            "[features]\n\
             \"math-scan\" = []\n\
             edge = [\"math-scan\", \"parsing\"]\n\
             parsing = [\"vyre-primitives/parsing\"]\n",
        );
    }

    /// Write a module file and the declaration that compiles it.
    ///
    /// A file no `mod` statement names is on no route, so no build reaches it
    /// and the rule does not judge it. A fixture that skipped the declaration
    /// made every case below pass for that reason rather than the one it names.
    fn write_module(dir: &Path, dialect: &str, name: &str, body: &str) -> PathBuf {
        let declaration = format!("vyre-libs/src/{dialect}/mod.rs");
        let mut declared = std::fs::read_to_string(dir.join(&declaration)).unwrap_or_default();
        declared.push_str(&format!("pub mod {name};\n"));
        write(dir, &declaration, &declared);
        write(dir, &format!("vyre-libs/src/{dialect}/{name}.rs"), body)
    }

    #[test]
    fn an_undeclared_dialect_import_is_a_finding() {
        let dir = TempDir::new().unwrap();
        crate_fixture(dir.path());
        let p = write_module(
            dir.path(),
            "math",
            "uses_parsing",
            "use crate::parsing::lexer;\nfn _f() {}\n",
        );
        let findings = check_cross_dialect(dir.path(), &[p]);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].category, "cross-dialect");
        assert!(findings[0].message.contains("parsing"));
    }

    /// WHY: composing another dialect's operation is what the composition
    /// policy asks for, and the manifest is where that coupling is declared. A
    /// check that flagged a declared edge would report the architecture working
    /// as designed, which is how a gate teaches a reader to ignore it.
    #[test]
    fn a_declared_dialect_edge_is_not_a_finding() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            "vyre-libs/src/lib.rs",
            "#[cfg(feature = \"edge\")]\npub mod math;\n\
             #[cfg(feature = \"parsing\")]\npub mod parsing;\n",
        );
        write(
            dir.path(),
            "vyre-libs/Cargo.toml",
            "[features]\nedge = [\"parsing\"]\nparsing = []\n",
        );
        let p = write_module(
            dir.path(),
            "math",
            "uses_parsing",
            "use crate::parsing::lexer;\nfn _f() {}\n",
        );
        assert!(check_cross_dialect(dir.path(), &[p]).is_empty());
    }

    /// WHY: the edge is transitive. `nn-inference` enables `nn-linear`, which
    /// enables `math-linalg`, and a walk that read only direct edges would call
    /// the matmul every linear layer composes an undeclared import.
    #[test]
    fn a_transitively_declared_edge_is_not_a_finding() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            "vyre-libs/src/lib.rs",
            "#[cfg(feature = \"outer\")]\npub mod math;\n\
             #[cfg(feature = \"parsing\")]\npub mod parsing;\n",
        );
        write(
            dir.path(),
            "vyre-libs/Cargo.toml",
            "[features]\nouter = [\"middle\"]\nmiddle = [\"parsing\"]\nparsing = []\n",
        );
        let p = write_module(
            dir.path(),
            "math",
            "uses_parsing",
            "use crate::parsing::lexer;\nfn _f() {}\n",
        );
        assert!(check_cross_dialect(dir.path(), &[p]).is_empty());
    }

    /// WHY: a nested declaration carries its own gate, and the importing side
    /// is the route to the file rather than the gate on its dialect. Judged by
    /// the dialect alone, one file that `encoding/mod.rs` declares behind the
    /// neural-network features produced 13 findings against imports every one
    /// of those features enables.
    #[test]
    fn a_nested_declaration_supplies_the_importing_features() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            "vyre-libs/src/lib.rs",
            "pub mod math;\n#[cfg(feature = \"parsing\")]\npub mod parsing;\n",
        );
        write(
            dir.path(),
            "vyre-libs/Cargo.toml",
            "[features]\nedge = [\"parsing\"]\nparsing = []\n",
        );
        write(
            dir.path(),
            "vyre-libs/src/math/mod.rs",
            "#[cfg(feature = \"edge\")]\npub mod uses_parsing;\n",
        );
        let p = write(
            dir.path(),
            "vyre-libs/src/math/uses_parsing.rs",
            "use crate::parsing::lexer;\nfn _f() {}\n",
        );
        assert!(check_cross_dialect(dir.path(), &[p]).is_empty());
    }

    /// WHY: a module only a test build reaches is not production source. The
    /// rule used to answer that question from the file name, which reported 6
    /// imports in one `#[cfg(test)]` module and would have missed the next one
    /// spelled differently.
    #[test]
    fn a_file_on_no_route_is_not_judged() {
        let dir = TempDir::new().unwrap();
        crate_fixture(dir.path());
        write(
            dir.path(),
            "vyre-libs/src/math/mod.rs",
            "#[cfg(test)]\nmod region_checks;\n",
        );
        let p = write(
            dir.path(),
            "vyre-libs/src/math/region_checks.rs",
            "use crate::parsing::lexer;\nfn _f() {}\n",
        );
        assert!(check_cross_dialect(dir.path(), &[p]).is_empty());
    }

    #[test]
    fn a_same_dialect_import_is_not_a_finding() {
        let dir = TempDir::new().unwrap();
        crate_fixture(dir.path());
        let p = write_module(
            dir.path(),
            "math",
            "uses_self",
            "use crate::math::reduce;\nfn _f() {}\n",
        );
        assert!(check_cross_dialect(dir.path(), &[p]).is_empty());
    }

    #[test]
    fn a_primitive_import_is_not_a_finding() {
        let dir = TempDir::new().unwrap();
        crate_fixture(dir.path());
        let p = write_module(
            dir.path(),
            "math",
            "uses_primitives",
            "use vyre_primitives::lane_grid;\nfn _f() {}\n",
        );
        assert!(check_cross_dialect(dir.path(), &[p]).is_empty());
    }

    /// WHY: a module `lib.rs` declares unconditionally is shared plumbing every
    /// dialect may reach, and the check has to read that from `lib.rs`. The
    /// exclusion list it replaced named five modules by hand and went stale
    /// whenever one was added or renamed.
    #[test]
    fn a_shared_module_is_not_a_dialect() {
        let dir = TempDir::new().unwrap();
        crate_fixture(dir.path());
        let dialects = dialect_features(&dir.path().join("vyre-libs/src"));
        assert!(dialects.contains_key("math"));
        assert!(dialects.contains_key("parsing"));
        assert!(!dialects.contains_key("builder"));
    }

    /// WHY: an edge that names another crate cannot gate a module in this one,
    /// and following it would let `vyre-primitives/parsing` stand in for the
    /// `parsing` feature and excuse an undeclared import.
    #[test]
    fn a_feature_edge_that_leaves_the_crate_is_not_followed() {
        let dir = TempDir::new().unwrap();
        crate_fixture(dir.path());
        let closure = feature_closure(&dir.path().join("vyre-libs/Cargo.toml"));
        let parsing = closure.get("parsing").expect("parsing feature is closed");
        assert_eq!(
            parsing.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["parsing"]
        );
        let edge = closure.get("edge").expect("edge feature is closed");
        assert!(edge.contains("parsing"));
        assert!(edge.contains("math-scan"));
    }
}
