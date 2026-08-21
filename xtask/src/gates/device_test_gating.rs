//! A test that acquires a real device is compiled only where the device is.
//!
//! Acquiring a concrete backend on a runner that has no driver is not a
//! recoverable error. `CudaBackend::acquire` reaches cudarc, which aborts the
//! process from inside the dependency, so one ungated test file turns every
//! hosted matrix leg red with a panic that names a third-party source line and
//! not the test that caused it. That is what
//! `vyre-bench/tests/release_macro_cuda_live.rs` did: it carried a
//! `cfg(not(target_os = "macos"))` guard, which is a compile-time fact about
//! the dependency graph and says nothing about whether a device is present.
//!
//! The rule: test code calls a backend constructor only where the test is
//! admitted to run on hardware. Two admissions count. `feature = "device-tests"`
//! is the one `gpu-parity.yml` turns on, on the runners that have the device.
//! `#[ignore]` is the other: an ignored test is not run by a default `cargo
//! test`, so it cannot abort a hosted leg, and the measurement instruments in
//! the CUDA driver are invoked deliberately with `--ignored`.
//!
//! The roster is read from source: every `pub struct *Backend` declared by a
//! `vyre-driver-*` member. Adding a backend crate extends the rule without
//! anyone editing this file. `vyre-driver-reference` is excluded because it is
//! the CPU parity oracle and acquiring it needs no hardware.
//!
//! Naming a backend type is not acquiring one: a helper that takes
//! `&CudaBackend`, an inherent `impl` block, and a doc link all mention the
//! type without touching hardware, so the signature is the constructor call.
//!
//! What this does not catch: a test that reaches a device through a helper in
//! another file that hides the constructor, or one that spawns a workspace
//! binary and lets the child acquire. The gate sees syntax, not call graphs,
//! and the subcommand that acquires is an argv fact no source scan can settle.
//!
//! The other half of the rule is that the admission has to lead somewhere. A
//! test moved behind `device-tests` in a package no lane builds with that
//! feature is not gated, it is deleted, and every remaining lane stays green
//! while the coverage is gone. So every member declaring the feature must be
//! named by a workflow step that turns it on.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use quote::ToTokens;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// The feature that admits a device-acquiring test.
const FEATURE: &str = "device-tests";

/// The CPU parity oracle. Acquiring it needs no hardware.
const CPU_ORACLE_CRATE: &str = "vyre-driver-reference";

/// Test code acquires a concrete backend only where hardware is admitted.
pub struct DeviceTestGating;

impl crate::gate::GateBehavior for DeviceTestGating {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let backends = backend_roster(&tree)?;
        report.note(format!(
            "{} concrete backend type(s) in the roster: {}",
            backends.len(),
            backends.iter().cloned().collect::<Vec<_>>().join(", ")
        ));
        if backends.is_empty() {
            report.find(Finding::new(
                "no concrete backend type was found in any vyre-driver-* member",
                "this gate reads `pub struct *Backend` from the driver crates; if a backend was \
                 renamed, teach the roster the new shape rather than leaving the rule vacuous",
            ));
            return Ok(report);
        }

        let sources = tree.all_rust();
        report.cover_complete("rust source files", sources.len());
        let mut parsed = BTreeMap::new();
        for path in &sources {
            if let Ok(file) = syn::parse_file(&tree.read(path)?) {
                parsed.insert(path.clone(), file);
            }
        }
        let admitted_files = admitted_closure(&parsed);
        for path in &sources {
            let Some(file) = parsed.get(path) else {
                continue;
            };
            if admitted_files.contains(path) {
                continue;
            }
            let display = path.display().to_string();
            let in_test = scan::is_test_tree(path);
            for name in ungated_acquisitions(&file.items, in_test, &backends) {
                report.find(Finding::new(
                    format!("{display}: test code acquires {name} with no hardware admission"),
                    format!(
                        "put the test behind `#[cfg(feature = \"{FEATURE}\")]`, or the whole file \
                         behind `#![cfg(feature = \"{FEATURE}\")]`, so it compiles on the runner \
                         that has the device instead of aborting on the one that does not; a \
                         measurement instrument run by hand takes `#[ignore]` instead"
                    ),
                ));
            }
        }
        for finding in unenabled_admissions(&tree)? {
            report.find(finding);
        }
        Ok(report)
    }
}

/// Every file the admission reaches, following `mod` declarations.
///
/// A module file carries no attributes of its own. `tests/foo.rs` gates the
/// whole target with one inner attribute and `tests/foo/bar.rs` inherits it,
/// so reading each file in isolation reports a submodule that cannot be
/// compiled without the feature at all. That reading also pushes the tree
/// toward restating the attribute in every submodule, which is a second copy
/// of one fact and drifts the moment a root's gate changes.
///
/// Both directions count: a root whose own attributes admit passes admission
/// to everything it declares, and an admitted `mod` item passes it to that one
/// child even when the parent is ungated.
fn admitted_closure(parsed: &BTreeMap<PathBuf, syn::File>) -> BTreeSet<PathBuf> {
    let mut admitted_files = BTreeSet::new();
    let mut frontier: Vec<PathBuf> = parsed
        .iter()
        .filter(|(_, file)| admitted(&file.attrs))
        .map(|(path, _)| path.clone())
        .collect();
    for (path, file) in parsed {
        for child in declared_modules(parsed, path, &file.items, true) {
            frontier.push(child);
        }
    }
    while let Some(path) = frontier.pop() {
        if !admitted_files.insert(path.clone()) {
            continue;
        }
        let Some(file) = parsed.get(&path) else {
            continue;
        };
        frontier.extend(declared_modules(parsed, &path, &file.items, false));
    }
    admitted_files
}

/// The files `items` declares as modules, resolved against the parsed tree.
///
/// `admitted_only` restricts the answer to `mod` items the admission governs
/// directly, which is how a gated declaration inside an ungated parent is
/// found. Otherwise every declaration is followed, because the parent is
/// already admitted and passes that on.
fn declared_modules(
    parsed: &BTreeMap<PathBuf, syn::File>,
    parent: &Path,
    items: &[syn::Item],
    admitted_only: bool,
) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for item in items {
        let syn::Item::Mod(module) = item else {
            continue;
        };
        if let Some((_, inner)) = &module.content {
            if !admitted_only || admitted(&module.attrs) {
                found.extend(declared_modules(parsed, parent, inner, false));
            }
            continue;
        }
        if admitted_only && !admitted(&module.attrs) {
            continue;
        }
        if let Some(path) = module_file(parsed, parent, module) {
            found.push(path);
        }
    }
    found
}

/// The file a `mod name;` in `parent` names.
///
/// A crate root, a `mod.rs` and a `#[path]` attribute each place children in a
/// different directory, and a test target root is read both ways in this tree.
/// Every shape is offered to the parsed set and the one that exists answers,
/// so the resolver never has to decide which convention a target follows.
fn module_file(
    parsed: &BTreeMap<PathBuf, syn::File>,
    parent: &Path,
    module: &syn::ItemMod,
) -> Option<PathBuf> {
    let dir = parent.parent().unwrap_or(Path::new(""));
    let is_root = parent
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|name| matches!(name, "mod.rs" | "lib.rs" | "main.rs"));
    let nested = parent
        .file_stem()
        .filter(|_| !is_root)
        .map(|stem| dir.join(stem));
    let bases: Vec<PathBuf> = std::iter::once(dir.to_path_buf()).chain(nested).collect();
    if let Some(declared) = path_attribute(&module.attrs) {
        return bases
            .iter()
            .map(|base| normalize(&base.join(&declared)))
            .find(|candidate| parsed.contains_key(candidate));
    }
    let name = module.ident.to_string();
    bases
        .iter()
        .flat_map(|base| {
            [
                base.join(format!("{name}.rs")),
                base.join(&name).join("mod.rs"),
            ]
        })
        .find(|candidate| parsed.contains_key(candidate))
}

/// The literal of a `#[path = "..."]` attribute.
fn path_attribute(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().find_map(|attr| {
        if !attr.path().is_ident("path") {
            return None;
        }
        let syn::Meta::NameValue(meta) = &attr.meta else {
            return None;
        };
        let syn::Expr::Lit(literal) = &meta.value else {
            return None;
        };
        match &literal.lit {
            syn::Lit::Str(text) => Some(text.value()),
            _ => None,
        }
    })
}

/// A path with `.` and `..` components resolved lexically.
fn normalize(path: &Path) -> PathBuf {
    let mut parts: Vec<std::ffi::OsString> = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                parts.pop();
            }
            other => parts.push(other.as_os_str().to_owned()),
        }
    }
    parts.iter().collect()
}

/// Every member declaring the admission feature is built with it somewhere.
fn unenabled_admissions(tree: &Tree) -> Result<Vec<Finding>, GateError> {
    let mut declaring = Vec::new();
    for member in tree.member_manifests()? {
        if member
            .manifest
            .get("features")
            .and_then(toml::Value::as_table)
            .is_some_and(|features| features.contains_key(FEATURE))
        {
            declaring.push(member.name);
        }
    }
    let enabling = enabling_lanes(tree)?;
    Ok(declaring
        .into_iter()
        .filter(|package| !enabling.contains(package))
        .map(|package| {
            Finding::new(
                format!(
                    "`{package}` declares `{FEATURE}` and no workflow step builds it with that \
                     feature"
                ),
                format!(
                    "add a step on a runner that owns a device running `./cargo_full test -p \
                     {package} --features {FEATURE}`, or delete the feature and the tests behind \
                     it rather than leaving them uncompiled everywhere"
                ),
            )
        })
        .collect())
}

/// The packages some workflow step builds with the admission feature.
fn enabling_lanes(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    let mut enabled = BTreeSet::new();
    let mut workflows: Vec<&std::path::Path> = tree
        .paths()
        .iter()
        .filter(|path| path.starts_with(WORKFLOWS) && path.extension().is_some_and(|e| e == "yml"))
        .map(std::path::PathBuf::as_path)
        .collect();
    workflows.sort_unstable();
    for path in workflows {
        enabled.extend(packages_built_with_the_feature(&tree.read(path)?));
    }
    Ok(enabled)
}

/// The packages one workflow's steps build with the admission feature.
///
/// Every invocation in these workflows begins `./cargo_full`, so splitting on
/// it yields one command per segment. A segment is cut at the next step header
/// so a `-p` in one step cannot borrow a `--features` from the next.
fn packages_built_with_the_feature(text: &str) -> BTreeSet<String> {
    let mut enabled = BTreeSet::new();
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    for segment in collapsed.split("./cargo_full").skip(1) {
        let command = segment.split("- name:").next().unwrap_or(segment);
        if !command.contains(FEATURE) {
            continue;
        }
        let mut tokens = command.split(' ');
        while let Some(token) = tokens.next() {
            if token == "-p" {
                if let Some(package) = tokens.next() {
                    enabled.insert(package.to_string());
                }
            }
        }
    }
    enabled
}

/// Where the live workflows are.
const WORKFLOWS: &str = ".github/workflows";

/// The concrete backend types, read from the driver members that own them.
fn backend_roster(tree: &Tree) -> Result<BTreeSet<String>, GateError> {
    let mut roster = BTreeSet::new();
    for member in tree.members()? {
        if !member.starts_with("vyre-driver-") || member == CPU_ORACLE_CRATE {
            continue;
        }
        for path in tree.rust(&[&format!("{member}/src")])? {
            for line in tree.read(&path)?.lines() {
                if let Some(name) = declared_backend(line) {
                    roster.insert(name);
                }
            }
        }
    }
    Ok(roster)
}

/// The type-name suffixes a hardware-owning handle is declared with.
///
/// `*Backend` was the whole roster once, and it let two real acquisitions
/// through: `SpirvBackendRegistration::acquire` opens a Vulkan device, and
/// `CudaDeviceHandle::acquire_ordinal` initialises a CUDA context. Both sat in
/// hosted CPU legs and failed there. A driver crate that owns hardware names
/// the owner with one of these.
const ROSTER_SUFFIXES: &[&str] = &["Backend", "BackendRegistration", "DeviceHandle"];

/// The type name a `pub struct` line declares, when it names a hardware owner.
fn declared_backend(line: &str) -> Option<String> {
    let rest = line.trim_start().strip_prefix("pub struct ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    ROSTER_SUFFIXES
        .iter()
        .any(|suffix| name.len() > suffix.len() && name.ends_with(suffix))
        .then_some(name)
}

/// The constructor name stems that reach hardware.
///
/// Matched as stems, not exact names: `acquire` is the CUDA and Vulkan entry
/// point and `new` the wgpu one, but the CUDA context is opened through
/// `acquire_ordinal`, so an exact-name list misses it. Every other association
/// on these types takes an already-live handle, so a helper signature and a
/// doc link are still not acquisitions.
const CONSTRUCTORS: &[&str] = &["acquire", "new"];

/// Whether `rendered` calls a hardware constructor on `name`.
///
/// The identifier after `::` is read out and matched whole, so `new` matches
/// `new(` and `new_from_parts(` but never `newtype_of(`, and the trailing `(`
/// is required because naming a path is not calling it.
fn acquires(rendered: &str, name: &str) -> bool {
    let needle = format!("{name}::");
    let mut rest = rendered;
    while let Some(at) = rest.find(&needle) {
        let after = &rest[at + needle.len()..];
        let ident: String = after
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if after[ident.len()..].starts_with('(')
            && CONSTRUCTORS
                .iter()
                .any(|stem| ident == *stem || ident.starts_with(&format!("{stem}_")))
        {
            return true;
        }
        rest = after;
    }
    false
}

/// Whether an attribute list admits the item to run on hardware.
///
/// Either the device-test feature governs it, or `#[ignore]` keeps it out of a
/// default run so only a deliberate `--ignored` invocation reaches the device.
fn admitted(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("ignore")
            || (attr.path().is_ident("cfg") && attr.to_token_stream().to_string().contains(FEATURE))
    })
}

/// One item's tokens with every literal dropped.
///
/// A backend name inside a string is text. Rendering it with the code would
/// make a table of expected messages, a doc test, or this gate's own fixtures
/// indistinguishable from a call. The delimiters stay, because the open paren
/// is what separates acquiring a backend from naming one.
fn code_tokens(stream: proc_macro2::TokenStream) -> String {
    let mut rendered = String::new();
    for tree in stream {
        match tree {
            proc_macro2::TokenTree::Literal(_) => {}
            proc_macro2::TokenTree::Group(group) => {
                let (open, close) = match group.delimiter() {
                    proc_macro2::Delimiter::Parenthesis => ("(", ")"),
                    proc_macro2::Delimiter::Brace => ("{", "}"),
                    proc_macro2::Delimiter::Bracket => ("[", "]"),
                    proc_macro2::Delimiter::None => ("", ""),
                };
                rendered.push_str(open);
                rendered.push_str(&code_tokens(group.stream()));
                rendered.push_str(close);
            }
            other => {
                rendered.push_str(&other.to_string());
                rendered.push(' ');
            }
        }
    }
    rendered
}

/// Backend acquisitions in test code that nothing admits to hardware.
fn ungated_acquisitions(
    items: &[syn::Item],
    in_test: bool,
    backends: &BTreeSet<String>,
) -> Vec<String> {
    let mut found = Vec::new();
    for item in items {
        let attrs = item_attrs(item);
        if attrs.is_some_and(admitted) {
            continue;
        }
        let test_here =
            in_test || attrs.is_some_and(|list| list.iter().any(scan::attribute_is_test_only));
        if let syn::Item::Mod(module) = item {
            if let Some((_, inner)) = &module.content {
                found.extend(ungated_acquisitions(inner, test_here, backends));
            }
            continue;
        }
        if !test_here {
            continue;
        }
        // Token rendering spaces every punctuation apart; the call shape is
        // what distinguishes acquiring a backend from naming its type. Literal
        // tokens are dropped first: a gate fixture that quotes an acquisition
        // is a string, not a call, and reading it as one made this file report
        // itself.
        let rendered = code_tokens(item.to_token_stream()).replace(' ', "");
        found.extend(
            backends
                .iter()
                .filter(|name| acquires(&rendered, name))
                .cloned(),
        );
    }
    found
}

/// The attributes of an item, for the item kinds that carry them.
fn item_attrs(item: &syn::Item) -> Option<&[syn::Attribute]> {
    match item {
        syn::Item::Const(i) => Some(&i.attrs),
        syn::Item::Enum(i) => Some(&i.attrs),
        syn::Item::Fn(i) => Some(&i.attrs),
        syn::Item::Impl(i) => Some(&i.attrs),
        syn::Item::Macro(i) => Some(&i.attrs),
        syn::Item::Mod(i) => Some(&i.attrs),
        syn::Item::Static(i) => Some(&i.attrs),
        syn::Item::Struct(i) => Some(&i.attrs),
        syn::Item::Trait(i) => Some(&i.attrs),
        syn::Item::Type(i) => Some(&i.attrs),
        syn::Item::Use(i) => Some(&i.attrs),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roster() -> BTreeSet<String> {
        ["CudaBackend".to_string(), "WgpuBackend".to_string()]
            .into_iter()
            .collect()
    }

    fn acquisitions(source: &str, in_test: bool) -> Vec<String> {
        let file = syn::parse_file(source).expect("fixture must parse");
        if admitted(&file.attrs) {
            return Vec::new();
        }
        ungated_acquisitions(&file.items, in_test, &roster())
    }

    #[test]
    fn a_backend_declaration_is_read_from_its_struct_line() {
        assert_eq!(
            declared_backend("pub struct CudaBackend {"),
            Some("CudaBackend".to_string())
        );
        assert_eq!(declared_backend("pub struct Backend {"), None);
        assert_eq!(declared_backend("struct CudaBackend {"), None);
    }

    #[test]
    fn an_ungated_integration_test_that_acquires_a_device_is_a_finding() {
        let source = r#"
            use vyre_driver_cuda::CudaBackend;
            #[test]
            fn live() {
                let backend = CudaBackend::acquire().unwrap();
                let _ = backend;
            }
        "#;
        assert_eq!(acquisitions(source, true), vec!["CudaBackend".to_string()]);
    }

    #[test]
    fn a_file_wide_device_cfg_admits_the_whole_file() {
        let source = r#"
            #![cfg(all(not(target_os = "macos"), feature = "device-tests"))]
            #[test]
            fn live() {
                let _ = CudaBackend::acquire();
            }
        "#;
        assert!(acquisitions(source, true).is_empty());
    }

    #[test]
    fn a_per_test_device_cfg_admits_that_test_and_not_its_neighbour() {
        let source = r#"
            #[cfg(test)]
            mod tests {
                #[cfg(feature = "device-tests")]
                #[test]
                fn gated() {
                    let _ = WgpuBackend::new();
                }

                #[test]
                fn ungated() {
                    let _ = CudaBackend::acquire();
                }
            }
        "#;
        assert_eq!(acquisitions(source, false), vec!["CudaBackend".to_string()]);
    }

    #[test]
    fn production_code_may_name_a_backend_freely() {
        let source = r#"
            pub fn dispatch() {
                let _ = CudaBackend::acquire();
            }
        "#;
        assert!(acquisitions(source, false).is_empty());
    }

    #[test]
    fn an_import_alone_is_not_an_acquisition() {
        let source = r#"
            use vyre_driver_cuda::CudaBackend;

            #[cfg(feature = "device-tests")]
            #[test]
            fn live() {
                let _ = CudaBackend::acquire();
            }
        "#;
        assert!(acquisitions(source, true).is_empty());
    }

    #[test]
    fn an_ignored_measurement_instrument_is_admitted() {
        let source = r#"
            #[test]
            #[ignore = "measurement instrument: run with --ignored"]
            fn instrument() {
                let backend = CudaBackend::acquire().unwrap();
                let _ = backend;
            }
        "#;
        assert!(acquisitions(source, true).is_empty());
    }

    #[test]
    fn naming_a_backend_type_is_not_acquiring_one() {
        let source = r#"
            #[test]
            fn accounting() {
                fn cost(backend: &CudaBackend) -> u64 {
                    backend.cost()
                }
                let _ = cost;
            }
        "#;
        assert!(acquisitions(source, true).is_empty());
    }
    /// WHY: moving a test behind `device-tests` and forgetting the lane that
    /// turns it on is indistinguishable from deleting the test, and it is
    /// quieter: every lane goes green because nothing compiles the test any
    /// more. The step has to be found for the package it names.
    #[test]
    fn a_step_building_a_package_with_the_feature_is_the_lane_for_it() {
        let workflow = "\
jobs:\n\
  device:\n\
    steps:\n\
      - name: certificates on a real device\n\
        run: ./cargo_full test -p vyre-conform --features device-tests\n";
        assert_eq!(
            packages_built_with_the_feature(workflow),
            BTreeSet::from(["vyre-conform".to_string()])
        );
    }

    /// WHY: the feature is commonly one entry in a comma list, and the step is
    /// commonly folded across lines by a `>-` scalar. Reading either shape as
    /// "no lane" would fail a package that is covered.
    #[test]
    fn a_folded_step_and_a_comma_list_both_count() {
        let workflow = "\
      - name: harness suites\n\
        run: >-\n\
          ./cargo_full test -p vyre-bench\n\
          --features cuda,device-tests --tests\n";
        assert_eq!(
            packages_built_with_the_feature(workflow),
            BTreeSet::from(["vyre-bench".to_string()])
        );
    }

    /// WHY: the whole point is to catch the package whose admission leads
    /// nowhere, so a step that names the package without the feature, and a
    /// step that names the feature for a different package, must both leave it
    /// uncovered. A `-p` must not borrow the next step's `--features` either.
    #[test]
    fn a_package_without_its_own_enabling_step_is_not_covered() {
        let workflow = "\
      - name: default build\n\
        run: ./cargo_full test -p vyre-conform\n\
      - name: somebody else's device lane\n\
        run: ./cargo_full test -p vyre-bench --features device-tests\n";
        assert_eq!(
            packages_built_with_the_feature(workflow),
            BTreeSet::from(["vyre-bench".to_string()])
        );
    }

    /// WHY: a workflow that builds nothing with the feature must derive to the
    /// empty set rather than to every package it mentions, or the rule passes
    /// for a tree in which no lane enables anything.
    #[test]
    fn a_workflow_that_enables_nothing_covers_nothing() {
        let workflow = "\
      - name: hosted matrix\n\
        run: ./cargo_full test -p vyre-conform\n\
      - name: docs\n\
        run: ./cargo_full doc --workspace\n";
        assert!(packages_built_with_the_feature(workflow).is_empty());
    }
}

#[cfg(test)]
mod literal_tests {
    use super::*;

    /// WHY: this gate's own fixtures quote `CudaBackend::acquire()` inside raw
    /// strings, and a scan over rendered tokens read every one of them as a
    /// call, so the file reported itself seven times. Any table of expected
    /// diagnostics has the same shape.
    #[test]
    fn a_backend_acquisition_inside_a_string_is_not_an_acquisition() {
        let source = r##"
            #[test]
            fn fixture_table() {
                let quoted = r#"let _ = CudaBackend::acquire();"#;
                assert!(quoted.contains("acquire"));
            }
        "##;
        let file = syn::parse_file(source).expect("fixture must parse");
        let backends = BTreeSet::from(["CudaBackend".to_string()]);
        assert!(ungated_acquisitions(&file.items, true, &backends).is_empty());
    }

    /// WHY: dropping literals must not drop the call shape with them. The open
    /// paren lives in a delimiter group, and a renderer that skipped groups
    /// would make every real acquisition invisible and the gate vacuous.
    #[test]
    fn a_real_acquisition_still_reads_as_one() {
        let source = "
            #[test]
            fn live() {
                let _ = CudaBackend::acquire().expect(\"device\");
            }
        ";
        let file = syn::parse_file(source).expect("fixture must parse");
        let backends = BTreeSet::from(["CudaBackend".to_string()]);
        assert_eq!(
            ungated_acquisitions(&file.items, true, &backends),
            vec!["CudaBackend".to_string()]
        );
    }
}

#[cfg(test)]
mod closure_tests {
    use super::*;

    fn tree(files: &[(&str, &str)]) -> BTreeMap<PathBuf, syn::File> {
        files
            .iter()
            .map(|(path, source)| {
                (
                    PathBuf::from(path),
                    syn::parse_file(source).expect("fixture must parse"),
                )
            })
            .collect()
    }

    /// WHY: a test target gates itself once, at its root, and its submodules
    /// carry no attribute of their own. Reading each file alone reported every
    /// one of them as an ungated acquisition, which named 47 files that cannot
    /// be compiled without the feature at all. The alternative the gate pushed
    /// the tree toward was restating the attribute in every submodule, which
    /// is a second copy of one fact.
    #[test]
    fn a_submodule_of_an_admitted_root_is_admitted() {
        let parsed = tree(&[
            (
                "vyre-driver-cuda/tests/resident.rs",
                "#![cfg(all(test, feature = \"device-tests\"))]\nmod lane;\n",
            ),
            (
                "vyre-driver-cuda/tests/resident/lane.rs",
                "#[test]\nfn live() { let _ = CudaBackend::acquire(); }\n",
            ),
        ]);
        let admitted_files = admitted_closure(&parsed);
        assert!(
            admitted_files.contains(Path::new("vyre-driver-cuda/tests/resident/lane.rs")),
            "Fix: the closure must follow `mod lane;` out of an admitted root"
        );
    }

    /// WHY: admission must reach the whole subtree, not one level. A module
    /// that declares another is the shape every `mod.rs` in this tree has.
    #[test]
    fn admission_reaches_a_transitive_submodule() {
        let parsed = tree(&[
            (
                "vyre-driver-wgpu/tests/parity.rs",
                "#![cfg(feature = \"device-tests\")]\nmod inner;\n",
            ),
            ("vyre-driver-wgpu/tests/parity/inner/mod.rs", "mod deep;\n"),
            (
                "vyre-driver-wgpu/tests/parity/inner/deep.rs",
                "#[test]\nfn live() { let _ = WgpuBackend::new(); }\n",
            ),
        ]);
        let admitted_files = admitted_closure(&parsed);
        assert!(admitted_files.contains(Path::new("vyre-driver-wgpu/tests/parity/inner/deep.rs")));
    }

    /// WHY: the closure must not admit a file nothing gated. A sibling target
    /// in the same directory that acquires a device is the defect this gate
    /// exists for, and a resolver that matched on directory rather than on the
    /// declaration would swallow it.
    #[test]
    fn a_sibling_target_no_admitted_root_declares_stays_ungated() {
        let parsed = tree(&[
            (
                "vyre-driver-cuda/tests/resident.rs",
                "#![cfg(feature = \"device-tests\")]\nmod lane;\n",
            ),
            (
                "vyre-driver-cuda/tests/resident/lane.rs",
                "pub fn helper() {}\n",
            ),
            (
                "vyre-driver-cuda/tests/loose.rs",
                "#[test]\nfn live() { let _ = CudaBackend::acquire(); }\n",
            ),
        ]);
        let admitted_files = admitted_closure(&parsed);
        assert!(!admitted_files.contains(Path::new("vyre-driver-cuda/tests/loose.rs")));
    }

    /// WHY: a gated `mod` item inside an ungated parent admits that one child.
    /// Without it the only way to gate one module of a shared root would be to
    /// gate the root, which takes the ungated tests with it.
    #[test]
    fn a_gated_module_declaration_admits_its_child() {
        let parsed = tree(&[
            (
                "conform/vyre-conform/tests/cert.rs",
                "#[cfg(feature = \"device-tests\")]\nmod gpu;\nmod cpu;\n",
            ),
            (
                "conform/vyre-conform/tests/cert/gpu.rs",
                "#[test]\nfn live() { let _ = WgpuBackend::new(); }\n",
            ),
            (
                "conform/vyre-conform/tests/cert/cpu.rs",
                "#[test]\nfn live() { let _ = WgpuBackend::new(); }\n",
            ),
        ]);
        let admitted_files = admitted_closure(&parsed);
        assert!(admitted_files.contains(Path::new("conform/vyre-conform/tests/cert/gpu.rs")));
        assert!(!admitted_files.contains(Path::new("conform/vyre-conform/tests/cert/cpu.rs")));
    }

    /// WHY: `#[path]` is how this tree points a `src` module at a file under
    /// `tests/`, and a resolver that ignored it would leave that file looking
    /// like it belongs to nobody.
    #[test]
    fn a_path_attribute_resolves_to_the_file_it_names() {
        let parsed = tree(&[
            (
                "vyre-driver-cuda/src/lib.rs",
                "#![cfg(feature = \"device-tests\")]\n#[path = \"../tests/internal/live.rs\"]\nmod live;\n",
            ),
            (
                "vyre-driver-cuda/tests/internal/live.rs",
                "#[test]\nfn live() { let _ = CudaBackend::acquire(); }\n",
            ),
        ]);
        let admitted_files = admitted_closure(&parsed);
        assert!(
            admitted_files.contains(Path::new("vyre-driver-cuda/tests/internal/live.rs")),
            "Fix: `#[path]` names the file, so the closure must follow it"
        );
    }
}
