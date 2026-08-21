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

use std::collections::BTreeSet;

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
        for path in sources {
            let source = tree.read(&path)?;
            let Ok(file) = syn::parse_file(&source) else {
                continue;
            };
            if admitted(&file.attrs) {
                continue;
            }
            let display = path.display().to_string();
            let in_test = scan::is_test_tree(&path);
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

/// The type name a `pub struct *Backend` line declares.
fn declared_backend(line: &str) -> Option<String> {
    let rest = line.trim_start().strip_prefix("pub struct ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    (name.ends_with("Backend") && name.len() > "Backend".len()).then_some(name)
}

/// The constructors that reach hardware.
///
/// `acquire` is the CUDA entry point and `new` the wgpu one. Every other
/// association on a backend type takes an already-live handle, so a helper
/// signature and a doc link are not acquisitions.
const CONSTRUCTORS: &[&str] = &["acquire", "new"];

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
        found.extend(backends.iter().filter_map(|name| {
            CONSTRUCTORS
                .iter()
                .any(|ctor| rendered.contains(&format!("{name}::{ctor}(")))
                .then(|| name.clone())
        }));
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
