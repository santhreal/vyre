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
//! another file that hides the constructor. The gate sees syntax, not call
//! graphs.

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
        Ok(report)
    }
}

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
        // what distinguishes acquiring a backend from naming its type.
        let rendered = item.to_token_stream().to_string().replace(' ', "");
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
}
