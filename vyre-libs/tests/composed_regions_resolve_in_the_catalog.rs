//! Every child region a built program emits names an operation the catalog
//! contains, under the feature selection that compiled this test.
//!
//! WHY: a composition names its shared building blocks by id inside the emitted
//! IR. When the skeleton that emits a region is compiled but the registration
//! that puts that id in the catalog is not, the build hands consumers programs
//! whose internal structure points at operations nothing can resolve. That is
//! invisible to `--all-features` and to `--workspace`, both of which unify
//! features until every registration is present, so this test carries no
//! feature gate: it measures whatever selection compiled it, and the default
//! selection is the one a consumer gets by asking for nothing.
//!
//! Both halves are derived from the registry at run time: the catalog is the
//! registry's own id set and the programs are every registered builder. Adding
//! an operation, a dialect, or a shared child region cannot leave this test
//! measuring a stale list, and a new registration that emits an unresolvable
//! region turns it red without anyone editing it.
//!
//! What it does not catch: an id present under one feature selection and absent
//! under another. Selections are the feature-isolation sweep's job; this covers
//! the one that is compiled.

use std::collections::BTreeSet;

use vyre_foundation::ir::{Node, Program};
use vyre_foundation::operation::OperationRegistry;
use vyre_libs::operation_catalog;

/// Prefix marking a child region that is deliberately outside the catalog.
const ANONYMOUS_PREFIX: &str = "anonymous::";

/// Every distinct region-generator identity reachable from a program's entry.
fn region_generators(program: &Program, out: &mut BTreeSet<String>) {
    fn walk(node: &Node, out: &mut BTreeSet<String>) {
        if let Node::Region { generator, .. } = node {
            out.insert(generator.as_str().to_string());
        }
        for body in vyre_foundation::visit::child_bodies(node) {
            for child in body {
                walk(child, out);
            }
        }
    }
    program.entry().iter().for_each(|node| walk(node, out));
}

#[test]
fn every_emitted_region_names_a_catalog_operation() {
    let registry = OperationRegistry::global();
    let catalog: BTreeSet<&'static str> = registry.iter().map(|operation| operation.id).collect();
    assert!(
        !catalog.is_empty(),
        "the operation catalog is empty, so this test measures nothing. Fix: link the crates \
         whose registrations it is meant to read."
    );

    let mut built = 0_usize;
    let mut unresolved: Vec<String> = Vec::new();
    for operation in operation_catalog::all_entries() {
        let Some(build) = operation.build else {
            continue;
        };
        built += 1;
        let mut generators = BTreeSet::new();
        region_generators(&build(), &mut generators);
        for generator in generators {
            if generator == Program::ROOT_REGION_GENERATOR
                || vyre_foundation::composition::is_anonymous_generator(&generator)
                || catalog.contains(generator.as_str())
                || catalog.iter().any(|op| {
                    generator.starts_with(op) && generator[op.len()..].starts_with("::")
                })
            {
                continue;
            }
            unresolved.push(format!("{} emits region {generator}", operation.id));
        }
    }

    assert!(
        built > 0,
        "no registered operation carries a program builder, so no region was walked. Fix: this \
         test reads OperationRegistration::build."
    );
    assert!(
        unresolved.is_empty(),
        "{} emitted region(s) name an operation this build's catalog does not contain:\n  {}\n\
         Fix: the feature that compiles the emitter must also enable the registration for the id \
         it emits. Name that feature from every dialect feature whose code composes the emitter, \
         or give the region an {ANONYMOUS_PREFIX} identity if it is deliberately not a catalog \
         entry.",
        unresolved.len(),
        unresolved.join("\n  ")
    );
}
