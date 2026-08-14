//! The live operation registry, with every crate that submits into it linked.
//!
//! WHY: `inventory` registrations live in the object file of the crate that
//! declares them, and a linker pulls an archive member out of an rlib only when
//! a symbol inside it is referenced. `use vyre_libs as _;` names a crate without
//! referencing any symbol in it, so the registrations were dropped from every
//! binary that did not otherwise call into those crates. The production binary
//! calls the catalogs while generating documents, so it saw 354 registrations;
//! the test binaries called nothing, so they saw zero and three registry gates
//! judged an empty registry while reporting success. One of them said so out
//! loud ("saw only 0 registrations, so this rule is judging nothing") and had
//! been failing for that reason rather than for anything about the tree.
//!
//! Reading the registry through [`live_operation_registry`] makes the reference
//! real: the accessor calls each source crate's catalog, which is a symbol in
//! that crate's object file, so the registrations are linked into whatever
//! binary reads them. The floor per source is asserted rather than assumed,
//! because a silently empty source is the failure mode this exists to prevent.

use std::sync::LazyLock;

use vyre_foundation::operation::OperationRegistry;

/// One entry per crate that submits operation registrations, with the count it
/// contributed. Computing this is what references the source crates.
///
/// Counting through each crate's own catalog rather than through the registry's
/// namespace field keeps the reference in the source crate: a filter over ids
/// would compile to a walk of the registry alone and link nothing.
static SOURCES: LazyLock<[(&str, usize); 2]> = LazyLock::new(|| {
    [
        (
            "vyre-libs",
            vyre_libs::operation_catalog::all_entries().count(),
        ),
        (
            "vyre-primitives",
            vyre_primitives::operation_catalog::all_entries().count(),
        ),
    ]
});

/// Every crate that submits operation registrations, with what it contributed.
#[must_use]
pub fn registration_sources() -> &'static [(&'static str, usize)] {
    &*SOURCES
}

/// The global operation registry, with every registration source linked in.
///
/// # Panics
/// Panics when a source crate contributed no registrations, which means its
/// object file was dropped at link time and every rule reading the registry
/// would be judging a partial tree.
#[must_use]
pub fn live_operation_registry() -> &'static OperationRegistry {
    for (source, count) in registration_sources() {
        assert!(
            *count > 0,
            "Fix: `{source}` contributed no operation registrations, so this run is judging a partial registry. Call into that crate from the binary under test, as `xtask_registry::live_registry` does, instead of naming it with a discarding import"
        );
    }
    OperationRegistry::global()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A crate declares that it owns operation registrations by publishing an
    /// `operation_catalog` module. Every such crate has to be linked here.
    ///
    /// WHY: the defect this module exists for is invisible by construction. An
    /// unlinked crate contributes nothing, so every count, every document and
    /// every rule agrees with itself while describing a partial registry. The
    /// candidate set is therefore read from the tree at run time: a new
    /// registration-owning crate turns this red until someone calls into it,
    /// rather than being absorbed in silence.
    #[test]
    fn every_crate_that_owns_a_catalog_is_linked_here() {
        let root = xtask::checkout::checkout_root();
        let mut owners: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&root).expect("Fix: the checkout root must be readable") {
            let path = entry
                .expect("Fix: a checkout entry must be readable")
                .path();
            if path.join("src/operation_catalog.rs").is_file() {
                owners.push(
                    path.file_name()
                        .expect("Fix: a member directory has a name")
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
        owners.sort();
        let mut linked: Vec<String> = registration_sources()
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect();
        linked.sort();
        assert_eq!(
            linked, owners,
            "Fix: every crate publishing an `operation_catalog` module must be counted in `SOURCES`, so its registrations are linked into whatever binary reads the registry"
        );
    }

    /// The registry holds exactly what the counted sources contributed.
    ///
    /// WHY: the per-source floor catches a source that vanished entirely. This
    /// catches the other direction: registrations reaching the registry from a
    /// crate nobody counted, which means the accessor's account of where
    /// registrations come from is out of date even though every rule still
    /// passes.
    #[test]
    fn the_registry_holds_exactly_what_the_counted_sources_contributed() {
        let registry = live_operation_registry();
        let counted: usize = registration_sources().iter().map(|(_, count)| count).sum();
        assert_eq!(
            registry.iter().len(),
            counted,
            "Fix: the live registry carries registrations from a crate `SOURCES` does not name; add it there or stop registering from it"
        );
    }
}
