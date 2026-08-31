//! The operation registry, with every crate that submits into it linked.
//!
//! Counting through each crate's own catalog rather than through the registry's
//! tier field keeps the reference in the source crate: a filter over ids would
//! compile to a walk of the registry alone and link nothing.

use std::sync::LazyLock;

use vyre_foundation::operation::OperationRegistry;

/// One entry per crate that submits operation registrations, with the count it
/// contributed. Computing this is what references the source crates.
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
            "Fix: `{source}` contributed no operation registrations, so this run is judging a partial registry. Read the registry through `vyre_registry_link::operation`, which calls that crate's catalog, instead of naming it with a discarding import"
        );
    }
    OperationRegistry::global()
}
