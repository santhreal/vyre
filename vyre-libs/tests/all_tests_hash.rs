//! One binary for every hash integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// WHY: a linker that drops unreferenced objects links no operation
/// registration into a binary that never names the facade anchor, and every
/// catalog read in it then walks an empty registry. The Apple linker does drop
/// them, so the five pinned hash witnesses reported the registration missing
/// instead of a wrong value, on that platform only.
///
/// # What it does not catch
///
/// The count proves the feature-selected partitions reached the registry. It
/// does not prove any one operation is among them; the witnesses below do.
#[cfg(feature = "hash")]
#[test]
fn the_operation_catalog_is_linked_into_this_binary() {
    assert!(
        vyre_libs::link_anchor() > 0,
        "Fix: name `vyre_libs::link_anchor` in the harness root; the registry this binary links \
         is empty, so every catalog case in it asserts over nothing"
    );
}

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[cfg(feature = "hash")]
#[allow(deprecated)]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/hash_registration_witnesses.rs`.
#[cfg(feature = "hash")]
#[path = "hash_registration_witnesses.rs"]
pub mod hash_registration_witnesses;
