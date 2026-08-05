//! Substring-search op id compatibility contract.
//!
//! `substring_search` moved from `vyre_libs::matching` to `vyre::scan`.
//! The move kept a deprecated alias at the old path, and that alias keeps
//! emitting the OLD op id so already-recorded conformance rows, cache keys,
//! and operator dashboards that key on `vyre-libs::matching::substring_search`
//! keep resolving. Two op ids for one builder is a deliberate, bounded
//! compatibility decision, not drift.
//!
//! This suite exists because the split was easy to get backwards: an
//! integration test asserted the canonical builder emitted the LEGACY id and
//! nothing caught it, because the workspace test run could not complete. Each
//! test below pins one half of the contract so neither side can silently take
//! the other's identity.
//!
//! What breaks if this regresses: if the canonical builder emits the legacy
//! id, every new conformance row is filed under a deprecated op id that the
//! removal condition says will disappear. If the alias emits the canonical id,
//! existing consumers pinned to the old id stop matching and their programs
//! fall out of the registry silently.

#![cfg(feature = "matching-substring")]
#![allow(deprecated)]

use vyre::ir::{Node, Program};
use vyre_libs::compat_aliases::{CompatibilityAlias, MATCHING_SUBSTRING_ALIAS};

/// The op id the canonical `vyre::scan` path must emit.
const CANONICAL_OP_ID: &str = "vyre-libs::scan::substring_search";
/// The op id the deprecated `vyre_libs::matching` path must keep emitting.
const LEGACY_OP_ID: &str = "vyre-libs::matching::substring_search";

/// Extract the single region generator name from a substring program.
///
/// Every vyre-libs builder emits exactly one top-level `Node::Region`, and the
/// generator is the op id. Anything else is a shape change that these tests
/// should fail on rather than silently look past.
fn region_generator(program: &Program) -> String {
    let entry = program.entry();
    assert_eq!(
        entry.len(),
        1,
        "a substring program is exactly one top-level region node, got {} nodes",
        entry.len()
    );
    match &entry[0] {
        Node::Region { generator, .. } => generator.as_str().to_string(),
        other => panic!("expected Node::Region at the program entry, got {other:?}"),
    }
}

fn canonical(haystack_len: u32, needle_len: u32) -> Program {
    vyre::scan::substring_search("haystack", "needle", "matches", haystack_len, needle_len)
}

fn legacy(haystack_len: u32, needle_len: u32) -> Program {
    vyre_libs::matching::substring::substring_search(
        "haystack",
        "needle",
        "matches",
        haystack_len,
        needle_len,
    )
}

/// The canonical path emits the canonical op id.
///
/// This is the assertion that was inverted in `integration.rs`. Locking it
/// here means the canonical builder cannot drift back onto the deprecated
/// identity that `MATCHING_SUBSTRING_ALIAS.removal_condition` schedules for
/// deletion.
#[test]
fn the_scan_path_emits_the_canonical_op_id() {
    assert_eq!(region_generator(&canonical(8, 3)), CANONICAL_OP_ID);
}

/// The deprecated path keeps the legacy op id.
///
/// The whole point of the alias is identity preservation. If it started
/// emitting the canonical id, consumers keyed on the old id would see their
/// programs vanish from the registry with no error, which is exactly the kind
/// of silent behavior change Law 10 forbids.
#[test]
fn the_matching_alias_emits_the_legacy_op_id() {
    assert_eq!(region_generator(&legacy(8, 3)), LEGACY_OP_ID);
}

/// The two ids are distinct.
///
/// A rename that collapsed both onto one string would make every other test in
/// this file pass trivially while destroying the compatibility guarantee.
#[test]
fn the_canonical_and_legacy_op_ids_are_not_the_same_string() {
    assert_ne!(CANONICAL_OP_ID, LEGACY_OP_ID);
    assert_ne!(
        region_generator(&canonical(8, 3)),
        region_generator(&legacy(8, 3))
    );
}

/// Re-exporting the canonical function from `matching` does not relabel it.
///
/// `vyre_libs::matching::substring_search` (the flat re-export, not the
/// `substring` leaf module) is a plain `pub use` of the canonical function, so
/// it must produce the canonical id. Only the `substring` leaf module carries
/// the legacy identity. These two neighbours under the same `matching` parent
/// are the easiest pair in the crate to confuse.
#[test]
fn the_flat_matching_reexport_is_the_canonical_function_not_the_alias() {
    let via_reexport = vyre_libs::matching::substring_search("haystack", "needle", "matches", 8, 3);
    assert_eq!(region_generator(&via_reexport), CANONICAL_OP_ID);
}

/// The op id is the ONLY difference between the two builders.
///
/// The alias must be a relabel, never a fork. If someone edits the canonical
/// builder and forgets the alias (or vice versa), the two programs diverge in
/// buffers or body and consumers on the old path silently get different work.
#[test]
fn the_alias_differs_from_the_canonical_builder_only_in_its_op_id() {
    let canonical = canonical(8, 3);
    let legacy = legacy(8, 3);

    assert_eq!(canonical.buffers().len(), legacy.buffers().len());
    for (c, l) in canonical.buffers().iter().zip(legacy.buffers()) {
        assert_eq!(c.name(), l.name(), "buffer names must match");
        assert_eq!(c.binding(), l.binding(), "buffer bindings must match");
        assert_eq!(c.access(), l.access(), "buffer access must match");
        assert_eq!(c.kind(), l.kind(), "buffer memory kinds must match");
        assert_eq!(c.element(), l.element(), "buffer element types must match");
        assert_eq!(c.count(), l.count(), "buffer element counts must match");
    }
    assert_eq!(canonical.workgroup_size(), legacy.workgroup_size());

    // Substituting the legacy id for the canonical one must make the two
    // regions byte-identical: same body, same everything but the label.
    let relabelled = format!("{:?}", legacy.entry()[0]).replace(LEGACY_OP_ID, CANONICAL_OP_ID);
    assert_eq!(
        format!("{:?}", canonical.entry()[0]),
        relabelled,
        "the alias must relabel the region, never fork its body"
    );
}

/// Both builders are deterministic across sizes.
///
/// The id must not be derived from, or perturbed by, the buffer sizes. A
/// size-dependent generator name would fragment the conformance matrix.
#[test]
fn both_op_ids_are_stable_across_haystack_and_needle_sizes() {
    for (haystack_len, needle_len) in [(1_u32, 1_u32), (8, 3), (64, 7), (4096, 32)] {
        assert_eq!(
            region_generator(&canonical(haystack_len, needle_len)),
            CANONICAL_OP_ID,
            "canonical id changed at haystack_len={haystack_len} needle_len={needle_len}"
        );
        assert_eq!(
            region_generator(&legacy(haystack_len, needle_len)),
            LEGACY_OP_ID,
            "legacy id changed at haystack_len={haystack_len} needle_len={needle_len}"
        );
    }
}

/// Both programs pass IR validation.
///
/// A compatibility path is still a shipped path. The alias does not get to be
/// a builder that produces an invalid program because "nobody new uses it".
#[test]
fn both_paths_produce_programs_that_validate() {
    for program in [canonical(8, 3), legacy(8, 3)] {
        let errors = vyre::ir::validate(&program);
        assert!(
            errors.is_empty(),
            "substring program failed validation: {:?}",
            errors
                .iter()
                .map(|e| e.message().to_string())
                .collect::<Vec<_>>()
        );
    }
}

/// The alias registry row names the real owner.
///
/// `compat_aliases` is the ONE place that records where a shim points. Facade
/// docs, release gates, and import audits all read it, so a row that drifts
/// from the module tree misdirects every one of them at once.
#[test]
fn the_alias_registry_row_points_at_the_canonical_substring_module() {
    let alias: CompatibilityAlias = MATCHING_SUBSTRING_ALIAS;
    assert_eq!(alias.deprecated_path, "vyre_libs::matching::substring");
    assert_eq!(alias.canonical_path, "vyre::scan::substring");
    assert_eq!(alias.canonical_owner, "vyre-libs/src/scan/substring");
    assert!(
        !alias.removal_condition.is_empty() && alias.removal_condition.contains("substring"),
        "the removal condition must name a concrete substring-specific gate, got {:?}",
        alias.removal_condition
    );
}

/// The registry row and the emitted op ids describe the same rename.
///
/// The row is written in Rust path syntax (`vyre::scan::substring`) and
/// the op ids in crate-name syntax (`vyre-libs::scan::substring_search`).
/// Nothing enforced that they agreed, so a future rename could move the module
/// and leave the op id behind. This ties the two spellings together.
#[test]
fn the_registry_row_and_the_op_ids_agree_on_which_side_is_canonical() {
    let canonical_module = MATCHING_SUBSTRING_ALIAS.canonical_path.replace('_', "-");
    let deprecated_module = MATCHING_SUBSTRING_ALIAS.deprecated_path.replace('_', "-");

    assert_eq!(canonical_module, "vyre-libs::scan::substring");
    assert_eq!(deprecated_module, "vyre-libs::matching::substring");

    // `vyre-libs::scan::substring` -> `vyre-libs::scan::` is the op id prefix.
    let canonical_prefix = canonical_module
        .rsplit_once("::")
        .expect("the canonical path has a module segment")
        .0;
    let deprecated_prefix = deprecated_module
        .rsplit_once("::")
        .expect("the deprecated path has a module segment")
        .0;

    assert_eq!(
        CANONICAL_OP_ID,
        format!("{canonical_prefix}::substring_search")
    );
    assert_eq!(
        LEGACY_OP_ID,
        format!("{deprecated_prefix}::substring_search")
    );
}
