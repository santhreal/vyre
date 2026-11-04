//! Smoke test for vyre-libs-builder.

use std::collections::BTreeSet;

use vyre_libs_builder::plumbing::registration::operation_catalog;

#[test]
fn library_registrations_survive_linking() {
    // link_anchor exists so the linker keeps the feature-selected registrations.
    // If that stopped working the catalog would come back empty here.
    let anchored = operation_catalog::link_anchor();
    assert!(
        anchored > 0,
        "the library tier must contribute registrations"
    );

    let mut ids = BTreeSet::new();
    for entry in operation_catalog::library_entries() {
        assert!(
            !entry.id.is_empty(),
            "every library operation needs a stable id"
        );
        assert!(
            ids.insert(entry.id),
            "duplicate library operation id: {}",
            entry.id
        );
    }

    assert_eq!(
        ids.len(),
        anchored,
        "link_anchor counts exactly the library entries it anchors"
    );
}
