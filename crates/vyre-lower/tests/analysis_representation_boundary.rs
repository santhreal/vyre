//! No-alias facts on lowered binding slots hold in both directions and never
//! relate a slot index to itself.

use vyre_lower::analyses::alias_facts::{AliasFactSet, NoAliasFact};

#[test]
fn descriptor_alias_facts_enforce_bidirectional_slot_isolation() {
    let mut alias_set = AliasFactSet::default();
    alias_set.insert_no_alias(NoAliasFact {
        left_binding: 0,
        left_index: 10,
        right_binding: 1,
        right_index: 20,
    });

    assert!(alias_set.proves_no_alias(0, 10, 1, 20));
    assert!(alias_set.proves_no_alias(1, 20, 0, 10));
    assert!(!alias_set.proves_no_alias(0, 10, 0, 10));
}
