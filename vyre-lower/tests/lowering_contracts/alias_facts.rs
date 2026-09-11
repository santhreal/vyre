//! Alias-fact set contracts.

use vyre_lower::analyses::alias_facts::{AliasFactSet, NoAliasFact};

#[test]
fn no_alias_facts_are_bidirectional() {
    let mut facts = AliasFactSet::default();
    facts.insert_no_alias(NoAliasFact {
        left_binding: 1,
        left_index: 7,
        right_binding: 2,
        right_index: 9,
    });
    assert!(facts.proves_no_alias(1, 7, 2, 9));
    assert!(facts.proves_no_alias(2, 9, 1, 7));
}
