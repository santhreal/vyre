use super::*;

#[test]
fn capacity_class_classifies_by_alignment_and_granularity() {
    assert_eq!(
        ReadbackRingSet::capacity_class_for(16).unwrap(),
        4096,
        "16-byte requests must promote to 4096-byte slot class"
    );
    assert_eq!(
        ReadbackRingSet::capacity_class_for(1).unwrap(),
        4096,
        "1-byte requests must promote to minimum aligned 4096-byte class"
    );
    assert_eq!(
        ReadbackRingSet::capacity_class_for(4097).unwrap(),
        8192,
        "4KB boundary crossings must promote to the next class"
    );
}

#[test]
fn existing_ring_for_and_capacity_variant_agree_on_lookup_key() {
    let ring_set = ReadbackRingSet::new();
    let from_raw = ring_set
        .existing_ring_for(16)
        .expect("Fix: lookup with raw byte length should not fail");
    let from_class = ring_set.existing_ring_for_capacity(4096);
    assert!(
        from_raw.is_none() && from_class.is_none(),
        "raw and capacity-based lookups should agree on an empty set"
    );
}
