//! What the CUDA materialized output cache must agree with: the backend-neutral
//! exact-input envelope in `vyre_driver::input_identity`.
//!
//! The envelope's own properties (tuple-boundary separation, single-byte
//! sensitivity, arity limits) are proven where the envelope lives, over 4096
//! generated cases each. Restating them here proved nothing about CUDA and hid
//! the question that matters: whether this cache keys by that envelope at all,
//! or by a second one that happens to behave the same way today.
//!
//! Not covered: whether the cache is collision-safe. It is not required to be.
//! The key is a hot-path filter and every hit still byte-compares the retained
//! inputs, which `matches_inputs` owns.

use vyre_driver::input_identity::domain_separated_exact_input_key;

use super::*;

/// Input tuples that separately exercise a single slot, a tuple boundary, an
/// empty slot between two non-empty ones, and a trailing empty slot.
const INPUT_TUPLES: &[&[&[u8]]] = &[
    &[b"state"],
    &[b"ab", b"c"],
    &[b"ab", b"", b"c"],
    &[b"abc", b""],
    &[b"", b"abc"],
];

fn outputs_for(inputs: &[&[u8]]) -> Vec<Vec<u8>> {
    vec![inputs
        .iter()
        .flat_map(|input| input.iter().rev().copied())
        .collect()]
}

#[test]
fn materialized_cache_keys_inputs_with_the_shared_driver_envelope() {
    for inputs in INPUT_TUPLES {
        let outputs = outputs_for(inputs);
        let envelope_key = exact_input_key(inputs)
            .expect("Fix: shared exact-input envelope must key the declared tuple");

        let entry = MaterializedPipelineOutputCacheEntry::new(inputs, &outputs)
            .expect("Fix: materialized cache entry construction must fit the declared tuple");
        assert_eq!(
            entry.input_key(),
            &envelope_key,
            "Fix: the CUDA materialized output cache must key inputs with vyre_driver::input_identity::exact_input_key rather than a CUDA-private envelope, for tuple {inputs:?}."
        );

        let mut cache = MaterializedPipelineOutputCache::default();
        cache
            .remember_entry(entry)
            .expect("Fix: materialized cache insertion must fit the declared tuple");
        assert!(
            cache.snapshot_with_key(inputs, &envelope_key).is_some(),
            "Fix: an entry the cache stored must be reachable by the shared envelope key, for tuple {inputs:?}."
        );
    }
}

#[test]
fn materialized_cache_rejects_a_resident_cache_domain_key_for_the_same_inputs() {
    for inputs in INPUT_TUPLES {
        let outputs = outputs_for(inputs);
        let mut cache = MaterializedPipelineOutputCache::default();
        cache
            .remember(inputs, &outputs)
            .expect("Fix: materialized cache remember must fit the declared tuple");

        let domain_key =
            domain_separated_exact_input_key(b"vyre.cuda.optimizer.static-upload.v1", 0, 0, inputs)
                .expect("Fix: domain-separated key must fit the declared tuple");

        assert!(
            cache.snapshot_with_key(inputs, &domain_key).is_none(),
            "Fix: a resident-cache domain key must not reach materialized replay outputs, or the two caches alias for tuple {inputs:?}."
        );
        assert!(
            cache
                .snapshot(inputs)
                .expect("Fix: materialized cache lookup must fit the declared tuple")
                .is_some(),
            "Fix: the same inputs must still hit under the plain replay envelope, for tuple {inputs:?}."
        );
    }
}
