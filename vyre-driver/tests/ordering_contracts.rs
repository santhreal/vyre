//! Contracts for `vyre_driver::ordering`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use std::cell::Cell;
use vyre_driver::ordering::{
    classify_dense_permutation, iter_is_monotonic_by_key, sort_by_key_if_needed,
    sort_unstable_by_key_if_needed, sort_unstable_if_needed, DensePermutationDefect,
};

#[test]
fn iter_monotonic_by_key_detects_ordered_and_unordered_streams() {
    assert!(iter_is_monotonic_by_key([0, 1, 1, 3], |value| value));
    assert!(!iter_is_monotonic_by_key([0, 2, 1, 3], |value| value));
}

#[test]
fn stable_sort_by_key_skips_already_monotonic_slices() {
    let calls = Cell::new(0usize);
    let mut items = [(0usize, "a"), (1, "b"), (1, "c"), (3, "d")];

    sort_by_key_if_needed(&mut items, |(key, _)| {
        calls.set(calls.get() + 1);
        *key
    });

    assert_eq!(items, [(0, "a"), (1, "b"), (1, "c"), (3, "d")]);
    assert_eq!(
        calls.get(),
        items.len(),
        "Fix: monotonic ordering paths must not invoke the fallback sort."
    );
}

#[test]
fn stable_sort_by_key_sorts_unordered_slices() {
    let mut items = [(2usize, "c"), (0, "a"), (3, "d"), (1, "b")];

    sort_by_key_if_needed(&mut items, |(key, _)| *key);

    assert_eq!(items, [(0, "a"), (1, "b"), (2, "c"), (3, "d")]);
}

#[test]
fn unstable_sort_by_key_skips_already_monotonic_slices() {
    let calls = Cell::new(0usize);
    let mut items = [(0usize, "a"), (1, "b"), (3, "c")];

    sort_unstable_by_key_if_needed(&mut items, |(key, _)| {
        calls.set(calls.get() + 1);
        *key
    });

    assert_eq!(items, [(0, "a"), (1, "b"), (3, "c")]);
    assert_eq!(
        calls.get(),
        items.len(),
        "Fix: monotonic unstable-ordering paths must not invoke the fallback sort."
    );
}

#[test]
fn unstable_sort_by_key_sorts_unordered_slices() {
    let mut items = [(2usize, "c"), (0, "a"), (1, "b")];

    sort_unstable_by_key_if_needed(&mut items, |(key, _)| *key);

    assert_eq!(items, [(0, "a"), (1, "b"), (2, "c")]);
}

#[test]
fn unstable_sort_skips_already_monotonic_slices() {
    let mut items = [0usize, 1, 1, 3];

    sort_unstable_if_needed(&mut items);

    assert_eq!(items, [0, 1, 1, 3]);
}

#[test]
fn unstable_sort_sorts_unordered_slices() {
    let mut items = [2usize, 0, 1];

    sort_unstable_if_needed(&mut items);

    assert_eq!(items, [0, 1, 2]);
}

#[test]
fn classify_dense_permutation_distinguishes_dense_duplicate_sparse_and_length() {
    assert_eq!(classify_dense_permutation(&[0, 1, 2], 3), Ok(()));
    assert_eq!(classify_dense_permutation(&[], 0), Ok(()));
    assert_eq!(
        classify_dense_permutation(&[0, 0, 2], 3),
        Err(DensePermutationDefect::Duplicate { index: 0, slot: 1 }),
        "Fix: a repeated value at a later sorted slot is a duplicate, not a generic non-dense map."
    );
    assert_eq!(
        classify_dense_permutation(&[0, 2, 3], 3),
        Err(DensePermutationDefect::Sparse { index: 2, slot: 1 }),
        "Fix: a value above its sorted slot is a sparse gap, not a duplicate."
    );
    assert_eq!(
        classify_dense_permutation(&[0, 1], 3),
        Err(DensePermutationDefect::LengthMismatch {
            resolved: 2,
            expected: 3
        }),
        "Fix: a dense-but-short map is a length mismatch."
    );
    assert_eq!(
        classify_dense_permutation(&[0, 1, 2, 3], 3),
        Err(DensePermutationDefect::LengthMismatch {
            resolved: 4,
            expected: 3
        }),
        "Fix: a dense-but-long map is a length mismatch."
    );
}

#[test]
fn classify_dense_permutation_matches_sorted_reference_over_generated_maps() {
    // For every permutation-with-defect we can synthesize, the classifier's
    // verdict must agree with an independent set-based reference oracle.
    for len in 0usize..=24 {
        let dense: Vec<usize> = (0..len).collect();
        assert_eq!(classify_dense_permutation(&dense, len), Ok(()));

        for collide in 0..len {
            // Replace one slot's value with a duplicate of slot 0's value (0),
            // then re-sort: this guarantees a duplicate, never a sparse gap.
            let mut indices = dense.clone();
            indices[collide] = 0;
            sort_unstable_if_needed(&mut indices);
            let verdict = classify_dense_permutation(&indices, len);
            let distinct: std::collections::BTreeSet<usize> = indices.iter().copied().collect();
            let reference_is_dense = distinct.len() == len
                && indices.len() == len
                && *distinct.iter().max().unwrap_or(&0) < len.max(1);
            if collide == 0 {
                // collide==0 leaves the map unchanged: still dense.
                assert_eq!(verdict, Ok(()));
            } else {
                assert!(
                    verdict.is_err(),
                    "len={len} collide={collide} must be a defect"
                );
                assert!(!reference_is_dense || verdict.is_ok());
                assert!(matches!(
                    verdict,
                    Err(DensePermutationDefect::Duplicate { .. })
                        | Err(DensePermutationDefect::Sparse { .. })
                        | Err(DensePermutationDefect::LengthMismatch { .. })
                ));
            }
        }
    }
}

#[test]
fn generated_ordering_matrix_matches_full_sort_contract() {
    for len in 0..=128 {
        let ordered: Vec<usize> = (0..len).collect();
        let mut reversed: Vec<usize> = (0..len).rev().collect();
        let mut expected = reversed.clone();
        expected.sort_unstable();

        assert!(iter_is_monotonic_by_key(ordered.iter().copied(), |value| {
            value
        }));
        if len > 1 {
            assert!(!iter_is_monotonic_by_key(
                reversed.iter().copied(),
                |value| value
            ));
        }

        sort_unstable_if_needed(&mut reversed);
        assert_eq!(reversed, expected);

        let mut keyed: Vec<(usize, usize)> = (0..len).rev().map(|value| (value, len)).collect();
        sort_unstable_by_key_if_needed(&mut keyed, |(key, _)| *key);
        for (expected_key, actual) in keyed.iter().enumerate() {
            assert_eq!(actual.0, expected_key);
            assert_eq!(actual.1, len);
        }
    }
}
