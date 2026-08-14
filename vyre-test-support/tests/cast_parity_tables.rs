//! The shared cast-parity pins agree with Rust `as` over their own corpora.
//!
//! # The class this closes
//!
//! Both driver suites read the cast probe words and the pinned result vectors
//! from one table instead of from a copy each. A pin is only load bearing while
//! it still describes the corpus next to it: extend the corpus and the pin is
//! short, reorder the corpus and the pin is stale, edit a pin to match a
//! miscompile and every consumer accepts the miscompile. None of those is a
//! compile error, and a driver target that only compares its own recomputed
//! reference against the pin would agree with a pin edited in the same commit.
//!
//! This gate recomputes every pin from the corpus with Rust `as` at run time, so
//! the pin is checked against the language rather than against another copy of
//! itself, and it checks lengths so a corpus that grew without its pin fails
//! here rather than truncating a live-GPU comparison.
//!
//! # What it does not catch
//!
//! It does not dispatch anything, so it cannot see a backend that lowers the
//! cast wrongly. That is what each driver's parity target proves on real
//! hardware, and it is why each of those keeps its own reference arm.

use vyre_test_support::cast_parity::{
    signed_widening_words, I32_TO_I64_EXPECTED, NARROWING_INPUTS, SIGNED_WIDENING_INPUTS,
    U32_TO_I16_EXPECTED, U32_TO_I8_EXPECTED, U32_TO_U16_EXPECTED, U32_TO_U8_EXPECTED,
    UNSIGNED_TO_SIGNED_WIDENING_INPUTS, UNSIGNED_WIDENING_INPUTS,
};

/// Minimum probe words a corpus must carry, so a corpus emptied or trimmed to a
/// single happy case fails here instead of reporting a clean sweep of nothing.
const CORPUS_FLOOR: usize = 5;

#[test]
fn every_pinned_widening_result_is_what_rust_as_computes() {
    assert_eq!(
        I32_TO_I64_EXPECTED.len(),
        SIGNED_WIDENING_INPUTS.len(),
        "the i32->i64 pin must answer every signed probe word"
    );
    for (&input, &pinned) in SIGNED_WIDENING_INPUTS.iter().zip(&I32_TO_I64_EXPECTED) {
        assert_eq!(
            pinned,
            i64::from(input) as u64,
            "pinned i32->i64 widening for {input} disagrees with Rust `as`"
        );
        assert_eq!(
            pinned, input as u64,
            "pinned i32->u64 widening for {input} disagrees with Rust `as`; \
             the source signedness, not the target's, drives the high word"
        );
    }
}

#[test]
fn every_pinned_narrowing_result_is_what_rust_as_computes() {
    for (label, pinned, computed) in [
        (
            "u32->u8",
            U32_TO_U8_EXPECTED.to_vec(),
            NARROWING_INPUTS
                .iter()
                .map(|&v| u32::from(v as u8))
                .collect::<Vec<u32>>(),
        ),
        (
            "u32->u16",
            U32_TO_U16_EXPECTED.to_vec(),
            NARROWING_INPUTS
                .iter()
                .map(|&v| u32::from(v as u16))
                .collect(),
        ),
    ] {
        assert_eq!(
            pinned, computed,
            "pinned {label} narrowing disagrees with Rust `as` over NARROWING_INPUTS"
        );
    }
    for (label, pinned, computed) in [
        (
            "u32->i8",
            U32_TO_I8_EXPECTED.to_vec(),
            NARROWING_INPUTS
                .iter()
                .map(|&v| i32::from(v as u8 as i8))
                .collect::<Vec<i32>>(),
        ),
        (
            "u32->i16",
            U32_TO_I16_EXPECTED.to_vec(),
            NARROWING_INPUTS
                .iter()
                .map(|&v| i32::from(v as u16 as i16))
                .collect(),
        ),
    ] {
        assert_eq!(
            pinned, computed,
            "pinned {label} narrowing disagrees with Rust `as` over NARROWING_INPUTS"
        );
    }
}

#[test]
fn every_corpus_still_spans_the_boundaries_it_was_built_for() {
    for (label, len) in [
        ("SIGNED_WIDENING_INPUTS", SIGNED_WIDENING_INPUTS.len()),
        ("UNSIGNED_WIDENING_INPUTS", UNSIGNED_WIDENING_INPUTS.len()),
        (
            "UNSIGNED_TO_SIGNED_WIDENING_INPUTS",
            UNSIGNED_TO_SIGNED_WIDENING_INPUTS.len(),
        ),
        ("NARROWING_INPUTS", NARROWING_INPUTS.len()),
    ] {
        assert!(
            len >= CORPUS_FLOOR,
            "{label} carries {len} probe words, below the {CORPUS_FLOOR}-word floor; \
             a trimmed corpus silently narrows every parity gate that reads it"
        );
    }
    assert!(
        SIGNED_WIDENING_INPUTS.contains(&i32::MIN) && SIGNED_WIDENING_INPUTS.contains(&i32::MAX),
        "the signed widening corpus must keep both 32-bit extremes"
    );
    assert!(
        SIGNED_WIDENING_INPUTS.iter().any(|&v| v < 0),
        "the signed widening corpus must keep a negative word; \
         without one a zero-extending miscompile passes"
    );
    assert!(
        UNSIGNED_WIDENING_INPUTS.contains(&0xFFFF_FFFF)
            && UNSIGNED_TO_SIGNED_WIDENING_INPUTS.contains(&0xFFFF_FFFF),
        "both unsigned widening corpora must keep 0xFFFFFFFF; \
         it is the word a sign-extending miscompile turns into -1"
    );
    assert!(
        NARROWING_INPUTS.iter().any(|&v| v > 0xFF),
        "the narrowing corpus must keep a word wider than a byte; \
         without one a cast that never truncates passes"
    );
    assert_eq!(
        signed_widening_words().len(),
        SIGNED_WIDENING_INPUTS.len(),
        "the packed signed word view must carry one word per probe input"
    );
}
