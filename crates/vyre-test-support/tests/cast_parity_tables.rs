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

use std::collections::BTreeSet;

use vyre_test_support::cast_parity::{
    signed_widening_words, I32_TO_I64_EXPECTED, NARROWING_CASES, NARROWING_INPUTS,
    SIGNED_WIDENING_INPUTS, UNSIGNED_TO_SIGNED_WIDENING_INPUTS, UNSIGNED_WIDENING_INPUTS,
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
    assert!(
        !NARROWING_CASES.is_empty(),
        "the narrowing matrix is empty, so this gate checks no pin at all"
    );
    for case in NARROWING_CASES {
        assert_eq!(
            case.reference_words().len(),
            NARROWING_INPUTS.len(),
            "the `u32 {}` reference must answer every probe word",
            case.label
        );
        case.assert_pin_holds();
    }
}

/// Every integer the IR admits as a cast-participating scalar and that is
/// narrower than the 32-bit source must have a case.
///
/// The variant space comes from `vyre_foundation::validate::cast`, the owner of
/// which types an integer cast may name, read at run time from its own source.
/// `DataType` also declares sub-byte quantization storage families such as `I4`
/// that the cast table deliberately excludes, so enumerating the enum instead
/// would demand a case for a type no backend can lower. Admit a narrower
/// integer there and this turns red until both backend arms dispatch it; a
/// hand-written list of four types would instead let it land proven nowhere,
/// which is indistinguishable from a clean sweep.
#[test]
fn the_matrix_covers_every_narrowing_target_the_cast_table_admits() {
    let root = structure_gate::workspace_root();
    let source_path =
        structure_gate::member_directory(&root, "vyre-foundation").join("src/validate/cast.rs");
    let source = std::fs::read_to_string(&source_path).unwrap_or_else(|error| {
        panic!(
            "Fix: the narrowing coverage gate cannot read {}: {error}",
            source_path.display()
        )
    });

    let admitted = cast_participating_scalars(&source);
    assert!(
        admitted.len() > 4,
        "Fix: only {} cast-participating scalar(s) were parsed out of {}; the gate is reading the wrong shape and would report success over nothing",
        admitted.len(),
        source_path.display()
    );
    let declared: BTreeSet<String> = admitted
        .into_iter()
        .filter(|variant| integer_bit_width(variant).is_some_and(|bits| bits < 32))
        .collect();

    let covered: BTreeSet<String> = NARROWING_CASES
        .iter()
        .map(|case| format!("{:?}", case.narrow))
        .collect();
    assert_eq!(
        covered, declared,
        "every cast-participating integer narrower than 32 bits needs a narrowing-cast case; \
         a target without one is dispatched by no backend"
    );
}

/// `DataType` variants the cast table treats as integer-like scalars.
///
/// Read from the `is_integer_like_scalar` predicate's own match arms, which is
/// where that decision lives.
fn cast_participating_scalars(source: &str) -> BTreeSet<String> {
    let Some(body) = source
        .split_once("fn is_integer_like_scalar")
        .map(|(_, rest)| rest)
    else {
        return BTreeSet::new();
    };
    let Some(arms) = body.split_once('}').map(|(head, _)| head) else {
        return BTreeSet::new();
    };
    arms.split("DataType::")
        .skip(1)
        .filter_map(|fragment| {
            let name: String = fragment
                .chars()
                .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// Bit width of an integer variant named as a signedness letter and a width,
/// or `None` for any other variant.
fn integer_bit_width(variant: &str) -> Option<u32> {
    let bits = variant
        .strip_prefix('U')
        .or_else(|| variant.strip_prefix('I'))?;
    bits.parse().ok()
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
