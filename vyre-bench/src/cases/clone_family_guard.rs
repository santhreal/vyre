//! Cross-entry-point guard for the merged clone families.
//!
//! Each test here pins behavior that used to exist in two or more copies. The
//! conditional and mixer cases now route through one owner, so these assert
//! that every entry point still classifies the same inputs the same way and
//! that the owner's classification is the intended one, not merely consistent.

use crate::api::case::{BenchError, Correctness};

fn classify(result: Result<Correctness, BenchError>) -> (&'static str, String) {
    match result {
        Ok(Correctness::Exact) => ("ok-exact", String::new()),
        Ok(_) => ("ok-other", String::new()),
        Err(error @ BenchError::CorrectnessViolation(_)) => ("err-correctness", error.to_string()),
        Err(error) => ("err-other", error.to_string()),
    }
}

fn sparse_output_shapes() -> Vec<(&'static str, Vec<Vec<u8>>, Option<Vec<Vec<u8>>>)> {
    let words = |values: &[u32]| -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    };
    vec![
        ("no baseline", vec![words(&[1]), words(&[7])], None),
        (
            "arity mismatch",
            vec![words(&[1])],
            Some(vec![words(&[1]), words(&[7])]),
        ),
        (
            "count mismatch",
            vec![words(&[2]), words(&[7, 9])],
            Some(vec![words(&[1]), words(&[7])]),
        ),
        (
            "truncated backend buffer",
            vec![words(&[3]), words(&[7])],
            Some(vec![words(&[3]), words(&[7, 9, 11])]),
        ),
        (
            "truncated baseline buffer",
            vec![words(&[3]), words(&[7, 9, 11])],
            Some(vec![words(&[3]), words(&[7])]),
        ),
        (
            "equal sets out of order",
            vec![words(&[3]), words(&[11, 7, 9])],
            Some(vec![words(&[3]), words(&[7, 9, 11])]),
        ),
        (
            "disjoint sets",
            vec![words(&[2]), words(&[7, 9])],
            Some(vec![words(&[2]), words(&[7, 10])]),
        ),
        ("empty sets", vec![words(&[0]), vec![]], Some(vec![words(&[0]), vec![]])),
    ]
}

/// Every entry point into the sparse fired-set verifier classifies each output
/// shape the same way, and classifies it correctly. The two conditional cases
/// differ only in the wording they pass in.
///
/// The verdict alone is not enough. The batched copy lacked the short-buffer
/// check, and without it a truncated buffer still ends as a correctness
/// violation, just one that blames the wrong thing. The expected message
/// fragment is what holds that check in place.
#[test]
fn conditional_sparse_verifier_classifies_every_shape_identically() {
    let expected = [
        ("err-correctness", "did not capture baseline sparse"),
        ("err-correctness", "sparse output count mismatch"),
        ("err-correctness", "count mismatch: backend returned 2, baseline returned 1"),
        ("err-correctness", "output buffer shorter than reported count"),
        ("err-correctness", "output buffer shorter than reported count"),
        ("ok-exact", ""),
        ("err-correctness", "set differs between backend and baseline"),
        ("ok-exact", ""),
    ];

    for ((label, outputs, baseline), (verdict, fragment)) in
        sparse_output_shapes().into_iter().zip(expected)
    {
        let eval = classify(crate::cases::conditional::verify_sparse_outputs(
            crate::cases::conditional_eval::LABELS,
            &outputs,
            baseline.as_deref(),
        ));
        let batch = classify(crate::cases::conditional::verify_sparse_outputs(
            crate::cases::conditional_batch::LABELS,
            &outputs,
            baseline.as_deref(),
        ));

        assert_eq!(eval.0, batch.0, "sparse verifier drift on `{label}`");
        assert_eq!(eval.0, verdict, "sparse verifier verdict on `{label}`");
        assert!(eval.1.contains(fragment), "eval `{label}`: {}", eval.1);
        assert!(batch.1.contains(fragment), "batch `{label}`: {}", batch.1);
    }
}

/// The exact words the bench-wide mixer produced before it was collapsed
/// from eleven byte-identical copies onto one owner.
const MIX32_PINNED_WORDS: [u32; 5] = [
    0x00000000,
    0x688990C0,
    0x6768824A,
    0x01FCE552,
    0x5CE575F0,
];

/// The bench-wide mixer has one owner. This pins the exact words it produces,
/// so a rewrite of the owner cannot silently change every generated fixture in
/// the crate at once.
#[test]
fn mix32_owner_produces_its_pinned_words() {
    assert_eq!(
        [
            crate::cases::mix32(0),
            crate::cases::mix32(1),
            crate::cases::mix32(0xFFFF_FFFF),
            crate::cases::mix32(0x9E37_79B9),
            crate::cases::mix32(0x517C_C1B7),
        ],
        MIX32_PINNED_WORDS
    );
}

/// The word codec has one owner. It refuses every out-of-range and overflowing
/// index rather than reading or writing past a buffer, and round-trips every
/// index it accepts.
///
/// Both megakernel cases used to carry their own copy; only the buffer name in
/// the message differed, and that name is now an argument.
#[test]
fn word_codec_refuses_out_of_range_and_round_trips_the_rest() {
    use crate::cases::byte_pack::{read_word, write_word};

    let mut buffer = vec![0_u8; 64];
    for word in 0..16_u32 {
        let value = word.wrapping_mul(2_654_435_761);

        write_word(&mut buffer, word, value, "guard buffer").expect("word is in range");

        assert_eq!(
            read_word(&buffer, word, "guard buffer").expect("word is in range"),
            value
        );
    }

    for out_of_range in [16_u32, 17, 4_000_000_000, u32::MAX] {
        let read = read_word(&buffer, out_of_range, "guard buffer")
            .expect_err("a word past the buffer must never read");
        let write = write_word(&mut buffer, out_of_range, 1, "guard buffer")
            .expect_err("a word past the buffer must never write");

        // The owner's own range check must be what rejects, not a short slice
        // handed to the wire decoder underneath it: only this branch can name
        // the index that was asked for.
        let expected = format!("guard buffer word {out_of_range} is outside output buffer");
        assert_eq!(read.to_string(), format!("Correctness violation: {expected}"));
        assert_eq!(write.to_string(), format!("Execution failed: {expected}"));
    }

    assert_eq!(buffer.len(), 64, "a rejected write must not resize the buffer");
}

/// Zero elapsed time reports zero, not an infinity or a division panic, and the
/// scaled rate saturates instead of overflowing.
#[test]
fn rate_helpers_are_total_over_their_whole_domain() {
    use crate::cases::byte_pack::{gb_per_second, rate_per_second_x1000};

    assert_eq!(gb_per_second(1_000, 0), 0.0);
    assert_eq!(rate_per_second_x1000(1_000, 0), 0);
    assert_eq!(gb_per_second(2_000, 1_000), 2.0);
    assert_eq!(rate_per_second_x1000(1, 1_000_000_000), 1_000);
    assert_eq!(rate_per_second_x1000(u64::MAX, 1), u64::MAX);
}
