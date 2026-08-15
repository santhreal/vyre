//! Every scalar dual evaluator against the u32 contract it must reproduce.
//!
//! WHY: the sweep restated the same four lines per operation, so an operation
//! was added by copying a block, and the corpus it swept was a fourth copy of
//! the anchor list the storage-graph matrices carried. Both are tables now: the
//! corpus has one owner in `tests/support/scalar_corpus.rs`, and each operation
//! is a row rather than a paragraph.
//!
//! Every ordered pair of corpus values is evaluated, so the corpus length is
//! part of the contract here in a way it is not for a sampled sweep.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use vyre_primitives::{
    ArithAdd, ArithMul, Clz, CompareEq, CompareLt, Popcount, ShiftLeft, ShiftRight,
};
use vyre_reference::dual_impls::{EvalError, ReferenceEvaluator};
use vyre_reference::workgroup::Memory;

#[path = "support/scalar_corpus.rs"]
mod scalar_corpus;

use scalar_corpus::{u32_anchors, u32_evaluator_corpus};

/// One binary evaluator and the contract its output word must satisfy.
struct BinaryRow {
    /// The marker type the evaluator is implemented for, and the name every
    /// failure message uses.
    name: &'static str,
    /// The evaluator under test.
    evaluate: fn(&[Memory]) -> Result<Memory, EvalError>,
    /// The word the evaluator must produce.
    expected: fn(u32, u32) -> u32,
}

/// One unary evaluator and the contract its output word must satisfy.
struct UnaryRow {
    /// The marker type the evaluator is implemented for, and the name every
    /// failure message uses.
    name: &'static str,
    /// The evaluator under test.
    evaluate: fn(&[Memory]) -> Result<Memory, EvalError>,
    /// The word the evaluator must produce.
    expected: fn(u32) -> u32,
}

fn binary_rows() -> Vec<BinaryRow> {
    vec![
        BinaryRow {
            name: "ArithAdd",
            evaluate: |inputs| ArithAdd.evaluate(inputs),
            expected: |left, right| left.wrapping_add(right),
        },
        BinaryRow {
            name: "ArithMul",
            evaluate: |inputs| ArithMul.evaluate(inputs),
            expected: |left, right| left.wrapping_mul(right),
        },
        BinaryRow {
            name: "CompareEq",
            evaluate: |inputs| CompareEq.evaluate(inputs),
            expected: |left, right| u32::from(left == right),
        },
        BinaryRow {
            name: "CompareLt",
            evaluate: |inputs| CompareLt.evaluate(inputs),
            expected: |left, right| u32::from(left < right),
        },
        BinaryRow {
            name: "ShiftLeft",
            evaluate: |inputs| ShiftLeft.evaluate(inputs),
            expected: |left, right| left << (right & 31),
        },
        BinaryRow {
            name: "ShiftRight",
            evaluate: |inputs| ShiftRight.evaluate(inputs),
            expected: |left, right| left >> (right & 31),
        },
    ]
}

fn unary_rows() -> Vec<UnaryRow> {
    vec![
        UnaryRow {
            name: "Clz",
            evaluate: |inputs| Clz.evaluate(inputs),
            expected: u32::leading_zeros,
        },
        UnaryRow {
            name: "Popcount",
            evaluate: |inputs| Popcount.evaluate(inputs),
            expected: u32::count_ones,
        },
    ]
}

/// The corpus, with the guard that keeps the case-count assertions meaningful:
/// an empty corpus makes every `checked == len * rows` comparison trivially
/// true, so the sweep would report success having evaluated nothing.
fn corpus() -> Vec<u32> {
    let values = u32_evaluator_corpus();
    let anchors = u32_anchors().len();
    assert!(
        values.len() > anchors,
        "Fix: the evaluator corpus holds {} value(s) against {anchors} anchors; its generator \
         walk contributes nothing and the sweep is only checking boundaries.",
        values.len()
    );
    values
}

fn word_payload(value: u32) -> Memory {
    Memory::from_bytes(value.to_le_bytes().to_vec())
}

/// A payload that is not a whole number of words, so every evaluator must
/// refuse it rather than read past what it was given.
fn unaligned_payload() -> Memory {
    Memory::from_bytes(vec![1, 2, 3])
}

fn word(memory: &Memory, name: &str) -> u32 {
    let bytes = memory.bytes();
    let [first, second, third, fourth] = bytes.as_slice() else {
        panic!(
            "Fix: {name} must return exactly one u32; it returned {} byte(s).",
            bytes.len()
        )
    };
    u32::from_le_bytes([*first, *second, *third, *fourth])
}

#[test]
fn binary_scalar_evaluators_match_the_u32_contract() {
    let values = corpus();
    let rows = binary_rows();
    let mut checked = 0usize;
    for &left in &values {
        for &right in &values {
            let inputs = [word_payload(left), word_payload(right)];
            for row in &rows {
                let output = (row.evaluate)(&inputs).unwrap_or_else(|error| {
                    panic!("Fix: {} must accept two u32 payloads: {error}", row.name)
                });
                assert_eq!(
                    word(&output, row.name),
                    (row.expected)(left, right),
                    "Fix: {} disagrees with its u32 contract for left={left:#010x} right={right:#010x}",
                    row.name
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, values.len() * values.len() * rows.len());
}

#[test]
fn unary_scalar_evaluators_match_the_u32_contract() {
    let values = corpus();
    let rows = unary_rows();
    let mut checked = 0usize;
    for value in values.iter().copied() {
        let inputs = [word_payload(value)];
        for row in &rows {
            let output = (row.evaluate)(&inputs).unwrap_or_else(|error| {
                panic!("Fix: {} must accept one u32 payload: {error}", row.name)
            });
            assert_eq!(
                word(&output, row.name),
                (row.expected)(value),
                "Fix: {} disagrees with its u32 contract for value={value:#010x}",
                row.name
            );
            checked += 1;
        }
    }
    assert_eq!(checked, values.len() * rows.len());
}

#[test]
fn scalar_evaluators_reject_wrong_arity_and_unaligned_payloads() {
    for row in binary_rows() {
        for inputs in [
            vec![word_payload(1)],
            vec![unaligned_payload(), word_payload(1)],
            vec![word_payload(1), unaligned_payload()],
            Vec::new(),
        ] {
            assert!(
                (row.evaluate)(&inputs).is_err(),
                "Fix: {} must refuse {} payload(s) that are not two whole words.",
                row.name,
                inputs.len()
            );
        }
    }

    for row in unary_rows() {
        for inputs in [Vec::new(), vec![unaligned_payload()]] {
            assert!(
                (row.evaluate)(&inputs).is_err(),
                "Fix: {} must refuse {} payload(s) that are not one whole word.",
                row.name,
                inputs.len()
            );
        }
    }
}

/// Markers that implement the evaluator but do not take a pair of words, with
/// the shape they take instead. A scalar sweep cannot feed them, and the
/// closure gate below refuses to let one be added silently: an entry here is a
/// recorded decision, not an exemption.
const NOT_A_SCALAR_WORD_PAIR: &[(&str, &str)] = &[
    ("Gather", "indexed read: a data buffer and an index buffer"),
    (
        "Scatter",
        "indexed write: a data buffer, an index buffer and values",
    ),
    ("Shuffle", "lane permutation across a workgroup payload"),
    ("Reduce", "folds a whole buffer to one word"),
    ("Scan", "prefix over a whole buffer"),
    ("HashBlake3", "digest over a byte payload of any length"),
    ("HashFnv1a", "digest over a byte payload of any length"),
    (
        "PatternMatchDfa",
        "compiled automaton plus a subject buffer",
    ),
    ("PatternMatchLiteral", "needle buffer plus a subject buffer"),
];

#[test]
fn every_reference_evaluator_is_swept_or_recorded_as_non_scalar() {
    let implemented = evaluator_markers();
    let swept: BTreeSet<&str> = binary_rows()
        .iter()
        .map(|row| row.name)
        .chain(unary_rows().iter().map(|row| row.name))
        .collect();
    let recorded: BTreeSet<&str> = NOT_A_SCALAR_WORD_PAIR
        .iter()
        .map(|(marker, _)| *marker)
        .collect();

    let unaccounted: Vec<&String> = implemented
        .iter()
        .filter(|marker| !swept.contains(marker.as_str()) && !recorded.contains(marker.as_str()))
        .collect();
    assert!(
        unaccounted.is_empty(),
        "Fix: {unaccounted:?} implement ReferenceEvaluator and this sweep neither exercises them \
         nor records why it cannot. Add a row, or add the marker to NOT_A_SCALAR_WORD_PAIR with \
         the payload shape it takes."
    );

    let both: Vec<&&str> = swept
        .iter()
        .filter(|marker| recorded.contains(**marker))
        .collect();
    assert!(
        both.is_empty(),
        "Fix: {both:?} are swept as scalar word evaluators and also recorded as taking another \
         shape. One of the two statements is wrong."
    );

    let stale: Vec<&&str> = swept
        .iter()
        .chain(recorded.iter())
        .filter(|marker| !implemented.contains(**marker))
        .collect();
    assert!(
        stale.is_empty(),
        "Fix: {stale:?} are named here but no longer implement ReferenceEvaluator in the frozen \
         surface. Drop them, or refresh the snapshot with \
         scripts/check_public_api_snapshot.sh --refresh vyre-reference."
    );
}

/// The markers that implement `ReferenceEvaluator`, from the frozen public-API
/// snapshot.
///
/// `scripts/check_public_api_snapshot.sh` regenerates the snapshot from rustdoc
/// and a byte-stability gate holds it equal to the crate's real surface, so a
/// new evaluator reaches this sweep through the gate that already forces a
/// snapshot refresh.
fn evaluator_markers() -> BTreeSet<String> {
    const PREFIX: &str =
        "impl vyre_reference::dual_impls::evaluator::ReferenceEvaluator for vyre_primitives::markers::";
    let path = reference_api_snapshot();
    let snapshot = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "Fix: the public-API snapshot at {} must be readable to enumerate the evaluators: {error}",
            path.display()
        )
    });
    let markers: BTreeSet<String> = snapshot
        .lines()
        .filter_map(|line| line.trim().strip_prefix(PREFIX))
        .map(str::to_string)
        .collect();
    assert!(
        !markers.is_empty(),
        "Fix: the public-API snapshot at {} lists no ReferenceEvaluator implementations. Refresh \
         it with scripts/check_public_api_snapshot.sh --refresh vyre-reference.",
        path.display()
    );
    markers
}

fn reference_api_snapshot() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .map(|directory| directory.join("docs/public-api/vyre-reference.txt"))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| {
            panic!(
                "Fix: no docs/public-api/vyre-reference.txt above {}. This sweep enumerates the \
                 evaluator surface from that snapshot.",
                manifest.display()
            )
        })
}
