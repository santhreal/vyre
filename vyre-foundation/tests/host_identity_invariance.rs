//! The class closed here: a persisted identity whose bytes depend on the host
//! that computed them.
//!
//! # The property proved here
//!
//! `Program::content_hash` is a cache identity. Two hosts that compile the
//! same program must file it under the same key, or a 32-bit consumer reads
//! past a 64-bit producer's cache and a big-endian consumer reads past both.
//! Every byte that reaches the digest is therefore fixed-width and
//! little-endian, and no value derived from `usize` or from native byte order
//! reaches it.
//!
//! # How it is proved
//!
//! - COMPILE TIME. The digest inputs come from
//!   `vyre_test_support::{expr_variants, ir_variants}`, whose fixture sets are
//!   checked against `EXPR_VARIANT_NAMES` and `NODE_VARIANT_NAMES` as the AST
//!   registry macro emits them. A variant added to the IR has no fixture, the
//!   coverage assertion fails, and the pinned digest below fails with it.
//! - RUN TIME, ONE CELL. [`CANONICAL_CORPUS_IDENTITY`] and
//!   [`FALLBACK_CORPUS_IDENTITY`] pin the digest. An encoder change that moves
//!   the identity is red until somebody records the new value.
//! - RUN TIME, EVERY CELL. `cargo xtask portability-evidence --write` runs this
//!   file on each host cell the support matrix claims, including a 32-bit cell
//!   and a big-endian cell, and fails when two cells disagree. The digest is
//!   printed below so the collector can read it out of the test output.
//!
//! Emulation carries the big-endian cell. That proves host arithmetic, byte
//! order and decoding, and it is never evidence of device support.

use std::collections::BTreeMap;

use vyre_foundation::hashing::domain_digest;
use vyre_foundation::ir::{DataType, Expr, Ident, Node, Program};
use vyre_foundation::serial::wire::MAX_TENSOR_RANK;
use vyre_test_support::expr_variants::{assert_covers_every_expr_variant, expr_variant_samples};
use vyre_test_support::ir_variants::{
    assert_covers_every_node_variant, node_variant_samples, single_u32_output_program,
};

/// Digest of the canonical identity of one program per declared IR variant.
///
/// Regenerate with `cargo xtask portability-evidence --write`, which recomputes
/// it on every claimed host cell and refuses to record a value two cells
/// disagree on.
const CANONICAL_CORPUS_IDENTITY: &str =
    "26c194baded92c255b685f4a9d06694d8c84644c9fe64e61def661404ffa0918";

/// Digest of the identity a program takes when canonical wire encoding fails.
///
/// The fallback used to mix a `rustc_hash::FxHasher` result, whose state is
/// `usize`-wide and whose byte reads are native-endian, so a program that
/// could not be wire-encoded took one identity on a 64-bit little-endian host
/// and a different one everywhere else.
const FALLBACK_CORPUS_IDENTITY: &str =
    "a4421238708c3f9e9f98eaca283f113edc30f6e09829c6fd92cd17ee013bfa7a";

/// Domain separator for the corpus digest.
const CORPUS_DOMAIN: &[u8] = b"vyre.host-identity.v1\0";

/// Render 32 digest bytes as lowercase hex.
fn hex(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(char::from_digit(u32::from(byte >> 4), 16).expect("nibble"));
        out.push(char::from_digit(u32::from(byte & 0x0f), 16).expect("nibble"));
    }
    out
}

/// Append one length-delimited labelled identity to the corpus buffer.
///
/// The length prefix is a fixed 8-byte little-endian count, not a host
/// `usize`, so the framing is the same width on every cell.
fn push_identity(buf: &mut Vec<u8>, label: &str, program: &Program) {
    let label_bytes = label.as_bytes();
    buf.extend_from_slice(&(label_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(label_bytes);
    buf.extend_from_slice(&program.content_hash());
}

/// One program per declared `Node` variant and per declared `Expr` variant,
/// keyed by variant name so the fold order is the declaration order.
fn canonical_corpus() -> BTreeMap<String, Program> {
    let node_samples = node_variant_samples();
    assert_covers_every_node_variant(&node_samples);
    let expr_samples = expr_variant_samples();
    assert_covers_every_expr_variant(&expr_samples);

    let mut corpus = BTreeMap::new();
    for sample in node_samples {
        corpus.insert(
            format!("node:{}", sample.variant),
            single_u32_output_program(vec![sample.node]),
        );
    }
    for sample in expr_samples {
        corpus.insert(
            format!("expr:{}", sample.variant),
            single_u32_output_program(vec![Node::Store {
                buffer: Ident::from("out"),
                index: Expr::u32(0),
                value: sample.expr,
            }]),
        );
    }
    corpus
}

/// A program the canonical wire encoder rejects, so its identity is taken by
/// the structural fallback.
///
/// The tensor rank is one past the wire-format bound, which the encoder
/// refuses rather than emit a blob no decoder accepts.
fn fallback_corpus() -> BTreeMap<String, Program> {
    let over_rank = DataType::TensorShaped {
        element: Box::new(DataType::F32),
        shape: core::iter::repeat_n(1u32, MAX_TENSOR_RANK + 1).collect(),
    };
    let mut corpus = BTreeMap::new();
    for sample in node_variant_samples() {
        corpus.insert(
            format!("fallback:{}", sample.variant),
            single_u32_output_program(vec![
                sample.node,
                Node::Store {
                    buffer: Ident::from("out"),
                    index: Expr::u32(0),
                    value: Expr::Cast {
                        target: over_rank.clone(),
                        value: Box::new(Expr::u32(1)),
                    },
                },
            ]),
        );
    }
    corpus
}

/// Fold a keyed corpus into one digest.
fn corpus_digest(corpus: &BTreeMap<String, Program>) -> String {
    let mut buf = Vec::new();
    for (label, program) in corpus {
        push_identity(&mut buf, label, program);
    }
    hex(&domain_digest(CORPUS_DOMAIN, &buf))
}

/// The fallback fixture must actually take the fallback path.
///
/// Without this the fallback digest would silently become a second copy of the
/// canonical one, and the pin would certify a path it never ran.
#[test]
fn fallback_fixture_is_rejected_by_the_canonical_wire_encoder() {
    let corpus = fallback_corpus();
    let (label, program) = corpus.iter().next().expect("fallback corpus is not empty");
    let error = program
        .to_wire()
        .expect_err("an over-rank tensor type must be refused by the wire encoder");
    let rendered = error.to_string();
    assert!(
        rendered.contains("rank"),
        "{label}: the refusal must name the rank bound it enforces, got: {rendered}"
    );
}

/// The canonical identity of the whole declared IR is one fixed value.
#[test]
fn canonical_identity_is_byte_identical_on_every_host_cell() {
    let digest = corpus_digest(&canonical_corpus());
    println!("VYRE-HOST-IDENTITY canonical {digest}");
    println!(
        "VYRE-HOST-CELL pointer_width={} endian={}",
        usize::BITS,
        if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        }
    );
    assert_eq!(
        digest, CANONICAL_CORPUS_IDENTITY,
        "canonical program identity moved; regenerate with `cargo xtask portability-evidence --write`"
    );
}

/// The fallback identity, taken when canonical encoding fails, is one fixed value.
#[test]
fn fallback_identity_is_byte_identical_on_every_host_cell() {
    let digest = corpus_digest(&fallback_corpus());
    println!("VYRE-HOST-IDENTITY fallback {digest}");
    assert_eq!(
        digest, FALLBACK_CORPUS_IDENTITY,
        "wire-hash fallback identity moved; regenerate with `cargo xtask portability-evidence --write`"
    );
}

/// The canonical and fallback digests are distinct.
///
/// A fallback that returned the canonical bytes would make both pins pass
/// while proving nothing about the fallback encoder.
#[test]
fn canonical_and_fallback_identities_are_distinct() {
    assert_ne!(
        corpus_digest(&canonical_corpus()),
        corpus_digest(&fallback_corpus()),
        "the fallback path must produce its own identity"
    );
}
