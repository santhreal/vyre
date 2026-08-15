//! Pipeline-cache fingerprints are exact authenticated artifact identities, and
//! they follow the canonical neutral artifact bytes rather than the host that
//! produced them.
//!
//! # The class this closes
//!
//! A cache key that is not exactly the artifact digest is a cache that can serve
//! the wrong megakernel: too coarse and two different artifacts collide, too fine
//! and one artifact compiled twice misses its own entry. Both failures are
//! silent, and the second one is only visible across hosts, where nothing but the
//! canonical bytes is shared.
//!
//! The fixtures are enumerated once and every identity assertion runs over all of
//! them, so a fixture added for one contract is covered by the others too. This
//! file previously had a twin, `fingerprint_cross_host.rs`, with a byte-identical
//! `artifact` helper and a different subset of the same assertions: each proved
//! its distinctness on a program pair the other did not use, so neither covered
//! the other's pair.
//!
//! # What it does not catch
//!
//! It does not prove the digest is collision resistant, and it does not prove two
//! genuinely different hosts agree: it proves the fingerprint reads nothing but
//! the artifact, which is what makes cross-host agreement follow from the
//! artifact being canonical.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_megakernel::{compile, CompileRequest, Digest, ExternalFacts, SearchBudget};
use vyre_runtime::pipeline_cache::PipelineFingerprint;

fn artifact(program: Program) -> vyre_megakernel::Artifact {
    let mut graph = ProgramGraph::new();
    for buffer in program.buffers() {
        graph
            .add_external_value(
                buffer.name(),
                ValueContract {
                    dtype: buffer.element(),
                    shape: vec![ShapeDim::Known(u64::from(buffer.count()))],
                    access: buffer.access(),
                    lifetime: ValueLifetime::Invocation,
                },
            )
            .unwrap();
    }
    graph
        .add_node("main", program, Vec::new(), Vec::new())
        .unwrap();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 0, 0, 1),
        1_000_000,
    )
    .validate()
    .unwrap();
    compile(&request).unwrap()
}

fn single_store() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("in", Expr::u32(0)),
        )],
    )
}

fn return_only(out_count: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(out_count)],
        [1, 1, 1],
        vec![Node::Return],
    )
}

/// Every program whose artifact identity is asserted, named for failure messages.
///
/// Built fresh per call so a test that compiles a fixture twice gets two
/// independent compiles rather than one artifact read twice.
fn fixtures() -> Vec<(&'static str, Program)> {
    vec![
        ("empty", Program::empty()),
        ("single store", single_store()),
        ("return with one output word", return_only(1)),
        ("return with two output words", return_only(2)),
    ]
}

/// The fingerprint of every fixture is exactly its artifact digest.
///
/// Not "derived from" or "consistent with": the same 32 bytes. Any transform on
/// the way in is a second identity function nothing else knows about, and the hex
/// form is what a cache directory is named after, so its width is pinned too.
#[test]
fn the_fingerprint_is_the_neutral_artifact_digest() {
    for (name, program) in fixtures() {
        let artifact = artifact(program);
        let fingerprint = PipelineFingerprint::of(&artifact);
        assert_eq!(
            fingerprint.0,
            artifact.digest().0,
            "the {name} fingerprint is not the artifact digest"
        );
        assert_eq!(
            fingerprint.hex().len(),
            64,
            "the {name} fingerprint hex is not 32 bytes wide"
        );
        // One type under both paths: the crate-root re-export and the owning
        // module. A second `PipelineFingerprint` would split callers silently.
        let _: PipelineFingerprint = vyre_runtime::PipelineFingerprint::of(&artifact);
    }
}

/// Reading the same artifact twice yields the same fingerprint.
#[test]
fn the_fingerprint_is_deterministic_for_one_artifact() {
    for (name, program) in fixtures() {
        let artifact = artifact(program);
        assert_eq!(
            PipelineFingerprint::of(&artifact),
            PipelineFingerprint::of(&artifact),
            "the {name} fingerprint changed between two reads of one artifact"
        );
    }
}

/// One program compiled twice lands on one cache identity.
///
/// The cross-host contract stated locally: two independent compiles share only
/// the canonical artifact bytes, so a fingerprint that read anything else -
/// a pointer, a timestamp, a compile counter - would diverge here and every host
/// would miss every other host's cache entry.
#[test]
fn independently_compiled_identical_programs_share_identity() {
    for ((name, first), (_, second)) in fixtures().into_iter().zip(fixtures()) {
        let first = artifact(first);
        let second = artifact(second);
        assert_eq!(
            PipelineFingerprint::of(&first),
            PipelineFingerprint::of(&second),
            "two independent compiles of {name} did not share a cache identity"
        );
    }
}

/// No two distinct fixtures share a cache identity.
///
/// Every pair, not one pair: the two files this replaces each picked a different
/// pair, so the empty-versus-store distinction and the output-count distinction
/// were proven in different binaries and neither covered both.
#[test]
fn distinct_programs_do_not_share_cache_identity() {
    let fingerprints: Vec<(&'static str, PipelineFingerprint)> = fixtures()
        .into_iter()
        .map(|(name, program)| (name, PipelineFingerprint::of(&artifact(program))))
        .collect();
    for (index, (left_name, left)) in fingerprints.iter().enumerate() {
        for (right_name, right) in &fingerprints[index + 1..] {
            assert_ne!(
                left, right,
                "{left_name} and {right_name} collide on one cache identity"
            );
        }
    }
}

/// The hex form is lowercase, so a cache path is one spelling per artifact.
#[test]
fn the_fingerprint_hex_is_lowercase() {
    for (name, program) in fixtures() {
        let hex = PipelineFingerprint::of(&artifact(program)).hex();
        assert!(
            hex.bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "the {name} fingerprint hex is not lowercase: {hex}"
        );
    }
}
