use super::*;

/// The exhaustive literal must describe the same buffer the builders produce.
///
/// Why this exists: the guard above is only a guard if it stays a faithful
/// stand-in for a real `BufferDecl`. If it drifted (say someone "fixed" a
/// compile error by changing `count` instead of handling their new field), the
/// guard would still compile while no longer describing anything production
/// builds. Comparing it against the builder output pins it.
#[test]
fn exhaustive_buffer_decl_literal_matches_the_builder() {
    assert_eq!(
        exhaustive_buffer_decl(),
        out_buf(),
        "Fix: the exhaustive BufferDecl literal drifted from BufferDecl::storage(...).with_count(4). \
         Update the literal to match, do not weaken the comparison."
    );
}

// ---------------------------------------------------------------------------
// 2. The root defect: the wire format is lossy
// ---------------------------------------------------------------------------

/// THE REASON THIS CHANGE EXISTS: ROUND-TRIPPING A PROGRAM MUST NOT CHANGE
/// WHETHER IT IS VALID.
///
/// Why this exists: before wire rev 6 the encoder never emitted
/// `bytes_extraction`, so `from_wire` reconstructed it as `false`. That field
/// is the per-buffer opt-in that gate V013 consults to admit a
/// `DataType::Bytes` load or store. A program that declared the opt-in, was
/// valid, and was then persisted and reloaded came back INVALID, with nobody
/// having edited it. Semantics changing under serialization is disqualifying on
/// its own, and it needs no cache to hurt anyone: persist, reload, and the
/// program no longer compiles.
///
/// What breaks if this regresses: a program becomes invalid purely by being
/// saved and loaded. Assert the VERDICT on both sides, not the field, because
/// the verdict is the observable a user experiences.
#[test]
fn wire_round_trip_must_not_change_whether_a_program_is_valid() {
    let bytes_buffer = BufferDecl::storage("b", 0, BufferAccess::ReadWrite, DataType::Bytes)
        .with_count(4)
        .with_bytes_extraction(true);
    let program = Program::wrapped(
        vec![bytes_buffer],
        [64, 1, 1],
        vec![Node::store("b", Expr::gid_x(), Expr::u32(1))],
    );

    let before = vyre_foundation::validate::validate(&program);
    assert_eq!(
        before.len(),
        0,
        "fixture must start VALID for this test to mean anything: {before:?}"
    );

    let wire = program.to_wire().expect("fixture must encode");
    let decoded = Program::from_wire(&wire).expect("fixture must decode");
    let after = vyre_foundation::validate::validate(&decoded);

    assert_eq!(
        after.len(),
        0,
        "Fix: round-tripping a valid program made it INVALID. The wire encoder dropped a field the \
         validator reads (bytes_extraction gates V013), so persisting and reloading a program \
         changed its meaning. Errors: {after:?}"
    );
}

/// All three previously-dropped `BufferDecl` fields now survive the wire round
/// trip, asserted on their exact non-default values.
///
/// Why this exists: these three fields (`shape_predicate`, `linear_type`,
/// `bytes_extraction`) were absent from the VIR0 encoding entirely, and
/// `from_wire` hardcoded each to its default. That made `Program::fingerprint`,
/// which hashes the canonical wire bytes, structurally blind to them, so two
/// materially different programs shared a cache identity. Wire rev 6 emits
/// them.
///
/// Asserts the DECODED VALUES, never merely that decode succeeded, because the
/// failure mode was a silently defaulted field that still looked like a
/// well-formed `BufferDecl`. "It decoded" was true the whole time the defect
/// existed.
#[test]
fn wire_round_trip_preserves_all_three_formerly_dropped_bufferdecl_fields() {
    let original = out_buf()
        .with_shape_predicate(ShapePredicate::AtLeast(64))
        .with_linear_type(LinearType::Affine)
        .with_bytes_extraction(true);
    let program = program_with(vec![original]);

    let wire = program.to_wire().expect("fixture must encode");
    let decoded = Program::from_wire(&wire).expect("fixture must decode");
    let after = &decoded.buffers()[0];

    assert_eq!(
        after.shape_predicate(),
        Some(&ShapePredicate::AtLeast(64)),
        "Fix: shape_predicate must survive the wire round trip with its exact refinement."
    );
    assert_eq!(
        after.linear_type(),
        LinearType::Affine,
        "Fix: linear_type must survive the wire round trip. Decoding it as Unrestricted silently \
         weakens the declared discipline."
    );
    assert!(
        after.bytes_extraction,
        "Fix: bytes_extraction must survive the wire round trip. It gates V013."
    );

    // Everything else on BufferDecl survives. Listed field by field so a
    // newly-lossy field is caught here rather than in a cache months later.
    assert_eq!(after.name(), "out");
    assert_eq!(after.binding, 0);
    assert_eq!(after.access(), BufferAccess::ReadWrite);
    assert_eq!(after.kind, MemoryKind::Global);
    assert_eq!(after.element, DataType::U32);
    assert_eq!(after.count(), 4);
    assert!(!after.is_output());
    assert!(!after.is_pipeline_live_out());
    assert_eq!(after.output_byte_range(), None::<Range<usize>>);
    assert_eq!(after.hints().coalesce_axis, None);
    assert_eq!(after.hints().preferred_alignment, 0);
    assert_eq!(after.hints().cache_locality, CacheLocality::Temporal);
}

/// Every `ShapePredicate` variant round-trips, including the recursive ones.
///
/// Why this exists: the encoder added for rev 6 tags nine variants by hand and
/// three of them (`And`, `Or`, `Not`) recurse. A tag swapped between two
/// same-shaped variants (`AtLeast` against `AtMost`, or the four `AffineRange`
/// operands reordered) produces a predicate that decodes cleanly and means
/// something different, which the optimizer then uses to prove a loop bound.
/// Only asserting the decoded value per variant catches that.
///
/// `AffineRange` carries NEGATIVE coefficients and the i64 extremes on purpose:
/// they are encoded as a two's-complement bit pattern, and a sign-extending or
/// truncating conversion corrupts exactly these and nothing else.
#[test]
fn every_shape_predicate_variant_survives_the_wire_round_trip() {
    let cases = vec![
        ShapePredicate::AtLeast(64),
        ShapePredicate::AtMost(4096),
        ShapePredicate::Exactly(256),
        ShapePredicate::MultipleOf(32),
        ShapePredicate::ModEquals {
            modulus: 64,
            remainder: 7,
        },
        ShapePredicate::AffineRange {
            scale: -3,
            offset: -17,
            min: i64::MIN,
            max: i64::MAX,
        },
        ShapePredicate::Not(Box::new(ShapePredicate::AtLeast(8))),
        ShapePredicate::And(
            Box::new(ShapePredicate::AtLeast(8)),
            Box::new(ShapePredicate::AtMost(64)),
        ),
        ShapePredicate::Or(
            Box::new(ShapePredicate::Exactly(1)),
            Box::new(ShapePredicate::Not(Box::new(ShapePredicate::MultipleOf(3)))),
        ),
    ];

    for predicate in cases {
        let program = program_with(vec![out_buf().with_shape_predicate(predicate.clone())]);
        let wire = program.to_wire().expect("fixture must encode");
        let decoded = Program::from_wire(&wire).expect("fixture must decode");
        assert_eq!(
            decoded.buffers()[0].shape_predicate(),
            Some(&predicate),
            "Fix: {predicate:?} did not survive the wire round trip."
        );
    }
}

/// BACKWARD COMPATIBILITY: the rev-6 decoder still reads rev-4 and rev-5
/// payloads, and the three new reads are genuinely GATED on the declared
/// version rather than performed unconditionally.
///
/// Why this exists: the three fields land INSIDE each buffer record, not
/// appended after the envelope, so an ungated read would consume bytes
/// belonging to the next buffer and decode a plausible WRONG buffer table from
/// a valid older blob. The version gate is the only thing preventing that, and
/// nothing else asserts the gate is wired to the version.
///
/// Method, and its boundary stated plainly: the second half takes a rev-6 blob
/// and relabels its version bytes as rev 5. The body still carries the three
/// fields, so a version-gated decoder skips them and reports trailing bytes,
/// while an UNGATED decoder would read them and succeed. The relabel therefore
/// MUST fail, and that failure is the proof the gate is version driven. This
/// tests the gate mechanism and the accepted version range; it does NOT decode
/// a byte-for-byte legacy blob, because synthesizing one means duplicating the
/// encoder's node-record framing inside this test, where a mis-synthesized blob
/// would fail for the wrong reason and prove nothing.
///
/// AND BE CLEAR ABOUT WHAT COMPATIBILITY MEANS HERE, because it will otherwise
/// be misread as a repair: old bytes REMAIN LOSSY FOREVER. A rev-4 or rev-5
/// blob written from a program that did carry a linear type, a bytes-extraction
/// opt-in, or a shape predicate lost that information at ENCODE time. No
/// decoder can recover a field the writer never emitted. The guarantee is
/// faithfulness to what those bytes say, not recovery of what the program was.
/// No REAL artifact is affected today: no production code writes any of the
/// three fields (every writer in the tree is a `#[cfg(test)]` fixture) and
/// exatok writes none of them, so the population of lossy stored programs is
/// empty. That is what makes this a clean version bump rather than a migration.
#[test]
fn rev_six_decoder_accepts_older_payloads_and_gates_the_new_reads_on_version() {
    use vyre_foundation::serial::wire::framing::{
        wire_format_version_is_supported, WIRE_FORMAT_VERSION,
    };

    assert_eq!(
        WIRE_FORMAT_VERSION, 6,
        "Fix: this test encodes the rev-6 compatibility contract. If the version moved, decide \
         what the new revision does to the three buffer fields and update this test."
    );
    assert!(
        !wire_format_version_is_supported(3),
        "Fix: rev 3 predates the metadata layout and must stay rejected."
    );
    assert!(
        wire_format_version_is_supported(4),
        "Fix: rev-4 payloads must keep decoding. Rejecting them would invalidate stored programs \
         for no gain, which this bump was specifically routed to avoid."
    );
    assert!(
        wire_format_version_is_supported(5),
        "Fix: rev-5 payloads must keep decoding."
    );
    assert!(
        wire_format_version_is_supported(6),
        "Fix: the current revision must be readable by its own decoder."
    );
    assert!(
        !wire_format_version_is_supported(7),
        "Fix: an unknown future revision must be refused with a version diagnostic, never parsed \
         on a guess."
    );

    // A rev-6 body relabelled as rev 5 must be REJECTED, which is only true if
    // the three reads are gated on the declared version.
    let program = program_with(vec![out_buf()]);
    let mut relabelled = program.to_wire().expect("fixture must encode");
    assert_eq!(
        u16::from_le_bytes([relabelled[4], relabelled[5]]),
        6,
        "fixture must be stamped rev 6 before relabelling"
    );
    relabelled[4] = 5;
    relabelled[5] = 0;
    assert!(
        Program::from_wire(&relabelled).is_err(),
        "Fix: a rev-6 body relabelled rev 5 decoded successfully, which means the three new buffer \
         reads are NOT gated on the version. An ungated read consumes bytes belonging to the next \
         buffer, so a genuine rev-5 blob would decode into a plausible wrong buffer table."
    );
}

/// The loss is CONFINED TO `BufferDecl`: every probed `Node` and `Expr` field
/// survives the wire round trip byte for byte.
///
/// Why this exists: a peer's cache digest hashes the program BODY through the
/// wire node encoder. If the lossiness extended to any Node or Expr field,
/// their key would silently under-discriminate too, and their fix would have
/// to change. Establishing the boundary is what makes "fix the buffer encoding"
/// a complete fix rather than a partial one.
///
/// Method: compare the raw `Debug` rendering of the entry tree before and
/// after the round trip. `Debug` prints every field of every variant, so a
/// dropped or defaulted field shows up as a textual difference. This is
/// stronger than comparing with `==`, because `Program`'s `PartialEq` is
/// itself blind to some fields (see
/// [`fingerprint_is_not_a_refinement_of_program_equality`]).
///
/// Stated boundary: this covers the node and expression shapes constructed
/// below. It does NOT cover `Node::Opaque` / extension nodes, collectives
/// (`AllReduce`, `AllGather`, `ReduceScatter`, `Broadcast`), `IndirectDispatch`,
/// the async family, or `Trap` / `Resume`, because those need registered
/// extensions or capabilities this test does not install. Those shapes are an
/// unestablished gap, recorded here rather than left silent.
#[test]
fn wire_round_trip_preserves_every_probed_node_and_expr_field() {
    let cases: Vec<(&str, Vec<Node>)> = vec![
        (
            "let+assign+store",
            vec![
                Node::let_bind("x", Expr::u32(3)),
                Node::assign("x", Expr::u32(4)),
                Node::store("out", Expr::u32(0), Expr::var("x")),
            ],
        ),
        (
            "if with both arms",
            vec![Node::If {
                cond: Expr::lt(Expr::gid_x(), Expr::u32(8)),
                then: vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
                otherwise: vec![Node::store("out", Expr::u32(1), Expr::u32(2))],
            }],
        ),
        (
            "loop",
            vec![Node::Loop {
                var: "i".into(),
                from: Expr::u32(0),
                to: Expr::u32(4),
                body: vec![Node::store("out", Expr::var("i"), Expr::u32(1))],
            }],
        ),
        ("return", vec![Node::return_()]),
        (
            "block non-empty",
            vec![Node::Block(vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::u32(1),
            )])],
        ),
        ("block empty", vec![Node::Block(Vec::new())]),
        (
            "select / cast / fma / load / buf_len",
            vec![
                Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::select(
                        Expr::lt(Expr::gid_x(), Expr::buf_len("out")),
                        Expr::cast(DataType::U32, Expr::load("out", Expr::u32(0))),
                        Expr::fma(Expr::u32(1), Expr::u32(2), Expr::u32(3)),
                    ),
                ),
                Node::store(
                    "out",
                    Expr::u32(1),
                    Expr::bitxor(Expr::gid_x(), Expr::u32(3)),
                ),
            ],
        ),
    ];

    for (label, entry) in cases {
        let program = Program::wrapped(vec![out_buf()], [64, 1, 1], entry);
        let wire = program
            .to_wire()
            .unwrap_or_else(|e| panic!("Fix: fixture `{label}` must encode: {e}"));
        let decoded = Program::from_wire(&wire)
            .unwrap_or_else(|e| panic!("Fix: fixture `{label}` must decode: {e}"));
        assert_eq!(
            format!("{:?}", program.entry()),
            format!("{:?}", decoded.entry()),
            "Fix: wire round trip changed the entry tree for `{label}`. A Node or Expr field is \
             now lossy, which under-discriminates every fingerprint-keyed cache AND the backend \
             pipeline cache digest that hashes the node list."
        );
    }
}

// ---------------------------------------------------------------------------
// 3. The key/artifact tabulation
// ---------------------------------------------------------------------------

/// BOTH DERIVED KEYS OVER `BufferDecl` now discriminate all three of the fields
/// that used to be invisible to them, and this test holds them to it TOGETHER.
///
/// Why this exists: every cache in this crate that keys on the fingerprint is
/// implicitly assuming `fingerprint(a) == fingerprint(b)` implies the two
/// programs are interchangeable, and a great deal of code assumes `a == b` means
/// the same thing. The enumeration behind this file found THREE independent keys
/// derived over one `BufferDecl`, each covering a DIFFERENT subset of its
/// fields:
///
///   1. `to_wire` / `Program::fingerprint`. Dropped `bytes_extraction`,
///      `linear_type` and `shape_predicate` entirely. FIXED at wire rev 6.
///   2. `buffer_decl_canonical_key`, which decides `Program::eq` through
///      `structural_eq`. Covered `bytes_extraction` and omitted the other two:
///      one of three wired in, two forgotten. FIXED, and it now calls the SAME
///      encoders as key 1 so the two cannot drift.
///   3. `normalized_cache_digest`, which EXCLUDES all three deliberately,
///      because a backend cache label must not vary on a declaration-level
///      discipline that cannot change generated code. Still excluded, on
///      purpose, and owned elsewhere.
///
/// How key 2 was found, because the method matters more than the fix: this test
/// originally asserted only that the FINGERPRINT discriminated each field, and
/// each iteration first asserted the PRECONDITION that `Program::eq`
/// distinguished that field, on the grounds that a fixture whose two halves
/// compare equal proves nothing about a key. The `bytes_extraction` iteration
/// passed and the `linear_type` iteration tripped ON THE PRECONDITION, first
/// run, in a test written for something else. The negative control found a
/// second defect that no amount of reading the encoder would have surfaced.
///
/// What breaks if this regresses: both formerly omitted fields DECIDE
/// VALIDATION VERDICTS, `linear_type` through `validate::linear_type` and
/// `shape_predicate` through `check_shape_predicates`. So while either key was
/// blind, a VALID program and an INVALID one compared equal or hashed alike, and
/// any code shaped as "a == b, therefore same program, therefore safe to reuse"
/// was wrong for them. If key 1 regresses every fingerprint-keyed cache goes
/// back to conflating materially different programs; if key 2 regresses the
/// optimizer's `changed` flag can report FALSE for a real change, and a fixpoint
/// loop stops early on a program it still needed to rewrite.
///
/// The test asserts the two keys AGREE per field rather than checking them in
/// separate tests, because agreement is the actual invariant: a field that moves
/// one key and not the other is how this bug class starts.
#[test]
fn both_derived_keys_discriminate_every_buffer_field_they_used_to_drop() {
    let plain = program_with(vec![out_buf()]);

    // One field at a time, asserting BOTH keys, so a partial regression (an
    // encoder keeping two of the three, or equality and the fingerprint drifting
    // apart on one field) cannot hide behind the others.
    for (label, buffer) in [
        ("bytes_extraction", out_buf().with_bytes_extraction(true)),
        (
            "linear_type",
            out_buf().with_linear_type(LinearType::Affine),
        ),
        (
            "shape_predicate",
            out_buf().with_shape_predicate(ShapePredicate::AtLeast(64)),
        ),
    ] {
        let varied = program_with(vec![buffer]);
        assert_ne!(
            plain, varied,
            "Fix: Program::eq must distinguish {label}. buffer_decl_canonical_key has stopped \
             covering it, so two programs differing only in {label} now compare EQUAL. Because \
             that field decides a validation verdict, a valid and an invalid program can now be \
             treated as the same program."
        );
        assert_ne!(
            plain.fingerprint(),
            varied.fingerprint(),
            "Fix: fingerprint must distinguish {label}. The wire encoder has stopped emitting it, \
             so every fingerprint-keyed cache can now serve one program's artifact for another."
        );
    }

    // Item 2: structure that canonicalization erases.
    let (target, primer) = (indexed_target(), indexed_primer());
    assert_ne!(
        target, primer,
        "Program::eq must distinguish a leading empty Block."
    );
    assert_eq!(
        target.fingerprint(),
        primer.fingerprint(),
        "PROBED: fingerprint cannot distinguish a Let-free Block because canonicalization \
         splices it before hashing."
    );
    // Exact node counts, so the difference is not merely nominal.
    assert_eq!(target.stats().node_count, 5);
    assert_eq!(primer.stats().node_count, 6);
}
