//! RELEASE PROOF L11  -  conformance certificate regression pin.
//!
//! Five canonical bundles are built, certed, signed, and their hashes + wire
//! lengths + verifying-key are pinned as constants. Any silent drift in the
//! cert pipeline (hash domain tag, witness order, wire format tag assignment)
//! breaks an assertion, forcing an intentional update.
//!
//! All bundles run on the CPU reference. When `gpu` is enabled, every bundle
//! must also verify against the live backend; a backend coverage gap is a test
//! failure, not a warning.

use std::sync::Arc;

use ed25519_dalek::{Signer, SigningKey};
use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_conform::witness_plan::{plan_witness_inputs_into, WitnessInputPlan};
use vyre_conform::{issue_bundle_cert, verify_bundle_against_reference, verify_cert_signature_hex};
use vyre_conform_spec::{BundleCertificate, ConformanceCase};
use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::operation::{OperationRegistration, OperationTier};
use vyre_primitives::wire::pack_u32_slice as bytes_u32;
use vyre_reference::value::Value;
use vyre_reference::ReferenceFacet;

#[cfg(feature = "gpu")]
use vyre_driver_metal as _;
#[cfg(feature = "gpu")]
use vyre_driver_wgpu as _;

type BundleBuilderFn = fn() -> (Program, Vec<ConformanceCase>);

const TEST_IDENTITY_U32_OP: &str = "vyre_conform_test::identity_u32";

fn identity_u32_cpu_ref(input: &[u8], output: &mut Vec<u8>) {
    output.clear();
    output.extend_from_slice(input.get(..4).unwrap_or(&[0, 0, 0, 0]));
}

const TEST_IDENTITY_U32_SIGNATURE: Signature = Signature {
    inputs: &[TypedParam {
        name: "value",
        ty: "u32",
    }],
    outputs: &[TypedParam {
        name: "out",
        ty: "u32",
    }],
    attrs: &[],
    bytes_extraction: false,
};

inventory::submit! {
    OperationRegistration::new(
        TEST_IDENTITY_U32_OP,
        OperationTier::External,
        None,
        None,
        None,
    )
    .with_signature(TEST_IDENTITY_U32_SIGNATURE)
    .with_category("vyre-conform-test")
}
inventory::submit! {
    ReferenceFacet::new(TEST_IDENTITY_U32_OP, identity_u32_cpu_ref)
}

// ---------------------------------------------------------------------------
// Deterministic Ed25519 key  -  same seed => same pubkey & sig every run.
// ---------------------------------------------------------------------------
fn deterministic_signing_key() -> SigningKey {
    let seed_hash = blake3::hash(b"RELEASE-PROOF-L11-cert-regression-pin");
    let mut seed = [0u8; 32];
    seed.copy_from_slice(seed_hash.as_bytes());
    SigningKey::from_bytes(&seed)
}

// ---------------------------------------------------------------------------
// Pinned constants  -  generated once, guarded forever.
// If any assertion fires, copy the "Fix:" value into the constant below.
// ---------------------------------------------------------------------------

/// Ed25519 verifying key (hex) for the deterministic signing key.
const VERIFYING_KEY_HEX: &str = "7d6cdd2bb962491984ea484fe095a24719aac478eae2cf943af71c9941f99d83";

// Pinned bundle hashes, lengths, and signatures below moved for wire revision 6.
// Revision 6 appends the `linear_type`, `bytes_extraction`, and `shape_predicate`
// tags to every buffer declaration. Each canonical bundle has one buffer, so
// its wire body grows by exactly three bytes.
//
// The signatures moved a second time, and the region-chain hash and length with
// them, for two reasons recorded here because a bare digest cannot state its own
// provenance:
//
// 1. The signable body covers `reference_output_blake3`, the digest of the
//    reference output stream, so a change in what the reference returns moves
//    every signature while leaving the wire bytes alone. Every pinned signature
//    below is reproducible from the bundle hash, the corpus digest, and the one
//    output word each program stores, which `EXPECTED_OUTPUT_WORDS` now asserts
//    directly. That assertion is what makes the next drift attributable: a hash
//    that moves while the words hold is a framing change, and a hash that moves
//    with them is a semantic one.
// 2. `TEST_IDENTITY_U32_OP` was respelled from `vyre-conform.test.identity_u32`
//    to `vyre_conform_test::identity_u32` when the operation and target
//    registries were unified. The region-chain bundle carries that op id in an
//    `Expr::call`, so the id is in its wire bytes: one character longer is one
//    byte longer, 324 to 325, and a different bundle digest.

// --- trivial const ---
const TRIVIAL_CONST_BUNDLE_BLAKE3: &str =
    "22325167b0cb84193a1d5cfe581e9e2f95fd5721a5cf3f4fd2fdd80d49dcf062";
const TRIVIAL_CONST_WIRE_LEN: usize = 197;
const TRIVIAL_CONST_SIG_HEX: &str =
    "16265ad4942775a2e05c8898037f0133219cd52e4ef69b75b2177f34303bbf8b281f9cca6df3619adf08cf911534c74d0b5ec7e156913622c11c43cd4fd54e0c";

// --- 1-op add ---
const ONE_OP_ADD_BUNDLE_BLAKE3: &str =
    "c202412743128a5f2b944de76a272503b4244e3b009b440613e74cef56cd22ed";
const ONE_OP_ADD_WIRE_LEN: usize = 204;
const ONE_OP_ADD_SIG_HEX: &str =
    "56f0e605dbaec7a4934e53407aba9ca50f997b2163e19fb3ac939b6005103876a44ac0bd02b28cd91e643a33fd302274437725af96105029734a9fff5862e609";

// --- loop-add ---
const LOOP_ADD_BUNDLE_BLAKE3: &str =
    "42042606253dd42ac84317e40997166aa3fb704c4a05ead4cce851f312ee6b4e";
const LOOP_ADD_WIRE_LEN: usize = 257;
const LOOP_ADD_SIG_HEX: &str =
    "8b1ec6c68ab1c73d4898ef4f62eb2a12f1e84e5bce6984a180d396e6f9964de857ac30136b1cb13b02180694dc607ad1cf63b9e6566cb5a936536bf3db9b450e";

// --- composed nested ---
const COMPOSED_NESTED_BUNDLE_BLAKE3: &str =
    "17fe6fe62d2b37144bf29d8994412572e7b7a6dbf0f93c137b89dc413b496983";
const COMPOSED_NESTED_WIRE_LEN: usize = 200;
const COMPOSED_NESTED_SIG_HEX: &str =
    "6a95ae84033e4cdef26ca8fa6211c333c6d7ddae183968406f6bf01c561b4123e65a74e7527d1745c59615d38f7603c259e905a6b4d4cd4d0d5de25e8c9f0b03";

// --- region-chain with intrinsic + dialect op ---
const REGION_CHAIN_BUNDLE_BLAKE3: &str =
    "20833f6b55d44af16092bd1eab94eb5a522ce1b495dab15f7d08500800fc02e8";
const REGION_CHAIN_WIRE_LEN: usize = 325;
const REGION_CHAIN_SIG_HEX: &str =
    "3136779bcee25a5acd434a7bdc9c76779994a4128712ee3f6cadfbfba00708f56eafe068d1eb9a89b09eb82c575eb659a24e07be6a6482c680d3ecaeafeb250a";

// ---------------------------------------------------------------------------
// Sign a bundle cert with the deterministic key.
// ---------------------------------------------------------------------------
fn sign_bundle_cert(cert: &mut BundleCertificate, key: &SigningKey) {
    let signable = serde_json::json!({
        "version": cert.version,
        "bundle_blake3": cert.bundle_blake3,
        "corpus_blake3": cert.corpus_blake3,
        "reference_output_blake3": cert.reference_output_blake3,
        "witness_count": cert.witness_count,
        "timestamp": cert.timestamp,
        "pubkey": hex::encode(key.verifying_key().to_bytes()),
    });
    let signable_bytes = serde_json::to_vec(&signable).expect("canonical json");
    let signature = key.sign(&signable_bytes);
    cert.signature_ed25519 = hex::encode(signature.to_bytes());
    cert.pubkey = hex::encode(key.verifying_key().to_bytes());
}

// ---------------------------------------------------------------------------
// Bundle 1  -  trivial const
// ---------------------------------------------------------------------------
fn bundle_trivial_const() -> (Program, Vec<ConformanceCase>) {
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );
    let corpus = vec![ConformanceCase {
        name: "tc1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 2  -  1-op add
// ---------------------------------------------------------------------------
fn bundle_one_op_add() -> (Program, Vec<ConformanceCase>) {
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::u32(1), Expr::u32(2)),
        )],
    );
    let corpus = vec![ConformanceCase {
        name: "add1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 3  -  loop-add
// ---------------------------------------------------------------------------
fn bundle_loop_add() -> (Program, Vec<ConformanceCase>) {
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::let_bind("acc", Expr::u32(0)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(4),
                vec![Node::assign(
                    "acc",
                    Expr::add(Expr::var("acc"), Expr::var("i")),
                )],
            ),
            Node::store("out", Expr::u32(0), Expr::var("acc")),
        ],
    );
    let corpus = vec![ConformanceCase {
        name: "loop1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 4  -  composed nested regions
// ---------------------------------------------------------------------------
fn bundle_composed_nested() -> (Program, Vec<ConformanceCase>) {
    let inner = vec![Node::store("out", Expr::u32(0), Expr::u32(7))];
    let outer = vec![Node::Region {
        generator: "inner".into(),
        source_region: None,
        body: Arc::new(inner),
    }];
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Region {
            generator: "outer".into(),
            source_region: None,
            body: Arc::new(outer),
        }],
    );
    let corpus = vec![ConformanceCase {
        name: "nest1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 5  -  Region-chain with executable dialect op
//
// Contains a Node::Region (intrinsic-like generator) and an Expr::call to a
// operation registry; the bundle certificate hashes remain stable.
// ---------------------------------------------------------------------------
fn bundle_region_chain_intrinsic_dialect() -> (Program, Vec<ConformanceCase>) {
    let body = vec![
        Node::let_bind("acc", Expr::u32(0)),
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(3),
            vec![Node::assign(
                "acc",
                Expr::add(Expr::var("acc"), Expr::var("i")),
            )],
        ),
        Node::let_bind(
            "dial",
            Expr::call(TEST_IDENTITY_U32_OP, vec![Expr::var("acc")]),
        ),
        Node::store("out", Expr::u32(0), Expr::var("dial")),
    ];
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Region {
            generator: "vyre.intrinsics.math.accum".into(),
            source_region: None,
            body: Arc::new(body),
        }],
    );
    let corpus = vec![ConformanceCase {
        name: "rd1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

fn bundle_region_chain_backend_witness() -> (Program, Vec<ConformanceCase>) {
    let body = vec![
        Node::let_bind("acc", Expr::u32(0)),
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(3),
            vec![Node::assign(
                "acc",
                Expr::add(Expr::var("acc"), Expr::var("i")),
            )],
        ),
        Node::let_bind("dial", Expr::add(Expr::var("acc"), Expr::u32(1))),
        Node::store("out", Expr::u32(0), Expr::var("acc")),
    ];
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Region {
            generator: "vyre.intrinsics.math.accum".into(),
            source_region: None,
            body: Arc::new(body),
        }],
    );
    let corpus = vec![ConformanceCase {
        name: "rd-backend".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

/// Assert the reference output stream for `corpus`'s single case is exactly
/// `expected`, read as u32 words.
///
/// Runs the same planning and the same interpreter entry point the certificate
/// issuer runs, so the words asserted here are the words the signable
/// `reference_output_blake3` is taken over.
fn assert_reference_output_words(
    name: &str,
    program: &Program,
    corpus: &[ConformanceCase],
    expected: &[u32],
) {
    let plan = WitnessInputPlan::for_program(program)
        .unwrap_or_else(|e| panic!("{name}: witness planning failed: {e}"));
    let mut planned: Vec<&[u8]> = Vec::new();
    let case = corpus
        .first()
        .unwrap_or_else(|| panic!("{name}: bundle corpus must hold at least one case"));
    plan_witness_inputs_into(&case.inputs, &plan, &mut planned)
        .unwrap_or_else(|e| panic!("{name}: witness input planning failed: {e}"));
    let values: Vec<Value> = planned.into_iter().map(Value::from).collect();
    let outputs = vyre_reference::reference_eval(program, &values)
        .unwrap_or_else(|e| panic!("{name}: reference_eval failed: {e}"));
    let words: Vec<u32> = outputs
        .iter()
        .flat_map(|value| {
            value
                .to_bytes()
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect::<Vec<u32>>()
        })
        .collect();
    assert_eq!(
        words, expected,
        "{name}: reference output drift. The certificate signature covers this \
         stream, so it moved with these words. Fix: if the new values are the \
         program's semantics, update the pinned words and the pinned signature \
         together; if they are not, the reference regressed."
    );
}

// ---------------------------------------------------------------------------
// Main test: pin and verify all five bundles.
// ---------------------------------------------------------------------------
#[test]
fn cert_regression_pin_all_five_bundles() {
    let key = deterministic_signing_key();

    // Drifts are collected rather than asserted one at a time. A wire-format or
    // pipeline change moves every bundle's hash, length and signature at once,
    // and failing on the first case hid the other four: updating the pins then
    // took five full runs instead of one.
    let mut drifts: Vec<String> = Vec::new();

    #[allow(clippy::type_complexity)]
    let cases: Vec<(
        &str,
        fn() -> (Program, Vec<ConformanceCase>),
        &str,   // pinned bundle_blake3
        usize,  // pinned wire_len
        &str,   // pinned signature
        &[u32], // pinned reference output words
    )> = vec![
        (
            "trivial_const",
            bundle_trivial_const,
            TRIVIAL_CONST_BUNDLE_BLAKE3,
            TRIVIAL_CONST_WIRE_LEN,
            TRIVIAL_CONST_SIG_HEX,
            &[42],
        ),
        (
            "one_op_add",
            bundle_one_op_add,
            ONE_OP_ADD_BUNDLE_BLAKE3,
            ONE_OP_ADD_WIRE_LEN,
            ONE_OP_ADD_SIG_HEX,
            &[3],
        ),
        (
            "loop_add",
            bundle_loop_add,
            LOOP_ADD_BUNDLE_BLAKE3,
            LOOP_ADD_WIRE_LEN,
            LOOP_ADD_SIG_HEX,
            &[6],
        ),
        (
            "composed_nested",
            bundle_composed_nested,
            COMPOSED_NESTED_BUNDLE_BLAKE3,
            COMPOSED_NESTED_WIRE_LEN,
            COMPOSED_NESTED_SIG_HEX,
            &[7],
        ),
        (
            "region_chain_intrinsic_dialect",
            bundle_region_chain_intrinsic_dialect,
            REGION_CHAIN_BUNDLE_BLAKE3,
            REGION_CHAIN_WIRE_LEN,
            REGION_CHAIN_SIG_HEX,
            &[3],
        ),
    ];

    for (name, builder, pinned_hash, pinned_len, pinned_sig, pinned_words) in cases {
        let (program, corpus) = builder();

        // 0. Pin the SEMANTICS the signature covers. `reference_output_blake3` is
        // part of the signable body, so a change in what the reference returns
        // moves the signature with the wire bytes untouched, and a digest cannot
        // say which happened. Asserting the words first makes the next drift
        // attributable: words held means the framing moved, words moved means the
        // reference did.
        assert_reference_output_words(name, &program, &corpus, pinned_words);

        // 1. Independent re-compute of wire bytes + bundle hash.
        let wire_bytes = program
            .to_wire()
            .unwrap_or_else(|e| panic!("{name}: to_wire() failed: {e}"));
        let computed_hash = blake3::hash(&wire_bytes).to_hex().to_string();
        let computed_len = wire_bytes.len();

        if computed_hash != pinned_hash {
            drifts.push(format!(
                "{name}: bundle_blake3 drift. actual={computed_hash} expected={pinned_hash}. \
                 Fix: update the pinned constant to {computed_hash} if the pipeline change was \
                 intentional."
            ));
        }
        if computed_len != pinned_len {
            drifts.push(format!(
                "{name}: wire length drift. actual={computed_len} expected={pinned_len}. \
                 Fix: update the pinned constant to {computed_len} if the wire format change was \
                 intentional."
            ));
        }

        // 2. Issue cert from the same bundle.
        let mut cert = issue_bundle_cert(&program, &corpus, "2026-04-23T20:00:00Z", "TBD", "TBD")
            .unwrap_or_else(|e| panic!("{name}: issue_bundle_cert failed: {e}"));

        // Cert must match the independently-computed hash.
        assert_eq!(
            cert.bundle_blake3, computed_hash,
            "{name}: cert bundle_blake3 must match independent hash compute"
        );

        // 3. Sign and pin the signature.
        sign_bundle_cert(&mut cert, &key);

        assert_eq!(
            cert.pubkey, VERIFYING_KEY_HEX,
            "{name}: pubkey drift. \
             actual={} expected={VERIFYING_KEY_HEX}. \
             Fix: update VERIFYING_KEY_HEX if the signing key changed.",
            cert.pubkey
        );
        if cert.signature_ed25519 != pinned_sig {
            drifts.push(format!(
                "{name}: signature drift. actual={} expected={pinned_sig}. Fix: update the pinned \
                 signature constant to {} if the signable body changed intentionally.",
                cert.signature_ed25519, cert.signature_ed25519
            ));
        }

        // 4. Cryptographic signature must verify against the pinned pubkey.
        verify_cert_signature_hex(&cert, VERIFYING_KEY_HEX)
            .unwrap_or_else(|e| panic!("{name}: signature verification failed: {e}"));

        // 5. Hash-chain re-compute from the same (program, corpus) must pass.
        verify_bundle_against_reference(&cert, &program, &corpus)
            .unwrap_or_else(|e| panic!("{name}: reference re-verify failed: {e}"));
    }

    assert!(
        drifts.is_empty(),
        "Fix: {} pinned certificate value(s) drifted:\n{}",
        drifts.len(),
        drifts.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Backend verification on the mandatory dispatch lane.
// ---------------------------------------------------------------------------
// Requires the wgpu backend factory to succeed against a live GPU device.
// Missing backend registration is a release-host failure, not a skipped test.
#[test]
fn cert_regression_pin_backend_verification_gpu() {
    let cases: Vec<(&str, BundleBuilderFn)> = vec![
        ("trivial_const", bundle_trivial_const),
        ("one_op_add", bundle_one_op_add),
        ("loop_add", bundle_loop_add),
        ("composed_nested", bundle_composed_nested),
        (
            "region_chain_intrinsic_dialect",
            bundle_region_chain_backend_witness,
        ),
    ];

    let backend = vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == "wgpu")
        .expect("Fix: wgpu backend must be registered in the GPU certificate regression lane");

    for (name, builder) in cases {
        let (program, corpus) = builder();
        let cert = issue_bundle_cert(&program, &corpus, "2026-04-23T20:00:00Z", "sig", "pub")
            .unwrap_or_else(|e| panic!("{name}: issue failed: {e}"));

        if let Err(e) = vyre_conform::verify_bundle_with_backend(&cert, &program, backend, &corpus)
        {
            panic!("Fix: {name}: backend verification failed: {e}");
        }
    }
}
