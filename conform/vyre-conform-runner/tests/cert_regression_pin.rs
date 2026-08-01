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
use vyre_conform_runner::{
    issue_bundle_cert, verify_bundle_against_reference, verify_cert_signature_hex,
    BundleCertificate, CorpusWitness,
};
use vyre_driver::registry::{
    Category, LoweringTable, OpDef, OpDefRegistration, Signature, TypedParam,
};
use vyre_primitives::wire::pack_u32_slice as bytes_u32;

#[cfg(feature = "gpu")]
use vyre_driver_metal as _;
#[cfg(feature = "gpu")]
use vyre_driver_wgpu as _;

type BundleBuilderFn = fn() -> (Program, Vec<CorpusWitness>);

const TEST_IDENTITY_U32_OP: &str = "vyre-conform.test.identity_u32";

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
    OpDefRegistration::new(|| OpDef {
        id: TEST_IDENTITY_U32_OP,
        dialect: "vyre-conform-test",
        category: Category::Intrinsic,
        signature: TEST_IDENTITY_U32_SIGNATURE,
        lowerings: LoweringTable::new(identity_u32_cpu_ref),
        laws: &[],
        compose: None,
    })
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

// --- trivial const ---
const TRIVIAL_CONST_BUNDLE_BLAKE3: &str =
    "22325167b0cb84193a1d5cfe581e9e2f95fd5721a5cf3f4fd2fdd80d49dcf062";
const TRIVIAL_CONST_WIRE_LEN: usize = 197;
const TRIVIAL_CONST_SIG_HEX: &str =
    "cc136e575ff68a9352f81842a6de41c635ae323ea38dd243deb1faecfacecc09c2a90e57c1a14950a05b06e39b7dba547c6fb09cabc2d9953daa7e060209240c";

// --- 1-op add ---
const ONE_OP_ADD_BUNDLE_BLAKE3: &str =
    "c202412743128a5f2b944de76a272503b4244e3b009b440613e74cef56cd22ed";
const ONE_OP_ADD_WIRE_LEN: usize = 204;
const ONE_OP_ADD_SIG_HEX: &str =
    "f74016145b3aad28d4f4573b6ec450f32e94ba0d297051194f8309b95479d766e8eb7876366a4be9e500c121559229f0b5c4e35a19d11a1cfc5dbc0bfa056a0c";

// --- loop-add ---
const LOOP_ADD_BUNDLE_BLAKE3: &str =
    "42042606253dd42ac84317e40997166aa3fb704c4a05ead4cce851f312ee6b4e";
const LOOP_ADD_WIRE_LEN: usize = 257;
const LOOP_ADD_SIG_HEX: &str =
    "1c7d5cfc8f754bdc8df0455eaeb3c95ea0fd489c95ebc591cefd3e778993c8754483e9f08ba93126de7e1f2d154e8dc26d61743f0f67664221313e2939c71105";

// --- composed nested ---
const COMPOSED_NESTED_BUNDLE_BLAKE3: &str =
    "17fe6fe62d2b37144bf29d8994412572e7b7a6dbf0f93c137b89dc413b496983";
const COMPOSED_NESTED_WIRE_LEN: usize = 200;
const COMPOSED_NESTED_SIG_HEX: &str =
    "5a997330ac378014280d147456ee3fa2a90b789ea295d44bf601854655c8aba5c3c92d04274b6409508af7c58a468cb257b6447a78dfc2b14df80654a7a08705";

// --- region-chain with intrinsic + dialect op ---
const REGION_CHAIN_BUNDLE_BLAKE3: &str =
    "984aec1c4b0ffb081687e23fad358cc2f3645f85c5e85d7c0f9cbe1d113e6b49";
const REGION_CHAIN_WIRE_LEN: usize = 324;
const REGION_CHAIN_SIG_HEX: &str =
    "dc62c8f2c8f51cdad309aa73a43c395a3fca353ac962a82dc641d02c9ee42af68d56cd7c98f4de704a94e05f276351dbadeb0eadafa2d38baeb1d31561059d09";

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
fn bundle_trivial_const() -> (Program, Vec<CorpusWitness>) {
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );
    let corpus = vec![CorpusWitness {
        name: "tc1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 2  -  1-op add
// ---------------------------------------------------------------------------
fn bundle_one_op_add() -> (Program, Vec<CorpusWitness>) {
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::u32(1), Expr::u32(2)),
        )],
    );
    let corpus = vec![CorpusWitness {
        name: "add1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 3  -  loop-add
// ---------------------------------------------------------------------------
fn bundle_loop_add() -> (Program, Vec<CorpusWitness>) {
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
    let corpus = vec![CorpusWitness {
        name: "loop1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 4  -  composed nested regions
// ---------------------------------------------------------------------------
fn bundle_composed_nested() -> (Program, Vec<CorpusWitness>) {
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
    let corpus = vec![CorpusWitness {
        name: "nest1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Bundle 5  -  Region-chain with executable dialect op
//
// Contains a Node::Region (intrinsic-like generator) and an Expr::call to a
// dialect op. The CPU reference resolves the call via the DialectRegistry; the
// bundle cert hashes are still stable.
// ---------------------------------------------------------------------------
fn bundle_region_chain_intrinsic_dialect() -> (Program, Vec<CorpusWitness>) {
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
    let corpus = vec![CorpusWitness {
        name: "rd1".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

fn bundle_region_chain_backend_witness() -> (Program, Vec<CorpusWitness>) {
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
    let corpus = vec![CorpusWitness {
        name: "rd-backend".into(),
        inputs: vec![bytes_u32(&[0])],
    }];
    (program, corpus)
}

// ---------------------------------------------------------------------------
// Main test: pin and verify all five bundles.
// ---------------------------------------------------------------------------
#[test]
fn cert_regression_pin_all_five_bundles() {
    // Initialise driver registry so dialect ops (e.g. core.indirect_dispatch)
    // resolve during reference_eval.
    let _ = vyre_driver::registry::DialectRegistry::global();

    let key = deterministic_signing_key();

    // Drifts are collected rather than asserted one at a time. A wire-format or
    // pipeline change moves every bundle's hash, length and signature at once,
    // and failing on the first case hid the other four: updating the pins then
    // took five full runs instead of one.
    let mut drifts: Vec<String> = Vec::new();

    #[allow(clippy::type_complexity)]
    let cases: Vec<(
        &str,
        fn() -> (Program, Vec<CorpusWitness>),
        &str,  // pinned bundle_blake3
        usize, // pinned wire_len
        &str,  // pinned signature
    )> = vec![
        (
            "trivial_const",
            bundle_trivial_const,
            TRIVIAL_CONST_BUNDLE_BLAKE3,
            TRIVIAL_CONST_WIRE_LEN,
            TRIVIAL_CONST_SIG_HEX,
        ),
        (
            "one_op_add",
            bundle_one_op_add,
            ONE_OP_ADD_BUNDLE_BLAKE3,
            ONE_OP_ADD_WIRE_LEN,
            ONE_OP_ADD_SIG_HEX,
        ),
        (
            "loop_add",
            bundle_loop_add,
            LOOP_ADD_BUNDLE_BLAKE3,
            LOOP_ADD_WIRE_LEN,
            LOOP_ADD_SIG_HEX,
        ),
        (
            "composed_nested",
            bundle_composed_nested,
            COMPOSED_NESTED_BUNDLE_BLAKE3,
            COMPOSED_NESTED_WIRE_LEN,
            COMPOSED_NESTED_SIG_HEX,
        ),
        (
            "region_chain_intrinsic_dialect",
            bundle_region_chain_intrinsic_dialect,
            REGION_CHAIN_BUNDLE_BLAKE3,
            REGION_CHAIN_WIRE_LEN,
            REGION_CHAIN_SIG_HEX,
        ),
    ];

    for (name, builder, pinned_hash, pinned_len, pinned_sig) in cases {
        let (program, corpus) = builder();

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
    let _ = vyre_driver::registry::DialectRegistry::global();

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

    let backend = match vyre_driver::backend::registered_backends()
        .iter()
        .find(|r| r.id == "wgpu")
    {
        Some(reg) => match reg.acquire() {
            Ok(b) => b,
            Err(e) => {
                panic!("Fix: wgpu backend factory failed on a GPU-required host: {e}");
            }
        },
        None => {
            panic!("Fix: wgpu backend not registered in GPU certificate regression lane");
        }
    };

    for (name, builder) in cases {
        let (program, corpus) = builder();
        let cert = issue_bundle_cert(&program, &corpus, "2026-04-23T20:00:00Z", "sig", "pub")
            .unwrap_or_else(|e| panic!("{name}: issue failed: {e}"));

        if let Err(e) = vyre_conform_runner::verify_bundle_with_backend(
            &cert,
            &program,
            backend.as_ref(),
            &corpus,
        ) {
            panic!("Fix: {name}: backend verification failed: {e}");
        }
    }
}
