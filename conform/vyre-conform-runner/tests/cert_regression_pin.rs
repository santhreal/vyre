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

// Pinned bundle hashes and signatures below moved in 0.7.0 and the wire lengths
// did not. `WIRE_FORMAT_VERSION` went 4 to 5 for the `Expr::BufferRef` tag, and
// that version is a fixed-width header field, so every serialized program's
// bytes changed in place. The lengths staying identical is the check that the
// change was the header field and not a body-layout change.

// --- trivial const ---
const TRIVIAL_CONST_BUNDLE_BLAKE3: &str =
    "6e5706282fc848f2c113988b3c19d0895f1261acb0746fe82ca9580fed097f52";
const TRIVIAL_CONST_WIRE_LEN: usize = 194;
const TRIVIAL_CONST_SIG_HEX: &str =
    "fed1660c51b6a83e660b8e0f460cc63d9a3ebdfc98863a15f81ce2273ec644bff405153c9cb362ae0538e190226fed0cb2758fb4af6554b2b6b0d7df58d3b70a";

// --- 1-op add ---
const ONE_OP_ADD_BUNDLE_BLAKE3: &str =
    "6a262a2d08e4d96cab82ee7b74c999a8332522ab86e7eb12223baf5c717a7741";
const ONE_OP_ADD_WIRE_LEN: usize = 201;
const ONE_OP_ADD_SIG_HEX: &str =
    "a159f418c4a61bf2978eadc576d9784fc8f9d499877770e404ad3148583e4757592eac91dd5e50ea077b6efdb1e8746c483ba3ab03ec31e25de180e22b52740d";

// --- loop-add ---
const LOOP_ADD_BUNDLE_BLAKE3: &str =
    "16c8865a770d9087fe16581c46bdda1a205a519095342b017dd6205c32d26b58";
const LOOP_ADD_WIRE_LEN: usize = 254;
const LOOP_ADD_SIG_HEX: &str =
    "ef8c673bcbf2af99d8d560140225e018847b407d744a6b4c70d6c5826070fe57aa646c29329dbad79881b64bee4f24202d68f1d43f069a8017539a086cf81d0b";

// --- composed nested ---
const COMPOSED_NESTED_BUNDLE_BLAKE3: &str =
    "bf285b0187bd169feff9a1ac71e320a7e4cca65e1fbb7a58ca89480420984274";
const COMPOSED_NESTED_WIRE_LEN: usize = 197;
const COMPOSED_NESTED_SIG_HEX: &str =
    "d573f23d3c74cdebcc0e1eb996962191b6c576b0d1f76bca11487ff72de15dfbf47bce087078a9f72f4569e8f05ad7668a9390a23879513b29aac2dd2e3e1d0c";

// --- region-chain with intrinsic + dialect op ---
const REGION_CHAIN_BUNDLE_BLAKE3: &str =
    "6dba5e0db7baa0d8c512c401f63b5536d844d0757510531b5e214660a09575d0";
const REGION_CHAIN_WIRE_LEN: usize = 321;
const REGION_CHAIN_SIG_HEX: &str =
    "85ed83cb200ad77f1e088801b0e3e21d41c08030f10772fa3e4fe46f20da4bbf74f6662757a87b8725fe4f61fba35520889f0073c539df783f17f789148aa602";

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
