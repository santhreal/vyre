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

use vyre::ir::Program;
use vyre_conform::witness_plan::{plan_witness_inputs_into, WitnessInputPlan};
use vyre_conform::{issue_bundle_cert, verify_bundle_against_reference, verify_cert_signature_hex};
use vyre_conform_spec::ConformanceCase;
use vyre_reference::value::Value;

#[path = "cert_regression_pin/bundles.rs"]
mod bundles;
#[path = "cert_regression_pin/pins.rs"]
mod pins;
#[path = "cert_regression_pin/test_operation.rs"]
mod test_operation;

use bundles::{
    bundle_composed_nested, bundle_loop_add, bundle_one_op_add,
    bundle_region_chain_backend_witness, bundle_region_chain_intrinsic_dialect,
    bundle_trivial_const, BundleBuilderFn,
};
use pins::{
    deterministic_signing_key, sign_bundle_cert, COMPOSED_NESTED_BUNDLE_BLAKE3,
    COMPOSED_NESTED_SIG_HEX, COMPOSED_NESTED_WIRE_LEN, LOOP_ADD_BUNDLE_BLAKE3, LOOP_ADD_SIG_HEX,
    LOOP_ADD_WIRE_LEN, ONE_OP_ADD_BUNDLE_BLAKE3, ONE_OP_ADD_SIG_HEX, ONE_OP_ADD_WIRE_LEN,
    REGION_CHAIN_BUNDLE_BLAKE3, REGION_CHAIN_SIG_HEX, REGION_CHAIN_WIRE_LEN,
    TRIVIAL_CONST_BUNDLE_BLAKE3, TRIVIAL_CONST_SIG_HEX, TRIVIAL_CONST_WIRE_LEN, VERIFYING_KEY_HEX,
};

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

    let backend = vyre_registry_link::backend::live_backend_registry()
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
