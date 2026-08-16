//! Pinned certificate values and the deterministic key that produces them.

use ed25519_dalek::{Signer, SigningKey};
use vyre_conform_spec::BundleCertificate;

// ---------------------------------------------------------------------------
// Deterministic Ed25519 key  -  same seed => same pubkey & sig every run.
// ---------------------------------------------------------------------------
pub(crate) fn deterministic_signing_key() -> SigningKey {
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
pub(crate) const VERIFYING_KEY_HEX: &str =
    "7d6cdd2bb962491984ea484fe095a24719aac478eae2cf943af71c9941f99d83";

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
pub(crate) const TRIVIAL_CONST_BUNDLE_BLAKE3: &str =
    "be566351414e368a5e41dfb7850692326ae5aeec4287d412b6575be5594ab45d";
pub(crate) const TRIVIAL_CONST_WIRE_LEN: usize = 197;
pub(crate) const TRIVIAL_CONST_SIG_HEX: &str =
    "260a5b784f180fc4990c7c38f3f662141cab6cf8c940f4f29c4536bcbc5a065ca23492c045d99ef23f0b52d12c1630b2d9000d13198102b28572bbeb08af7a03";

// --- 1-op add ---
pub(crate) const ONE_OP_ADD_BUNDLE_BLAKE3: &str =
    "98c84692dec52b46d16b4766664277a7df006cfe28f09b317ce687b566c0bcf0";
pub(crate) const ONE_OP_ADD_WIRE_LEN: usize = 204;
pub(crate) const ONE_OP_ADD_SIG_HEX: &str =
    "ef679e5a4b8997bd066a99b22283eef6147eb3b6bcd1a19b568060162b7a3e4eec6c2f7cc2349b6b155b22024838ea5dc3155d8464bf3837edeed4956d116804";

// --- loop-add ---
pub(crate) const LOOP_ADD_BUNDLE_BLAKE3: &str =
    "73b09f2cdf249b2ed6d40ce24e9c54771fa774373f51a753946f519e665798a9";
pub(crate) const LOOP_ADD_WIRE_LEN: usize = 257;
pub(crate) const LOOP_ADD_SIG_HEX: &str =
    "6d6b070e4a592689b1330bd8c942acc97e9c119dd4bb8c7f53622340ba2b1838f9ca5b70dad69b35ba5029214a166edf84492b986ab235d62bc034ec44fba203";

// --- composed nested ---
pub(crate) const COMPOSED_NESTED_BUNDLE_BLAKE3: &str =
    "d411a26526ab5eb50f70c3ff94ca9761fbfdfc0cd1a0430030a52b0bf0893d49";
pub(crate) const COMPOSED_NESTED_WIRE_LEN: usize = 200;
pub(crate) const COMPOSED_NESTED_SIG_HEX: &str =
    "5d045ba6b76c30cb4cecf46dee0bad808ff130a072c990ba7d9e1f1b1a7c22143a050fdfe88ff47533d9028be2ba2220bb926c7239df7a833bdc69a3726fe708";

// --- region-chain with intrinsic + dialect op ---
pub(crate) const REGION_CHAIN_BUNDLE_BLAKE3: &str =
    "c36b7bee0f307bc97b9b42b3e0da4a3a80aeb2f843880a41a9acf9a993c23827";
pub(crate) const REGION_CHAIN_WIRE_LEN: usize = 325;
pub(crate) const REGION_CHAIN_SIG_HEX: &str =
    "f44960073a844f264a728336712dcf13924f9ef50786a952dccfd2e6f3f2f905a1a3a31e3161b886ba6e430298697ecd8401efea69d8f82556049b566149f903";
// ---------------------------------------------------------------------------
// Sign a bundle cert with the deterministic key.
// ---------------------------------------------------------------------------
pub(crate) fn sign_bundle_cert(cert: &mut BundleCertificate, key: &SigningKey) {
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
