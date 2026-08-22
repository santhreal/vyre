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

// The three bundles whose entry is a plain statement list moved once more, and
// only those three. `Program::wrapped` gives such an entry a synthetic root
// region, and `Program::ROOT_REGION_GENERATOR` was respelled from
// `vyre.program.root` to `anonymous::vyre.program.root` when region identities
// were canonicalized. That name is in the wire bytes, eleven characters longer
// is eleven bytes longer, and the bundle digest and signature moved with it.
// `composed_nested` and `region_chain_intrinsic_dialect` pass `Node::Region`
// nodes at the top level, take no synthetic root, and did not move. The pinned
// reference output words held across the change, which is what says the framing
// moved and the semantics did not.

// --- trivial const ---
pub(crate) const TRIVIAL_CONST_BUNDLE_BLAKE3: &str =
    "7c043d3f71b03daf29c4ee3e3881aa3ecac18a0dbdcbd404a85c7433fdc0afed";
pub(crate) const TRIVIAL_CONST_WIRE_LEN: usize = 208;
pub(crate) const TRIVIAL_CONST_SIG_HEX: &str =
    "41333198b0e00fc84852610bcb2485763fe7ea633231e352a1eaca8adc26ad284f36b32f69016e3f5b9a3eba2051f4c7d2df650bf9a44098decfa0711793ea03";

// --- 1-op add ---
pub(crate) const ONE_OP_ADD_BUNDLE_BLAKE3: &str =
    "571f0039b16d178740a1e20eb3daa054621d4511801a01d7d20eb070413d1fd0";
pub(crate) const ONE_OP_ADD_WIRE_LEN: usize = 215;
pub(crate) const ONE_OP_ADD_SIG_HEX: &str =
    "e659957783b637506819989d8a2e00e673f684b30db3bf036aaf86cec2b980926c4ec65695c919ff64816d018c26f4bc65dd825222ef813faef5b65dda278301";

// --- loop-add ---
pub(crate) const LOOP_ADD_BUNDLE_BLAKE3: &str =
    "8fe6a8a71838ef9edbf772f676cd28f48a313d144c25a4373f8ad220f3a30dfb";
pub(crate) const LOOP_ADD_WIRE_LEN: usize = 268;
pub(crate) const LOOP_ADD_SIG_HEX: &str =
    "e15732a2c60737978bd89223921751504cc2d0b67288dcd44cd9431f677882ca535d1a456f3afe9b8f1797be73fe5641c718e664c585a936a7485dc3def21c02";

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
