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
    "22325167b0cb84193a1d5cfe581e9e2f95fd5721a5cf3f4fd2fdd80d49dcf062";
pub(crate) const TRIVIAL_CONST_WIRE_LEN: usize = 197;
pub(crate) const TRIVIAL_CONST_SIG_HEX: &str =
    "16265ad4942775a2e05c8898037f0133219cd52e4ef69b75b2177f34303bbf8b281f9cca6df3619adf08cf911534c74d0b5ec7e156913622c11c43cd4fd54e0c";

// --- 1-op add ---
pub(crate) const ONE_OP_ADD_BUNDLE_BLAKE3: &str =
    "c202412743128a5f2b944de76a272503b4244e3b009b440613e74cef56cd22ed";
pub(crate) const ONE_OP_ADD_WIRE_LEN: usize = 204;
pub(crate) const ONE_OP_ADD_SIG_HEX: &str =
    "56f0e605dbaec7a4934e53407aba9ca50f997b2163e19fb3ac939b6005103876a44ac0bd02b28cd91e643a33fd302274437725af96105029734a9fff5862e609";

// --- loop-add ---
pub(crate) const LOOP_ADD_BUNDLE_BLAKE3: &str =
    "42042606253dd42ac84317e40997166aa3fb704c4a05ead4cce851f312ee6b4e";
pub(crate) const LOOP_ADD_WIRE_LEN: usize = 257;
pub(crate) const LOOP_ADD_SIG_HEX: &str =
    "8b1ec6c68ab1c73d4898ef4f62eb2a12f1e84e5bce6984a180d396e6f9964de857ac30136b1cb13b02180694dc607ad1cf63b9e6566cb5a936536bf3db9b450e";

// --- composed nested ---
pub(crate) const COMPOSED_NESTED_BUNDLE_BLAKE3: &str =
    "17fe6fe62d2b37144bf29d8994412572e7b7a6dbf0f93c137b89dc413b496983";
pub(crate) const COMPOSED_NESTED_WIRE_LEN: usize = 200;
pub(crate) const COMPOSED_NESTED_SIG_HEX: &str =
    "6a95ae84033e4cdef26ca8fa6211c333c6d7ddae183968406f6bf01c561b4123e65a74e7527d1745c59615d38f7603c259e905a6b4d4cd4d0d5de25e8c9f0b03";

// --- region-chain with intrinsic + dialect op ---
pub(crate) const REGION_CHAIN_BUNDLE_BLAKE3: &str =
    "20833f6b55d44af16092bd1eab94eb5a522ce1b495dab15f7d08500800fc02e8";
pub(crate) const REGION_CHAIN_WIRE_LEN: usize = 325;
pub(crate) const REGION_CHAIN_SIG_HEX: &str =
    "3136779bcee25a5acd434a7bdc9c76779994a4128712ee3f6cadfbfba00708f56eafe068d1eb9a89b09eb82c575eb659a24e07be6a6482c680d3ecaeafeb250a";

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
