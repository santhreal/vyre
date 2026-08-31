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

// Every bundle moved once more, hash and signature together, with every wire
// length unchanged: `WIRE_FORMAT_VERSION` went from 7 to 8 for the schedule-free
// logical identity and synchronization variants. The version is a fixed-width
// little-endian `u16` in the header, so a bump rewrites two header bytes and no
// body bytes, which is exactly the shape of a drift with constant lengths. The
// pinned reference output words held, and `PINNED_WIRE_FORMAT_VERSION` below now
// states the schema these digests were taken under so the next bump reports
// itself instead of five opaque digests.

/// Wire schema version the pinned digests below were taken under.
pub(crate) const PINNED_WIRE_FORMAT_VERSION: u16 = 8;

// --- trivial const ---
pub(crate) const TRIVIAL_CONST_BUNDLE_BLAKE3: &str =
    "630f514ac2a978eefec3988c432ae4944544fcd35c20307f5217ba7af07c01df";
pub(crate) const TRIVIAL_CONST_WIRE_LEN: usize = 208;
pub(crate) const TRIVIAL_CONST_SIG_HEX: &str =
    "87964bae569dd9ccda99e53c4f020d398ce97c3725a40952f5c6f876f806af0729758522998c18cbf3ccc0b50a283d7e07ce0781c7ad20af462d41eba58d370c";

// --- 1-op add ---
pub(crate) const ONE_OP_ADD_BUNDLE_BLAKE3: &str =
    "f939b667b6b9916dc0a3cf20fd6dc7be636d85dac3c77473d1e660392d0f1af9";
pub(crate) const ONE_OP_ADD_WIRE_LEN: usize = 215;
pub(crate) const ONE_OP_ADD_SIG_HEX: &str =
    "ac34ef91b691aab5e61e1b0eaf115b776ccaa31e800fbe11a04d9753e611c7bce23dbbfc8326f54c89363b406a2ce37a70a2f013cc517275162538e0f8353208";

// --- loop-add ---
pub(crate) const LOOP_ADD_BUNDLE_BLAKE3: &str =
    "3ec9a54beac0e9f2e4e687aa6acee458e790e1d58f543eb2f892befb943ee15f";
pub(crate) const LOOP_ADD_WIRE_LEN: usize = 268;
pub(crate) const LOOP_ADD_SIG_HEX: &str =
    "09d76d3bff98b1b9c39dceb66b377f893e3603be71f98cd95d28a10306588225c072b991eed194cf1da330d78d91eed37fdd3fb178ba29f201129502538e1c04";

// --- composed nested ---
pub(crate) const COMPOSED_NESTED_BUNDLE_BLAKE3: &str =
    "6f5dbe12dd9769635341998d67ab6d2e4b69afd6d5a9ee61aae97faaa2abfa69";
pub(crate) const COMPOSED_NESTED_WIRE_LEN: usize = 200;
pub(crate) const COMPOSED_NESTED_SIG_HEX: &str =
    "d4bb1b68e57862d06e6427c919230e468c8b25c6c04bc1022bc37d6320c3144514690e1467e0a119be5d6ce2ba66d96b9a42abcfce78f6df3baab12b30a18208";

// --- region-chain with intrinsic + dialect op ---
pub(crate) const REGION_CHAIN_BUNDLE_BLAKE3: &str =
    "82793e2e1147d9050a94cbe3001f7cb80ab18643c1c3c0860af9f1535649e8b7";
pub(crate) const REGION_CHAIN_WIRE_LEN: usize = 325;
pub(crate) const REGION_CHAIN_SIG_HEX: &str =
    "85799ff11863311c0486e7cd82c3f4c3e19495a588b2a11f2f4e7728cfb51d9d0d84bfc83df5840d42863e2fbaaa04303e2c5d9ede283d6ace0b35cedfa1a701";
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
