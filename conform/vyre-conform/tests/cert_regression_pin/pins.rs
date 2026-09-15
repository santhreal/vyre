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
pub(crate) const PINNED_WIRE_FORMAT_VERSION: u16 = 9;

// --- trivial const ---
pub(crate) const TRIVIAL_CONST_BUNDLE_BLAKE3: &str =
    "53717d96801368d2c0728b54d00922901124a2513a622bd7fa1efa0e0e735b36";
pub(crate) const TRIVIAL_CONST_WIRE_LEN: usize = 208;
pub(crate) const TRIVIAL_CONST_SIG_HEX: &str =
    "2fbf5a4d728cd280dea52bc8b770ba2894e15194b91424ea7e79b8ebd05a79e288dd10aaec0e357f914043b546d396b4839e334bd93d0fd9c6c3b4eff7c7ec0b";

// --- 1-op add ---
pub(crate) const ONE_OP_ADD_BUNDLE_BLAKE3: &str =
    "94a1949e735e333a6fe83fc8257b3afe4721cf672e753baeb3571a108c08f37c";
pub(crate) const ONE_OP_ADD_WIRE_LEN: usize = 215;
pub(crate) const ONE_OP_ADD_SIG_HEX: &str =
    "0bc130979b22facbb5c04c1682e1b573b340601a9ee18946717ff6dfacf14c4bbda742e73ec2e5afcf85f64b925719c160ad4c14edc4391e5bfac234363e750d";

// --- loop-add ---
pub(crate) const LOOP_ADD_BUNDLE_BLAKE3: &str =
    "7efc94ba0331c134bad1ba3ea15f79e08e96acc3c74d11debff8ae171f697a66";
pub(crate) const LOOP_ADD_WIRE_LEN: usize = 268;
pub(crate) const LOOP_ADD_SIG_HEX: &str =
    "50a34428247775400c9b8c0eb612c1b082b4afd35447d22da107a8e1c73a3c790bc7e433af0d82ae9ae2b312e804f2bfdb9378a05a39162172b186f94d7dff0a";

// --- composed nested ---
pub(crate) const COMPOSED_NESTED_BUNDLE_BLAKE3: &str =
    "317862046a2c4982d772185cbadb1237841c9e12bc96153ca13e13a63d3b67a4";
pub(crate) const COMPOSED_NESTED_WIRE_LEN: usize = 200;
pub(crate) const COMPOSED_NESTED_SIG_HEX: &str =
    "ff85e7ba654cad9de41f95ba4327670a20b1e31eb43386fa35f59d4f848a52bf2669b76b239809a317fa818959016d6a42c447ec62b72fc391e346daf6a4cb04";

// --- region-chain with intrinsic + dialect op ---
pub(crate) const REGION_CHAIN_BUNDLE_BLAKE3: &str =
    "828ffd0b0b066a199b5622ec003b484a0cb1c6a381c50d925e3540a64e5c8b3b";
pub(crate) const REGION_CHAIN_WIRE_LEN: usize = 325;
pub(crate) const REGION_CHAIN_SIG_HEX: &str =
    "7f5a4635431e97f1bf94c84f55c2395b40110f0e27837d5adde1f00f71780be9150ebf3c0bb1c2b64c0e792ebd4d079aadc1b8dd116635bb47a1f2fd84a62e04";
// ---------------------------------------------------------------------------
// Sign a bundle cert with the deterministic key.
// ---------------------------------------------------------------------------

/// Sign `cert` in place with `key`.
///
/// The signable body is `BundleCertificate::to_signable_bytes`, which covers
/// `pubkey`, so the verifying key is recorded on the certificate before the
/// bytes are taken.
pub(crate) fn sign_bundle_cert(cert: &mut BundleCertificate, key: &SigningKey) {
    cert.pubkey = hex::encode(key.verifying_key().to_bytes());
    let signable_bytes = cert.to_signable_bytes().expect("canonical json");
    let signature = key.sign(&signable_bytes);
    cert.signature_ed25519 = hex::encode(signature.to_bytes());
}

// ---------------------------------------------------------------------------
// Bundle 1  -  trivial const
