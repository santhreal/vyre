//! Cryptographic authentication of a certificate body.
//!
//! The hash-chain verifiers prove that a corpus still produces the outputs a
//! cert records. They prove nothing about who issued it. This does, against a
//! public key the caller supplies out of band.

use vyre_conform_spec::BundleCertificate;

use super::error::BundleCertError;

/// 2026-04-23 C1 (companion API): cryptographically
/// verify the Ed25519 signature on a BundleCertificate. The
/// existing `verify_bundle_with_backend` + `verify_bundle_against_reference`
/// entrypoints check the hash chain; they do **not** check the
/// signature. An attacker who can tamper with the hex strings on a
/// shipped cert still has to match the hash chain to produce a
/// cert that verifies via the legacy path, but a bug-compatible
/// downstream consumer that treats "signature field non-empty" as
/// "cryptographically authenticated" is mistaken.
///
/// Callers that require cryptographic authentication must invoke
/// this helper alongside the hash-chain verifier, providing the
/// trusted public key out-of-band. The helper:
/// 1. Validates hex length of `signature_ed25519` (128 hex chars) +
///    the cert-declared `pubkey` (64 hex chars).
/// 2. Confirms the declared pubkey matches the caller-provided
///    trusted key.
/// 3. Verifies the signature over the canonical JSON body of the
///    cert (every field except `signature_ed25519` itself, serialised
///    in a stable field order).
///
/// # Errors
///
/// Returns [`BundleCertError::UnsetField`] when unsigned sentinel fields,
/// malformed hex, key mismatch, or Ed25519 verification failure makes the
/// certificate unauthenticated.
#[must_use = "the signature-verification result must be inspected; dropping it accepts an unverified cert"]
pub fn verify_cert_signature_hex(
    cert: &BundleCertificate,
    trusted_pubkey_hex: &str,
) -> Result<(), BundleCertError> {
    cert.validate_schema_version()
        .map_err(|error| BundleCertError::UnsupportedSchemaVersion(error.found().to_string()))?;
    // Hex-length sanity on declared fields.
    if cert.signature_ed25519 == "TBD" || cert.pubkey == "TBD" {
        return Err(BundleCertError::UnsetField(
            "signature_ed25519 or pubkey still set to 'TBD'  -  sign the cert before shipping.",
        ));
    }
    if cert.signature_ed25519.len() != 128
        || !cert
            .signature_ed25519
            .chars()
            .all(|c| c.is_ascii_hexdigit())
    {
        return Err(BundleCertError::UnsetField(
            "signature_ed25519 must be 128 lowercase hex chars (64 raw bytes)",
        ));
    }
    if cert.pubkey.len() != 64 || !cert.pubkey.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(BundleCertError::UnsetField(
            "pubkey must be 64 lowercase hex chars (32 raw bytes)",
        ));
    }
    if trusted_pubkey_hex.len() != 64 || !trusted_pubkey_hex.chars().all(|c| c.is_ascii_hexdigit())
    {
        return Err(BundleCertError::UnsetField(
            "trusted_pubkey_hex must be 64 lowercase hex chars (32 raw bytes)",
        ));
    }
    if !cert.pubkey.eq_ignore_ascii_case(trusted_pubkey_hex) {
        return Err(BundleCertError::UnsetField(
            "cert pubkey does not match trusted_pubkey_hex  -  the cert was signed by a different key than the one the caller trusts. This is a fraud signal.",
        ));
    }
    // Cryptographic verification of the signature over the cert's
    // canonical JSON body (every field except signature_ed25519
    // itself, serialised with field order fixed by the
    // BundleCertificate struct declaration). The ed25519-dalek
    // dep already ships in vyre-conform for issue_bundle_cert.
    let sig_bytes = hex::decode(&cert.signature_ed25519).map_err(|_| {
        BundleCertError::UnsetField(
            "signature_ed25519 is not valid hex; impossible after length check, but defensive.",
        )
    })?;
    let pk_bytes = hex::decode(&cert.pubkey).map_err(|_| {
        BundleCertError::UnsetField(
            "pubkey is not valid hex; impossible after length check, but defensive.",
        )
    })?;
    let pk_array: [u8; 32] = pk_bytes.as_slice().try_into().map_err(|_| {
        BundleCertError::UnsetField("pubkey decoded to the wrong byte length; defensive.")
    })?;
    let sig_array: [u8; 64] = sig_bytes.as_slice().try_into().map_err(|_| {
        BundleCertError::UnsetField(
            "signature_ed25519 decoded to the wrong byte length; defensive.",
        )
    })?;
    let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pk_array).map_err(|_| {
        BundleCertError::UnsetField(
            "pubkey is not a valid Ed25519 compressed point  -  the cert cannot have been signed by this key.",
        )
    })?;
    let signature = ed25519_dalek::Signature::from_bytes(&sig_array);
    // Reconstruct the signable body the same way issue_bundle_cert
    // does: every field except signature_ed25519 itself, in struct-
    // declaration order. A stable JSON encoder keeps this
    // deterministic across runs.
    let signable = serde_json::json!({
        "version": cert.version,
        "bundle_blake3": cert.bundle_blake3,
        "corpus_blake3": cert.corpus_blake3,
        "reference_output_blake3": cert.reference_output_blake3,
        "witness_count": cert.witness_count,
        "timestamp": cert.timestamp,
        "pubkey": cert.pubkey,
    });
    let signable_bytes = serde_json::to_vec(&signable).map_err(|_| {
        BundleCertError::UnsetField(
            "failed to serialise cert body for signature verification  -  impossible on well-formed cert.",
        )
    })?;
    use ed25519_dalek::Verifier;
    verifying_key
        .verify(&signable_bytes, &signature)
        .map_err(|_| {
            BundleCertError::UnsetField(
                "Ed25519 signature does not match cert body. The cert was tampered or signed by a different key.",
            )
        })?;
    Ok(())
}
