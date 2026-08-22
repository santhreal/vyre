//! Conformance certificate.
//!
//! Emitted by the runner after verifying an op satisfies its laws on a
//! backend. Byte-identical across backends (modulo `backend_id`) = portable op.

use vyre::ir::OpId;
use vyre_conform_spec::{Certificate, CERTIFICATE_SCHEMA_VERSION};

/// P5.4  -  Input to [`issue_certificate`]. The issuer runs the
/// witness corpus through the target backend and through the CPU
/// reference, then supplies the pair to this helper which
/// computes `program_blake3`, `witness_set_blake3`, the UTC
/// timestamp, and fills in the provided signature fields.
#[derive(Debug, Clone)]
pub struct IssueInput<'a> {
    /// Stable op id  -  matches the op's fingerprint entry.
    pub op_id: &'a OpId,
    /// `to_wire()` bytes of the **canonicalized** Program.
    pub program_wire_bytes: &'a [u8],
    /// Concatenated witness-input bytes (sorted for determinism).
    pub witness_bytes: &'a [u8],
    /// Backend that produced the outputs.
    pub backend_id: &'a str,
    /// `backend.version()` at issue time.
    pub backend_version: &'a str,
    /// Algebraic laws the runner verified hold on the witness set.
    pub laws_verified: Vec<String>,
    /// UTC ISO-8601 timestamp ("2026-04-20T00:00:00Z").
    pub timestamp: &'a str,
    /// Ed25519 signature over the canonical JSON body (hex).
    pub signature_ed25519: &'a str,
    /// Ed25519 public key (hex).
    pub pubkey: &'a str,
}

/// Compute an OCC certificate from runner inputs. Fills every
/// derived field; the caller supplies the signature + pubkey.
///
/// # Errors
///
/// Returns [`CertificateError::EmptyProgramWire`] if the program
/// bytes are empty (indicating a serialization bug upstream) or
/// [`CertificateError::EmptyWitnessSet`] if no witnesses ran.
pub fn issue_certificate(input: IssueInput<'_>) -> Result<Certificate, CertificateError> {
    if input.program_wire_bytes.is_empty() {
        return Err(CertificateError::EmptyProgramWire);
    }
    if input.witness_bytes.is_empty() {
        return Err(CertificateError::EmptyWitnessSet);
    }
    let program_blake3 = blake3::hash(input.program_wire_bytes).to_hex().to_string();
    let witness_set_blake3 = blake3::hash(input.witness_bytes).to_hex().to_string();

    Ok(Certificate {
        version: CERTIFICATE_SCHEMA_VERSION.to_string(),
        op_id: input.op_id.to_string(),
        wire_format_version: 1,
        program_blake3,
        witness_set_blake3,
        backend_id: input.backend_id.to_string(),
        backend_version: input.backend_version.to_string(),
        laws_verified: input.laws_verified,
        timestamp: input.timestamp.to_string(),
        signature_ed25519: input.signature_ed25519.to_string(),
        pubkey: input.pubkey.to_string(),
    })
}

/// P5.4  -  Structural verification of an OCC. Checks that every
/// field is populated with something other than the "TBD"
/// sentinel and the structural fingerprints parse as 64-char hex.
/// Cryptographic signature verification is a separate step that
/// requires an `ed25519_dalek`-style verifier; kept out of the
/// core crate so the base dep graph stays minimal.
///
/// # Errors
///
/// [`CertificateError::UnsetField`] when any field still carries
/// the "TBD" sentinel; [`CertificateError::BadFingerprint`] when
/// a blake3 field isn't 64 hex chars.
pub fn verify_structural(cert: &Certificate) -> Result<(), CertificateError> {
    cert.validate_schema_version()
        .map_err(|error| CertificateError::UnsupportedSchemaVersion(error.found().to_string()))?;
    for (name, value) in [
        ("program_blake3", &cert.program_blake3),
        ("witness_set_blake3", &cert.witness_set_blake3),
        ("signature_ed25519", &cert.signature_ed25519),
        ("pubkey", &cert.pubkey),
    ] {
        if value == "TBD" {
            return Err(CertificateError::UnsetField(name.to_string()));
        }
    }
    for (name, value) in [
        ("program_blake3", &cert.program_blake3),
        ("witness_set_blake3", &cert.witness_set_blake3),
    ] {
        if value.len() != 64 || !value.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(CertificateError::BadFingerprint(name.to_string()));
        }
    }
    Ok(())
}

/// Errors from OCC issuing / structural verification.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CertificateError {
    /// `program_wire_bytes` was empty at issue time  -  upstream
    /// serialization bug.
    #[error("empty program wire bytes  -  Program::to_wire() failed upstream")]
    EmptyProgramWire,
    /// Witness set was empty at issue time  -  nothing to certify.
    #[error("empty witness set  -  no witnesses ran through the backend")]
    EmptyWitnessSet,
    /// Certificate schema version is not supported by this runner.
    #[error(
        "unsupported certificate schema version `{0}`; supported version is `{CERTIFICATE_SCHEMA_VERSION}`"
    )]
    UnsupportedSchemaVersion(String),
    /// A cert field is still the "TBD" sentinel.
    #[error("cert field `{0}` is still set to the reserved value 'TBD'")]
    UnsetField(String),
    /// A blake3 fingerprint field isn't 64 hex chars.
    #[error("cert field `{0}` is not a 64-char hex fingerprint")]
    BadFingerprint(String),
}
