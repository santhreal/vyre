//! The signed conformance certificate on the wire, and what its signature covers.
//!
//! The bytes an Ed25519 signature is computed over are a field list in a fixed
//! order. Two copies of that list drift: a reader that rebuilds the body from
//! its own struct drops a field the writer added, and the signature stops
//! verifying over a certificate nothing tampered with. Adding
//! `unavailable_backends` to the plan did exactly that to the live-GPU
//! certificate contracts, whose own mirror of these types had no such field.
//!
//! So the wire shape and the verification live here, in the library both the
//! `vyre-conform` binary and every test link, and neither rebuilds the body for
//! itself.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use vyre_conform_spec::ConformanceResult;

/// One backend a run did not cover, and why.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnavailableBackendRecord {
    /// Stable backend identifier.
    pub id: String,
    /// The acquisition refusal, verbatim.
    pub reason: String,
}

/// What a proof run selected out of the registered universe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProofSelectionSummary {
    /// Backend filter the run was invoked with.
    pub backend_filter: String,
    /// Operation filter the run was invoked with.
    pub ops_filter: String,
    /// Shard index, for a sharded run.
    pub shard_index: Option<usize>,
    /// Shard count, for a sharded run.
    pub shard_count: Option<usize>,
    /// Registered backends before filtering.
    pub universe_backend_count: usize,
    /// Registered operations before filtering.
    pub universe_op_count: usize,
    /// Backends this run proved.
    pub selected_backend_count: usize,
    /// Operations this run proved.
    pub selected_op_count: usize,
    /// Registered backends this host cannot acquire, and what refused each.
    ///
    /// A certificate covers the backends the host can run. Naming the rest, with
    /// the refusal, is what keeps `selected_backend_count` from reading as the
    /// whole registered set on a host that carries more registrations than
    /// devices.
    ///
    /// A host that ran every registered backend omits the field rather than
    /// writing an empty list, so its certificate is byte-identical to one
    /// signed before this field existed and the signature over it still
    /// verifies.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unavailable_backends: Vec<UnavailableBackendRecord>,
}

/// The plan a proof run executed, and the hashes that identify it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProofPlanSummary {
    /// Backends the run proved.
    pub backend_count: usize,
    /// Operations the run proved.
    pub op_count: usize,
    /// `(backend, operation)` pairs the run executed.
    pub pair_count: usize,
    /// Witness cases across every pair.
    pub witness_case_count: usize,
    /// Identity of the operation catalog the run read.
    pub catalog_hash: String,
    /// Identity of the executed selection.
    pub execution_hash: String,
    /// What the run selected out of the registered universe.
    pub selection: ProofSelectionSummary,
}

/// One declared law and the witness that proved it on the reference oracle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LawRecord {
    /// Operation the law is declared on.
    pub op_id: String,
    /// Declared law.
    pub law: String,
    /// Witness that proved it, or `none`.
    pub witness: String,
    /// Fixture cases the proof ran.
    pub cases: usize,
}

/// A signed conformance certificate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProveArtifact {
    /// Schema version of this certificate.
    pub wire_format_version: u32,
    /// Identity of the proved program set.
    pub program_hash: String,
    /// Backend filter the certificate covers, or `merged`.
    pub backend_id: String,
    /// The plan the run executed.
    pub plan: ProofPlanSummary,
    /// Ed25519 signature over [`ProveSignableBody`], hex.
    pub signature: String,
    /// Ed25519 verifying key, hex.
    pub public_key: String,
    /// Every `(executor, operation)` pair the run proved.
    pub pairs: Vec<ConformanceResult>,
    /// Every declared law the run proved.
    #[serde(default)]
    pub laws: Vec<LawRecord>,
}

/// Exactly the fields a certificate signature is computed over, in order.
///
/// Borrowed from a [`ProveArtifact`] rather than rebuilt, so a signer and a
/// verifier cannot disagree about the field list.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProveSignableBody<'a> {
    /// Schema version of the certificate.
    pub wire_format_version: u32,
    /// Identity of the proved program set.
    pub program_hash: &'a str,
    /// Backend filter the certificate covers.
    pub backend_id: &'a str,
    /// The plan the run executed.
    pub plan: &'a ProofPlanSummary,
    /// Every proved pair.
    pub pairs: &'a [ConformanceResult],
    /// Every proved law.
    pub laws: &'a [LawRecord],
}

impl ProveArtifact {
    /// The exact bytes this certificate's signature covers.
    #[must_use]
    pub fn signable_body(&self) -> ProveSignableBody<'_> {
        ProveSignableBody {
            wire_format_version: self.wire_format_version,
            program_hash: &self.program_hash,
            backend_id: &self.backend_id,
            plan: &self.plan,
            pairs: &self.pairs,
            laws: &self.laws,
        }
    }

    /// Verify this certificate against the key it carries.
    ///
    /// # Errors
    ///
    /// Returns when the signature or key is not hex, is the wrong length, is
    /// not a valid Ed25519 value, when the body does not serialize, or when the
    /// signature does not cover the body.
    pub fn verify_signature(&self) -> Result<(), String> {
        let signature_bytes = hex::decode(&self.signature)
            .map_err(|error| format!("certificate signature is not hex: {error}"))?;
        let public_key_bytes = hex::decode(&self.public_key)
            .map_err(|error| format!("certificate public_key is not hex: {error}"))?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|error| format!("certificate signature is invalid: {error}"))?;
        let public_key_array: [u8; 32] = public_key_bytes
            .as_slice()
            .try_into()
            .map_err(|_| "certificate public_key must decode to 32 bytes".to_string())?;
        let verifying_key = VerifyingKey::from_bytes(&public_key_array)
            .map_err(|error| format!("certificate public_key is invalid: {error}"))?;
        let signable_bytes = serde_json::to_vec(&self.signable_body())
            .map_err(|error| format!("certificate signable body does not serialize: {error}"))?;
        verifying_key
            .verify(&signable_bytes, &signature)
            .map_err(|error| format!("certificate signature verification failed: {error}"))
    }
}
