//! The failure surface of bundle-cert issue and verify.

use vyre_conform_spec::CERTIFICATE_SCHEMA_VERSION;

/// Errors surfaced by bundle-cert issue / verify.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BundleCertError {
    /// Corpus is empty  -  nothing to certify.
    #[error("bundle cert requires at least one witness. Fix: supply a non-empty corpus.")]
    EmptyCorpus,
    /// Bundle wire bytes are empty  -  upstream serialization bug.
    #[error(
        "bundle wire bytes empty  -  upstream to_wire() regressed. Fix: re-run security-analysis-consumer compile."
    )]
    EmptyBundle,
    /// CPU reference refused a witness.
    #[error(
        "reference interpreter rejected witness `{witness}`: {message}. Fix: inspect the Program body  -  reference must accept what the backend does."
    )]
    ReferenceFailed {
        /// Name of the witness that tripped.
        witness: String,
        /// Rendered error surfaced by `vyre_reference::reference_eval`.
        message: String,
    },
    /// Canonical compiler, payload, materialization, or submission failed.
    #[error(
        "production artifact route rejected witness `{witness}` during cert verification: {message}. Fix: repair the compiler, target payload, materializer, bindings, or submission stage."
    )]
    ProductionFailed {
        /// Name of the witness or setup stage that failed.
        witness: String,
        /// Structured production-route failure rendered at the boundary.
        message: String,
    },
    /// `bundle_blake3` mismatched on verify.
    #[error(
        "bundle wire hash mismatch: cert declares {expected}, observed {observed}. Fix: the Program has been modified since the cert was issued  -  recompile."
    )]
    BundleHashMismatch {
        /// Hash the cert declared.
        expected: String,
        /// Hash observed at verify time.
        observed: String,
    },
    /// `corpus_blake3` mismatched on verify.
    #[error(
        "corpus input hash mismatch: cert declares {expected}, observed {observed}. Fix: the corpus has drifted since cert was issued  -  ensure identical witnesses."
    )]
    CorpusHashMismatch {
        /// Hash the cert declared.
        expected: String,
        /// Hash observed at verify time.
        observed: String,
    },
    /// `reference_output_blake3` mismatched on verify  -  divergence.
    #[error(
        "reference output hash mismatch: cert declares {expected}, backend produced {observed}. Fix: backend diverges from the certified reference. Either the backend regressed or the bundle was certified on a stale reference."
    )]
    OutputHashMismatch {
        /// Hash the cert declared.
        expected: String,
        /// Hash observed at verify time.
        observed: String,
    },
    /// Certificate schema version is not supported by this runner.
    #[error(
        "unsupported bundle certificate schema version `{0}`; supported version is `{CERTIFICATE_SCHEMA_VERSION}`"
    )]
    UnsupportedSchemaVersion(String),
    /// A cert field is still the "TBD" sentinel.
    #[error("cert field `{0}` is still set to the reserved value 'TBD'  -  sign before shipping")]
    UnsetField(&'static str),
    /// Two witnesses in the corpus share the same name.
    ///
    /// 2026-04-23 H5: duplicate witness names hash
    /// deterministically (the canonicalisation sort is stable), so
    /// the cert verifies, but any downstream display or cache that
    /// indexes by name silently overwrites one entry with the other.
    /// A forged corpus pairing one benign and one malicious witness
    /// with the same name can therefore smuggle through.
    #[error(
        "duplicate witness name `{name}` in corpus. Fix: witness names must be unique  -  rename one of the colliding entries before issuing the cert."
    )]
    DuplicateWitnessName {
        /// Name that appeared more than once.
        name: String,
    },
    /// Cert-declared witness count doesn't match the corpus it
    /// was built against.
    ///
    /// 2026-04-23 L1: `witness_count` was stored
    /// but never validated on verify. A tampered cert could claim
    /// a misleading count without affecting the hash chain. Now
    /// verify rejects the mismatch with both values named.
    #[error(
        "witness count mismatch: cert declares {expected}, corpus has {observed}. Fix: the cert was built against a different corpus size  -  re-issue with the current corpus."
    )]
    WitnessCountMismatch {
        /// Count declared in the cert.
        expected: u64,
        /// Count observed on verify.
        observed: u64,
    },
    /// Logical witness inputs could not be expanded into the Program dispatch
    /// input stream.
    #[error(
        "witness planner rejected witness `{witness}`: {message}. Fix: make corpus inputs follow logical fixture order and provide bytes for runtime-sized read-write buffers."
    )]
    WitnessPlanningFailed {
        /// Name of the witness that tripped.
        witness: String,
        /// Planner diagnostic.
        message: String,
    },
}
