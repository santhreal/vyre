//! Bundle-level conformance certificate.
//!
//! The per-op [`Certificate`](vyre_conform_spec::Certificate) proves that a
//! single op behaves identically on a backend and on the reference
//! backend. A bundle cert widens that guarantee to a whole fused
//! document: every rule Program, dispatched over a named corpus,
//! produces a byte-identical output stream on the issuing reference
//! backend.
//!
//! ## Why
//!
//! warpscan ships with rule bundles. At internet scale the cost of a
//! single cross-backend divergence is "malware missed" (LAW 8). The
//! bundle cert collapses that risk to a startup check:
//!
//! 1. `security-analysis-consumer compile --cert-corpus <dir>`  -  running reference sweep
//!    captures `reference_output_blake3` into `<bundle>.cert`.
//! 2. `warpscan scan --verify-cert`  -  on startup, warpscan re-runs the
//!    same corpus through the live backend, blake3s the outputs, and
//!    refuses to proceed if it diverges from the cert.
//!
//! The cert is content-addressable: identical inputs produce byte-
//! identical certs, so the same file serves as both integrity proof
//! and pipeline-cache key for the compiled backend pipeline.
//!
//! ## Design
//!
//! - **Input determinism**: the corpus is supplied as a list of
//!   [`ConformanceCase`] records, each naming one dispatch. They're
//!   sorted by `name` before hashing so the same logical corpus
//!   produces the same cert regardless of enumeration order.
//! - **Output determinism**: outputs are captured as
//!   `Vec<Vec<u8>>` (one byte buffer per output buffer in the
//!   Program) and length-prefixed in the hash stream, so unlike plain
//!   concatenation the hash survives empty outputs without
//!   collision.
//! - **Signatures**: the cert body hashes, not the sig. Callers
//!   supply `signature_ed25519` + `pubkey` separately so the cert
//!   can round-trip through CI systems that sign artifacts
//!   out-of-band.

//!
//! ## Layout
//!
//! - `error` the failure surface every stage reports through
//! - `canonical` corpus canonicalisation and the hash stream
//! - `issue` producing a cert from the reference interpreter
//! - `verify` re-running a corpus and comparing against a cert
//! - `signature` cryptographic authentication of a cert body

pub mod canonical;
pub mod error;
pub mod issue;
pub mod signature;
pub mod verify;

#[cfg(test)]
mod tests;
