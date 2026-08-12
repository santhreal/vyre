#![forbid(unsafe_code)]

//! Canonical conformance engine for proof execution, replay, minimization,
//! algebraic-law checking, and certificate verification.

pub mod bundle_cert;
pub mod cert;
pub mod convergence_lens;
pub mod dispatch_grid;
mod fp_contract;
pub mod fp_parity;
pub mod lens;
pub mod minimizer;
pub mod production;
pub mod prover;
#[doc(hidden)]
pub mod witness_plan;

pub use bundle_cert::{
    issue_bundle_cert, verify_bundle_against_reference, verify_bundle_with_backend,
    verify_cert_signature_hex, BundleCertError,
};
pub use cert::{issue_certificate, verify_structural, CertificateError, IssueInput};
pub use minimizer::CounterexampleMinimizer;
pub use production::{ProductionError, ProductionSession};
pub use prover::{LawProver, LawVerdict};
