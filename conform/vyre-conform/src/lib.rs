//! Canonical conformance engine for proof execution, replay, minimization,
//! algebraic-law checking, and certificate verification.

pub mod bundle_cert;
pub mod cert;
pub mod convergence_lens;
pub mod dispatch_grid;
pub mod lens;
pub mod minimizer;
pub mod production;
pub mod prover;
#[doc(hidden)]
pub mod witness_plan;

pub use bundle_cert::error::BundleCertError;
pub use bundle_cert::issue::issue_bundle_cert;
pub use bundle_cert::signature::verify_cert_signature_hex;
pub use bundle_cert::verify::{verify_bundle_against_reference, verify_bundle_with_backend};
pub use cert::{issue_certificate, verify_structural, CertificateError, IssueInput};
pub use minimizer::{CounterexampleMinimizer, MinimizationBudget, MinimizerReport};
pub use production::{ExecutionRoute, ProductionError, ProductionSession, ReplayCapsule};
pub use prover::{LawProver, LawVerdict};
