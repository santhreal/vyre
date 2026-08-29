//! Canonical conformance engine for proof execution, replay, minimization,
//! algebraic-law checking, and certificate verification.

pub mod bundle_cert;
pub mod cert;
pub mod convergence_lens;
pub mod lens;
pub mod minimizer;
pub mod panic_payload;
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
pub use panic_payload::panic_message;
pub use production::{
    check_family_outputs, check_schedule_agreement, submit_under_every_schedule, ProductionError,
    ProductionExecution, ProductionSession, ReplayCapsule, ScheduleAgreement,
    ScheduleAgreementReport, ScheduleDisagreement, ScheduleOutcome, CONFORMANCE_SCHEDULES,
};
pub use prover::{LawProver, LawVerdict};
