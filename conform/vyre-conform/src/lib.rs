//! Canonical conformance engine for proof execution, replay, minimization,
//! algebraic-law proof, and certificate verification.

pub mod backend_selection;
pub mod bundle_cert;
pub mod cert;
pub mod convergence_lens;
pub mod coordinator;
pub mod law_proof;
pub mod lens;
pub mod minimizer;
pub mod oracle;
pub mod panic_payload;
pub mod production;
#[doc(hidden)]
pub mod witness_plan;
pub mod worker;

pub use backend_selection::{backend_registration, select_backends, semantic_execution_backends};
pub use bundle_cert::error::BundleCertError;
pub use bundle_cert::issue::issue_bundle_cert;
pub use bundle_cert::signature::verify_cert_signature_hex;
pub use bundle_cert::verify::{verify_bundle_against_reference, verify_bundle_with_backend};
pub use cert::{issue_certificate, verify_structural, CertificateError, IssueInput};
pub use coordinator::{DeviceLeaseManager, WorkerCoordinator};
pub use law_proof::{
    prove_declared_laws, prove_law, LawProof, LawVerdict, LawWitness, UnprovenKind,
};
pub use minimizer::{CounterexampleMinimizer, MinimizationBudget, MinimizerReport};
pub use oracle::{OracleError, OracleSession};
pub use panic_payload::panic_message;
pub use production::{
    check_family_outputs, check_schedule_agreement, submit_under_every_schedule, ProductionError,
    ProductionExecution, ProductionSession, ReplayCapsule, ScheduleAgreement,
    ScheduleAgreementReport, ScheduleDisagreement, ScheduleOutcome, CONFORMANCE_SCHEDULES,
};
pub use worker::{
    current_binary_digest, current_environment_digest, execute_worker_request,
    run_worker_from_env_or_exit, run_worker_stdio, DEFAULT_WORKER_SECRET,
};
