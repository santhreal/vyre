//! xtask subcommands that read the live vyre operation registry.
//!
//! A subcommand belongs here when it must observe something that does not exist
//! in source text: which operations are actually registered, which primitives
//! they compose from, or which rewrite proofs the optimizer submitted. That is
//! why this crate links the compiler, and why the subcommands that answer their
//! question from source text stay in `xtask`, which links neither. The crates
//! that submit into a registry are linked through `vyre-registry-link`, which
//! owns those anchors and the floor per source.

pub mod compile;
pub mod docs;
pub mod gates;
pub mod print_composition;
pub mod release;
pub mod trace_f32;

/// Every gate this crate implements, in the shape the dispatcher runs them.
///
/// `xtask` links no vyre crate, so it cannot call these directly. It builds
/// this crate's binary and runs it, and the binary resolves the name against
/// this table, runs the gate and prints one `Report` on stdout.
pub static GATES: &[&dyn xtask::gate::Gate] = &[
    &docs::op_matrix::OpMatrixGate,
    &release::optimization_corpus::OptimizationCorpusGate,
    &release::optimization_matrix::OptimizationMatrixGate,
];

/// Every subcommand this crate implements, keyed by the name typed on the
/// command line. `xtask` routes to this crate by name and this table resolves
/// it, so the two lists have to agree; the test below is what enforces that.
pub const IMPLEMENTED: &[(&str, fn(&[String]))] = &[
    ("abstraction-gate", gates::abstraction_gate::run),
    ("catalog", docs::catalog::run),
    ("compile", compile::run),
    ("conformance-matrix", release::conformance_matrix::run),
    ("gate1", gates::gate1::run),
    ("heuristic-audit", gates::heuristic_audit::run),
    ("lego-audit", gates::lego_audit::run),
    ("lego-quick", gates::lego_quick::run),
    ("list-ops", docs::list_ops::run),
    ("operation-schema", docs::operation_schema::run),
    ("optimization-docs", docs::optimization_docs::run),
    ("primitive-admission-gate", |_args| {
        gates::lego_audit::run_primitive_admission_gate()
    }),
    ("print-composition", print_composition::run),
    ("trace-f32", trace_f32::run_cmd),
    ("verify-rewrite-proofs", gates::verify_rewrite_proofs::run),
    ("whats-similar", gates::whats_similar::run),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: this crate is built and run only when `xtask` decides a subcommand
    /// belongs here, so its table and the assignment in `xtask` are two
    /// declarations that must agree exactly, once each, and answer to nothing
    /// else. The checker derives both sides at call time.
    #[test]
    fn the_dispatch_table_agrees_with_the_assignment_in_xtask() {
        assert_eq!(
            xtask::subcommands::delegate_table_problems("xtask-registry", IMPLEMENTED),
            Vec::<String>::new()
        );
    }
}
