//! xtask gates that read the live vyre operation registry.
//!
//! A gate belongs here when it must observe something that does not exist in
//! source text: which operations are actually registered, which primitives they
//! compose from, or which rewrite proofs the optimizer submitted. That is why
//! this crate links the compiler, and why the gates that answer their question
//! from source text stay in `xtask`, which links neither. The crates that submit
//! into a registry are linked through `vyre-registry-link`, which owns those
//! anchors and the floor per source.

pub mod compile;
pub mod corpus;
pub mod docs;
pub mod gates;
pub mod print_composition;
pub mod release;
pub mod shrink;
pub mod trace_f32;

use xtask::gate::GateBehavior;

/// Every gate this crate implements. `xtask` assigns a gate to this package by
/// name and this table resolves it, so the two lists have to agree; the test
/// below is what enforces that.
pub static GATES: &[(&'static str, &'static dyn GateBehavior)] = &[
    ("compile", &compile::Compile),
    ("catalog", &docs::catalog::Catalog),
    ("list-ops", &docs::list_ops::ListOps),
    ("op-matrix", &docs::op_matrix::OpMatrixGate),
    (
        "operation-schema",
        &docs::operation_schema::OperationSchemaGate,
    ),
    (
        "optimization-docs",
        &docs::optimization_docs::OptimizationDocs,
    ),
    (
        "abstraction-gate",
        &gates::abstraction_gate::AbstractionGate,
    ),
    ("cross-target", &gates::cross_target::CrossTarget),
    ("gate1", &gates::gate1::Gate1),
    ("heuristic-audit", &gates::heuristic_audit::HeuristicAudit),
    ("lego-composability", &gates::lego_audit::LegoComposability),
    (
        "lego-composition-chains",
        &gates::lego_audit::LegoCompositionChains,
    ),
    (
        "lego-composition-depth",
        &gates::lego_audit::LegoCompositionDepth,
    ),
    ("lego-cross-dialect", &gates::lego_audit::LegoCrossDialect),
    (
        "lego-exemption-liveness",
        &gates::lego_audit::LegoExemptionLiveness,
    ),
    ("lego-name-stems", &gates::lego_audit::LegoNameStems),
    ("lego-no-reinvention", &gates::lego_audit::LegoNoReinvention),
    ("lego-operand-shapes", &gates::lego_audit::LegoOperandShapes),
    (
        "lego-primitive-coverage",
        &gates::lego_audit::LegoPrimitiveCoverage,
    ),
    (
        "lego-semantic-organization",
        &gates::lego_audit::LegoSemanticOrganization,
    ),
    ("lego-trend", &gates::lego_audit::LegoTrend),
    (
        "verify-rewrite-proofs",
        &gates::verify_rewrite_proofs::VerifyRewriteProofs,
    ),
    ("whats-similar", &gates::whats_similar::WhatsSimilar),
    ("print-composition", &print_composition::PrintComposition),
    (
        "conformance-matrix",
        &release::conformance_matrix::ConformanceMatrixGate,
    ),
    (
        "optimization-corpus",
        &release::optimization_corpus::OptimizationCorpusGate,
    ),
    (
        "optimization-matrix",
        &release::optimization_matrix::OptimizationMatrixGate,
    ),
    ("shrink", &shrink::Shrink),
    ("trace-f32", &trace_f32::TraceF32),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: this crate is built and run only when `xtask` decides a gate belongs
    /// here, so its table and the assignment in `xtask` are two declarations that
    /// must agree exactly, once each, and answer to nothing else. The checker
    /// derives both sides at call time.
    #[test]
    fn the_gate_table_agrees_with_the_assignment_in_xtask() {
        assert_eq!(
            xtask::subcommands::delegate_table_problems("xtask-registry", GATES),
            Vec::<String>::new()
        );
    }

    /// WHY: `--help` is a question about a gate, and a gate that reads the
    /// registry to answer it has run the check the caller asked it not to run.
    /// The answer is built from what the gate declares, so an option named in a
    /// help line and answered nowhere is a gap, and a gate added to this table
    /// is judged without being listed anywhere else.
    #[test]
    fn every_gate_answers_help_with_the_options_it_names() {
        assert_eq!(
            xtask::gate::usage_gaps_delegated(GATES),
            Vec::<String>::new()
        );
    }
}
