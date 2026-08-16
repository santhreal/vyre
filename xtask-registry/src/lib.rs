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

use xtask::gate::Gate;

/// Every gate this crate implements. `xtask` assigns a gate to this package by
/// name and this table resolves it, so the two lists have to agree; the test
/// below is what enforces that.
pub static GATES: &[&dyn Gate] = &[
    &compile::Compile,
    &docs::catalog::Catalog,
    &docs::list_ops::ListOps,
    &docs::op_matrix::OpMatrixGate,
    &docs::operation_schema::OperationSchemaGate,
    &docs::optimization_docs::OptimizationDocs,
    &gates::abstraction_gate::AbstractionGate,
    &gates::cross_target::CrossTarget,
    &gates::gate1::Gate1,
    &gates::heuristic_audit::HeuristicAudit,
    &gates::lego_audit::LegoAudit,
    &gates::lego_audit::PrimitiveAdmissionGate,
    &gates::verify_rewrite_proofs::VerifyRewriteProofs,
    &gates::whats_similar::WhatsSimilar,
    &print_composition::PrintComposition,
    &release::conformance_matrix::ConformanceMatrixGate,
    &release::optimization_corpus::OptimizationCorpusGate,
    &release::optimization_matrix::OptimizationMatrixGate,
    &shrink::Shrink,
    &trace_f32::TraceF32,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delegated_contracts() {
        xtask::delegate::assert_delegated_crate_contracts("xtask-registry", GATES);
    }
}
