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
    ("op-matrix", docs::op_matrix::run),
    ("operation-schema", docs::operation_schema::run),
    ("optimization-corpus", release::optimization_corpus::run),
    ("optimization-docs", docs::optimization_docs::run),
    ("optimization-matrix", release::optimization_matrix::run),
    ("primitive-admission-gate", |_args| {
        gates::lego_audit::run_primitive_admission_gate()
    }),
    ("print-composition", print_composition::run),
    ("trace-f32", trace_f32::run_cmd),
    ("verify-rewrite-proofs", gates::verify_rewrite_proofs::run),
    ("whats-similar", gates::whats_similar::run),
];

/// Run the named subcommand, or report that it is not implemented here.
///
/// `args` is the process argument vector, so the subcommand name is `args[1]`
/// and every subcommand reads its own options from the same slice `xtask` would
/// have passed it.
pub fn dispatch(name: &str, args: &[String]) -> bool {
    xtask::subcommands::dispatch(IMPLEMENTED, name, args)
}

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
