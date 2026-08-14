//! xtask subcommands that read the live vyre operation registry.
//!
//! A subcommand belongs here when it must observe something that does not exist
//! in source text: which operations are actually registered, which primitives
//! they compose from, or which rewrite proofs the optimizer submitted. That is
//! why this crate links the compiler and the drivers, and why the subcommands
//! that answer their question from source text stay in `xtask`, which links
//! neither.


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
    ("primitive-admission-gate", |_args| gates::lego_audit::run_primitive_admission_gate()),
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
    match IMPLEMENTED.iter().find(|(row, _)| *row == name) {
        Some((_, run)) => {
            run(args);
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xtask::subcommands;

    /// WHY: `xtask` decides which crate owns a subcommand and builds only that
    /// one. A row it assigns here with no entry in `IMPLEMENTED` would fail as
    /// an unknown subcommand after paying for the build, and an entry here that
    /// `xtask` assigns elsewhere is unreachable. Both lists are derived at run
    /// time, so adding a subcommand to one and not the other turns this red.
    #[test]
    fn dispatch_matches_the_subcommands_xtask_assigns_here() {
        let mut assigned = subcommands::owned_by("xtask-registry");
        let mut implemented: Vec<&str> = IMPLEMENTED.iter().map(|(name, _)| *name).collect();
        assigned.sort_unstable();
        implemented.sort_unstable();
        assert_eq!(assigned, implemented);
    }

    /// WHY: dispatch resolves by linear search, so a duplicated name silently
    /// shadows the second entry and the equality test above would still pass.
    #[test]
    fn every_implemented_name_appears_once() {
        let mut names: Vec<&str> = IMPLEMENTED.iter().map(|(name, _)| *name).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(before, names.len());
    }

    /// WHY: an unknown name must be reported by `xtask`, not swallowed here.
    #[test]
    fn an_unknown_name_is_not_dispatched() {
        assert!(!dispatch("dep-drift", &["xtask".to_string()]));
    }
}
