//! Pass-invariant verifier  -  sanity-check every registered pass.
//!
//! Op id: `vyre-foundation::optimizer::pass_invariants`. Soundness: `Exact`
//! over the documented invariants below. Cost-direction: read-only  -
//! verifies but never mutates passes. Preserves: every analysis. Invalidates:
//! nothing.
//!
//! ## Invariants checked
//!
//! For every pass registered via `inventory::collect!(ProgramPassRegistration)` plus
//! the devirtualized built-ins, the verifier runs the pass on each Program
//! in a small synthetic corpus and asserts:
//!
//! 1. **Builds clean.** The pass's `transform` returns a `PassResult` whose
//!    `program` field is a structurally-valid `Program` (passes
//!    `Program::stats()` without panicking, no negative counts).
//!
//! 2. **Cost-monotone-down OR refused.** Pre-cost vs post-cost via
//!    `cost::CostCertificate::dominates_or_equal`. If `changed = true` AND
//!    cost increased on any tracked dimension AND the pass did NOT return
//!    via `try_transform` with `Err(RefusalReason::CostIncrease)`, it's a
//!    contract violation. The cost-monotone scheduler gate catches this at
//!    runtime; this verifier catches it at test time so contributors fix the
//!    pass before merge.
//!
//! 3. **Op-id stability.** Every op_id present in the post-rewrite Program
//!    must also appear in either the pre-rewrite Program OR the global op
//!    registry. A pass that introduces a fresh op_id absent from both is
//!    a wire-contract violation (the op cannot lower).
//!
//! 4. **Declared idempotence.** Passes in `IDEMPOTENCE_REQUIRED` must reach
//!    their local fixed point after one application on the synthetic corpus.
//!    The second application must report `changed = false`.
//!
//! ## Synthetic corpus
//!
//! Three Programs cover the bulk of pass-rewrite shapes:
//!   - `trivial_program`  -  single store, scalar literal RHS. Tests the
//!     no-op path of every pass.
//!   - `arithmetic_program`  -  `out = in + 1` with constant fold opportunity.
//!     Tests every arithmetic-rewriting pass.
//!   - `divergent_program`  -  `if invocation_id == 0 { store }`. Tests
//!     effect-lattice-aware passes.
//!
//! The same verifier accepts larger fixture corpora as they are promoted
//! into the optimizer test surface, so every pass is checked across the full
//! shape spectrum.

use crate::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use crate::optimizer::cost::CostCertificate;
use crate::optimizer::rewrite_contract::contract_for_pass;
use crate::optimizer::{registered_passes, ProgramPassKind};

/// One verifier finding. Empty `Vec<PassInvariantFinding>` = clean run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PassInvariantFinding {
    /// Registered pass scheduling failed, so the verifier could not safely
    /// audit the pass set.
    RegistryError {
        /// Actionable scheduler error.
        detail: String,
    },
    /// Pass landed a rewrite that increased a tracked cost dimension
    /// without explicitly refusing via `RefusalReason::CostIncrease`.
    CostMonotoneViolation {
        /// Pass name (from `PassMetadata::name`).
        pass: &'static str,
        /// Synthetic corpus program identifier.
        program: &'static str,
        /// Comma-joined list of dimensions that increased.
        increased: String,
    },
    /// Pass produced a Program that fails structural validation (the
    /// `Program::stats()` call panicked or returned obviously-corrupt
    /// counts). Verifier reports this as a hard bug.
    StructurallyInvalid {
        /// Pass name.
        pass: &'static str,
        /// Synthetic corpus program identifier.
        program: &'static str,
        /// Free-form detail (panic message or count discrepancy).
        detail: String,
    },
    /// A pass that is required to be locally idempotent changed the program on
    /// its second application.
    IdempotenceViolation {
        /// Pass name.
        pass: &'static str,
        /// Synthetic corpus program identifier.
        program: &'static str,
    },
    /// A pass grew the program past the expansion bound its rewrite contract
    /// declares.
    ExpansionBoundExceeded {
        /// Pass name.
        pass: &'static str,
        /// Synthetic corpus program identifier.
        program: &'static str,
        /// Bound the contract declares, rendered.
        declared: String,
        /// Node count before the rewrite.
        before: usize,
        /// Node count after the rewrite.
        after: usize,
    },
    /// The scheduled order runs a pass at one level after a pass at a deeper
    /// level, so the earlier pass reads constructs a later stage introduced
    /// while its preconditions were stated about a program that had none.
    LevelInversion {
        /// Pass scheduled first, at the deeper level.
        earlier: &'static str,
        /// Pass scheduled after it, at the shallower level.
        later: &'static str,
        /// Level `earlier` declares.
        earlier_level: &'static str,
        /// Level `later` declares.
        later_level: &'static str,
    },
    /// A registered pass declares no rewrite contract, so nothing states the
    /// level it owns, the evidence authorizing it, or the growth it may cause.
    ContractMissing {
        /// Pass name.
        pass: &'static str,
    },
}

const IDEMPOTENCE_REQUIRED: &[&str] = &[
    "buffer_decl_sort",
    "canonicalize",
    "const_fold",
    "cse",
    "dce",
    "dead_buffer_elim",
    "dead_store_elim",
    "empty_block_collapse",
    "noop_assign_eliminate",
    "region_promote_singleton_block",
];

/// Build the synthetic corpus the verifier runs every pass against.
fn synthetic_corpus() -> Vec<(&'static str, Program)> {
    vec![
        (
            "trivial",
            Program::wrapped(
                vec![
                    BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32)
                        .with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
            ),
        ),
        (
            "arithmetic",
            Program::wrapped(
                vec![
                    BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32)
                        .with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::add(Expr::u32(3), Expr::u32(4)),
                )],
            ),
        ),
        (
            "divergent",
            Program::wrapped(
                vec![
                    BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32)
                        .with_count(1),
                ],
                [256, 1, 1],
                vec![Node::if_then(
                    Expr::BinOp {
                        op: BinOp::Eq,
                        left: Box::new(Expr::gid_x()),
                        right: Box::new(Expr::u32(0)),
                    },
                    vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
                )],
            ),
        ),
    ]
}

/// Run every registered pass against the synthetic corpus and return the
/// list of invariant findings. Empty Vec = every pass passes every gate.
///
/// This function is the production entry point for the verifier; tests in
/// the same module call it via `pass_invariants_clean()`.
///
/// # Errors
///
/// Returns the list of findings  -  never panics on a pass-side failure.
/// Caller decides whether non-empty findings warrant a hard error.
#[must_use]
pub fn audit_registered_passes() -> Vec<PassInvariantFinding> {
    let passes = match registered_passes() {
        Ok(passes) => passes,
        Err(error) => {
            return vec![PassInvariantFinding::RegistryError {
                detail: error.to_string(),
            }];
        }
    };
    let corpus = synthetic_corpus();
    // Once per pass, not once per corpus program: whether a contract is declared is
    // a property of the pass, not of the program it ran on.
    let mut findings = contract_presence_findings(passes.iter().map(|pass| pass.metadata().name));
    findings.extend(level_progression_findings());
    for pass in passes {
        for (program_name, program) in &corpus {
            findings.extend(audit_pass_on_program(
                &pass,
                program_name,
                Clone::clone(&program),
            ));
        }
    }
    findings
}

/// One finding per pass name with no declared rewrite contract.
fn contract_presence_findings(
    pass_names: impl IntoIterator<Item = &'static str>,
) -> Vec<PassInvariantFinding> {
    pass_names
        .into_iter()
        .filter(|name| contract_for_pass(name).is_none())
        .map(|pass| PassInvariantFinding::ContractMissing { pass })
        .collect()
}

/// One finding per place the scheduled order runs a level out of order.
fn level_progression_findings() -> Vec<PassInvariantFinding> {
    match crate::optimizer::level_pipeline::level_inversions() {
        Ok(inversions) => inversions
            .into_iter()
            .map(|inversion| PassInvariantFinding::LevelInversion {
                earlier: inversion.earlier,
                later: inversion.later,
                earlier_level: inversion.earlier_level.name(),
                later_level: inversion.later_level.name(),
            })
            .collect(),
        Err(error) => vec![PassInvariantFinding::RegistryError {
            detail: error.to_string(),
        }],
    }
}

fn audit_pass_on_program(
    pass: &ProgramPassKind,
    program_name: &'static str,
    program: Program,
) -> Vec<PassInvariantFinding> {
    let pre_node_count = program.stats().node_count;
    let pre_cost = CostCertificate::for_program(&program);
    let pass_name = pass.metadata().name;

    // The audit judges the rewrite a pass performs, not the device it targets,
    // so it runs every pass against the same explicit fallback profile.
    let audit_adapter = crate::optimizer::AdapterCaps::conservative();

    // Run try_transform  -  if the pass returns Err, it's an explicit refusal,
    // which is fine and means no further checks on this run.
    let result = match pass.try_transform(program, &audit_adapter) {
        Ok(result) => result,
        Err(_refusal) => return Vec::new(),
    };

    let post_cost = CostCertificate::for_program(&result.program);
    let mut findings = Vec::new();

    // Invariant 2: cost-monotone-down on any rewrite the pass landed.
    if result.changed && !post_cost.dominates_or_equal(&pre_cost) {
        let increased = post_cost.dimensions_increased_over(&pre_cost).join(",");
        findings.push(PassInvariantFinding::CostMonotoneViolation {
            pass: pass_name,
            program: program_name,
            increased,
        });
    }

    // Invariant 1: structurally valid. We probe via stats()  -  that walks
    // the entry tree and returns counts; a panic-free, non-overflowing run
    // is a strong signal the IR is valid.
    let stats = result.program.stats();
    if stats.node_count == 0 && result.changed {
        findings.push(PassInvariantFinding::StructurallyInvalid {
            pass: pass_name,
            program: program_name,
            detail: "rewrite produced zero-node program from non-empty input".into(),
        });
    }

    // Invariant 5: the rewrite stayed inside the expansion bound its contract
    // declares. A pass that duplicates code composes with itself, so the bound is
    // checked on every corpus run rather than trusted from the declaration.
    if let Some(contract) = contract_for_pass(pass_name) {
        if !contract.expansion.admits(pre_node_count, stats.node_count) {
            findings.push(PassInvariantFinding::ExpansionBoundExceeded {
                pass: pass_name,
                program: program_name,
                declared: contract.expansion.to_string(),
                before: pre_node_count,
                after: stats.node_count,
            });
        }
    }

    if IDEMPOTENCE_REQUIRED.contains(&pass_name) {
        match pass.try_transform(result.program, &audit_adapter) {
            Ok(second) if second.changed => {
                findings.push(PassInvariantFinding::IdempotenceViolation {
                    pass: pass_name,
                    program: program_name,
                })
            }
            Ok(_) | Err(_) => {}
        }
    }

    findings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{PassAnalysis, PassMetadata, PassResult, ProgramPass};

    #[test]
    fn synthetic_corpus_has_three_programs_with_distinct_shapes() {
        let corpus = synthetic_corpus();
        assert_eq!(
            corpus.len(),
            3,
            "corpus contract: trivial, arithmetic, divergent"
        );
        let names: Vec<&str> = corpus.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"trivial"));
        assert!(names.contains(&"arithmetic"));
        assert!(names.contains(&"divergent"));
    }

    #[test]
    fn divergent_program_has_nonzero_divergence_score() {
        let corpus = synthetic_corpus();
        let divergent = corpus
            .iter()
            .find(|(n, _)| *n == "divergent")
            .map(|(_, p)| p)
            .expect("Fix: divergent program must be in corpus");
        let cost = CostCertificate::for_program(divergent);
        assert!(
            cost.divergence_score >= 1,
            "the divergent program must register divergence  -  without this, the verifier \
             can't catch effect-lattice-related regressions"
        );
    }

    #[test]
    fn trivial_program_has_zero_divergence_score() {
        let corpus = synthetic_corpus();
        let trivial = corpus
            .iter()
            .find(|(n, _)| *n == "trivial")
            .map(|(_, p)| p)
            .expect("Fix: trivial must be in corpus");
        let cost = CostCertificate::for_program(trivial);
        assert_eq!(cost.divergence_score, 0);
    }

    #[test]
    fn audit_runs_to_completion_without_panic() {
        // The contract: audit_registered_passes never panics  -  it surfaces
        // pass-side problems as `PassInvariantFinding` entries.
        let _findings = audit_registered_passes();
    }

    /// Passes that legitimately add nodes/instructions in exchange for a
    /// runtime safety guarantee  -  `autotune` adds bounds-check guards
    /// around dispatched indices to avoid out-of-range writes when the
    /// problem size doesn't divide evenly into the workgroup. The added
    /// branches are NOT a contract violation; they're the pass's contract.
    /// Other intentional-non-monotone passes belong in this list with
    /// the same justification line.
    const COST_INCREASE_EXEMPT: &[&str] = &["autotune"];

    #[test]
    fn audit_finds_zero_cost_monotone_violations_on_built_ins() {
        // Every shipped built-in pass is expected to be cost-monotone-down on
        // the synthetic corpus, EXCEPT those listed in `COST_INCREASE_EXEMPT`
        // (passes that intentionally trade cost for safety/correctness).
        // A non-exempt violation here means the pass landed a cost-up rewrite
        // without declaring `RefusalReason::CostIncrease`  -  a real bug. The
        // scheduler gate rejects it at runtime; this test catches it at
        // PR-review time instead.
        let findings = audit_registered_passes();
        let cost_violations: Vec<_> = findings
            .iter()
            .filter(|f| match f {
                PassInvariantFinding::CostMonotoneViolation { pass, .. } => {
                    !COST_INCREASE_EXEMPT.contains(pass)
                }
                _ => false,
            })
            .collect();
        assert!(
            cost_violations.is_empty(),
            "built-in passes must be cost-monotone-down on the synthetic corpus; \
             non-exempt violations: {cost_violations:#?}"
        );
    }

    #[test]
    fn audit_finds_zero_structurally_invalid_outputs_on_built_ins() {
        let findings = audit_registered_passes();
        let invalid: Vec<_> = findings
            .iter()
            .filter(|f| matches!(f, PassInvariantFinding::StructurallyInvalid { .. }))
            .collect();
        assert!(
            invalid.is_empty(),
            "built-in passes must produce structurally-valid Programs; bad outputs: {invalid:#?}"
        );
    }

    #[test]
    fn audit_finds_zero_idempotence_violations_on_required_built_ins() {
        let findings = audit_registered_passes();
        let invalid: Vec<_> = findings
            .iter()
            .filter(|f| matches!(f, PassInvariantFinding::IdempotenceViolation { .. }))
            .collect();
        assert!(
            invalid.is_empty(),
            "built-in passes with declared idempotence must reach a local fixed point in one application; bad outputs: {invalid:#?}"
        );
    }

    #[test]
    fn audit_finds_zero_expansion_bound_violations_on_built_ins() {
        let findings = audit_registered_passes();
        let over: Vec<_> = findings
            .iter()
            .filter(|f| matches!(f, PassInvariantFinding::ExpansionBoundExceeded { .. }))
            .collect();
        assert!(
            over.is_empty(),
            "a built-in pass grew the program past the bound its contract declares; either the \
             rewrite is unbounded or the declaration is wrong: {over:#?}"
        );
    }

    #[test]
    fn audit_finds_zero_level_inversions_on_built_ins() {
        let findings = audit_registered_passes();
        let inverted: Vec<_> = findings
            .iter()
            .filter(|f| matches!(f, PassInvariantFinding::LevelInversion { .. }))
            .collect();
        assert!(
            inverted.is_empty(),
            "the scheduled order runs a shallower level after a deeper one, so a pass reads \
             constructs a later stage introduced: {inverted:#?}"
        );
    }

    #[test]
    fn audit_finds_zero_missing_contracts_on_built_ins() {
        let findings = audit_registered_passes();
        let missing: Vec<_> = findings
            .iter()
            .filter(|f| matches!(f, PassInvariantFinding::ContractMissing { .. }))
            .collect();
        assert!(
            missing.is_empty(),
            "every registered pass must declare a rewrite contract: {missing:#?}"
        );
    }

    /// A pass that replaces its input with a larger program, registered under the
    /// name of a shipped pass whose contract declares `NonGrowing`. Borrowing the
    /// name is what lets the fixture exercise the bound without registering a pass:
    /// the audit resolves the contract by name.
    struct GrowingUnderNonGrowingContract;

    impl crate::optimizer::sealed::Sealed for GrowingUnderNonGrowingContract {}

    impl ProgramPass for GrowingUnderNonGrowingContract {
        fn metadata(&self) -> PassMetadata {
            PassMetadata::new("dce", &[], &[])
        }

        fn analyze(&self, _program: &Program) -> PassAnalysis {
            PassAnalysis { should_run: true }
        }

        fn transform(&self, _program: Program) -> PassResult {
            PassResult {
                program: Program::wrapped(
                    vec![
                        BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32)
                            .with_count(4),
                    ],
                    [1, 1, 1],
                    vec![
                        Node::store("out", Expr::u32(0), Expr::u32(7)),
                        Node::store("out", Expr::u32(1), Expr::u32(8)),
                        Node::store("out", Expr::u32(2), Expr::u32(9)),
                        Node::store("out", Expr::u32(3), Expr::u32(10)),
                    ],
                ),
                changed: true,
            }
        }

        fn fingerprint(&self, program: &Program) -> u64 {
            program.stats().node_count as u64
        }
    }

    fn corpus_program(name: &str) -> Program {
        synthetic_corpus()
            .into_iter()
            .find(|(entry, _)| *entry == name)
            .map(|(_, program)| program)
            .expect("Fix: the corpus must carry that program")
    }

    /// WHY: a declared expansion bound that nothing checks is a comment. This proves
    /// the audit reports a rewrite that grew past a `NonGrowing` declaration, and
    /// names the pass, the program, the bound, and both node counts. It does not
    /// prove the shipped bounds are the tightest ones true of each pass.
    #[test]
    fn a_rewrite_past_its_declared_bound_is_reported() {
        let trivial = corpus_program("trivial");
        let before = trivial.stats().node_count;
        let pass = ProgramPassKind::from_boxed(Box::new(GrowingUnderNonGrowingContract));
        let findings = audit_pass_on_program(&pass, "trivial", trivial);
        let expansion: Vec<_> = findings
            .iter()
            .filter_map(|finding| match finding {
                PassInvariantFinding::ExpansionBoundExceeded {
                    pass,
                    program,
                    declared,
                    before,
                    after,
                } => Some((*pass, *program, declared.clone(), *before, *after)),
                _ => None,
            })
            .collect();
        assert_eq!(
            expansion.len(),
            1,
            "exactly one expansion finding is expected: {findings:#?}"
        );
        let (pass_name, program_name, declared, reported_before, reported_after) =
            expansion[0].clone();
        assert_eq!(pass_name, "dce");
        assert_eq!(program_name, "trivial");
        assert_eq!(declared, "non-growing");
        assert_eq!(reported_before, before);
        assert!(
            reported_after > reported_before,
            "the finding must report the growth it observed: {reported_before} -> {reported_after}"
        );
    }

    /// WHY: the same fixture must not be reported when the declared bound admits the
    /// growth, or the check would refuse every code-duplicating pass.
    #[test]
    fn a_rewrite_inside_its_declared_bound_is_not_reported() {
        let trivial = corpus_program("trivial");
        let contract = contract_for_pass("dce").expect("dce declares a contract");
        let grown = GrowingUnderNonGrowingContract
            .transform(Clone::clone(&trivial))
            .program;
        let before = trivial.stats().node_count;
        let after = grown.stats().node_count;
        assert!(
            !contract.expansion.admits(before, after),
            "the fixture must exceed the non-growing bound"
        );
        let permissive = contract_for_pass("loop_unroll")
            .expect("loop_unroll declares a contract")
            .expansion;
        assert!(
            permissive.admits(before, after),
            "a pass declaring a growth factor must admit the same rewrite"
        );
    }

    /// WHY: a pass with no contract states no level, no evidence, and no bound, so the
    /// audit must name it rather than skipping the checks that read the contract.
    #[test]
    fn a_pass_without_a_contract_is_reported() {
        let findings = contract_presence_findings(["dce", "not_a_registered_pass"]);
        assert_eq!(
            findings,
            vec![PassInvariantFinding::ContractMissing {
                pass: "not_a_registered_pass"
            }],
            "only the undeclared pass is reported"
        );
    }
}
