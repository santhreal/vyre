//! Behavioral mutation fixtures with positive controls.
//!
//! Each fixture has a valid baseline with expected output bytes. A mutation must
//! produce its specified validation diagnostic, changed output, or executed race
//! finding. Representative fixtures do not certify every optimizer branch or the
//! full device interleaving space. The caller supplies the execution oracle.

use vyre_foundation::ir::{BinOp, BufferDecl, DataType, Expr, MemoryOrdering, Node, Program};
use vyre_foundation::validate;

/// Classification of an invariant mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationKind {
    /// Off-by-one boundary mutation.
    OffByOneBound,
    /// Omission of an AST traversal arm or handler.
    MissingTraversalArm,
    /// Mutation of an algebraic identity.
    IncorrectAlgebraicLaw,
    /// Inversion of a purity or legality predicate.
    InvertedPredicate,
    /// Omission of an observable side effect.
    OmittedEffect,
    /// Exceeding allocated resource limits or invalid dispatch geometry.
    ResourceBoundaryError,
}

/// Observable evidence required to detect a mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationDetection {
    /// The validator must emit this rule identity.
    ValidationCode(&'static str),
    /// The valid mutant must execute successfully with different output bytes.
    OutputDifference,
    /// Executed memory accesses must report a race absent from the baseline.
    DataRace,
}

/// Results observed while executing and exploring one program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationObservation {
    /// Output bytes in binding order.
    pub outputs: Vec<Vec<u8>>,
    /// Number of findings from executed bounded race exploration.
    pub race_count: usize,
}

/// Execute a mutation fixture and explore its accesses on the reference oracle.
///
/// # Errors
/// Returns an error when execution or bounded race exploration fails.
#[cfg(feature = "parity-oracles")]
pub fn reference_mutation_observation(program: &Program) -> Result<MutationObservation, String> {
    use vyre_reference::{value::Value, ReferenceBudget, ReferenceRequest};
    let outputs = ReferenceRequest::new(program, &[], ReferenceBudget::bounded(100_000))
        .with_grid([1, 1, 1])
        .outputs()
        .map_err(|error| error.to_string())?;
    let request =
        ReferenceRequest::new(program, &[], ReferenceBudget::bounded(100_000)).with_grid([1, 1, 1]);
    let declared_orders = request.declared_race_exploration_orders();
    let races = request.explore_races().map_err(|error| error.to_string())?;
    if races.orders_explored != declared_orders {
        return Err(format!(
            "incomplete race exploration: {} of {declared_orders} orders",
            races.orders_explored
        ));
    }
    Ok(MutationObservation {
        outputs: outputs.iter().map(Value::to_bytes).collect(),
        race_count: races.findings.len(),
    })
}

/// One invariant mutation and its positive control.
#[derive(Debug, Clone)]
pub struct MutationDescriptor {
    /// Unique mutation identity.
    pub id: &'static str,
    /// Category of mutation.
    pub kind: MutationKind,
    /// Description of the injected fault.
    pub description: &'static str,
    /// Valid program before the fault is injected.
    pub baseline_program: Program,
    /// Expected baseline outputs in binding order.
    pub expected_outputs: Vec<Vec<u8>>,
    /// Program containing the injected fault.
    pub mutated_program: Program,
    /// Required diagnostic or execution difference.
    pub detection: MutationDetection,
}

fn scalar_program(nodes: Vec<Node>) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        nodes,
    )
}

fn store(value: u32) -> Node {
    Node::store("out", Expr::u32(0), Expr::u32(value))
}

fn semantic_mutation(
    id: &'static str,
    kind: MutationKind,
    description: &'static str,
    baseline: Vec<Node>,
    mutated: Vec<Node>,
) -> MutationDescriptor {
    MutationDescriptor {
        id,
        kind,
        description,
        baseline_program: scalar_program(baseline),
        expected_outputs: vec![7_u32.to_le_bytes().to_vec()],
        mutated_program: scalar_program(mutated),
        detection: MutationDetection::OutputDifference,
    }
}

fn neighbour_exchange(synchronized: bool) -> Program {
    let mut nodes = vec![Node::store(
        "shared",
        Expr::LocalId { axis: 0 },
        Expr::u32(1),
    )];
    if synchronized {
        nodes.push(Node::barrier_with_ordering(MemoryOrdering::SeqCst));
    }
    nodes.push(Node::store(
        "out",
        Expr::LocalId { axis: 0 },
        Expr::load(
            "shared",
            Expr::BinOp {
                op: BinOp::BitXor,
                left: Box::new(Expr::LocalId { axis: 0 }),
                right: Box::new(Expr::u32(1)),
            },
        ),
    ));
    Program::wrapped(
        vec![
            BufferDecl::workgroup("shared", 0, DataType::U32).with_count(64),
            BufferDecl::output("out", 1, DataType::U32).with_count(64),
        ],
        [64, 1, 1],
        nodes,
    )
}

/// Generate positive controls and mutations for every declared mutation class.
#[must_use]
pub fn representative_mutations() -> Vec<MutationDescriptor> {
    vec![
        MutationDescriptor {
            id: "mut_off_by_one_buffer_access",
            kind: MutationKind::OffByOneBound,
            description: "Store at the element count instead of the last valid index",
            baseline_program: Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
                [1, 1, 1],
                vec![Node::store("out", Expr::u32(3), Expr::u32(100))],
            ),
            expected_outputs: vec![[0_u32, 0, 0, 100]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect()],
            mutated_program: Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
                [1, 1, 1],
                vec![Node::store("out", Expr::u32(4), Expr::u32(100))],
            ),
            detection: MutationDetection::ValidationCode("V036"),
        },
        semantic_mutation(
            "mut_missing_nested_body",
            MutationKind::MissingTraversalArm,
            "Omit a nested conditional body during traversal",
            vec![Node::if_then_else(
                Expr::bool(true),
                vec![store(7)],
                vec![store(9)],
            )],
            vec![Node::if_then_else(Expr::bool(true), vec![], vec![store(9)])],
        ),
        semantic_mutation(
            "mut_incorrect_additive_identity",
            MutationKind::IncorrectAlgebraicLaw,
            "Replace x + 0 with 0 instead of x",
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(Expr::u32(7)),
                    right: Box::new(Expr::u32(0)),
                },
            )],
            vec![store(0)],
        ),
        semantic_mutation(
            "mut_inverted_predicate",
            MutationKind::InvertedPredicate,
            "Invert the predicate selecting an observable store",
            vec![Node::if_then_else(
                Expr::bool(true),
                vec![store(7)],
                vec![store(9)],
            )],
            vec![Node::if_then_else(
                Expr::bool(false),
                vec![store(7)],
                vec![store(9)],
            )],
        ),
        semantic_mutation(
            "mut_omitted_store_effect",
            MutationKind::OmittedEffect,
            "Omit the store that determines the output",
            vec![store(7)],
            vec![],
        ),
        MutationDescriptor {
            id: "mut_omitted_barrier_effect",
            kind: MutationKind::OmittedEffect,
            description: "Read a peer's shared-memory store without synchronization",
            baseline_program: neighbour_exchange(true),
            expected_outputs: vec![(0..64).flat_map(|_| 1_u32.to_le_bytes()).collect()],
            mutated_program: neighbour_exchange(false),
            detection: MutationDetection::DataRace,
        },
        MutationDescriptor {
            id: "mut_zero_workgroup_geometry",
            kind: MutationKind::ResourceBoundaryError,
            description: "Dispatch with zero invocations along one axis",
            baseline_program: scalar_program(vec![store(7)]),
            expected_outputs: vec![7_u32.to_le_bytes().to_vec()],
            mutated_program: Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [0, 1, 1],
                vec![store(7)],
            ),
            detection: MutationDetection::ValidationCode("V106"),
        },
    ]
}

/// Verify each positive control and detect every representative mutation.
///
/// `execute` must dispatch one workgroup without host inputs, return outputs in
/// binding order, and explore executed memory accesses. The caller must enforce
/// an execution deadline.
///
/// # Panics
/// Panics if a baseline is invalid or wrong, execution fails, a specified
/// diagnostic is absent, or a semantic mutation produces unchanged outputs.
pub fn assert_mutations_are_detected(
    mut execute: impl FnMut(&Program) -> Result<MutationObservation, String>,
) {
    for mutation in representative_mutations() {
        assert_mutation_is_detected(&mutation, &mut execute);
    }
}

fn assert_mutation_is_detected(
    mutation: &MutationDescriptor,
    execute: &mut impl FnMut(&Program) -> Result<MutationObservation, String>,
) {
    let baseline_errors = validate::validate(&mutation.baseline_program);
    assert!(
        baseline_errors.is_empty(),
        "{}: invalid positive control: {baseline_errors:?}",
        mutation.id
    );
    let baseline = execute(&mutation.baseline_program).unwrap_or_else(|error| {
        panic!(
            "{}: positive control execution failed: {error}",
            mutation.id
        )
    });
    assert_eq!(
        baseline.outputs, mutation.expected_outputs,
        "{}: wrong positive control output",
        mutation.id
    );
    assert_eq!(
        baseline.race_count, 0,
        "{}: positive control has a race",
        mutation.id
    );
    let errors = validate::validate(&mutation.mutated_program);
    match mutation.detection {
        MutationDetection::ValidationCode(code) => {
            assert!(
                errors.iter().any(|error| error.code().as_str() == code),
                "{}: expected {code}, received {errors:?}",
                mutation.id
            );
        }
        MutationDetection::OutputDifference | MutationDetection::DataRace => {
            assert!(
                errors.is_empty(),
                "{}: semantic mutation is invalid: {errors:?}",
                mutation.id
            );
            let observed = execute(&mutation.mutated_program).unwrap_or_else(|error| {
                panic!("{}: mutant execution failed: {error}", mutation.id)
            });
            if mutation.detection == MutationDetection::OutputDifference {
                assert_eq!(
                    observed.race_count, 0,
                    "{}: output mutation has a race",
                    mutation.id
                );
                assert_ne!(
                    observed.outputs, baseline.outputs,
                    "{}: mutation was not detected",
                    mutation.id
                );
            } else {
                assert!(
                    observed.race_count > 0,
                    "{}: missing synchronization was not detected",
                    mutation.id
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// WHY: a new class or duplicate fixture must not silently escape behavioral
    /// coverage. Source enumeration establishes membership, not detection.
    #[test]
    fn fixtures_cover_the_declared_mutation_classes() {
        let path = crate::monorepo::vyre_workspace_root()
            .join("vyre-test-support/src/mutation_testing.rs");
        let source = crate::read_source_file_bounded(&path).unwrap();
        let body = crate::braced_body(&source, "pub enum MutationKind {").unwrap();
        let declared = crate::top_level_variant_names(body);
        assert!(!declared.is_empty());
        let mutations = representative_mutations();
        let covered: BTreeSet<_> = mutations
            .iter()
            .map(|mutation| format!("{:?}", mutation.kind))
            .collect();
        assert_eq!(covered, declared);
        let ids: BTreeSet<_> = mutations.iter().map(|mutation| mutation.id).collect();
        assert_eq!(ids.len(), mutations.len());
        let detection_body = crate::braced_body(&source, "pub enum MutationDetection {").unwrap();
        let declared_detectors = crate::top_level_variant_names(detection_body);
        let covered_detectors: BTreeSet<_> = mutations
            .iter()
            .map(|mutation| {
                format!("{:?}", mutation.detection)
                    .split('(')
                    .next()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert_eq!(covered_detectors, declared_detectors);
    }

    #[cfg(feature = "parity-oracles")]
    mod behavioral {
        use super::*;
        use std::panic::{catch_unwind, AssertUnwindSafe};

        /// WHY: only observed validator identities or changed execution outputs
        /// establish detection. Nonempty IR, unrelated errors, and broken positive
        /// controls do not. These tests do not measure mutation coverage of passes.
        #[test]
        fn real_execution_and_validation_detect_every_registered_mutation() {
            assert_mutations_are_detected(reference_mutation_observation);
        }

        #[test]
        fn unchanged_programs_never_count_as_detected() {
            for mut mutation in representative_mutations() {
                mutation.mutated_program = mutation.baseline_program.clone();
                let result = catch_unwind(AssertUnwindSafe(|| {
                    assert_mutation_is_detected(&mutation, &mut reference_mutation_observation)
                }));
                assert!(
                    result.is_err(),
                    "{} incorrectly qualified an unchanged program",
                    mutation.id
                );
            }
        }

        #[test]
        fn wrong_positive_controls_and_execution_errors_are_not_detection() {
            for mutation in representative_mutations() {
                let mut wrong = mutation.clone();
                wrong.expected_outputs[0][0] ^= 1;
                assert!(
                    catch_unwind(AssertUnwindSafe(|| assert_mutation_is_detected(
                        &wrong,
                        &mut reference_mutation_observation
                    )))
                    .is_err()
                );
                assert!(catch_unwind(AssertUnwindSafe(|| {
                    assert_mutation_is_detected(&mutation, &mut |program| {
                        let mut observation = reference_mutation_observation(program)?;
                        observation.race_count += 1;
                        Ok(observation)
                    });
                }))
                .is_err());
                let mut calls = 0;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    assert_mutation_is_detected(&mutation, &mut |program| {
                        calls += 1;
                        if calls == 1 {
                            Err("positive control error".into())
                        } else {
                            reference_mutation_observation(program)
                        }
                    });
                }));
                assert!(result.is_err());
                assert_eq!(calls, 1, "a failed positive control must stop evaluation");
                if !matches!(mutation.detection, MutationDetection::ValidationCode(_)) {
                    let mut calls = 0;
                    assert!(catch_unwind(AssertUnwindSafe(|| {
                        assert_mutation_is_detected(&mutation, &mut |program| {
                            calls += 1;
                            if calls == 2 {
                                Err("mutant error".into())
                            } else {
                                reference_mutation_observation(program)
                            }
                        });
                    }))
                    .is_err());
                    assert_eq!(calls, 2);
                }
            }
        }

        #[test]
        fn unrelated_validation_errors_do_not_qualify() {
            let mutations = representative_mutations();
            for mutation in &mutations {
                if let MutationDetection::ValidationCode(code) = mutation.detection {
                    let other = mutations
                        .iter()
                        .find(|candidate| {
                            matches!(candidate.detection,
                        MutationDetection::ValidationCode(other_code) if other_code != code)
                        })
                        .unwrap();
                    let mut wrong = mutation.clone();
                    wrong.mutated_program = other.mutated_program.clone();
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| assert_mutation_is_detected(
                            &wrong,
                            &mut reference_mutation_observation
                        )))
                        .is_err()
                    );
                }
            }
        }

        #[test]
        fn invalid_programs_stop_before_the_executor_can_claim_success() {
            let mutations = representative_mutations();
            let invalid = &mutations
                .iter()
                .find(|mutation| mutation.kind == MutationKind::ResourceBoundaryError)
                .unwrap()
                .mutated_program;
            for mutation in &mutations {
                let mut wrong = mutation.clone();
                wrong.baseline_program = invalid.clone();
                let mut calls = 0;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    assert_mutation_is_detected(&wrong, &mut |_| {
                        calls += 1;
                        Ok(MutationObservation {
                            outputs: wrong.expected_outputs.clone(),
                            race_count: 0,
                        })
                    });
                }));
                assert!(result.is_err());
                assert_eq!(calls, 0, "{} executed an invalid baseline", mutation.id);
                if !matches!(mutation.detection, MutationDetection::ValidationCode(_)) {
                    let mut wrong = mutation.clone();
                    wrong.mutated_program = invalid.clone();
                    let mut calls = 0;
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        assert_mutation_is_detected(&wrong, &mut |program| {
                            calls += 1;
                            reference_mutation_observation(program)
                        });
                    }));
                    assert!(result.is_err());
                    assert_eq!(
                        calls, 1,
                        "{} executed an invalid semantic mutant",
                        mutation.id
                    );
                }
            }
        }

        #[test]
        fn output_drift_with_a_race_is_not_a_semantic_proof() {
            for mutation in representative_mutations()
                .into_iter()
                .filter(|mutation| mutation.detection == MutationDetection::OutputDifference)
            {
                let mut calls = 0;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    assert_mutation_is_detected(&mutation, &mut |program| {
                        calls += 1;
                        let mut observed = reference_mutation_observation(program)?;
                        if calls == 2 {
                            observed.race_count = 1;
                        }
                        Ok(observed)
                    });
                }));
                assert!(result.is_err());
                assert_eq!(calls, 2);
            }
        }
    }
}
