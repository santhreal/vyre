//! Whole-composition schedule, liveness, and allocation contracts.

#![forbid(unsafe_code)]

use proptest::prelude::*;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, GraphInput, GraphNodeId, GraphOutput, GraphValueId,
    Program, ProgramGraph, ProgramGraphError, ShapeDim, ValueContract, ValueLifetime,
};

fn contract(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::F32,
        shape: vec![ShapeDim::Symbol("tokens".into()), ShapeDim::Known(8)],
        access,
        lifetime,
    }
}

fn copy_program(input: &str, output: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::F32),
        ],
        [1, 1, 1],
        Vec::new(),
    )
}

fn linear_graph(node_count: usize) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let mut current = graph
        .add_external_value(
            "input",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("Fix: analysis fixture input must register");
    for index in 0..node_count {
        let is_last = index + 1 == node_count;
        let output_name = if is_last {
            "output".to_owned()
        } else {
            format!("temporary.{index}")
        };
        let output_lifetime = if is_last {
            ValueLifetime::Output
        } else {
            ValueLifetime::Invocation
        };
        let (_, outputs) = graph
            .add_node(
                format!("node.{index}"),
                copy_program("input", "output"),
                vec![GraphInput {
                    buffer: "input".into(),
                    value: current,
                    contract: contract(
                        BufferAccess::ReadOnly,
                        if index == 0 {
                            ValueLifetime::Invocation
                        } else {
                            ValueLifetime::Invocation
                        },
                    ),
                }],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: output_name,
                    contract: contract(BufferAccess::ReadWrite, output_lifetime),
                    retained_successor_of: None,
                }],
            )
            .expect("Fix: analysis fixture node must connect");
        current = outputs[0];
    }
    graph
}

/// Locks out allocation plans that reuse values while a consuming node still needs them.
#[test]
fn interval_coloring_reuses_only_disjoint_invocation_values() {
    let graph = linear_graph(3);
    let analysis = graph.analyze().expect("Fix: valid graph must analyze");

    assert_eq!(
        analysis.schedule,
        vec![GraphNodeId(0), GraphNodeId(1), GraphNodeId(2)]
    );
    assert_eq!(analysis.reusable_slot_count, 2);
    assert_eq!(analysis.allocations[0].reusable_slot, Some(0));
    assert_eq!(analysis.allocations[1].reusable_slot, Some(1));
    assert_eq!(analysis.allocations[2].reusable_slot, Some(0));
    assert_eq!(analysis.allocations[3].reusable_slot, None);
    assert_eq!(
        (
            analysis.allocations[0].interval.start,
            analysis.allocations[0].interval.end
        ),
        (0, 0)
    );
    assert_eq!(
        (
            analysis.allocations[1].interval.start,
            analysis.allocations[1].interval.end
        ),
        (0, 1)
    );
    assert_eq!(
        (
            analysis.allocations[2].interval.start,
            analysis.allocations[2].interval.end
        ),
        (1, 2)
    );
    assert_eq!(
        (
            analysis.allocations[3].interval.start,
            analysis.allocations[3].interval.end
        ),
        (2, 2)
    );
}

/// Prevents immutable weights, sequence state, and caller-visible outputs from entering scratch reuse.
#[test]
fn non_invocation_lifetimes_always_receive_dedicated_storage() {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_values(vec![
            (
                "weight".into(),
                contract(BufferAccess::ReadOnly, ValueLifetime::Constant),
            ),
            (
                "state".into(),
                contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            ),
        ])
        .expect("Fix: dedicated values must register");

    let analysis = graph
        .analyze()
        .expect("Fix: external-only graph must analyze");
    assert_eq!(analysis.reusable_slot_count, 0);
    assert_eq!(analysis.allocations.len(), 2);
    assert!(analysis
        .allocations
        .iter()
        .all(|allocation| allocation.reusable_slot.is_none()));
}

/// Proves loop-carried state remains dedicated while predecessor and successor liveness meet at the update node.
#[test]
fn state_transition_analysis_preserves_explicit_generations() {
    let state_contract = contract(BufferAccess::ReadWrite, ValueLifetime::Retained);
    let mut graph = ProgramGraph::new();
    let prior = graph
        .add_external_value("state.0", state_contract.clone())
        .expect("Fix: state predecessor must register");
    let (_, outputs) = graph
        .add_node(
            "state.update",
            Program::wrapped(
                vec![BufferDecl::storage(
                    "state",
                    0,
                    BufferAccess::ReadWrite,
                    DataType::F32,
                )],
                [1, 1, 1],
                Vec::new(),
            ),
            vec![GraphInput {
                buffer: "state".into(),
                value: prior,
                contract: state_contract.clone(),
            }],
            vec![GraphOutput {
                buffer: "state".into(),
                name: "state.1".into(),
                contract: state_contract,
                retained_successor_of: Some(prior),
            }],
        )
        .expect("Fix: state successor must connect");

    let analysis = graph
        .analyze()
        .expect("Fix: connected state graph must analyze");
    assert_eq!(analysis.schedule, vec![GraphNodeId(0)]);
    assert_eq!(analysis.reusable_slot_count, 0);
    assert_eq!(analysis.allocations[0].value, prior);
    assert_eq!(analysis.allocations[1].value, outputs[0]);
    assert_eq!(
        (
            analysis.allocations[0].interval.start,
            analysis.allocations[0].interval.end
        ),
        (0, 0)
    );
    assert_eq!(
        (
            analysis.allocations[1].interval.start,
            analysis.allocations[1].interval.end
        ),
        (0, 0)
    );
    assert!(analysis
        .allocations
        .iter()
        .all(|allocation| allocation.reusable_slot.is_none()));
}

/// WHY: a retained-to-output transition terminates loop-carried state into a caller-visible
/// Program result buffer. Analysis must validate the transition and assign dedicated storage.
#[test]
fn caller_output_retained_transition_analysis_succeeds() {
    let retained = contract(BufferAccess::ReadWrite, ValueLifetime::Retained);
    let mut graph = ProgramGraph::new();
    let prior = graph
        .add_external_value("cache.0", retained.clone())
        .expect("Fix: initial retained state must register");
    let program = Program::wrapped(
        vec![BufferDecl::output("cache", 0, DataType::F32)],
        [1, 1, 1],
        Vec::new(),
    );
    let (_, outputs) = graph
        .add_node(
            "decode.final",
            program,
            vec![GraphInput {
                buffer: "cache".into(),
                value: prior,
                contract: retained,
            }],
            vec![GraphOutput {
                buffer: "cache".into(),
                name: "cache.output".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: Some(prior),
            }],
        )
        .expect("Fix: caller-visible output transition must connect");

    let analysis = graph
        .analyze()
        .expect("Fix: caller-output retained transition must pass graph analysis");
    assert_eq!(analysis.schedule, vec![GraphNodeId(0)]);
    assert_eq!(analysis.reusable_slot_count, 0);
    assert_eq!(analysis.allocations[0].value, prior);
    assert_eq!(analysis.allocations[1].value, outputs[0]);
    assert!(analysis
        .allocations
        .iter()
        .all(|allocation| allocation.reusable_slot.is_none()));
}

/// Ensures serialized compositions derive exactly the same schedule and memory facts after validation.
#[test]
fn wire_round_trip_preserves_complete_graph_analysis() {
    let graph = linear_graph(8);
    let expected = graph.analyze().expect("Fix: source graph must analyze");
    let bytes = graph.to_wire().expect("Fix: source graph must encode");
    let decoded = ProgramGraph::from_wire(&bytes).expect("Fix: graph wire must decode");
    assert_eq!(
        decoded.analyze().expect("Fix: decoded graph must analyze"),
        expected
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    /// Compares arbitrary linear graph schedules, liveness, and minimum slot count with a closed-form oracle.
    #[test]
    fn generated_linear_graphs_match_independent_schedule_and_liveness_oracle(node_count in 1_usize..65) {
        let graph = linear_graph(node_count);
        let analysis = graph.analyze().expect("Fix: generated valid graph must analyze");
        let expected_schedule = (0..node_count)
            .map(|index| GraphNodeId(u32::try_from(index).expect("Fix: bounded generator")))
            .collect::<Vec<_>>();
        prop_assert_eq!(&analysis.schedule, &expected_schedule);
        prop_assert_eq!(analysis.allocations.len(), node_count + 1);
        prop_assert_eq!(analysis.reusable_slot_count, if node_count == 1 { 1 } else { 2 });

        for (index, allocation) in analysis.allocations.iter().enumerate() {
            if index == node_count {
                prop_assert_eq!(allocation.reusable_slot, None);
                prop_assert_eq!((allocation.interval.start, allocation.interval.end), (node_count - 1, node_count - 1));
            } else {
                let expected_start = index.saturating_sub(1);
                let expected_end = index.min(node_count - 1);
                prop_assert_eq!((allocation.interval.start, allocation.interval.end), (expected_start, expected_end));
                prop_assert_eq!(allocation.reusable_slot, Some(u32::try_from(index % 2).expect("Fix: modulo is bounded")));
            }
        }
    }

    /// Generates missing identities to prove invalid graph extensions return the first exact counterexample without mutation.
    #[test]
    fn generated_dangling_inputs_fail_transactionally(missing in 1_u32..u32::MAX) {
        let mut graph = ProgramGraph::new();
        let before = graph.to_wire().expect("Fix: valid empty graph must encode");
        let error = graph
            .add_node(
                "invalid",
                copy_program("input", "output"),
                vec![GraphInput {
                    buffer: "input".into(),
                    value: GraphValueId(missing),
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                }],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: "unreachable".into(),
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                    retained_successor_of: None,
                }],
            )
            .expect_err("Fix: dangling generated identity must fail");
        prop_assert_eq!(error, ProgramGraphError::MissingValue(GraphValueId(missing)));
        prop_assert_eq!(graph.to_wire().expect("Fix: rejected graph must still encode"), before);
    }
}
