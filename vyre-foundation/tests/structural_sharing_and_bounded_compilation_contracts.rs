//! Contract and regression tests for structural sharing and bounded compilation (BACKLOG row 67).
//!
//! # Proofs
//! - Budget overruns across CPU steps, transformation steps, memory bytes, and code size
//!   terminate with an actionable refusal stating the exact budget ceiling in the error message,
//!   never with a silent fallback or partial artifact.
//! - Repeated subgraphs in `ProgramGraph` and `LogicalProgramGraph` achieve structural sharing
//!   with `Arc::ptr_eq` pointer reuse, bounded memory footprint, and measurable sharing ratios.

use std::collections::BTreeMap;
use std::sync::Arc;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ProgramGraphTemplate, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_foundation::optimizer::compile_budget::CompileBudget;
use vyre_foundation::optimizer::{
    registered_passes_for_profile, OptimizerError, OptimizerProfile, PassScheduler,
};

fn sample_arithmetic_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("in_a", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("in_b", 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("out", 2, BufferAccess::ReadWrite, DataType::U32),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("gid", Expr::gid_x()),
            Node::store(
                "out",
                Expr::var("gid"),
                Expr::add(
                    Expr::load("in_a", Expr::var("gid")),
                    Expr::load("in_b", Expr::var("gid")),
                ),
            ),
        ],
    )
}

fn sample_optimizable_program() -> Program {
    // A program containing redundant computation that passes will optimize
    Program::wrapped(
        vec![
            BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("dst", 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("c1", Expr::add(Expr::u32(10), Expr::u32(20))),
            Node::let_bind("c2", Expr::add(Expr::u32(10), Expr::u32(20))),
            Node::store("dst", Expr::u32(0), Expr::add(Expr::var("c1"), Expr::var("c2"))),
        ],
    )
}

#[test]
fn bounded_compilation_cpu_steps_overrun_reported_as_refusal_with_budget_in_message() {
    let program = sample_arithmetic_program();
    let passes = registered_passes_for_profile(OptimizerProfile::Release).unwrap();
    let budget = CompileBudget::unbounded().with_max_cpu_steps(0);

    let scheduler = PassScheduler::with_passes(passes).with_budget(budget);
    let result = scheduler.run(program);

    match result {
        Err(OptimizerError::BudgetExceeded { resource, budget, consumed }) => {
            assert_eq!(resource, "cpu_steps");
            assert_eq!(budget, 0);
            assert!(consumed > 0);
            let msg = format!(
                "{}",
                OptimizerError::BudgetExceeded {
                    resource,
                    budget,
                    consumed
                }
            );
            assert!(
                msg.contains("budget of 0"),
                "error message must contain the budget ceiling, got: {msg}"
            );
            assert!(
                msg.contains("cpu_steps"),
                "error message must name the resource, got: {msg}"
            );
        }
        other => panic!("expected BudgetExceeded for cpu_steps, got: {other:?}"),
    }
}

#[test]
fn bounded_compilation_transform_steps_overrun_reported_as_refusal_with_budget_in_message() {
    let program = sample_optimizable_program();
    let passes = registered_passes_for_profile(OptimizerProfile::Release).unwrap();
    let budget = CompileBudget::unbounded().with_max_transform_steps(0);

    let scheduler = PassScheduler::with_passes(passes).with_budget(budget);
    let result = scheduler.run(program);

    match result {
        Err(OptimizerError::BudgetExceeded { resource, budget, consumed }) => {
            assert_eq!(resource, "transform_steps");
            assert_eq!(budget, 0);
            assert!(consumed > 0);
            let msg = format!(
                "{}",
                OptimizerError::BudgetExceeded {
                    resource,
                    budget,
                    consumed
                }
            );
            assert!(
                msg.contains("budget of 0"),
                "error message must contain the budget ceiling, got: {msg}"
            );
            assert!(
                msg.contains("transform_steps"),
                "error message must name the resource, got: {msg}"
            );
        }
        other => panic!("expected BudgetExceeded for transform_steps, got: {other:?}"),
    }
}

#[test]
fn bounded_compilation_code_size_overrun_reported_as_refusal_with_budget_in_message() {
    let program = sample_optimizable_program();
    let passes = registered_passes_for_profile(OptimizerProfile::Release).unwrap();
    let budget = CompileBudget::unbounded().with_max_code_size(1);

    let scheduler = PassScheduler::with_passes(passes).with_budget(budget);
    let result = scheduler.run(program);

    match result {
        Err(OptimizerError::BudgetExceeded { resource, budget, consumed }) => {
            assert_eq!(resource, "code_size");
            assert_eq!(budget, 1);
            let msg = format!(
                "{}",
                OptimizerError::BudgetExceeded {
                    resource,
                    budget,
                    consumed
                }
            );
            assert!(
                msg.contains("budget of 1"),
                "error message must contain the budget ceiling, got: {msg}"
            );
        }
        other => panic!("expected BudgetExceeded for code_size, got: {other:?}"),
    }
}

#[test]
fn bounded_compilation_memory_bytes_overrun_reported_as_refusal_with_budget_in_message() {
    let program = sample_optimizable_program();
    let passes = registered_passes_for_profile(OptimizerProfile::Release).unwrap();
    let budget = CompileBudget::unbounded().with_max_memory_bytes(1);

    let scheduler = PassScheduler::with_passes(passes).with_budget(budget);
    let result = scheduler.run(program);

    match result {
        Err(OptimizerError::BudgetExceeded { resource, budget, consumed }) => {
            assert_eq!(resource, "memory_bytes");
            assert_eq!(budget, 1);
            let msg = format!(
                "{}",
                OptimizerError::BudgetExceeded {
                    resource,
                    budget,
                    consumed
                }
            );
            assert!(
                msg.contains("budget of 1"),
                "error message must contain the budget ceiling, got: {msg}"
            );
        }
        other => panic!("expected BudgetExceeded for memory_bytes, got: {other:?}"),
    }
}

#[test]
fn structural_sharing_proven_by_measurement_on_repeated_subgraph() {
    let template_program = sample_arithmetic_program();
    let template = ProgramGraphTemplate::new(
        "vector_add_block",
        template_program,
        vec!["in_a".into(), "in_b".into()],
        vec![GraphOutput {
            buffer: "out".into(),
            name: "out".into(),
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(1024)],
                access: BufferAccess::ReadWrite,
                lifetime: ValueLifetime::Invocation,
            },
            retained_successor_of: None,
        }],
    );

    let mut graph = ProgramGraph::new();
    let val_in_a = graph
        .add_external_value(
            "global_in_a",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(1024)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .unwrap();
    let val_in_b = graph
        .add_external_value(
            "global_in_b",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(1024)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .unwrap();

    let repeat_count = 12;
    let mut current_input = val_in_a;
    let mut node_ids = Vec::new();

    for i in 0..repeat_count {
        let (node_id, out_ids) = template
            .instantiate(
                &mut graph,
                format!("block_{i}"),
                vec![
                    (
                        "in_a".into(),
                        current_input,
                        ValueContract {
                            dtype: DataType::U32,
                            shape: vec![ShapeDim::Known(1024)],
                            access: BufferAccess::ReadOnly,
                            lifetime: ValueLifetime::Invocation,
                        },
                    ),
                    (
                        "in_b".into(),
                        val_in_b,
                        ValueContract {
                            dtype: DataType::U32,
                            shape: vec![ShapeDim::Known(1024)],
                            access: BufferAccess::ReadOnly,
                            lifetime: ValueLifetime::Invocation,
                        },
                    ),
                ],
                &format!("blk_{i}"),
            )
            .unwrap();
        node_ids.push(node_id);
        current_input = out_ids[0];
    }

    // 1. Measurement proof: metrics record total vs unique program bodies and sharing ratio
    let metrics = graph.structural_sharing_metrics();
    assert_eq!(metrics.total_nodes, repeat_count);
    assert_eq!(metrics.unique_program_bodies, 1);
    assert_eq!(metrics.shared_instances, repeat_count - 1);
    assert!((metrics.sharing_ratio - (repeat_count as f64)).abs() < 1e-6);
    assert!(metrics.shared_estimated_bytes < metrics.unshared_estimated_bytes);

    // 2. Physical pointer sharing proof: node AST and buffers share exact Arc allocations
    let nodes = graph.nodes();
    for i in 1..repeat_count {
        assert!(
            Arc::ptr_eq(&nodes[0].program.entry, &nodes[i].program.entry),
            "entry node AST Arc must be structurally shared"
        );
        assert!(
            Arc::ptr_eq(&nodes[0].program.buffers, &nodes[i].program.buffers),
            "buffer declaration Arc must be structurally shared"
        );
    }

    // 3. LogicalProgramGraph verification
    let bindings = BTreeMap::new();
    let logical = LogicalProgramGraph::validate(&graph, &bindings).unwrap();
    let logical_metrics = logical.structural_sharing_metrics();
    assert_eq!(logical_metrics.total_nodes, repeat_count);
    assert_eq!(logical_metrics.unique_program_bodies, 1);
    assert_eq!(logical.canonical_region_count(), 1);
}

#[test]
fn structural_sharing_canonicalization_deduplicates_independent_copies() {
    let mut graph = ProgramGraph::new();
    let v_a = graph
        .add_external_value(
            "in_a",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(512)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .unwrap();
    let v_b = graph
        .add_external_value(
            "in_b",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(512)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .unwrap();

    // Add 4 independently-constructed identical programs
    for i in 0..4 {
        let p = sample_arithmetic_program();
        graph
            .add_node(
                format!("node_{i}"),
                p,
                vec![
                    GraphInput {
                        buffer: "in_a".into(),
                        value: v_a,
                        contract: ValueContract {
                            dtype: DataType::U32,
                            shape: vec![ShapeDim::Known(512)],
                            access: BufferAccess::ReadOnly,
                            lifetime: ValueLifetime::Invocation,
                        },
                    },
                    GraphInput {
                        buffer: "in_b".into(),
                        value: v_b,
                        contract: ValueContract {
                            dtype: DataType::U32,
                            shape: vec![ShapeDim::Known(512)],
                            access: BufferAccess::ReadOnly,
                            lifetime: ValueLifetime::Invocation,
                        },
                    },
                ],
                vec![GraphOutput {
                    buffer: "out".into(),
                    name: format!("out_{i}"),
                    contract: ValueContract {
                        dtype: DataType::U32,
                        shape: vec![ShapeDim::Known(512)],
                        access: BufferAccess::ReadWrite,
                        lifetime: ValueLifetime::Invocation,
                    },
                    retained_successor_of: None,
                }],
            )
            .unwrap();
    }

    // Before canonicalization: 4 separate allocations
    graph.canonicalize_structural_sharing();

    // After canonicalization: all 4 share the same Arc pointers
    let nodes = graph.nodes();
    for i in 1..4 {
        assert!(Arc::ptr_eq(&nodes[0].program.entry, &nodes[i].program.entry));
        assert!(Arc::ptr_eq(&nodes[0].program.buffers, &nodes[i].program.buffers));
    }
}
