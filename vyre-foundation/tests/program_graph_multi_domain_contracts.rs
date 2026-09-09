//! Multi-domain ProgramGraph composition, validation, analysis, and wire contracts.
//!
//! BACKLOG row 48 requires a domain-neutral graph construction API that composes
//! arbitrary operations, nested subgraphs, retained state, streams, and effects
//! across unrelated domains through one production path without special cases.

#![forbid(unsafe_code)]

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ProgramGraphError, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_foundation::types::ShapeExprId;

fn contract(
    dtype: DataType,
    shape: Vec<ShapeDim>,
    access: BufferAccess,
    lifetime: ValueLifetime,
) -> ValueContract {
    ValueContract {
        dtype,
        shape,
        access,
        lifetime,
    }
}

/// Domain 1: Dense Numerical / Neural Multi-Layer Linear Graph.
/// Stage 1: Matmul projection (inputs: x, w1 -> h1)
/// Stage 2: Bias add + ReLU activation (inputs: h1, b1 -> h2)
/// Stage 3: Second projection (inputs: h2, w2 -> out)
fn build_dense_numerical_domain_graph() -> Result<ProgramGraph, ProgramGraphError> {
    let mut graph = ProgramGraph::new();

    let x = graph.add_external_value(
        "x",
        contract(
            DataType::F32,
            vec![ShapeDim::Known(4), ShapeDim::Known(16)],
            BufferAccess::ReadOnly,
            ValueLifetime::Invocation,
        ),
    )?;
    let w1 = graph.add_external_value(
        "w1",
        contract(
            DataType::F32,
            vec![ShapeDim::Known(16), ShapeDim::Known(32)],
            BufferAccess::ReadOnly,
            ValueLifetime::Constant,
        ),
    )?;
    let b1 = graph.add_external_value(
        "b1",
        contract(
            DataType::F32,
            vec![ShapeDim::Known(32)],
            BufferAccess::ReadOnly,
            ValueLifetime::Constant,
        ),
    )?;
    let w2 = graph.add_external_value(
        "w2",
        contract(
            DataType::F32,
            vec![ShapeDim::Known(32), ShapeDim::Known(8)],
            BufferAccess::ReadOnly,
            ValueLifetime::Constant,
        ),
    )?;

    // Node 1: Projection 1
    let p1 = Program::wrapped(
        vec![
            BufferDecl::read("x", 0, DataType::F32).with_count(64),
            BufferDecl::read("w1", 1, DataType::F32).with_count(512),
            BufferDecl::output("h1", 2, DataType::F32).with_count(128),
        ],
        [64, 1, 1],
        vec![Node::store(
            "h1",
            Expr::gid_x(),
            Expr::mul(
                Expr::load("x", Expr::gid_x()),
                Expr::load("w1", Expr::gid_x()),
            ),
        )],
    );
    let (_, h1_outs) = graph.add_node(
        "dense_proj1",
        p1,
        vec![
            GraphInput {
                buffer: "x".into(),
                value: x,
                contract: contract(
                    DataType::F32,
                    vec![ShapeDim::Known(4), ShapeDim::Known(16)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                ),
            },
            GraphInput {
                buffer: "w1".into(),
                value: w1,
                contract: contract(
                    DataType::F32,
                    vec![ShapeDim::Known(16), ShapeDim::Known(32)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Constant,
                ),
            },
        ],
        vec![GraphOutput {
            buffer: "h1".into(),
            name: "h1_val".into(),
            contract: contract(
                DataType::F32,
                vec![ShapeDim::Known(4), ShapeDim::Known(32)],
                BufferAccess::ReadWrite,
                ValueLifetime::Invocation,
            ),
            retained_successor_of: None,
        }],
    )?;

    // Node 2: Bias add + Activation
    let p2 = Program::wrapped(
        vec![
            BufferDecl::read("h1", 0, DataType::F32).with_count(128),
            BufferDecl::read("b1", 1, DataType::F32).with_count(32),
            BufferDecl::output("h2", 2, DataType::F32).with_count(128),
        ],
        [64, 1, 1],
        vec![Node::store(
            "h2",
            Expr::gid_x(),
            Expr::add(
                Expr::load("h1", Expr::gid_x()),
                Expr::load("b1", Expr::gid_x()),
            ),
        )],
    );
    let (_, h2_outs) = graph.add_node(
        "dense_bias_act",
        p2,
        vec![
            GraphInput {
                buffer: "h1".into(),
                value: h1_outs[0],
                contract: contract(
                    DataType::F32,
                    vec![ShapeDim::Known(4), ShapeDim::Known(32)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                ),
            },
            GraphInput {
                buffer: "b1".into(),
                value: b1,
                contract: contract(
                    DataType::F32,
                    vec![ShapeDim::Known(32)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Constant,
                ),
            },
        ],
        vec![GraphOutput {
            buffer: "h2".into(),
            name: "h2_val".into(),
            contract: contract(
                DataType::F32,
                vec![ShapeDim::Known(4), ShapeDim::Known(32)],
                BufferAccess::ReadWrite,
                ValueLifetime::Invocation,
            ),
            retained_successor_of: None,
        }],
    )?;

    // Node 3: Final Projection
    let p3 = Program::wrapped(
        vec![
            BufferDecl::read("h2", 0, DataType::F32).with_count(128),
            BufferDecl::read("w2", 1, DataType::F32).with_count(256),
            BufferDecl::output("out", 2, DataType::F32).with_count(32),
        ],
        [32, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::mul(
                Expr::load("h2", Expr::gid_x()),
                Expr::load("w2", Expr::gid_x()),
            ),
        )],
    );
    graph.add_node(
        "dense_proj2",
        p3,
        vec![
            GraphInput {
                buffer: "h2".into(),
                value: h2_outs[0],
                contract: contract(
                    DataType::F32,
                    vec![ShapeDim::Known(4), ShapeDim::Known(32)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                ),
            },
            GraphInput {
                buffer: "w2".into(),
                value: w2,
                contract: contract(
                    DataType::F32,
                    vec![ShapeDim::Known(32), ShapeDim::Known(8)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Constant,
                ),
            },
        ],
        vec![GraphOutput {
            buffer: "out".into(),
            name: "dense_out".into(),
            contract: contract(
                DataType::F32,
                vec![ShapeDim::Known(4), ShapeDim::Known(8)],
                BufferAccess::WriteOnly,
                ValueLifetime::Output,
            ),
            retained_successor_of: None,
        }],
    )?;

    Ok(graph)
}

/// Domain 2: Graph / Topological CSR Traversal with Retained State.
/// Stage 1: Frontier expansion from row offsets and col indices
/// Stage 2: Visited bitmap accumulation and retained frontier advance
fn build_graph_csr_domain_graph() -> Result<ProgramGraph, ProgramGraphError> {
    let mut graph = ProgramGraph::new();

    let row_offsets = graph.add_external_value(
        "row_offsets",
        contract(
            DataType::U32,
            vec![ShapeDim::Known(257)],
            BufferAccess::ReadOnly,
            ValueLifetime::Constant,
        ),
    )?;
    let col_indices = graph.add_external_value(
        "col_indices",
        contract(
            DataType::U32,
            vec![ShapeDim::Known(1024)],
            BufferAccess::ReadOnly,
            ValueLifetime::Constant,
        ),
    )?;
    let frontier_0 = graph.add_external_value(
        "frontier.0",
        contract(
            DataType::U32,
            vec![ShapeDim::Known(8)],
            BufferAccess::ReadWrite,
            ValueLifetime::Retained,
        ),
    )?;
    let visited_0 = graph.add_external_value(
        "visited.0",
        contract(
            DataType::U32,
            vec![ShapeDim::Known(8)],
            BufferAccess::ReadWrite,
            ValueLifetime::Retained,
        ),
    )?;

    // Node 1: Expand frontier
    let p_expand = Program::wrapped(
        vec![
            BufferDecl::read("row_offsets", 0, DataType::U32).with_count(257),
            BufferDecl::read("col_indices", 1, DataType::U32).with_count(1024),
            BufferDecl::storage("frontier", 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(8),
            BufferDecl::output("next_frontier", 3, DataType::U32).with_count(8),
        ],
        [64, 1, 1],
        vec![Node::store(
            "next_frontier",
            Expr::gid_x(),
            Expr::load("frontier", Expr::gid_x()),
        )],
    );
    let (_, expand_outs) = graph.add_node(
        "csr_expand",
        p_expand,
        vec![
            GraphInput {
                buffer: "row_offsets".into(),
                value: row_offsets,
                contract: contract(
                    DataType::U32,
                    vec![ShapeDim::Known(257)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Constant,
                ),
            },
            GraphInput {
                buffer: "col_indices".into(),
                value: col_indices,
                contract: contract(
                    DataType::U32,
                    vec![ShapeDim::Known(1024)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Constant,
                ),
            },
            GraphInput {
                buffer: "frontier".into(),
                value: frontier_0,
                contract: contract(
                    DataType::U32,
                    vec![ShapeDim::Known(8)],
                    BufferAccess::ReadWrite,
                    ValueLifetime::Retained,
                ),
            },
        ],
        vec![GraphOutput {
            buffer: "next_frontier".into(),
            name: "frontier.1".into(),
            contract: contract(
                DataType::U32,
                vec![ShapeDim::Known(8)],
                BufferAccess::ReadWrite,
                ValueLifetime::Retained,
            ),
            retained_successor_of: Some(frontier_0),
        }],
    )?;

    // Node 2: Update visited mask and produce report
    let p_update = Program::wrapped(
        vec![
            BufferDecl::storage("visited", 0, BufferAccess::ReadWrite, DataType::U32).with_count(8),
            BufferDecl::storage("next_frontier", 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(8),
            BufferDecl::output("active_count", 2, DataType::U32).with_count(1),
        ],
        [64, 1, 1],
        vec![
            Node::store(
                "visited",
                Expr::gid_x(),
                Expr::or(
                    Expr::load("visited", Expr::gid_x()),
                    Expr::load("next_frontier", Expr::gid_x()),
                ),
            ),
            Node::store("active_count", Expr::u32(0), Expr::u32(1)),
        ],
    );
    graph.add_node(
        "csr_update_visited",
        p_update,
        vec![
            GraphInput {
                buffer: "visited".into(),
                value: visited_0,
                contract: contract(
                    DataType::U32,
                    vec![ShapeDim::Known(8)],
                    BufferAccess::ReadWrite,
                    ValueLifetime::Retained,
                ),
            },
            GraphInput {
                buffer: "next_frontier".into(),
                value: expand_outs[0],
                contract: contract(
                    DataType::U32,
                    vec![ShapeDim::Known(8)],
                    BufferAccess::ReadWrite,
                    ValueLifetime::Retained,
                ),
            },
        ],
        vec![
            GraphOutput {
                buffer: "visited".into(),
                name: "visited.1".into(),
                contract: contract(
                    DataType::U32,
                    vec![ShapeDim::Known(8)],
                    BufferAccess::ReadWrite,
                    ValueLifetime::Retained,
                ),
                retained_successor_of: Some(visited_0),
            },
            GraphOutput {
                buffer: "active_count".into(),
                name: "graph_active_count".into(),
                contract: contract(
                    DataType::U32,
                    vec![ShapeDim::Known(1)],
                    BufferAccess::WriteOnly,
                    ValueLifetime::Output,
                ),
                retained_successor_of: None,
            },
        ],
    )?;

    Ok(graph)
}

/// Domain 3: Streaming Parsing / Security Tokenizer State-Machine.
/// Stage 1: Byte-stream scan against transition table
/// Stage 2: Token classification & emission
fn build_parser_streaming_domain_graph() -> Result<ProgramGraph, ProgramGraphError> {
    let mut graph = ProgramGraph::new();

    let stream_bytes = graph.add_external_value(
        "stream_bytes",
        contract(
            DataType::U8,
            vec![ShapeDim::Symbol("stream_len".into())],
            BufferAccess::ReadOnly,
            ValueLifetime::Invocation,
        ),
    )?;
    let dfa_table = graph.add_external_value(
        "dfa_table",
        contract(
            DataType::U16,
            vec![ShapeDim::Known(256), ShapeDim::Known(256)],
            BufferAccess::ReadOnly,
            ValueLifetime::Constant,
        ),
    )?;
    let lexer_state_0 = graph.add_external_value(
        "lexer_state.0",
        contract(
            DataType::U32,
            vec![ShapeDim::Known(1)],
            BufferAccess::ReadWrite,
            ValueLifetime::Retained,
        ),
    )?;

    // Node 1: DFA Scan
    let p_scan = Program::wrapped(
        vec![
            BufferDecl::read("stream_bytes", 0, DataType::U8),
            BufferDecl::read("dfa_table", 1, DataType::U16).with_count(65536),
            BufferDecl::storage("lexer_state", 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            BufferDecl::output("match_indices", 3, DataType::U32).with_count(1024),
        ],
        [64, 1, 1],
        vec![
            Node::store("match_indices", Expr::gid_x(), Expr::u32(0)),
            Node::store("lexer_state", Expr::u32(0), Expr::u32(1)),
        ],
    );
    let (_, scan_outs) = graph.add_node(
        "dfa_scan",
        p_scan,
        vec![
            GraphInput {
                buffer: "stream_bytes".into(),
                value: stream_bytes,
                contract: contract(
                    DataType::U8,
                    vec![ShapeDim::Symbol("stream_len".into())],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                ),
            },
            GraphInput {
                buffer: "dfa_table".into(),
                value: dfa_table,
                contract: contract(
                    DataType::U16,
                    vec![ShapeDim::Known(256), ShapeDim::Known(256)],
                    BufferAccess::ReadOnly,
                    ValueLifetime::Constant,
                ),
            },
            GraphInput {
                buffer: "lexer_state".into(),
                value: lexer_state_0,
                contract: contract(
                    DataType::U32,
                    vec![ShapeDim::Known(1)],
                    BufferAccess::ReadWrite,
                    ValueLifetime::Retained,
                ),
            },
        ],
        vec![
            GraphOutput {
                buffer: "lexer_state".into(),
                name: "lexer_state.1".into(),
                contract: contract(
                    DataType::U32,
                    vec![ShapeDim::Known(1)],
                    BufferAccess::ReadWrite,
                    ValueLifetime::Retained,
                ),
                retained_successor_of: Some(lexer_state_0),
            },
            GraphOutput {
                buffer: "match_indices".into(),
                name: "raw_matches".into(),
                contract: contract(
                    DataType::U32,
                    vec![ShapeDim::Known(1024)],
                    BufferAccess::ReadWrite,
                    ValueLifetime::Invocation,
                ),
                retained_successor_of: None,
            },
        ],
    )?;

    // Node 2: Token Classifier
    let p_classify = Program::wrapped(
        vec![
            BufferDecl::read("raw_matches", 0, DataType::U32).with_count(1024),
            BufferDecl::output("tokens", 1, DataType::U32).with_count(1024),
        ],
        [64, 1, 1],
        vec![Node::store(
            "tokens",
            Expr::gid_x(),
            Expr::load("raw_matches", Expr::gid_x()),
        )],
    );
    graph.add_node(
        "token_classify",
        p_classify,
        vec![GraphInput {
            buffer: "raw_matches".into(),
            value: scan_outs[1],
            contract: contract(
                DataType::U32,
                vec![ShapeDim::Known(1024)],
                BufferAccess::ReadOnly,
                ValueLifetime::Invocation,
            ),
        }],
        vec![GraphOutput {
            buffer: "tokens".into(),
            name: "emitted_tokens".into(),
            contract: contract(
                DataType::U32,
                vec![ShapeDim::Known(1024)],
                BufferAccess::WriteOnly,
                ValueLifetime::Output,
            ),
            retained_successor_of: None,
        }],
    )?;

    Ok(graph)
}

#[test]
fn three_unrelated_domains_validate_and_analyze_identically() {
    let dense_graph = build_dense_numerical_domain_graph().expect("dense graph must construct");
    let csr_graph = build_graph_csr_domain_graph().expect("csr graph must construct");
    let parser_graph = build_parser_streaming_domain_graph().expect("parser graph must construct");

    // All graphs pass the identical analysis pipeline
    let dense_analysis = dense_graph
        .analyze()
        .expect("dense graph analysis must succeed");
    let csr_analysis = csr_graph
        .analyze()
        .expect("csr graph analysis must succeed");
    let parser_analysis = parser_graph
        .analyze()
        .expect("parser graph analysis must succeed");

    assert_eq!(dense_analysis.schedule.len(), 3);
    assert_eq!(csr_analysis.schedule.len(), 2);
    assert_eq!(parser_analysis.schedule.len(), 2);

    // Liveness intervals must cover all values
    assert_eq!(dense_analysis.allocations.len(), dense_graph.values().len());
    assert_eq!(csr_analysis.allocations.len(), csr_graph.values().len());
    assert_eq!(
        parser_analysis.allocations.len(),
        parser_graph.values().len()
    );
    for (idx, alloc) in dense_analysis.allocations.iter().enumerate() {
        assert_eq!(alloc.value.0 as usize, idx);
    }
    for (idx, alloc) in csr_analysis.allocations.iter().enumerate() {
        assert_eq!(alloc.value.0 as usize, idx);
    }
    for (idx, alloc) in parser_analysis.allocations.iter().enumerate() {
        assert_eq!(alloc.value.0 as usize, idx);
    }
}

#[test]
fn multi_domain_wire_round_trip_is_lossless() {
    for (name, graph_res) in [
        ("dense", build_dense_numerical_domain_graph()),
        ("csr", build_graph_csr_domain_graph()),
        ("parser", build_parser_streaming_domain_graph()),
    ] {
        let graph = graph_res.expect("graph must build");
        let wire = graph
            .to_wire()
            .unwrap_or_else(|e| panic!("Fix: {name} must encode: {e}"));
        let decoded = ProgramGraph::from_wire(&wire)
            .unwrap_or_else(|e| panic!("Fix: {name} must decode: {e}"));
        assert_eq!(
            decoded.to_wire().unwrap(),
            wire,
            "canonical wire encoding for {name} must be idempotent"
        );
        assert_eq!(decoded.nodes().len(), graph.nodes().len());
        assert_eq!(decoded.values().len(), graph.values().len());
    }
}

#[test]
fn lifetime_and_shape_enum_exhaustive_closure() {
    // Compile-time closure over ValueLifetime
    let lifetimes = [
        ValueLifetime::Constant,
        ValueLifetime::Invocation,
        ValueLifetime::Retained,
        ValueLifetime::Output,
        ValueLifetime::Stream,
    ];
    for lt in lifetimes {
        match lt {
            ValueLifetime::Constant => assert_eq!(lt, ValueLifetime::Constant),
            ValueLifetime::Invocation => assert_eq!(lt, ValueLifetime::Invocation),
            ValueLifetime::Retained => assert_eq!(lt, ValueLifetime::Retained),
            ValueLifetime::Output => assert_eq!(lt, ValueLifetime::Output),
            ValueLifetime::Stream => assert_eq!(lt, ValueLifetime::Stream),
        }
    }

    // Compile-time closure over ShapeDim
    let dims = [
        ShapeDim::Known(42),
        ShapeDim::Unresolved,
        ShapeDim::Symbol("dim".into()),
        ShapeDim::Expr(ShapeExprId(7)),
    ];
    for dim in dims {
        match dim {
            ShapeDim::Known(k) => assert_eq!(k, 42),
            ShapeDim::Unresolved => assert_eq!(dim, ShapeDim::Unresolved),
            ShapeDim::Symbol(s) => assert_eq!(s, "dim"),
            ShapeDim::Expr(id) => assert_eq!(id, ShapeExprId(7)),
        }
    }

    // Compile-time closure over BufferAccess
    for access in BufferAccess::ALL {
        match access {
            BufferAccess::ReadOnly => assert_eq!(access, BufferAccess::ReadOnly),
            BufferAccess::WriteOnly => assert_eq!(access, BufferAccess::WriteOnly),
            BufferAccess::ReadWrite => assert_eq!(access, BufferAccess::ReadWrite),
            BufferAccess::Uniform => assert_eq!(access, BufferAccess::Uniform),
            BufferAccess::Workgroup => assert_eq!(access, BufferAccess::Workgroup),
        }
    }
}
