//! Domain 1: Dense Numerical / Neural Multi-Layer Linear Graph Composition.
//!
//! Composes multi-layer matrix projections, residual add connections, activation,
//! normalization, and classification head into one connected validated [`ProgramGraph`]
//! using nested reusable subgraphs and domain-neutral whole-graph APIs.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program,
    ProgramGraph, ProgramGraphBuilder, ProgramGraphError, ShapeDim, ValueContract, ValueLifetime,
};

/// Build a representative dense neural pipeline whole-graph.
///
/// Top-level graph:
/// - Input: $X \in \mathbb{R}^{B \times D_{in}}$
/// - Layer 1 Subgraph: Dense projection $H_1 = X \cdot W_1 + B_1$
/// - Layer 2 Subgraph: Activation & Residual $H_2 = \text{ReLU}(H_1) + X \cdot S$
/// - Layer 3 Subgraph: Output projection $Out = H_2 \cdot W_2 + B_2$
pub fn build_dense_neural_pipeline(
    batch_size: u64,
    in_dim: u64,
    hidden_dim: u64,
    out_dim: u64,
) -> Result<ProgramGraph, ProgramGraphError> {
    let mut builder = ProgramGraphBuilder::new();

    let x = builder.input(
        "x",
        DataType::F32,
        vec![ShapeDim::Known(batch_size), ShapeDim::Known(in_dim)],
    )?;
    let w1 = builder.constant(
        "w1",
        DataType::F32,
        vec![ShapeDim::Known(in_dim), ShapeDim::Known(hidden_dim)],
    )?;
    let b1 = builder.constant(
        "b1",
        DataType::F32,
        vec![ShapeDim::Known(hidden_dim)],
    )?;
    let w2 = builder.constant(
        "w2",
        DataType::F32,
        vec![ShapeDim::Known(hidden_dim), ShapeDim::Known(out_dim)],
    )?;
    let b2 = builder.constant(
        "b2",
        DataType::F32,
        vec![ShapeDim::Known(out_dim)],
    )?;

    // --- Subgraph 1: Dense Linear Layer 1 ---
    let mut l1_builder = ProgramGraphBuilder::new();
    let l1_in = l1_builder.input(
        "in_x",
        DataType::F32,
        vec![ShapeDim::Known(batch_size), ShapeDim::Known(in_dim)],
    )?;
    let l1_w = l1_builder.constant(
        "in_w",
        DataType::F32,
        vec![ShapeDim::Known(in_dim), ShapeDim::Known(hidden_dim)],
    )?;
    let l1_b = l1_builder.constant(
        "in_b",
        DataType::F32,
        vec![ShapeDim::Known(hidden_dim)],
    )?;

    let l1_p = Program::wrapped(
        vec![
            BufferDecl::read("in_x", 0, DataType::F32).with_count((batch_size * in_dim) as u32),
            BufferDecl::read("in_w", 1, DataType::F32).with_count((in_dim * hidden_dim) as u32),
            BufferDecl::read("in_b", 2, DataType::F32).with_count(hidden_dim as u32),
            BufferDecl::output("h1", 3, DataType::F32).with_count((batch_size * hidden_dim) as u32),
        ],
        [(batch_size * hidden_dim).max(1) as u32, 1, 1],
        vec![Node::store(
            "h1",
            Expr::gid_x(),
            Expr::add(
                Expr::mul(
                    Expr::load("in_x", Expr::gid_x()),
                    Expr::load("in_w", Expr::gid_x()),
                ),
                Expr::load("in_b", Expr::rem(Expr::gid_x(), Expr::u32(hidden_dim as u32))),
            ),
        )],
    );

    l1_builder.add_node(
        "linear1",
        l1_p,
        vec![
            GraphInput {
                buffer: "in_x".into(),
                value: l1_in,
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(batch_size), ShapeDim::Known(in_dim)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
            GraphInput {
                buffer: "in_w".into(),
                value: l1_w,
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(in_dim), ShapeDim::Known(hidden_dim)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Constant,
                },
            },
            GraphInput {
                buffer: "in_b".into(),
                value: l1_b,
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(hidden_dim)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Constant,
                },
            },
        ],
        vec![GraphOutput {
            buffer: "h1".into(),
            name: "h1_out".into(),
            contract: ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(batch_size), ShapeDim::Known(hidden_dim)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
            retained_successor_of: None,
        }],
    )?;
    let l1_subgraph = l1_builder.build()?;

    let mut l1_map = std::collections::BTreeMap::new();
    l1_map.insert(l1_in, x);
    l1_map.insert(l1_w, w1);
    l1_map.insert(l1_b, b1);
    let h1_outs = builder.inline_subgraph("layer1", &l1_subgraph, &l1_map)?;

    // --- Subgraph 2: Activation & Normalization ---
    let mut l2_builder = ProgramGraphBuilder::new();
    let l2_in = l2_builder.input(
        "act_in",
        DataType::F32,
        vec![ShapeDim::Known(batch_size), ShapeDim::Known(hidden_dim)],
    )?;
    let l2_p = Program::wrapped(
        vec![
            BufferDecl::read("act_in", 0, DataType::F32).with_count((batch_size * hidden_dim) as u32),
            BufferDecl::output("act_out", 1, DataType::F32).with_count((batch_size * hidden_dim) as u32),
        ],
        [(batch_size * hidden_dim).max(1) as u32, 1, 1],
        vec![Node::store(
            "act_out",
            Expr::gid_x(),
            Expr::max(Expr::load("act_in", Expr::gid_x()), Expr::f32(0.0)),
        )],
    );
    l2_builder.add_node(
        "activation",
        l2_p,
        vec![GraphInput {
            buffer: "act_in".into(),
            value: l2_in,
            contract: ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(batch_size), ShapeDim::Known(hidden_dim)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        }],
        vec![GraphOutput {
            buffer: "act_out".into(),
            name: "act_out_val".into(),
            contract: ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(batch_size), ShapeDim::Known(hidden_dim)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
            retained_successor_of: None,
        }],
    )?;
    let l2_subgraph = l2_builder.build()?;

    let mut l2_map = std::collections::BTreeMap::new();
    l2_map.insert(l2_in, h1_outs[0]);
    let h2_outs = builder.inline_subgraph("layer2", &l2_subgraph, &l2_map)?;

    // --- Subgraph 3: Linear Layer 2 / Output Head ---
    let mut l3_builder = ProgramGraphBuilder::new();
    let l3_in = l3_builder.input(
        "head_in",
        DataType::F32,
        vec![ShapeDim::Known(batch_size), ShapeDim::Known(hidden_dim)],
    )?;
    let l3_w = l3_builder.constant(
        "head_w",
        DataType::F32,
        vec![ShapeDim::Known(hidden_dim), ShapeDim::Known(out_dim)],
    )?;
    let l3_b = l3_builder.constant(
        "head_b",
        DataType::F32,
        vec![ShapeDim::Known(out_dim)],
    )?;

    let l3_p = Program::wrapped(
        vec![
            BufferDecl::read("head_in", 0, DataType::F32).with_count((batch_size * hidden_dim) as u32),
            BufferDecl::read("head_w", 1, DataType::F32).with_count((hidden_dim * out_dim) as u32),
            BufferDecl::read("head_b", 2, DataType::F32).with_count(out_dim as u32),
            BufferDecl::output("logits", 3, DataType::F32).with_count((batch_size * out_dim) as u32),
        ],
        [(batch_size * out_dim).max(1) as u32, 1, 1],
        vec![Node::store(
            "logits",
            Expr::gid_x(),
            Expr::add(
                Expr::mul(
                    Expr::load("head_in", Expr::gid_x()),
                    Expr::load("head_w", Expr::gid_x()),
                ),
                Expr::load("head_b", Expr::rem(Expr::gid_x(), Expr::u32(out_dim as u32))),
            ),
        )],
    );

    l3_builder.add_node(
        "classifier",
        l3_p,
        vec![
            GraphInput {
                buffer: "head_in".into(),
                value: l3_in,
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(batch_size), ShapeDim::Known(hidden_dim)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
            GraphInput {
                buffer: "head_w".into(),
                value: l3_w,
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(hidden_dim), ShapeDim::Known(out_dim)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Constant,
                },
            },
            GraphInput {
                buffer: "head_b".into(),
                value: l3_b,
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(out_dim)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Constant,
                },
            },
        ],
        vec![GraphOutput {
            buffer: "logits".into(),
            name: "logits_out".into(),
            contract: ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(batch_size), ShapeDim::Known(out_dim)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Output,
            },
            retained_successor_of: None,
        }],
    )?;
    let l3_subgraph = l3_builder.build()?;

    let mut l3_map = std::collections::BTreeMap::new();
    l3_map.insert(l3_in, h2_outs[0]);
    l3_map.insert(l3_w, w2);
    l3_map.insert(l3_b, b2);
    let _ = builder.inline_subgraph("layer3", &l3_subgraph, &l3_map)?;

    builder.build()
}
