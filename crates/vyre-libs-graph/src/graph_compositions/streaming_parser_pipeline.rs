//! Domain 3: Parsing / Streaming / Security Dataflow Composition.
//!
//! Composes streaming packet channels, sliding-window CRC checksums, token scanning,
//! and external diagnostic effect barriers into one connected validated [`ProgramGraph`].

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, ExternalEffect, GraphInput, GraphOutput, Node,
    Program, ProgramGraph, ProgramGraphBuilder, ProgramGraphError, ShapeDim, ValueContract,
    ValueLifetime,
};

/// Build a representative streaming packet / log parser whole-graph.
///
/// Features exercised:
/// - Streaming dataflow values (`ValueLifetime::Stream`)
/// - Sliding-window checksum & token classification subgraphs
/// - External effect barriers (`ExternalEffect::StorageBarrier`, `TraceMarker`)
pub fn build_streaming_parser_pipeline(chunk_size: u64) -> Result<ProgramGraph, ProgramGraphError> {
    let mut builder = ProgramGraphBuilder::new();

    let packet_stream = builder.stream(
        "packet_stream",
        DataType::U8,
        vec![ShapeDim::Known(chunk_size)],
    )?;

    // Subgraph 1: Windowed Checksum / CRC stage
    let mut crc_builder = ProgramGraphBuilder::new();
    let crc_in =
        crc_builder.stream("raw_bytes", DataType::U8, vec![ShapeDim::Known(chunk_size)])?;

    let crc_p = Program::wrapped(
        vec![
            BufferDecl::read("raw_bytes", 0, DataType::U8).with_count(chunk_size as u32),
            BufferDecl::output("crc_hashes", 1, DataType::U32).with_count(chunk_size as u32),
        ],
        [chunk_size.max(1) as u32, 1, 1],
        vec![Node::store(
            "crc_hashes",
            Expr::gid_x(),
            Expr::bitxor(
                Expr::cast(DataType::U32, Expr::load("raw_bytes", Expr::gid_x())),
                Expr::u32(0xEDB88320),
            ),
        )],
    );

    crc_builder.add_node(
        "sliding_crc",
        crc_p,
        vec![GraphInput {
            buffer: "raw_bytes".into(),
            value: crc_in,
            contract: ValueContract {
                dtype: DataType::U8,
                shape: vec![ShapeDim::Known(chunk_size)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Stream,
            },
        }],
        vec![GraphOutput {
            buffer: "crc_hashes".into(),
            name: "crc_hashes_out".into(),
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(chunk_size)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
            retained_successor_of: None,
        }],
    )?;
    let crc_subgraph = crc_builder.build()?;

    let mut crc_map = std::collections::BTreeMap::new();
    crc_map.insert(crc_in, packet_stream);
    let crc_outs = builder.inline_subgraph("crc_stage", &crc_subgraph, &crc_map)?;

    // Subgraph 2: Token classification & Delimiter matching
    let mut tok_builder = ProgramGraphBuilder::new();
    let tok_in = tok_builder.input(
        "hashes_in",
        DataType::U32,
        vec![ShapeDim::Known(chunk_size)],
    )?;

    let tok_p = Program::wrapped(
        vec![
            BufferDecl::read("hashes_in", 0, DataType::U32).with_count(chunk_size as u32),
            BufferDecl::output("token_ids", 1, DataType::U32).with_count(chunk_size as u32),
        ],
        [chunk_size.max(1) as u32, 1, 1],
        vec![Node::store(
            "token_ids",
            Expr::gid_x(),
            Expr::rem(Expr::load("hashes_in", Expr::gid_x()), Expr::u32(128)),
        )],
    );

    tok_builder.add_node(
        "token_scanner",
        tok_p,
        vec![GraphInput {
            buffer: "hashes_in".into(),
            value: tok_in,
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(chunk_size)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        }],
        vec![GraphOutput {
            buffer: "token_ids".into(),
            name: "token_ids_out".into(),
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(chunk_size)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
            retained_successor_of: None,
        }],
    )?;
    let tok_subgraph = tok_builder.build()?;

    let mut tok_map = std::collections::BTreeMap::new();
    tok_map.insert(tok_in, crc_outs[0]);
    let tok_outs = builder.inline_subgraph("tok_stage", &tok_subgraph, &tok_map)?;

    // Stage 3: Effect Barrier + Output Diagnostic aggregation
    let (_, effect_outs) = builder.add_effect_barrier(
        "storage_sync_barrier",
        ExternalEffect::StorageBarrier,
        vec![GraphInput {
            buffer: "in_tokens".into(),
            value: tok_outs[0],
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(chunk_size)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        }],
        vec![GraphOutput {
            buffer: "synced_tokens".into(),
            name: "synced_tokens_val".into(),
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(chunk_size)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
            retained_successor_of: None,
        }],
    )?;

    // Final output stage
    let out_p = Program::wrapped(
        vec![
            BufferDecl::read("synced_tokens", 0, DataType::U32).with_count(chunk_size as u32),
            BufferDecl::output("parsed_events", 1, DataType::U32).with_count(chunk_size as u32),
        ],
        [chunk_size.max(1) as u32, 1, 1],
        vec![Node::store(
            "parsed_events",
            Expr::gid_x(),
            Expr::load("synced_tokens", Expr::gid_x()),
        )],
    );

    builder.add_node(
        "output_filter",
        out_p,
        vec![GraphInput {
            buffer: "synced_tokens".into(),
            value: effect_outs[0],
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(chunk_size)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        }],
        vec![GraphOutput {
            buffer: "parsed_events".into(),
            name: "final_parsed_events".into(),
            contract: ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(chunk_size)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Output,
            },
            retained_successor_of: None,
        }],
    )?;

    builder.build()
}
