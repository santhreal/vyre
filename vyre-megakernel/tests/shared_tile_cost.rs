//! The class closed here: a fusion group charged once per member for a tile
//! its members share.
//!
//! Fusion unions buffers by name. `merge_programs_shared` keeps one declaration
//! per name and takes the larger count, so two ops fused over one tile hold one
//! tile in the generated kernel. The selection cost model summed each member's
//! declared total, which charged that tile twice, pushed the group over the
//! device scratch budget, and ranked the tile-sharing fusion below the pair it
//! beats.
//!
//! That is the shape a fused attention group has. The score tile is written by
//! one op and read by the next without reaching memory, which is the whole
//! reason to fuse them, and the cost model has to be able to say so.
//!
//! The published `selection_cost` on the compiled artifact is what these tests
//! read, so they hold the contract a caller sees rather than the arithmetic
//! behind it.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, ProgramGraph};
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    compile, ArtifactNodeId, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget,
};

use graph_fixtures::producer_consumer_pair;

mod graph_fixtures;

/// Elements in each declared tile. A u32 tile of 8192 elements is 32 KiB, which
/// fits the 48 KiB budget below once and not twice.
const TILE_ELEMENTS: u32 = 8192;

/// Bytes one tile occupies.
const TILE_BYTES: u64 = TILE_ELEMENTS as u64 * 4;

/// A program that stages its input through a workgroup tile of the given name.
fn tiled_program(input: &str, output: &str, tile: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::workgroup(tile, TILE_ELEMENTS, DataType::U32),
        ],
        [32, 1, 1],
        vec![
            Node::store(tile, Expr::u32(0), Expr::load(input, Expr::u32(0))),
            Node::store(output, Expr::u32(0), Expr::load(tile, Expr::u32(0))),
        ],
    )
}

/// A producer and a consumer joined by one invocation-scoped value, each
/// staging through a tile of the given name.
fn tiled_pair(producer_tile: &str, consumer_tile: &str) -> ProgramGraph {
    producer_consumer_pair(
        tiled_program("input", "intermediate", producer_tile),
        tiled_program("intermediate", "output", consumer_tile),
    )
}

/// A device that holds shared memory and reports a 48 KiB workgroup budget.
fn device() -> DeviceFacts {
    DeviceFacts::new(
        BackendCapabilities {
            has_shared_memory: true,
            ..BackendCapabilities::default()
        },
        256,
    )
    .with_occupancy(0, 48 * 1024)
    .with_bandwidth_facts(1000, 1000)
}

fn plan(graph: ProgramGraph) -> vyre_megakernel::Artifact {
    let facts = ExternalFacts::new(Digest([0xA5; 32]), BTreeMap::from([("items".into(), 17)]));
    let request = CompileRequest::new(
        graph,
        facts,
        device(),
        SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()
    .expect("fixture request must validate");
    compile(&request).expect("fixture must compile")
}

#[test]
fn two_members_sharing_one_tile_are_charged_for_one() {
    let artifact = plan(tiled_pair("tile", "tile"));

    assert_eq!(
        artifact.fusion()[0].members,
        [ArtifactNodeId(0), ArtifactNodeId(1)],
        "a pair joined by an invocation value with one tile between them fuses"
    );
    let cost = artifact.selected_plan().selection_cost;
    assert_eq!(
        cost.shared_scratch_bytes, TILE_BYTES,
        "fusion unions the tile by name, so the group holds one"
    );
    assert_eq!(
        cost.occupancy_passes_peak, 1,
        "one 32 KiB tile fits the 48 KiB workgroup budget"
    );
    assert_eq!(cost.occupancy_ns, 0);
}

#[test]
fn two_members_with_differently_named_tiles_are_charged_for_both() {
    let artifact = plan(tiled_pair("score", "weights"));

    let cost = artifact.selected_plan().selection_cost;
    assert_eq!(
        cost.shared_scratch_bytes,
        2 * TILE_BYTES,
        "two names are two tiles in the generated kernel"
    );
    assert_eq!(
        cost.occupancy_passes_peak, 2,
        "64 KiB of distinct tiles exceeds the 48 KiB workgroup budget"
    );
    assert!(
        cost.occupancy_ns > 0,
        "a group over the budget moves its traffic a second time"
    );
}
