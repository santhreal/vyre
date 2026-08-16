//! The class closed here: a published selection cost whose launch term ignored
//! the device it was computed for, and a cost model document that described a
//! weighting the code had stopped using.
//!
//! `DeviceFacts` carries a measured `per_launch_overhead_ns`. `execution_mode`
//! already trades it against persistent setup, and the selection cost did not
//! read it at all: every launch was priced at a recorded floor. A host whose
//! launches cost five times that floor had its fusions ranked as though they
//! saved the floor, so it fused too little, and the ranking a caller reads on
//! the artifact said so with a number no clock on that host produced.
//!
//! The document is the second half. It claimed a launch weighed 1000 and a
//! materialization weighed 100, which were the weights of an earlier model with
//! no unit. Nothing compared it to the type, so it drifted for as long as it
//! took someone to read both. The roster test below derives the fields from
//! `CostBreakdown` at run time, so a twelfth field turns the suite red until the
//! table names it.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::cost::CostBreakdown;
use vyre_megakernel::{compile, CompileRequest, DeviceFacts, Digest, ExternalFacts, SearchBudget};

/// The floor the cost model prices a launch at when the device measured none.
///
/// Recorded as `dispatch_ns` p50 4224 for `foundation.elementwise.add.1m` in
/// `vyre-bench/snapshots/59a7d71f36292424c99b7530da59f7361bfab607.json`. Held
/// here as the published contract: a caller reading `launch_ns` off an artifact
/// compiled for a device with no measurement gets this figure.
const LAUNCH_COST_FLOOR_NS: u64 = 4_224;

/// A measured overhead well clear of the floor, so a fallback cannot pass as a
/// measurement.
const MEASURED_LAUNCH_NS: u64 = 20_000;

fn invocation_contract() -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Symbol("items".into())],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::Invocation,
    }
}

fn copy_program(input: &str, output: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            output,
            Expr::u32(0),
            Expr::load(input, Expr::u32(0)),
        )],
    )
}

/// A producer and a consumer joined by one invocation-scoped value.
fn pair() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value("input", invocation_contract())
        .unwrap();
    let (_, intermediate) = graph
        .add_node(
            "producer",
            copy_program("input", "intermediate"),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "intermediate".into(),
                name: "intermediate".into(),
                contract: invocation_contract(),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
        .add_node(
            "consumer",
            copy_program("intermediate", "output"),
            vec![GraphInput {
                buffer: "intermediate".into(),
                value: intermediate[0],
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract: ValueContract {
                    lifetime: ValueLifetime::Output,
                    ..invocation_contract()
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
}

fn cost_for(device: DeviceFacts) -> CostBreakdown {
    let facts = ExternalFacts::new(Digest([0x5A; 32]), BTreeMap::from([("items".into(), 17)]));
    let request = CompileRequest::new(
        pair(),
        facts,
        device,
        SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .expect("fixture request must validate");
    compile(&request)
        .expect("fixture must compile")
        .selected_plan()
        .selection_cost
}

fn device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256).with_occupancy(0, 0)
}

#[test]
fn a_measured_launch_overhead_prices_the_launch_term() {
    let cost = cost_for(device().with_launch_costs(MEASURED_LAUNCH_NS, 0));

    assert!(cost.launches > 0, "the fixture launches at least once");
    assert_eq!(
        cost.launch_ns,
        cost.launches * MEASURED_LAUNCH_NS,
        "the launch term is the device's measured overhead per launch, not a constant"
    );
}

#[test]
fn a_device_that_measured_nothing_is_priced_at_the_recorded_floor() {
    let cost = cost_for(device());

    assert!(cost.launches > 0, "the fixture launches at least once");
    assert_eq!(
        cost.launch_ns,
        cost.launches * LAUNCH_COST_FLOOR_NS,
        "a device with no measurement falls back to the recorded cheapest dispatch"
    );
}

#[test]
fn the_total_is_the_sum_of_the_three_priced_terms() {
    let cost = cost_for(device().with_launch_costs(MEASURED_LAUNCH_NS, 0));

    assert_eq!(
        cost.total,
        cost.launch_ns + cost.materialization_ns + cost.occupancy_ns,
        "semantic_work is evidence and is excluded, and there is no fourth term"
    );
}

/// Every field name `CostBreakdown` carries, read off the derived `Debug` at run
/// time rather than listed, so a field added to the type appears here without an
/// edit.
fn cost_breakdown_fields() -> Vec<String> {
    let rendered = format!("{:?}", CostBreakdown::default());
    let body = rendered
        .split_once('{')
        .expect("derived Debug renders a braced struct")
        .1;
    body.trim_end_matches([' ', '}'])
        .split(',')
        .filter_map(|entry| entry.split_once(':'))
        .map(|(name, _)| name.trim().to_string())
        .collect()
}

/// The field names the architecture page tabulates, in order.
///
/// Scoped to the cost-model section, because the page carries a second
/// backticked table for the fusion rejection codes.
fn documented_fields(page: &str) -> Vec<String> {
    let after = page
        .split_once("## The cost model is open")
        .expect("the architecture page must carry a cost model section")
        .1;
    let section = match after.split_once("\n## ") {
        Some((body, _)) => body,
        None => after,
    };
    section
        .lines()
        .filter_map(|line| line.strip_prefix("| `"))
        .filter_map(|line| line.split_once('`'))
        .map(|(name, _)| name.to_string())
        .collect()
}

#[test]
fn the_architecture_page_tabulates_every_cost_field() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/architecture/compile-search.md")
        .canonicalize()
        .expect("the architecture page must exist beside the workspace manifest");
    let page = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));

    assert_eq!(
        documented_fields(&page),
        cost_breakdown_fields(),
        "docs/architecture/compile-search.md must tabulate every CostBreakdown field in \
         declaration order. Fix: add the row, or correct the one that no longer names a field."
    );
}
