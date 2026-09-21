//! Single-program graph adaptation contracts.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, ValueLifetime,
};

/// WHY: `BufferDecl::output` is read-write for backend allocation but remains a caller-visible
/// output, while an ordinary read-write buffer carries retained state into the next invocation.
#[test]
fn output_marker_takes_precedence_over_read_write_access_for_graph_lifetime() {
    let program = Program::from_raw_parts(
        vec![
            BufferDecl::read_write("state", 0, DataType::U32).with_count(1),
            BufferDecl::output("result", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::new(),
    );
    let graph = ProgramGraph::from_program("main", program).expect("program graph must validate");
    let values = graph.values();
    let state = values
        .iter()
        .find(|value| value.name == "state")
        .expect("state graph value");
    let result = values
        .iter()
        .find(|value| value.name == "result")
        .expect("result graph value");

    assert_eq!(state.contract.lifetime, ValueLifetime::Retained);
    assert_eq!(result.contract.lifetime, ValueLifetime::Output);
}

/// WHY: a read-write pipeline live-out is allocated by every backend and must
/// not become a retained host input when a Program is lifted into an artifact.
#[test]
fn pipeline_live_out_read_write_buffer_is_a_graph_output() {
    let program = Program::from_raw_parts(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(4),
            BufferDecl::read_write("intermediate", 1, DataType::U32)
                .with_count(4)
                .with_pipeline_live_out(true),
        ],
        [4, 1, 1],
        Vec::new(),
    );

    let graph = ProgramGraph::from_program("main", program).expect("program graph must validate");
    let node = &graph.nodes()[0];
    assert_eq!(
        node.inputs
            .iter()
            .map(|input| input.buffer.as_str())
            .collect::<Vec<_>>(),
        ["input"]
    );
    assert_eq!(
        node.output_ports
            .iter()
            .map(|output| output.buffer.as_str())
            .collect::<Vec<_>>(),
        ["intermediate"]
    );
    assert_eq!(
        graph
            .values()
            .iter()
            .find(|value| value.name == "intermediate")
            .expect("pipeline output graph value")
            .contract
            .lifetime,
        ValueLifetime::Output
    );
}

/// WHY: workgroup scratch is node-local storage. Projecting it as an external value makes the
/// canonical graph wire format reject otherwise runnable programs and falsely asks callers to bind it.
#[test]
fn workgroup_scratch_remains_internal_to_single_program_graph_nodes() {
    let program = Program::from_raw_parts(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(64),
            BufferDecl::workgroup("scratch", 64, DataType::U32),
            BufferDecl::output("result", 1, DataType::U32).with_count(1),
        ],
        [64, 1, 1],
        Vec::new(),
    );

    let graph = ProgramGraph::from_program("main", program).expect("program graph must validate");
    let names = graph
        .values()
        .iter()
        .map(|value| value.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(names, ["input", "result"]);
    graph
        .to_wire()
        .expect("host-visible graph boundary must encode");
}

/// WHY: target compilation resolves every lowered binding through the node ABI.
/// A single-Program graph must therefore record invocation/retained buffers as
/// inputs and caller-visible buffers as outputs, not only create unconnected values.
#[test]
fn single_program_graph_connects_every_host_visible_buffer_to_the_node_abi() {
    let program = Program::from_raw_parts(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(4),
            BufferDecl::read_write("state", 1, DataType::U32).with_count(4),
            BufferDecl::output("result", 2, DataType::U32).with_count(4),
            BufferDecl::workgroup("scratch", 4, DataType::U32),
        ],
        [4, 1, 1],
        Vec::new(),
    );

    let graph = ProgramGraph::from_program("main", program).expect("program graph must validate");
    let node = &graph.nodes()[0];
    assert_eq!(
        node.inputs
            .iter()
            .map(|input| input.buffer.as_str())
            .collect::<Vec<_>>(),
        ["input", "state"]
    );
    assert_eq!(
        node.output_ports
            .iter()
            .map(|output| output.buffer.as_str())
            .collect::<Vec<_>>(),
        ["result"]
    );
    assert_eq!(node.outputs.len(), 1);
}

/// WHY: a runtime-sized Program buffer remains dynamic in executable IR while
/// its artifact graph records the exact caller-provided resource extent.
#[test]
fn runtime_counts_specialize_graph_resources_without_rewriting_program_ir() {
    let program = Program::from_raw_parts(
        vec![
            BufferDecl::read("input", 0, DataType::U32),
            BufferDecl::output("result", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::new(),
    );
    let graph = ProgramGraph::from_program_with_runtime_counts(
        "main",
        program,
        &BTreeMap::from([("input".to_string(), 6)]),
    )
    .expect("runtime count must specialize the graph resource");

    let input = graph
        .values()
        .iter()
        .find(|value| value.name == "input")
        .expect("input graph value");
    assert_eq!(input.contract.shape, [ShapeDim::Known(6)]);
    assert_eq!(graph.nodes()[0].program.buffers()[0].count(), 0);
}

/// WHY: runtime extent evidence belongs only to an existing dynamic host
/// buffer; stale or static overrides must not silently alter graph identity.
#[test]
fn runtime_count_overrides_fail_closed_on_unknown_and_static_buffers() {
    let program = Program::from_raw_parts(
        vec![BufferDecl::read("static", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        Vec::new(),
    );
    for (name, expected) in [
        ("missing", "has no buffer `missing`"),
        (
            "static",
            "requires a host-visible declaration with count == 0",
        ),
    ] {
        let error = ProgramGraph::from_program_with_runtime_counts(
            "main",
            program.clone(),
            &BTreeMap::from([(name.to_string(), 4)]),
        )
        .expect_err("invalid runtime count override must fail");
        assert!(
            error.to_string().contains(expected),
            "unexpected runtime-count diagnostic: {error}"
        );
    }
}

/// The three declaration shapes a backend allocates rather than reads from the
/// dispatch inputs, each countless.
///
/// `BufferDecl::is_backend_allocated_output` is the single definition of the
/// set, and each fixture is checked against it below, so a shape that stops
/// being backend-allocated fails here instead of quietly leaving the sweep.
fn countless_backend_allocated_declarations() -> Vec<(&'static str, BufferDecl)> {
    vec![
        ("output", BufferDecl::output("out", 0, DataType::U32)),
        (
            "write-only",
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32),
        ),
        (
            "pipeline live-out read-write",
            BufferDecl::read_write("out", 0, DataType::U32).with_pipeline_live_out(true),
        ),
    ]
}

/// WHY: a countless backend-allocated output has no caller bytes and no static
/// count, so nothing downstream can size it. Lifting it produced a graph value
/// with a zero extent, and the failure surfaced two stages later as an
/// unresolved extent at a `GraphValueId`, which names neither the declaration
/// that is wrong nor what to write instead. Every member of the set refuses
/// here, at the one place the declaration is still in hand.
///
/// Does not catch: a shape that is sized by something other than a static
/// count, caller bytes, or a runtime override. Such a source would have to
/// teach this lift about itself.
#[test]
fn a_countless_backend_allocated_output_is_refused_and_names_the_remedy() {
    for (label, decl) in countless_backend_allocated_declarations() {
        assert!(
            decl.is_backend_allocated_output(),
            "{label} fixture is no longer a backend-allocated output"
        );
        assert_eq!(decl.count(), 0, "{label} fixture must be countless");
        let program = Program::from_raw_parts(vec![decl], [1, 1, 1], Vec::new());
        let error = ProgramGraph::from_program("main", program)
            .expect_err("a countless backend-allocated output must be refused");
        let message = error.to_string();
        assert!(
            message.contains("out"),
            "{label} refusal must name the buffer, got: {message}"
        );
        assert!(
            message.contains(".with_count(n)"),
            "{label} refusal must name the remedy, got: {message}"
        );
    }
}

/// WHY: the refusal above must fire on exactly the un-sizable case. Each source
/// of a size accepts the same declaration: a static count, a runtime override,
/// and a declared output byte range that states the buffer is empty.
#[test]
fn a_backend_allocated_output_with_any_source_of_a_size_is_lifted() {
    let counted = Program::from_raw_parts(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        Vec::new(),
    );
    let graph = ProgramGraph::from_program("main", counted).expect("a counted output must lift");
    assert_eq!(graph.values()[0].contract.shape, [ShapeDim::Known(4)]);

    let countless = Program::from_raw_parts(
        vec![BufferDecl::output("out", 0, DataType::U32)],
        [1, 1, 1],
        Vec::new(),
    );
    let graph = ProgramGraph::from_program_with_runtime_counts(
        "main",
        countless.clone(),
        &BTreeMap::from([("out".to_string(), 6)]),
    )
    .expect("a runtime count must size the output");
    assert_eq!(graph.values()[0].contract.shape, [ShapeDim::Known(6)]);

    let empty = Program::from_raw_parts(
        vec![BufferDecl::output("out", 0, DataType::U32).with_output_byte_range(0_u64..0_u64)],
        [1, 1, 1],
        Vec::new(),
    );
    let graph =
        ProgramGraph::from_program("main", empty).expect("a declared empty output must lift");
    assert_eq!(graph.values()[0].contract.shape, [ShapeDim::Known(0)]);
}
