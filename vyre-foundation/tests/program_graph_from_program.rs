//! Single-program graph adaptation contracts.

use vyre_foundation::ir::{BufferDecl, DataType, Program, ProgramGraph, ValueLifetime};

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
