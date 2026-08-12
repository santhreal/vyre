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
