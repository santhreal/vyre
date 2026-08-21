//! Contracts for hashmap interpreter async and indirect-dispatch nodes.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::{reference_eval, value::Value};

fn run(program: &Program, inputs: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>, String> {
    let values = vyre_reference::reference_inputs(program, inputs);
    reference_eval(program, &values)
        .map(|outputs| outputs.into_iter().map(|value| value.to_bytes()).collect())
        .map_err(|error| error.to_string())
}

#[test]
fn async_load_copies_only_when_waited() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("src", 0, DataType::U32).with_count(2),
            BufferDecl::output("dst", 1, DataType::U32).with_count(2),
        ],
        [1, 1, 1],
        vec![
            Node::async_load_gpu_driven("src", "dst", Expr::u32(0), Expr::u32(8), "copy"),
            Node::store("dst", Expr::u32(0), Expr::u32(0xdead_beef)),
            Node::async_wait("copy"),
        ],
    );

    let outputs = run(
        &program,
        vec![
            [11_u32, 22]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
            vec![0; 8],
        ],
    )
    .expect("async copy with wait must execute");

    let words = outputs[0]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(words, vec![11, 22], "AsyncWait must publish the transfer");
}

#[test]
fn async_transfer_without_wait_is_rejected() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("src", 0, DataType::U32).with_count(1),
            BufferDecl::output("dst", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::async_load_gpu_driven(
            "src",
            "dst",
            Expr::u32(0),
            Expr::u32(4),
            "copy",
        )],
    );

    let error = run(&program, vec![1_u32.to_le_bytes().to_vec(), vec![0; 4]])
        .expect_err("pending async transfer must not be silently dropped");
    assert!(
        (error.contains("still pending") || error.contains("in flight"))
            && error.contains("AsyncWait"),
        "missing wait error must be actionable, got: {error}"
    );
}

#[test]
fn async_load_in_multi_invocation_workgroup_executes_once_and_synchronizes() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("src", 0, DataType::U32).with_count(4),
            BufferDecl::output("dst", 1, DataType::U32).with_count(4),
        ],
        [16, 1, 1],
        vec![
            Node::async_load_gpu_driven("src", "dst", Expr::u32(0), Expr::u32(16), "copy"),
            Node::async_wait("copy"),
        ],
    );

    let src_bytes = [10_u32, 20, 30, 40]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();

    let forward = run(&program, vec![src_bytes, vec![0; 16]])
        .expect("multi-invocation workgroup async transfer must execute");
    let words = forward[0]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        words,
        vec![10, 20, 30, 40],
        "AsyncLoad in wide workgroup must publish correctly"
    );

    // Reversed step order
    let values = vec![Value::from(
        [10_u32, 20, 30, 40]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<u8>>(),
    )];
    let reversed = vyre_reference::reference_eval_lane_reversed(&program, &values)
        .expect("reversed lane stepping must execute correctly");
    let reversed_bytes: Vec<Vec<u8>> = reversed.into_iter().map(|v| v.to_bytes()).collect();
    assert_eq!(
        forward, reversed_bytes,
        "Reversed lane order must produce identical output"
    );
}

#[test]
fn indirect_dispatch_returns_structured_reference_error() {
    let program = Program::wrapped(
        vec![BufferDecl::read("counts", 0, DataType::U32).with_count(3)],
        [1, 1, 1],
        vec![Node::indirect_dispatch("counts", 0)],
    );

    let error = run(
        &program,
        vec![[1_u32, 2, 3]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect()],
    )
    .expect_err("hashmap reference cannot silently no-op indirect dispatch");
    assert!(
        error.contains("Node::IndirectDispatch") && error.contains("dynamic indirect dispatch"),
        "indirect dispatch error must name the required runtime capability, got: {error}"
    );
}
