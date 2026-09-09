//! What the oracle owes a program that declares a whole-grid fence.
//!
//! A fence orders every invocation in the dispatch, so the interpreter advances
//! the whole grid through one inter-fence segment before any workgroup enters
//! the next. The segmentation is
//! `vyre_foundation::transform::grid_sync_split`, the same transform a backend
//! without a native cooperative launch runs, so the oracle and the device agree
//! on where the boundaries are and on what survives them.
//!
//! This crate used to carry its own splitter. It flattened only unconditional
//! wrappers and rebuilt each segment from bare nodes, so a `Let` bound before a
//! fence and read after it was rejected as an undeclared variable and a fence
//! hoistable out of a wrapper was never found. These cases pin both.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, MemoryOrdering, Node, Program,
};
use vyre_reference::reference_eval_with_grid;
use vyre_reference::value::Value;

fn u32_words(value: &Value) -> Vec<u32> {
    value
        .to_bytes()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

fn out_buffer(count: u32) -> BufferDecl {
    BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(count)
}

#[test]
fn a_binding_made_before_the_fence_is_readable_after_it() {
    // The regression this closes: the segment after the fence used to be rebuilt
    // without the `Let`, so evaluation failed with "reference to undeclared
    // variable `carried`" rather than producing a value.
    let program = Program::wrapped(
        vec![out_buffer(4)],
        [1, 1, 1],
        vec![
            Node::let_bind("carried", Expr::add(Expr::gid_x(), Expr::u32(100))),
            Node::barrier_with_ordering(MemoryOrdering::GridSync),
            Node::store("out", Expr::gid_x(), Expr::var("carried")),
        ],
    );

    let outputs = reference_eval_with_grid(
        &program,
        &[Value::from(vec![0u8; 16].as_slice())],
        [4, 1, 1],
    )
    .expect("a binding must survive the fence it was made before");

    assert_eq!(u32_words(&outputs[0]), vec![100, 101, 102, 103]);
}

#[test]
fn a_read_after_the_fence_sees_every_lanes_write() {
    let width = 8;
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("scratch", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(width),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(width),
        ],
        [1, 1, 1],
        vec![
            Node::store(
                "scratch",
                Expr::gid_x(),
                Expr::add(Expr::gid_x(), Expr::u32(1)),
            ),
            Node::barrier_with_ordering(MemoryOrdering::GridSync),
            Node::store(
                "out",
                Expr::gid_x(),
                Expr::load(
                    "scratch",
                    Expr::rem(Expr::add(Expr::gid_x(), Expr::u32(1)), Expr::u32(width)),
                ),
            ),
        ],
    );
    let zeros = Value::from(vec![0u8; width as usize * 4].as_slice());

    let outputs = reference_eval_with_grid(&program, &[zeros.clone(), zeros], [width, 1, 1])
        .expect("a fenced program must evaluate");

    // `out[i] == scratch[(i + 1) % width] == (i + 1) % width + 1`.
    let expected: Vec<u32> = (0..width).map(|lane| (lane + 1) % width + 1).collect();
    assert_eq!(
        u32_words(&outputs[1]),
        expected,
        "every lane's pre-fence store must be visible to every post-fence read"
    );
}

#[test]
fn a_fence_inside_a_wrapper_still_partitions_the_program() {
    // The fence is one block deep. A splitter that only looks at top-level
    // nodes finds none, runs the program as a single segment, and the read of a
    // neighbour's slot sees whatever that lane had written by then rather than
    // the value the whole grid agreed on at the fence.
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("scratch", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::block(vec![
            Node::store(
                "scratch",
                Expr::gid_x(),
                Expr::add(Expr::gid_x(), Expr::u32(7)),
            ),
            Node::barrier_with_ordering(MemoryOrdering::GridSync),
            Node::store(
                "out",
                Expr::gid_x(),
                Expr::load(
                    "scratch",
                    Expr::rem(Expr::add(Expr::gid_x(), Expr::u32(1)), Expr::u32(4)),
                ),
            ),
        ])],
    );
    let zeros = Value::from(vec![0u8; 16].as_slice());

    let outputs = reference_eval_with_grid(&program, &[zeros.clone(), zeros], [4, 1, 1])
        .expect("a wrapped fence must still evaluate");

    assert_eq!(
        u32_words(&outputs[1]),
        vec![8, 9, 10, 7],
        "every lane must read the value its neighbour stored before the fence"
    );
}

#[test]
fn a_workgroup_barrier_is_not_a_grid_fence() {
    // Only `GridSync` partitions. A workgroup-scoped barrier must leave the
    // program as one segment, so this pins the predicate that decides.
    let program = Program::wrapped(
        vec![out_buffer(4)],
        [1, 1, 1],
        vec![
            Node::let_bind("local", Expr::mul(Expr::gid_x(), Expr::u32(2))),
            Node::barrier_with_ordering(MemoryOrdering::SeqCst),
            Node::store("out", Expr::gid_x(), Expr::var("local")),
        ],
    );

    let outputs = reference_eval_with_grid(
        &program,
        &[Value::from(vec![0u8; 16].as_slice())],
        [4, 1, 1],
    )
    .expect("a workgroup barrier must not split the program");

    assert_eq!(u32_words(&outputs[0]), vec![0, 2, 4, 6]);
}
