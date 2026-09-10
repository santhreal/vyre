//! What the oracle owes a program that declares a whole-grid fence.
//!
//! A fence orders every invocation in the dispatch, so the whole grid arrives
//! before any lane goes on. The interpreter enforces that on the program as
//! submitted: a lane that reaches the fence suspends, every other workgroup
//! runs to its own fence, and only then does any lane resume.
//!
//! The interpreter used to reach that ordering by cutting the program into
//! segments with `vyre_foundation::transform::grid_sync_split`, the transform a
//! backend without a cooperative launch runs. A shared cut is a shared answer:
//! a fence that transform declines to hoist stayed inside its container and
//! degraded here to a workgroup barrier, so the oracle certified an
//! unsynchronized result for the one shape the backend also gets wrong. These
//! cases pin both what the fence orders and that the ordering survives a
//! container the cut cannot lift it out of.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, MemoryOrdering, Node, Program,
};
use vyre_reference::node;
use vyre_reference::reference_eval_with_grid;
use vyre_reference::value::Value;
use vyre_reference::ReferenceError;
use vyre_reference::workgroup::{Invocation, InvocationIds, Memory};

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

/// Advance one lane until it finishes or suspends.
///
/// A wrapped program enters through a `Region`, so a single `node::step` only
/// pushes the container frame. The bound is part of the contract: a route that
/// neither advances nor suspends is a hang, and it fails here instead of
/// stalling the run.
fn step_until_settled<'a>(
    invocation: &mut Invocation<'a>,
    memory: &mut Memory,
    program: &'a Program,
) -> Result<(), ReferenceError> {
    for _ in 0..64 {
        if invocation.done() || invocation.waiting_at_barrier {
            return Ok(());
        }
        node::step(invocation, memory, program)?;
    }
    panic!("the single-workgroup stepper neither finished nor suspended within 64 steps");
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

/// A fence the production cut cannot hoist still orders the whole grid.
///
/// `grid_sync_split` promotes a fence only out of an unconditional `Block` or
/// `Region`, because lifting one out of a `Loop` would change which
/// invocations reach it. A loop-nested fence is therefore copied into its
/// segment verbatim, and an interpreter that took its ordering from that cut
/// saw a program with no boundary at all: each workgroup ran its whole loop
/// alone, every read after the fence saw only what its own lane had written,
/// and the oracle answered with the unsynchronized result. Interpreting the
/// fence where it stands orders the grid on every iteration.
#[test]
fn a_fence_inside_a_loop_orders_the_grid_on_every_iteration() {
    const LANES: u32 = 2;
    const ITERATIONS: u32 = 2;
    let neighbour = Expr::rem(Expr::add(Expr::gid_x(), Expr::u32(1)), Expr::u32(LANES));
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("scratch", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(LANES),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(LANES * ITERATIONS),
        ],
        [1, 1, 1],
        vec![Node::loop_for(
            "k",
            Expr::u32(0),
            Expr::u32(ITERATIONS),
            vec![
                // This iteration's value, one slot per lane.
                Node::store(
                    "scratch",
                    Expr::gid_x(),
                    Expr::add(Expr::gid_x(), Expr::mul(Expr::var("k"), Expr::u32(10))),
                ),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                // Read the neighbour's value for THIS iteration, which exists
                // only if every lane wrote before any lane read.
                Node::store(
                    "out",
                    Expr::add(Expr::mul(Expr::gid_x(), Expr::u32(ITERATIONS)), Expr::var("k")),
                    Expr::load("scratch", neighbour.clone()),
                ),
                // The next iteration overwrites `scratch`, so the reads above
                // have to complete grid-wide first.
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
            ],
        )],
    );

    let outputs = reference_eval_with_grid(
        &program,
        &[
            Value::from(vec![0u8; LANES as usize * 4].as_slice()),
            Value::from(vec![0u8; (LANES * ITERATIONS) as usize * 4].as_slice()),
        ],
        [LANES, 1, 1],
    )
    .expect("a loop-nested fence must evaluate");

    // out[lane * 2 + k] == scratch[(lane + 1) % 2] == (lane + 1) % 2 + 10 * k.
    assert_eq!(
        u32_words(&outputs[1]),
        vec![1, 11, 0, 10],
        "every iteration must read the value the neighbour wrote in that same iteration"
    );
}

/// The same contract for a fence under a uniform `If`, the other container the
/// production cut leaves in place.
#[test]
fn a_fence_inside_a_uniform_branch_orders_the_grid() {
    const LANES: u32 = 2;
    let neighbour = Expr::rem(Expr::add(Expr::gid_x(), Expr::u32(1)), Expr::u32(LANES));
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("scratch", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(LANES),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(LANES),
        ],
        [1, 1, 1],
        vec![Node::if_then(
            Expr::lt(Expr::WorkgroupId { axis: 0 }, Expr::u32(LANES)),
            vec![
                Node::store(
                    "scratch",
                    Expr::gid_x(),
                    Expr::add(Expr::gid_x(), Expr::u32(7)),
                ),
                Node::barrier_with_ordering(MemoryOrdering::GridSync),
                Node::store("out", Expr::gid_x(), Expr::load("scratch", neighbour.clone())),
            ],
        )],
    );

    let outputs = reference_eval_with_grid(
        &program,
        &[
            Value::from(vec![0u8; LANES as usize * 4].as_slice()),
            Value::from(vec![0u8; LANES as usize * 4].as_slice()),
        ],
        [LANES, 1, 1],
    )
    .expect("a branch-nested fence must evaluate");

    assert_eq!(
        u32_words(&outputs[1]),
        vec![8, 7],
        "a fence under a uniform branch must order the whole grid, not one workgroup"
    );
}

/// Workgroup memory is not destroyed by a fence.
///
/// A lane holding at the fence is still resident, so the shared allocation it
/// wrote before the fence is the one it reads after. Segmenting the program
/// into separate dispatches loses that: each segment re-entered every
/// workgroup with a zeroed workgroup allocation, so a value staged in shared
/// memory before the fence read back as zero.
#[test]
fn workgroup_memory_survives_a_fence() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("shared", 0, BufferAccess::Workgroup, DataType::U32).with_count(1),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(2),
        ],
        [1, 1, 1],
        vec![
            Node::store(
                "shared",
                Expr::u32(0),
                Expr::add(Expr::gid_x(), Expr::u32(5)),
            ),
            Node::barrier_with_ordering(MemoryOrdering::GridSync),
            Node::store("out", Expr::gid_x(), Expr::load("shared", Expr::u32(0))),
        ],
    );

    let outputs = reference_eval_with_grid(
        &program,
        &[Value::from(vec![0u8; 8].as_slice())],
        [2, 1, 1],
    )
    .expect("a fence over workgroup memory must evaluate");

    assert_eq!(
        u32_words(&outputs[0]),
        vec![5, 6],
        "each workgroup must read back what it staged in shared memory before the fence"
    );
}

/// The single-workgroup statement executor refuses a fence it cannot order.
///
/// `node::step` advances one invocation and hands the barrier release back to
/// its caller, which drives one workgroup. Treating `GridSync` as a suspend
/// there releases as soon as that workgroup's lanes are waiting, so a program
/// with no cross-workgroup ordering at all reads as correct. The route refuses
/// instead and names the entry point that does order the grid.
#[test]
fn the_single_workgroup_stepper_refuses_a_grid_fence() {
    let program = Program::wrapped(
        vec![out_buffer(1)],
        [1, 1, 1],
        vec![Node::barrier_with_ordering(MemoryOrdering::GridSync)],
    );
    let mut invocation = Invocation::new(InvocationIds::ZERO, program.entry());
    let mut memory = Memory::empty();

    let error = step_until_settled(&mut invocation, &mut memory, &program)
        .expect_err("a route with no grid driver must refuse a whole-grid fence");

    assert!(
        error.to_string().contains("whole-grid fence"),
        "the refusal must name what it cannot order, got: {error}"
    );
}

/// The same route still suspends on a workgroup-scoped barrier.
#[test]
fn the_single_workgroup_stepper_suspends_on_a_workgroup_barrier() {
    let program = Program::wrapped(
        vec![out_buffer(1)],
        [1, 1, 1],
        vec![Node::barrier_with_ordering(MemoryOrdering::SeqCst)],
    );
    let mut invocation = Invocation::new(InvocationIds::ZERO, program.entry());
    let mut memory = Memory::empty();

    step_until_settled(&mut invocation, &mut memory, &program)
        .expect("a workgroup barrier must suspend rather than fail");
    assert!(
        invocation.waiting_at_barrier,
        "the lane must be waiting for the rest of its workgroup"
    );
}
