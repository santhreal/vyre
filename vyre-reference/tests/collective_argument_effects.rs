//! A subgroup collective evaluates its argument once per lane, never once per
//! lane per lane.
//!
//! WHY: a collective gathers a value from every lane in the subgroup, so a
//! lane that reaches `subgroupAdd(e)` evaluates `e` for all `width` lanes. Every
//! lane in the subgroup reaches the collective, so `e` is evaluated `width`
//! times per lane and `width * width` times per subgroup. For a pure `e` that
//! is only wasted work. For an `e` that writes memory it is a wrong answer: the
//! interpreter stores buffer bytes behind a shared handle, so an atomic in a
//! collective argument commits `width * width` times where the device commits
//! once per lane, and the oracle then certifies a buffer no device produces.
//!
//! The oracle has no defined answer here to pick instead. Which lane's write
//! lands last, and how many times each lane's write is applied, is a property
//! of how the interpreter gathers lanes rather than of the program, so this
//! returns a structured refusal and no expected output.
//!
//! What this does NOT catch: an effectful expression reaching a collective
//! through an operation the registry declares pure. The refusal reads the
//! expression tree, so a `Call` whose registered CPU reference writes through a
//! captured handle is invisible to it.
#![cfg(feature = "subgroup-ops")]

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::value::Value;

/// One subgroup exactly, so the expected per-lane count and the observed
/// per-lane-per-lane count differ by the subgroup width and nothing else.
const LANES: u32 = 32;

fn words(values: &[Value], index: usize) -> Vec<u32> {
    values[index]
        .to_bytes()
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// `sink[i] = subgroupAdd(atomicAdd(counter[0], 1))`.
///
/// The argument is the atomic itself, so the value left in `counter` is exactly
/// the number of times the interpreter evaluated that argument. Validation
/// admits one output buffer, so the gathered value lands in a read-write
/// storage buffer and the counter is the one result read back.
fn collective_over_an_atomic() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("sink", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(LANES),
            BufferDecl::output("counter", 1, DataType::U32).with_count(1),
        ],
        [LANES, 1, 1],
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::let_bind(
                "gathered",
                Expr::SubgroupReduce {
                    op: vyre_foundation::ir::SubgroupReduceOp::Add,
                    value: Box::new(Expr::atomic_add("counter", Expr::u32(0), Expr::u32(1))),
                },
            ),
            Node::store("sink", Expr::var("idx"), Expr::var("gathered")),
        ],
    )
}

/// Zeroed inputs for the one non-output buffer `collective_over_an_atomic`
/// declares.
fn atomic_program_inputs() -> Vec<Value> {
    vec![Value::from(vec![0u8; (LANES * 4) as usize])]
}

/// A pure argument keeps working, so the refusal is scoped to the effect and
/// not to collectives.
fn collective_over_a_load() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32).with_count(LANES),
            BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
        ],
        [LANES, 1, 1],
        vec![
            Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
            Node::let_bind(
                "gathered",
                Expr::SubgroupReduce {
                    op: vyre_foundation::ir::SubgroupReduceOp::Add,
                    value: Box::new(Expr::load("src", Expr::var("idx"))),
                },
            ),
            Node::store("out", Expr::var("idx"), Expr::var("gathered")),
        ],
    )
}

/// The reproduction, as the answer the interpreter used to certify.
///
/// Before the refusal this dispatch succeeded and certified a wrong buffer:
/// `counter` held 1024 rather than the 32 a device leaves, and `sink` held a
/// DIFFERENT reduction per lane (496, 1520, 2544, ...) where every lane of a
/// subgroup reads the same sum.
/// Both come from the same cause: each of the 32 lanes gathered 32 lane values
/// and every gather committed the atomic again.
#[test]
fn an_atomic_in_a_collective_argument_is_refused() {
    let error = vyre_reference::ReferenceRequest::standard(
        &collective_over_an_atomic(),
        &atomic_program_inputs(),
    )
    .outputs()
    .expect_err(
        "an atomic under a subgroup collective has no defined commit count, so the oracle must \
         refuse it instead of committing it once per lane per lane",
    );
    let message = error.to_string();
    assert!(
        message.contains("subgroup collective")
            && message.contains("writes memory")
            && message.contains("Fix:"),
        "the refusal must name the collective, the effect and the repair, got: {message}"
    );
}

/// Permissive mode absorbs out-of-bounds access and nothing else, so an
/// effectful collective is refused there too. The report type carries no
/// output value at all, so there is no path on which a hypothetical commit
/// count becomes an expected output.
#[test]
fn the_permissive_mode_issues_no_output_for_an_effectful_collective() {
    vyre_reference::ReferenceRequest::standard(
        &collective_over_an_atomic(),
        &atomic_program_inputs(),
    )
        .execute_permissive()
        .expect_err(
            "permissive mode absorbs out-of-bounds access only, so an effectful collective \
             argument stays a refusal rather than becoming a recorded anomaly",
        );
}

/// A pure collective argument still evaluates, and still to the right value.
#[test]
fn a_pure_collective_argument_still_evaluates() {
    let src: Vec<u8> = (0..LANES).flat_map(|lane| lane.to_le_bytes()).collect();
    let outputs = vyre_reference::ReferenceRequest::standard(
        &collective_over_a_load(),
        &[Value::from(src)],
    )
    .outputs()
    .expect("a load under a collective is pure and has one defined answer");
    let sum: u32 = (0..LANES).sum();
    assert_eq!(
        words(&outputs, 0),
        vec![sum; LANES as usize],
        "every lane in the one subgroup must read the same reduction of every lane's value"
    );
}
