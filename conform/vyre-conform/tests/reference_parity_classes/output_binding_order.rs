//! WHY: closes the class "the artifact route returns a Program's writable buffers
//! in an order the Program did not declare", which took `(cuda,
//! vyre-libs::decode::base64)`, `(wgpu, decode::base64)`, `(cuda,
//! decode::inflate_stored_block)`, `(cuda, matching::emit_hit)`, `(wgpu,
//! matching::emit_hit)`, `(cuda, nn::ln_scale_backward)` and `(wgpu, same)` out of
//! the conformance certificate. Canonical ABI slot order numbers graph values, and
//! a graph lifted from one Program mints an external value for every retained
//! read-write buffer before the node that produces the declared outputs, so slot
//! order is retained-then-output. A Program that declares an output buffer before a
//! retained one is therefore read transposed, which reported as a length mismatch
//! when the two buffers differ in size and as an f32 value disagreement when they
//! do not.
//!
//! The roster is the operation registry: every op the registry carries a builder
//! for is lifted through the real graph boundary and its declared writable buffers
//! are compared against the canonical values that boundary produced. A new op whose
//! declaration order the graph cannot express joins these assertions without anyone
//! editing a list.
//!
//! What it does not catch: whether a materializer projects the value it was asked
//! for. This judges the identity and the order of the writable set at the boundary
//! where the order is decided; `prove` judges the bytes, and the seven pairs above
//! are its record of this class going red.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, ProgramGraph};
use vyre_registry_link::operation::live_operation_registry;

/// Writable buffer names in Program declaration order.
///
/// This is the order `Program::output_buffer_indices` reports and the order a
/// caller that authored the Program binds and reads.
fn declared_writable_names(program: &Program) -> Vec<&str> {
    let buffers = program.buffers();
    program
        .output_buffer_indices()
        .iter()
        .map(|index| {
            buffers
                .get(*index as usize)
                .expect("Fix: a declared writable buffer index must address a declared buffer.")
                .name()
        })
        .collect()
}

/// Writable canonical value names in graph value order, which is ABI slot order.
fn canonical_writable_names(graph: &ProgramGraph) -> Vec<&str> {
    let mut writable = graph
        .values()
        .iter()
        .filter(|value| {
            matches!(
                value.contract.access,
                BufferAccess::ReadWrite | BufferAccess::WriteOnly
            )
        })
        .collect::<Vec<_>>();
    writable.sort_unstable_by_key(|value| value.id.0);
    writable
        .into_iter()
        .map(|value| value.name.as_str())
        .collect()
}

#[test]
fn every_registered_op_has_one_canonical_value_per_declared_writable_buffer() {
    let mut missing = Vec::new();
    let mut judged = 0usize;
    for operation in live_operation_registry().iter() {
        let Some(program) = operation.program() else {
            continue;
        };
        let Ok(graph) = ProgramGraph::from_program("main", program.clone()) else {
            continue;
        };
        judged += 1;
        let declared = declared_writable_names(&program);
        let canonical = canonical_writable_names(&graph);
        let mut declared_sorted = declared.clone();
        let mut canonical_sorted = canonical.clone();
        declared_sorted.sort_unstable();
        canonical_sorted.sort_unstable();
        if declared_sorted != canonical_sorted {
            missing.push(format!(
                "{}: declares writable {declared_sorted:?} but the graph boundary carries {canonical_sorted:?}",
                operation.id
            ));
        }
    }
    assert!(
        judged > 0,
        "Fix: no registered op lifted through the graph boundary, so this test judged nothing."
    );
    assert!(
        missing.is_empty(),
        "Fix: the artifact route projects a writable buffer by its canonical resource name, so a declared writable buffer with no canonical value cannot be read back at all. Carry every declared writable buffer across the graph boundary.\n{}",
        missing.join("\n")
    );
}

/// The transposition this class is about, built rather than looked for.
///
/// A Program that declares an output buffer before a retained read-write one is
/// read in the opposite order at the graph boundary, because `from_program` mints
/// an external value for every retained buffer before the node that produces the
/// declared outputs exists. That is not a defect to fix at the boundary: an
/// output value cannot exist before its producer. It is why a projection keyed on
/// slot order returns a caller's buffers transposed, and why
/// `ArtifactSession::program_outputs` exists.
///
/// Built here instead of found in the registry: an assertion that some registered
/// op exhibits the case goes red when the op roster changes, which says nothing
/// about the projection. This Program is the case, so the contract holds whatever
/// `vyre-libs` and `vyre-primitives` carry.
#[test]
fn the_graph_boundary_transposes_an_output_declared_before_a_retained_buffer() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("src", 0, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
            BufferDecl::read_write("state", 2, DataType::U32).with_count(4),
        ],
        [4, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::load("src", Expr::u32(0))),
            Node::store("state", Expr::u32(0), Expr::load("src", Expr::u32(0))),
        ],
    );
    let graph = ProgramGraph::from_program("main", program.clone())
        .expect("Fix: a Program with one read, one output and one retained buffer must lift.");

    assert_eq!(
        declared_writable_names(&program),
        vec!["out", "state"],
        "Fix: declaration order is the order `Program::output_buffer_indices` reports, which is buffer slot order."
    );
    assert_eq!(
        canonical_writable_names(&graph),
        vec!["state", "out"],
        "Fix: a retained buffer becomes an external value before the node that produces the declared outputs, so canonical slot order is retained-then-output. If this stops being true the projection keyed on declaration order can be reconsidered, and `prove` is what reads the bytes."
    );
}
