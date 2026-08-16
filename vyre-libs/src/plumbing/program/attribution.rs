//! Attribute a built `Program` to the composition that selected it.
//!
//! A composition that picks an existing operation and runs it has to say so in
//! the IR, or the region it emits looks like a body it wrote itself. The edge
//! is a `source_region` on the child's own region: the generator name stays the
//! operation that owns the body, and the parent name is the composition that
//! called it. That pair is what the composition-chain gates read, and it is
//! what separates a selection from a relabel.
//!
//! Only the entry is rebuilt. `Program::wrapped` would deep-clone the buffer
//! table and reset the metadata flags, both of which the caller already has
//! correct.

#[cfg(feature = "llm")]
use vyre_foundation::composition::single_invocation;
use vyre_foundation::composition::wrap_child_region;
use vyre_foundation::ir::{GeneratorRef, Node, Program};

/// `program`'s body, re-emitted as one child region of `parent_op_id` named
/// `child_op_id`.
///
/// `child_op_id` must name a registered operation. A child region naming an
/// unregistered id claims a building block that does not exist, which is what
/// the `anonymous::` prefix is for.
pub(crate) fn attribute_child_nodes(
    parent_op_id: &str,
    child_op_id: &str,
    program: &Program,
) -> Vec<Node> {
    vec![attributed_region(
        parent_op_id,
        child_op_id,
        entry_body(program),
    )]
}

/// `program` with its entry replaced by [`attribute_child_nodes`].
///
/// This is the shape for an arm of a fused composition, where the fused entry
/// already carries the parent's own region. A composition whose entry is one
/// selected operation wraps the nodes in its own anonymous region instead.
#[cfg(feature = "llm")]
pub(crate) fn attribute_child(parent_op_id: &str, child_op_id: &str, program: Program) -> Program {
    let nodes = attribute_child_nodes(parent_op_id, child_op_id, &program);
    program.with_rewritten_wrapped_entry(nodes)
}

/// [`attribute_child`] for an operation whose body runs on one invocation.
///
/// A serial operation dispatched on its own launches one invocation, so its
/// body needs no guard. Fusion runs every arm under the widest geometry in the
/// batch, so the same body under a parallel arm runs once per invocation, and
/// every copy performs the same read-modify-write on the arm's own scratch.
/// The gate names the invocation the body belongs to. Fusion reads the gate
/// too: an invocation-gated store makes the arm a grid-sync writer, so the
/// arms that consume its result wait for it.
#[cfg(feature = "llm")]
pub(crate) fn attribute_serial_child(
    parent_op_id: &str,
    child_op_id: &str,
    program: Program,
) -> Program {
    let gated = single_invocation(entry_body(&program));
    let node = attributed_region(parent_op_id, child_op_id, gated);
    program.with_rewritten_wrapped_entry(vec![node])
}

/// The nodes a built operation runs, without the region it wrapped them in.
///
/// Every builder in this crate emits one region as its entry. A program that
/// does not is taken as its own body rather than being rejected: the caller is
/// attributing the nodes it already holds, and an empty or multi-node entry is
/// still exactly those nodes.
fn entry_body(program: &Program) -> Vec<Node> {
    match program.entry() {
        [Node::Region { body, .. }] => body.as_ref().clone(),
        other => other.to_vec(),
    }
}

fn attributed_region(parent_op_id: &str, child_op_id: &str, body: Vec<Node>) -> Node {
    wrap_child_region(
        child_op_id,
        GeneratorRef {
            name: parent_op_id.to_string(),
        },
        body,
    )
}
