use vyre_foundation::composition::wrap_child_region;
use vyre_foundation::ir::GeneratorRef;
use vyre_foundation::ir::Node;

pub(crate) fn child_phase(parent_op_id: &str, phase_op_id: &str, body: Vec<Node>) -> Node {
    wrap_child_region(
        phase_op_id,
        GeneratorRef {
            name: parent_op_id.to_string(),
        },
        body,
    )
}
