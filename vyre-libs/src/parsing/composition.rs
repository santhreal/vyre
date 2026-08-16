use vyre_foundation::composition::wrap_child_region;
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::Node;

pub(crate) fn child_phase(parent_op_id: &str, phase_op_id: &str, body: Vec<Node>) -> Node {
    wrap_child_region(phase_op_id, Ident::from(parent_op_id), body)
}
