use crate::ir::{Ident, Node};

/// The set of identifiers live at a program point.
///
/// A persistent set, because liveness propagates backwards through branches and
/// each arm needs the set as it stood at the join; a std set would clone every
/// element per arm. This alias is the one place the concrete set type is named,
/// so the DCE passes below never spell it.
pub(crate) type LiveSet = imbl::HashSet<Ident>;

pub(crate) struct LiveResult {
    pub(crate) nodes: Vec<Node>,
    pub(crate) live_in: LiveSet,
}
