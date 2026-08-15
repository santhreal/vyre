//! Region builder. Every Category C hardware intrinsic wraps its body in one
//! of these, so the IR records which operation built which nodes.

use std::sync::Arc;
use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};
use vyre_foundation::ir::Node;

/// Wrap `body` in a `Node::Region` tagged with `generator`. If `source_region`
/// is `Some`, it records the composition chain from caller to callee  -  the
/// invariant every Cat-C intrinsic must uphold: a body built by calling
/// another operation's builder carries that operation's generator and a
/// `source_region` naming the caller.
#[must_use]
pub fn wrap(generator: &str, body: Vec<Node>, source_region: Option<GeneratorRef>) -> Node {
    Node::Region {
        generator: Ident::from(generator),
        source_region,
        body: Arc::new(body),
    }
}

/// Shorthand for `wrap(generator, body, None)`  -  used when an intrinsic has no
/// composition parent (the op is the root of its region chain).
#[must_use]
pub fn wrap_anonymous(generator: &str, body: Vec<Node>) -> Node {
    wrap(generator, body, None)
}

/// Shorthand for `wrap(generator, body, Some(parent))`  -  used when an intrinsic
/// is invoked from inside another registered op's body.
#[must_use]
pub fn wrap_child(generator: &str, parent: GeneratorRef, body: Vec<Node>) -> Node {
    wrap(generator, body, Some(parent))
}
