//! Region builder. Every hardware intrinsic wraps its body in exactly one
//! `Node::Region` naming its generator, so a pass can treat the op as atomic
//! and a composition chain stays visible from caller to callee.

use std::sync::Arc;
use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};
use vyre_foundation::ir::Node;

/// Wrap `body` in a `Node::Region` tagged with `generator`. A `source_region`
/// records the composition edge from caller to callee, which every intrinsic
/// built by calling another operation's builder must carry.
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
