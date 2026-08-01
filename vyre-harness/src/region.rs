//! Shared Region builder.
//!
//! Every public composition in `vyre-libs` routes its produced `Vec<Node>`
//! through `wrap` so optimizer passes treat the library call as an
//! opaque unit by default. Explicit inline passes can unroll the Region
//! at lower levels of the pipeline.
//!
//! The `generator` name is load-bearing  -  it's what shows up in
//! BackendError stack traces, conform certificates, and tracing spans.
//! Every library function uses its fully-qualified path as the
//! generator name so a consumer looking at a trace can grep exactly
//! where the IR came from.

use std::sync::Arc;
use vyre::ir::Node;
use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};

pub use vyre_foundation::composition::{reparent_program_children, tag_program};

/// Wrap a list of Nodes into a single `Node::Region`.
///
/// The `generator` argument is the stable identifier consumers see in
/// errors and traces. Convention: fully-qualified module path, e.g.
/// `"vyre-libs::nn::linear"`, `"vyre-libs::crypto::fnv1a"`.
///
/// The `source_region` argument is optional caller-provided span
/// information. Library functions pass `None`; a higher-level compiler
/// that tracks source positions can construct `GeneratorRef` with
/// line + column.
#[must_use]
pub fn wrap(generator: &str, body: Vec<Node>, source_region: Option<GeneratorRef>) -> Node {
    Node::Region {
        generator: Ident::from(generator),
        source_region,
        body: Arc::new(body),
    }
}

/// Construct a Region with no source-region annotation. Convenience
/// shortcut for the common library-call case where the caller isn't
/// tracking source positions.
#[must_use]
pub fn wrap_anonymous(generator: &str, body: Vec<Node>) -> Node {
    wrap(generator, body, None)
}

/// Construct a Region whose `source_region` names the composing parent op.
#[must_use]
pub fn wrap_child(generator: &str, parent: GeneratorRef, body: Vec<Node>) -> Node {
    wrap(generator, body, Some(parent))
}
