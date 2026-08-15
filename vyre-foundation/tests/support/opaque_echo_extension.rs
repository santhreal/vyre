//! The test-only opaque extension pair the wire suites register.
//!
//! Three suites needed an `ExprNode` and a `NodeExtension` that survive a wire
//! round trip by echoing their payload, and each carried its own copy: the
//! opaque round trip, the adversarial wire cases, and the round-trip property.
//! The copies differed only in the kind strings and the struct names, and no
//! assertion in any of them reads either, so the difference was accidental
//! rather than a distinction any test depends on.
//!
//! It matters that this is one file and not three because the pair IS the
//! contract under test: an extension whose `wire_payload` and registered
//! `deserialize` disagree makes every one of those suites pass against a
//! resolver that does not round-trip. One definition means one place where that
//! can be wrong, and it is wrong for all three at once instead of for the one
//! nobody updated.
//!
//! Each consumer includes this file with `#[path]`, so every test binary gets
//! its own `inventory` registration. That is required rather than incidental:
//! the resolver table is per-binary.

#![allow(dead_code)]

use std::sync::Arc;

use vyre_foundation::extension::{OpaqueExprResolver, OpaqueNodeResolver};
use vyre_foundation::ir::{DataType, ExprNode, NodeExtension};

/// Extension kind of the echoing expression node.
pub(crate) const EXPR_KIND: &str = "test.opaque.echo_expr";

/// Extension kind of the echoing statement node.
pub(crate) const NODE_KIND: &str = "test.opaque.echo_node";

/// The payload prefix the statement resolver refuses.
///
/// One of the three copies this file replaced carried this rule and the other
/// two did not, so it was not an accidental difference after all: the
/// adversarial suite needs a registered decoder that refuses in order to prove
/// that `Program::from_wire` reports the refusal as a structured error instead
/// of panicking past it or accepting the payload. An echo that accepts every
/// byte string cannot prove that, so the rule belongs here with the pair rather
/// than in whichever suite happened to need it.
///
/// It applies to the statement half only. The expression half echoes every byte
/// string, because the round-trip property builds one program holding every
/// expression variant with payloads it does not choose, and a refusal there
/// makes that property fail on the payload rather than on the round trip.
pub(crate) const REFUSED_NODE_PREFIX: [u8; 2] = [0xDE, 0xAD];

/// An expression extension whose wire payload is exactly its own bytes.
#[derive(Debug)]
pub(crate) struct EchoExpr {
    /// The bytes this node writes to the wire and reads back.
    pub(crate) payload: Vec<u8>,
}

impl ExprNode for EchoExpr {
    fn extension_kind(&self) -> &'static str {
        EXPR_KIND
    }

    fn debug_identity(&self) -> &str {
        "echo-expr"
    }

    fn result_type(&self) -> Option<DataType> {
        Some(DataType::U32)
    }

    fn cse_safe(&self) -> bool {
        true
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        *blake3::hash(&self.payload).as_bytes()
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn wire_payload(&self) -> Vec<u8> {
        self.payload.clone()
    }
}

fn deserialize_echo_expr(bytes: &[u8]) -> Result<Arc<dyn ExprNode>, String> {
    Ok(Arc::new(EchoExpr {
        payload: bytes.to_vec(),
    }))
}

inventory::submit! {
    OpaqueExprResolver {
        kind: EXPR_KIND,
        deserialize: deserialize_echo_expr,
    }
}

/// A statement extension whose wire payload is exactly its own bytes.
#[derive(Debug)]
pub(crate) struct EchoNode {
    /// The bytes this node writes to the wire and reads back.
    pub(crate) payload: Vec<u8>,
}

impl NodeExtension for EchoNode {
    fn extension_kind(&self) -> &'static str {
        NODE_KIND
    }

    fn debug_identity(&self) -> &str {
        "echo-node"
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        *blake3::hash(&self.payload).as_bytes()
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn wire_payload(&self) -> Vec<u8> {
        self.payload.clone()
    }
}

fn deserialize_echo_node(bytes: &[u8]) -> Result<Arc<dyn NodeExtension>, String> {
    if bytes.starts_with(&REFUSED_NODE_PREFIX) {
        return Err(format!(
            "Fix: {NODE_KIND} refuses a payload beginning {:#04X} {:#04X}; encode a payload the resolver accepts",
            REFUSED_NODE_PREFIX[0], REFUSED_NODE_PREFIX[1]
        ));
    }
    Ok(Arc::new(EchoNode {
        payload: bytes.to_vec(),
    }))
}

inventory::submit! {
    OpaqueNodeResolver {
        kind: NODE_KIND,
        deserialize: deserialize_echo_node,
    }
}
