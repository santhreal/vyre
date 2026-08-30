// Statement nodes  -  execute effects.
//
// Statements modify state: bind variables, write to buffers, branch, loop.
// A program's entry point is a sequence of statements.

use std::fmt;

pub use crate::ir_inner::model::generated::Node;

/// Public contract for downstream statement extension nodes.
///
/// Implementors own their payload and semantics. Core uses the stable
/// metadata here to validate, compare, and diagnose opaque nodes without
/// pretending it can execute or serialize them.
pub trait NodeExtension: fmt::Debug + Send + Sync + 'static {
    /// Stable extension namespace, for example `my_backend.speculate`.
    fn extension_kind(&self) -> &'static str;

    /// Human-readable identity used in diagnostics and debug logs.
    fn debug_identity(&self) -> &str;

    /// Stable, content-addressed identity for equality and optimizer keys.
    fn stable_fingerprint(&self) -> [u8; 32];

    /// Validate extension-local invariants.
    ///
    /// The returned error must explain the bad invariant and include `Fix:`.
    ///
    /// # Errors
    ///
    /// Returns an extension-defined diagnostic when the payload violates its
    /// local invariants.
    fn validate_extension(&self) -> Result<(), String>;

    /// Downcast to Any to allow backend-specific dispatch from opaque payloads.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Serialize the extension payload into stable bytes used by the wire
    /// encoder's `Node::Opaque` path (tag `0x80`). Default: empty payload.
    ///
    /// The payload contract is endian-fixed: any numeric field wider than
    /// one byte MUST be written with `to_le_bytes` (or the
    /// [`crate::opaque_payload`] helpers) and the matching decoder MUST
    /// reconstruct it with `from_le_bytes`. Host-endian encodings such as
    /// `to_ne_bytes` are forbidden because the wire format must stay
    /// byte-identical across architectures: a Program encoded on a
    /// little-endian host and decoded on a big-endian host must produce
    /// the same `crate::ir::Program::hash` and the same IR.
    ///
    /// Extension authors should use [`crate::opaque_payload::endian::LeBytesWriter`] when
    /// building payloads because it makes the required endianness explicit in the type.
    fn wire_payload(&self) -> Vec<u8> {
        Vec::new()
    }
}

mod impl_node;

/// Canonical string op id for every statement-node variant.
///
/// Wire-format encode/decode keys on these names to route an encoded node
/// back to its variant. Adding a new `Node` variant REQUIRES extending this
/// function with a matching arm  -  the wire decoder depends on round-tripping
/// the exact name this function returns.
#[must_use]
pub fn node_op_id(node: &Node) -> &'static str {
    match node {
        Node::Let { .. } => "vyre.node.let",
        Node::Assign { .. } => "vyre.node.assign",
        Node::Store { .. } => "vyre.node.store",
        Node::If { .. } => "vyre.node.if",
        Node::Loop { .. } => "vyre.node.loop",
        Node::Return => "vyre.node.return",
        Node::Block(_) => "vyre.node.block",
        Node::Barrier { .. } => "vyre.node.barrier",
        Node::LogicalBarrier { .. } => "vyre.node.logical_barrier",
        Node::Region { .. } => "vyre.node.region",
        Node::IndirectDispatch { .. } => "vyre.node.indirect_dispatch",
        Node::AsyncLoad { .. } => "vyre.node.async_load",
        Node::AsyncStore { .. } => "vyre.node.async_store",
        Node::AsyncWait { .. } => "vyre.node.async_wait",
        Node::Trap { .. } => "vyre.node.trap",
        Node::Resume { .. } => "vyre.node.resume",
        Node::AllReduce { .. } => "vyre.node.all_reduce",
        Node::AllGather { .. } => "vyre.node.all_gather",
        Node::ReduceScatter { .. } => "vyre.node.reduce_scatter",
        Node::Broadcast { .. } => "vyre.node.broadcast",
        Node::TileLoad { .. } => "vyre.node.tile_load",
        Node::TileStore { .. } => "vyre.node.tile_store",
        Node::TileMatmul { .. } => "vyre.node.tile_matmul",
        Node::TileReduce { .. } => "vyre.node.tile_reduce",
        Node::TileElementwise { .. } => "vyre.node.tile_elementwise",
        Node::TileDecl { .. } => "vyre.node.tile_decl",
        Node::Opaque(extension) => extension.extension_kind(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::ir::{BufferAccess, BufferDecl, DataType, Node, Program};

    #[test]
    fn indirect_dispatch_round_trip() {
        let program = Program::wrapped(
            vec![BufferDecl::storage(
                "counts",
                0,
                BufferAccess::ReadOnly,
                DataType::U32,
            )],
            [64, 1, 1],
            vec![Node::indirect_dispatch("counts", 16)],
        );

        let wire = program
            .to_wire()
            .expect("Fix: indirect dispatch must serialize into VIR0");
        let decoded =
            Program::from_wire(&wire).expect("Fix: indirect dispatch must decode from VIR0");

        assert_eq!(decoded, program);
    }

    #[test]
    fn async_load_async_wait_round_trip() {
        let program = Program::wrapped(
            vec![BufferDecl::storage(
                "out",
                0,
                BufferAccess::ReadWrite,
                DataType::U32,
            )],
            [1, 1, 1],
            vec![
                Node::async_load("chunk-0"),
                Node::store("out", crate::ir::Expr::u32(0), crate::ir::Expr::u32(1)),
                Node::async_wait("chunk-0"),
            ],
        );

        let wire = program
            .to_wire()
            .expect("Fix: async stream nodes must serialize into VIR0");
        let decoded =
            Program::from_wire(&wire).expect("Fix: async stream nodes must decode from VIR0");

        assert_eq!(decoded, program);
    }

    /// WHY: `node_op_id` is exhaustive with no catch-all, so the compiler
    /// already forces a new variant to state an id. It does not force that id
    /// to be new. Two arms returning one string make the wire decoder route a
    /// decoded node to the wrong variant, and make a backend that admits one
    /// variant silently admit the other, because
    /// `vyre-driver`'s program validation keys admission on this string. The
    /// arm roster is read from this file at run time rather than restated, so
    /// a copied arm is caught instead of a stale list.
    #[test]
    fn every_node_op_id_arm_states_a_distinct_id() {
        let source = include_str!("mod.rs");
        let body = source
            .split_once("pub fn node_op_id(node: &Node) -> &'static str {")
            .expect("Fix: node_op_id must remain declared in this file")
            .1
            .split_once("\n}\n")
            .expect("Fix: node_op_id must remain a closed function body")
            .0;

        let mut arms = 0_usize;
        let mut ids = std::collections::BTreeSet::new();
        for line in body.lines() {
            let Some((pattern, result)) = line.split_once("=>") else {
                continue;
            };
            if !pattern.trim_start().starts_with("Node::") {
                continue;
            }
            arms += 1;
            // The extension arm defers to the extension's own kind, so it
            // states no literal here and cannot collide with one.
            if let Some(id) = result.trim().strip_prefix('"') {
                let id = id
                    .strip_suffix("\",")
                    .expect("Fix: a literal node op id must end the match arm");
                assert!(
                    id.starts_with("vyre.node."),
                    "Fix: node op id `{id}` must stay in the `vyre.node.` namespace"
                );
                assert!(
                    ids.insert(id),
                    "Fix: node op id `{id}` is stated by more than one arm; wire decode and backend admission both key on it"
                );
            }
        }

        assert!(
            arms > 25,
            "Fix: only {arms} node_op_id arms were read; the parse no longer sees the match"
        );
        assert_eq!(
            ids.len() + 1,
            arms,
            "Fix: {arms} arms produced {} distinct literal ids; every arm but the extension arm must state its own",
            ids.len()
        );
    }
}
