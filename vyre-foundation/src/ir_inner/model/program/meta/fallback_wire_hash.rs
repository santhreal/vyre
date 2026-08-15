//! Bounded IR structure digest for the wire-hash fallback.
//!
//! The fallback runs when a program cannot be encoded to the wire format, and
//! it must never format the full IR through `Debug`: the digest is built from
//! per-variant discriminants and bounded scalars instead.

use std::hash::{Hash, Hasher as _};

use rustc_hash::FxHasher;
use crate::ir::{Expr, Node};
use crate::transform::visit::{ExprVisitor, NodeVisitor};

fn mix_wire_fallback_hashable<T: Hash>(hasher: &mut blake3::Hasher, value: &T) {
    let mut state = FxHasher::default();
    value.hash(&mut state);
    hasher.update(&state.finish().to_le_bytes());
}

/// Bounded IR structure digest for wire-hash fallback (never formats full IR via `Debug`).
pub(super) struct FallbackWireHasher<'a>(pub(super) &'a mut blake3::Hasher);

impl NodeVisitor for FallbackWireHasher<'_> {
    fn visit_node(&mut self, node: &Node) {
        let h = &mut *self.0;
        match node {
            Node::Let { name, .. } => {
                h.update(b"n:Let\0");
                h.update(name.as_bytes());
            }
            Node::Assign { name, .. } => {
                h.update(b"n:Assign\0");
                h.update(name.as_bytes());
            }
            Node::Store { buffer, .. } => {
                h.update(b"n:Store\0");
                h.update(buffer.as_bytes());
            }
            Node::If { .. } => {
                h.update(b"n:If\0");
            }
            Node::Loop { var, .. } => {
                h.update(b"n:Loop\0");
                h.update(var.as_bytes());
            }
            Node::IndirectDispatch {
                count_buffer,
                count_offset,
            } => {
                h.update(b"n:IndirectDispatch\0");
                h.update(count_buffer.as_bytes());
                h.update(&count_offset.to_le_bytes());
            }
            Node::AsyncLoad {
                source,
                destination,
                tag,
                ..
            } => {
                h.update(b"n:AsyncLoad\0");
                h.update(source.as_bytes());
                h.update(destination.as_bytes());
                h.update(tag.as_bytes());
            }
            Node::AsyncStore {
                source,
                destination,
                tag,
                ..
            } => {
                h.update(b"n:AsyncStore\0");
                h.update(source.as_bytes());
                h.update(destination.as_bytes());
                h.update(tag.as_bytes());
            }
            Node::AsyncWait { tag } => {
                h.update(b"n:AsyncWait\0");
                h.update(tag.as_bytes());
            }
            Node::Trap { tag, .. } => {
                h.update(b"n:Trap\0");
                h.update(tag.as_bytes());
            }
            Node::Resume { tag } => {
                h.update(b"n:Resume\0");
                h.update(tag.as_bytes());
            }
            Node::AllReduce { buffer, op, group } => {
                h.update(b"n:AllReduce\0");
                h.update(buffer.as_bytes());
                h.update(&op.builtin_wire_tag().to_le_bytes());
                h.update(&group.as_u32().to_le_bytes());
            }
            Node::AllGather {
                input,
                output,
                group,
            } => {
                h.update(b"n:AllGather\0");
                h.update(input.as_bytes());
                h.update(output.as_bytes());
                h.update(&group.as_u32().to_le_bytes());
            }
            Node::ReduceScatter {
                input,
                output,
                op,
                group,
            } => {
                h.update(b"n:ReduceScatter\0");
                h.update(input.as_bytes());
                h.update(output.as_bytes());
                h.update(&op.builtin_wire_tag().to_le_bytes());
                h.update(&group.as_u32().to_le_bytes());
            }
            Node::Broadcast {
                buffer,
                root,
                group,
            } => {
                h.update(b"n:Broadcast\0");
                h.update(buffer.as_bytes());
                h.update(&root.to_le_bytes());
                h.update(&group.as_u32().to_le_bytes());
            }
            Node::Return => {
                h.update(b"n:Return\0");
            }
            Node::Barrier { ordering } => {
                h.update(b"n:Barrier\0");
                mix_wire_fallback_hashable(h, ordering);
            }
            Node::Block(_) => {
                h.update(b"n:Block\0");
            }
            Node::Region {
                generator,
                source_region,
                ..
            } => {
                h.update(b"n:Region\0");
                h.update(generator.as_bytes());
                if let Some(source_gen) = source_region {
                    h.update(source_gen.name.as_bytes());
                }
            }
            Node::Opaque(ext) => {
                h.update(b"n:Opaque\0");
                h.update(ext.extension_kind().as_bytes());
            }
        }
    }
}

impl ExprVisitor for FallbackWireHasher<'_> {
    fn visit_expr(&mut self, expr: &Expr) {
        let h = &mut *self.0;
        match expr {
            Expr::LitU32(v) => {
                h.update(b"e:LitU32\0");
                h.update(&v.to_le_bytes());
            }
            Expr::LitI32(v) => {
                h.update(b"e:LitI32\0");
                h.update(&v.to_le_bytes());
            }
            Expr::LitF32(v) => {
                h.update(b"e:LitF32\0");
                h.update(&v.to_le_bytes());
            }
            Expr::LitBool(v) => {
                h.update(b"e:LitBool\0");
                h.update(&[u8::from(*v)]);
            }
            Expr::Var(name) => {
                h.update(b"e:Var\0");
                h.update(name.as_bytes());
            }
            Expr::Load { buffer, .. } => {
                h.update(b"e:Load\0");
                h.update(buffer.as_bytes());
            }
            Expr::BufLen { buffer } => {
                h.update(b"e:BufLen\0");
                h.update(buffer.as_bytes());
            }
            Expr::BufferRef { buffer } => {
                h.update(b"e:BufferRef\0");
                h.update(buffer.as_bytes());
            }
            Expr::InvocationId { axis } => {
                h.update(b"e:InvocationId\0");
                h.update(&[*axis]);
            }
            Expr::WorkgroupId { axis } => {
                h.update(b"e:WorkgroupId\0");
                h.update(&[*axis]);
            }
            Expr::LocalId { axis } => {
                h.update(b"e:LocalId\0");
                h.update(&[*axis]);
            }
            Expr::BinOp { op, .. } => {
                h.update(b"e:BinOp\0");
                mix_wire_fallback_hashable(h, op);
            }
            Expr::UnOp { op, .. } => {
                h.update(b"e:UnOp\0");
                mix_wire_fallback_hashable(h, op);
            }
            Expr::Call { op_id, .. } => {
                h.update(b"e:Call\0");
                h.update(op_id.as_bytes());
            }
            Expr::Select { .. } => {
                h.update(b"e:Select\0");
            }
            Expr::Cast { target, .. } => {
                h.update(b"e:Cast\0");
                mix_wire_fallback_hashable(h, target);
            }
            Expr::Fma { .. } => {
                h.update(b"e:Fma\0");
            }
            Expr::Atomic {
                op,
                buffer,
                ordering,
                ..
            } => {
                h.update(b"e:Atomic\0");
                mix_wire_fallback_hashable(h, op);
                h.update(buffer.as_bytes());
                mix_wire_fallback_hashable(h, ordering);
            }
            Expr::SubgroupBallot { .. } => {
                h.update(b"e:SubgroupBallot\0");
            }
            Expr::SubgroupShuffle { .. } => {
                h.update(b"e:SubgroupShuffle\0");
            }
            Expr::SubgroupReduce { op, .. } => {
                h.update(b"e:SubgroupReduce\0");
                h.update(&[op.builtin_wire_tag()]);
            }
            Expr::SubgroupLocalId => {
                h.update(b"e:SubgroupLocalId\0");
            }
            Expr::SubgroupSize => {
                h.update(b"e:SubgroupSize\0");
            }
            Expr::Opaque(ext) => {
                h.update(b"e:Opaque\0");
                h.update(ext.extension_kind().as_bytes());
            }
        }
    }
}
