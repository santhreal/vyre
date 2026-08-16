//! A structural fingerprint of one program.
//!
//! The fingerprint is a byte string over node and expression kinds, so two
//! programs that differ only in buffer names or in a literal that falls in the
//! same bucket agree. Checks 1 and 10 both compare fingerprints, and they
//! compare different parts of one: check 1 scores the whole string, check 10
//! fixes the first [`PREFIX_LEN`] bytes as a bucket key and scores the rest.

use super::*;

/// Build a compact byte sequence representing the node-kind tree
/// structure of a Program's body. Two programs with identical
/// structural shape produce identical fingerprints; one-byte edits
/// produce minor differences. Used for check 1 similarity scoring.
pub(super) fn fingerprint_program(program: &Program) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for node in program.entry() {
        fingerprint_node(node, &mut out);
    }
    out
}

pub(super) fn fingerprint_node(node: &Node, out: &mut Vec<u8>) {
    match node {
        Node::Let { value, .. } => {
            out.push(0x01);
            fingerprint_expr(value, out);
        }
        Node::Assign { value, .. } => {
            out.push(0x02);
            fingerprint_expr(value, out);
        }
        Node::Store { index, value, .. } => {
            out.push(0x03);
            fingerprint_expr(index, out);
            fingerprint_expr(value, out);
        }
        Node::If {
            cond,
            then,
            otherwise,
            ..
        } => {
            out.push(0x04);
            fingerprint_expr(cond, out);
            out.push(0xFE);
            for n in then {
                fingerprint_node(n, out);
            }
            out.push(0xFF);
            for n in otherwise {
                fingerprint_node(n, out);
            }
            out.push(0xFF);
        }
        Node::Loop { from, to, body, .. } => {
            out.push(0x05);
            fingerprint_expr(from, out);
            fingerprint_expr(to, out);
            out.push(0xFE);
            for n in body {
                fingerprint_node(n, out);
            }
            out.push(0xFF);
        }
        Node::Return => out.push(0x06),
        Node::Block(nodes) => {
            out.push(0x07);
            for n in nodes {
                fingerprint_node(n, out);
            }
            out.push(0xFF);
        }
        Node::Barrier {
            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
        } => out.push(0x08),
        Node::Region {
            source_region,
            body,
            generator,
        } => {
            out.push(0x09);
            if source_region.is_some() {
                out.extend_from_slice(&fingerprint_name(generator.as_str()));
            } else {
                for n in body.iter() {
                    fingerprint_node(n, out);
                }
            }
            out.push(0xFF);
        }
        Node::IndirectDispatch { .. } => out.push(0x0A),
        Node::AsyncLoad { offset, size, .. } => {
            out.push(0x0B);
            fingerprint_expr(offset, out);
            fingerprint_expr(size, out);
        }
        Node::AsyncStore { offset, size, .. } => {
            out.push(0x0C);
            fingerprint_expr(offset, out);
            fingerprint_expr(size, out);
        }
        Node::AsyncWait { .. } => out.push(0x0D),
        Node::Trap { address, .. } => {
            out.push(0x0E);
            fingerprint_expr(address, out);
        }
        Node::Resume { .. } => out.push(0x0F),
        _ => out.push(0x80),
    }
}

pub(super) fn fingerprint_expr(expr: &Expr, out: &mut Vec<u8>) {
    match expr {
        Expr::LitU32(value) => {
            out.push(0x21);
            out.push(literal_bucket_u32(*value));
        }
        Expr::LitI32(value) => {
            out.push(0x22);
            out.push(literal_bucket_u32(*value as u32));
        }
        Expr::LitF32(value) => {
            out.push(0x23);
            out.push(literal_bucket_u32(value.to_bits()));
        }
        Expr::LitBool(value) => {
            out.push(0x24);
            out.push(u8::from(*value));
        }
        Expr::Var(_) => out.push(0x25),
        Expr::Load { index, .. } => {
            out.push(0x26);
            fingerprint_expr(index, out);
        }
        Expr::BufLen { .. } => out.push(0x27),
        Expr::InvocationId { axis } => {
            out.push(0x28);
            out.push(*axis);
        }
        Expr::WorkgroupId { axis } => {
            out.push(0x29);
            out.push(*axis);
        }
        Expr::LocalId { axis } => {
            out.push(0x2A);
            out.push(*axis);
        }
        Expr::BinOp { op, left, right } => {
            out.push(0x2B);
            out.push(fingerprint_name(&format!("bin::{op:?}"))[0]);
            fingerprint_expr(left, out);
            fingerprint_expr(right, out);
        }
        Expr::UnOp { op, operand } => {
            out.push(0x2C);
            out.push(fingerprint_name(&format!("un::{op:?}"))[0]);
            fingerprint_expr(operand, out);
        }
        Expr::Call { op_id, args } => {
            out.push(0x2D);
            out.push(fingerprint_name(op_id.as_str())[0]);
            for arg in args {
                fingerprint_expr(arg, out);
            }
            out.push(0xFD);
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            out.push(0x2E);
            fingerprint_expr(cond, out);
            fingerprint_expr(true_val, out);
            fingerprint_expr(false_val, out);
        }
        Expr::Cast { target, value } => {
            out.push(0x2F);
            out.push(fingerprint_name(&format!("cast::{target:?}"))[0]);
            fingerprint_expr(value, out);
        }
        Expr::Fma { a, b, c } => {
            out.push(0x30);
            fingerprint_expr(a, out);
            fingerprint_expr(b, out);
            fingerprint_expr(c, out);
        }
        Expr::Atomic {
            op,
            index,
            expected,
            value,
            ordering,
            ..
        } => {
            out.push(0x31);
            out.push(fingerprint_name(&format!("atomic::{op:?}::{ordering:?}"))[0]);
            fingerprint_expr(index, out);
            if let Some(expected) = expected.as_deref() {
                fingerprint_expr(expected, out);
            }
            out.push(0xFC);
            fingerprint_expr(value, out);
        }
        Expr::SubgroupBallot { cond } => {
            out.push(0x32);
            fingerprint_expr(cond, out);
        }
        Expr::SubgroupShuffle { value, lane } => {
            out.push(0x33);
            fingerprint_expr(value, out);
            fingerprint_expr(lane, out);
        }
        Expr::SubgroupReduce { value, .. } => {
            out.push(0x34);
            fingerprint_expr(value, out);
        }
        Expr::SubgroupLocalId => out.push(0x35),
        Expr::SubgroupSize => out.push(0x36),
        Expr::Opaque(extension) => {
            out.push(0x37);
            out.push(extension.stable_fingerprint()[0]);
        }
        _ => out.push(0xBF),
    }
}

pub(super) fn literal_bucket_u32(value: u32) -> u8 {
    match value {
        0 => 0,
        1 => 1,
        2..=4 => 2,
        5..=31 => 3,
        32..=255 => 4,
        256..=4096 => 5,
        _ => 6,
    }
}

pub(super) fn fingerprint_name(name: &str) -> [u8; 4] {
    let mut hash = 0x811C_9DC5u32;
    for byte in name.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash.to_le_bytes()
}

pub(super) const PREFIX_LEN: usize = 16;

/// The part of a fingerprint the bucket key did not already fix.
pub(super) fn fingerprint_past_prefix(fingerprint: &[u8]) -> &[u8] {
    fingerprint.get(PREFIX_LEN..).unwrap_or(&[])
}
