//! Shared f32 backend-parity contract for Cat-A and conform gates.
//!
//! Integer and boolean outputs remain byte-identical. F32 outputs use a
//! bounded ULP window because GPU backends may contract multiply-add
//! sequences and may use native approximate transcendental instructions.

use crate::ir::{DataType, Expr, Node, Program, UnOp};
use crate::operation::OperationRegistry;

/// Maximum accepted reference-oracle error against correctly-rounded f32
/// transcendentals.
pub const REFERENCE_TRANSCENDENTAL_ULP_BUDGET: u32 = 4;

/// Maximum accepted backend-vs-reference error for programs containing f32
/// transcendentals.
pub const BACKEND_TRANSCENDENTAL_ULP_BUDGET: u32 = 128;

/// Maximum accepted backend-vs-reference error for elementary f32 programs.
///
/// This is the Q6 contraction contract: WGSL/Naga backends are allowed to
/// fuse `a*b+c` into one FMA while the reference may evaluate as two
/// operations. The budget is program-level, not an op-id whitelist.
pub const BACKEND_ELEMENTARY_F32_ULP_BUDGET: u32 = 4;

/// Return the allowed f32 ULP tolerance for backend-vs-reference parity checks.
///
/// Every caller compares a hardware backend against the CPU reference, so the
/// window can never be zero: contraction is a backend right, stated at the top
/// of this module, and cuda and wgpu both fold `a*b+c` into one FMA. A
/// `strict-fp` feature used to force 0 here. It forbade nothing, because no
/// emitter consulted it; its only effect was to fail every elementary f32 op
/// that contracts, so `cargo test --workspace --all-features`, which the release
/// procedure requires, could not pass. `newton_schulz_poly5_f32` drifted 4 ULP,
/// `newton_schulz_5step` 2 and `ema_apply` 1, with cuda and wgpu agreeing
/// bit-for-bit with each other. Bounding contraction has to happen in the
/// emitters before a tolerance can claim to.
#[must_use]
pub fn f32_ulp_tolerance(program: &Program) -> u32 {
    if program_has_transcendental(program) {
        BACKEND_TRANSCENDENTAL_ULP_BUDGET
    } else {
        BACKEND_ELEMENTARY_F32_ULP_BUDGET
    }
}

/// Combine an op-id-specific tolerance with the program-level f32 policy.
#[must_use]
pub fn effective_tolerance(op_id: &str, program: &Program) -> u32 {
    OperationRegistry::global()
        .get(op_id)
        .map_or(0, |entry| entry.tolerance())
        .max(f32_ulp_tolerance(program))
}

fn program_has_transcendental(program: &Program) -> bool {
    program.entry().iter().any(node_has_transcendental)
}

fn node_has_transcendental(node: &Node) -> bool {
    match node {
        Node::Let { value, .. } | Node::Assign { value, .. } => expr_has_transcendental(value),
        Node::Store { index, value, .. } => {
            expr_has_transcendental(index) || expr_has_transcendental(value)
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            expr_has_transcendental(cond)
                || then.iter().any(node_has_transcendental)
                || otherwise.iter().any(node_has_transcendental)
        }
        Node::Loop { from, to, body, .. } => {
            expr_has_transcendental(from)
                || expr_has_transcendental(to)
                || body.iter().any(node_has_transcendental)
        }
        Node::Block(body) => body.iter().any(node_has_transcendental),
        Node::Region { body, .. } => body.iter().any(node_has_transcendental),
        Node::AsyncLoad { offset, size, .. } | Node::AsyncStore { offset, size, .. } => {
            expr_has_transcendental(offset) || expr_has_transcendental(size)
        }
        Node::Trap { address, .. } => expr_has_transcendental(address),
        Node::IndirectDispatch { .. }
        | Node::AsyncWait { .. }
        | Node::Barrier { .. }
        | Node::Resume { .. }
        | Node::Return => false,
        Node::Opaque(_) => false,
        _ => false,
    }
}

/// Whether `expr` reaches an f32 op a backend may lower to an approximate
/// native instruction.
///
/// The set is the policy, so a `UnOp` left out of it asserts that backends
/// agree with the reference to `BACKEND_ELEMENTARY_F32_ULP_BUDGET` on that op.
/// `UnOp::Reciprocal` is deliberately outside: cuda and wgpu both lower it to
/// a division rather than an approximate reciprocal instruction, so it stays
/// in the elementary window.
fn expr_has_transcendental(expr: &Expr) -> bool {
    match expr {
        Expr::UnOp { op, operand } => {
            matches!(
                op,
                UnOp::Exp
                    | UnOp::Exp2
                    | UnOp::Log
                    | UnOp::Log2
                    | UnOp::Sqrt
                    | UnOp::InverseSqrt
                    | UnOp::Sin
                    | UnOp::Cos
                    | UnOp::Tan
                    | UnOp::Asin
                    | UnOp::Acos
                    | UnOp::Atan
                    | UnOp::Sinh
                    | UnOp::Cosh
                    | UnOp::Tanh
            ) || expr_has_transcendental(operand)
        }
        Expr::BinOp { left, right, .. } => {
            expr_has_transcendental(left) || expr_has_transcendental(right)
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            expr_has_transcendental(cond)
                || expr_has_transcendental(true_val)
                || expr_has_transcendental(false_val)
        }
        Expr::Cast { value, .. } => expr_has_transcendental(value),
        Expr::Fma { a, b, c } => {
            expr_has_transcendental(a) || expr_has_transcendental(b) || expr_has_transcendental(c)
        }
        Expr::Load { index, .. } => expr_has_transcendental(index),
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            expr_has_transcendental(index)
                || expected.as_deref().is_some_and(expr_has_transcendental)
                || expr_has_transcendental(value)
        }
        Expr::SubgroupReduce { value, .. } | Expr::SubgroupBallot { cond: value } => {
            expr_has_transcendental(value)
        }
        Expr::SubgroupShuffle { value, lane } => {
            expr_has_transcendental(value) || expr_has_transcendental(lane)
        }
        Expr::Call { args, .. } => args.iter().any(expr_has_transcendental),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType};

    #[test]
    fn elementary_f32_program_gets_contraction_budget() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::F32).with_count(1)],
            [1, 1, 1],
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::add(Expr::mul(Expr::f32(1.25), Expr::f32(2.0)), Expr::f32(0.5)),
            )],
        );

        // An elementary a*b+c program gets the FMA-contraction window under
        // every feature combination. The window used to collapse to 0 under a
        // `strict-fp` feature that no emitter honoured, which made this the one
        // assertion that had to branch on features to stay true.
        assert_eq!(
            f32_ulp_tolerance(&program),
            BACKEND_ELEMENTARY_F32_ULP_BUDGET
        );
    }

    #[test]
    fn transcendental_program_gets_native_backend_budget() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::F32).with_count(1)],
            [1, 1, 1],
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::UnOp {
                    op: UnOp::Tanh,
                    operand: Box::new(Expr::f32(1.0)),
                },
            )],
        );

        assert_eq!(
            f32_ulp_tolerance(&program),
            BACKEND_TRANSCENDENTAL_ULP_BUDGET
        );
    }
}

// ───────────────────────────────────────────────────────────────────
// Buffer parity comparison
// ───────────────────────────────────────────────────────────────────

/// Per-buffer comparison outcome for `compare_output_buffers`.
#[derive(Debug)]
pub enum BufferParity {
    /// Every output buffer matched the reference (byte-exact for
    /// non-F32, within the ULP window for F32).
    Ok,
    /// A specific buffer diverged; human-readable explanation.
    Mismatch(String),
}

/// Compare two output-buffer vectors against the program's declared
/// buffer layout. F32 buffers use [`f32_buffer_matches`] with the program-level
/// floating-point policy; every other element type requires byte identity.
/// Returns [`BufferParity::Ok`] only when every slot passed.
pub fn compare_output_buffers(
    program: &Program,
    outputs_a: &[Vec<u8>],
    outputs_b: &[Vec<u8>],
) -> BufferParity {
    compare_output_buffers_with_tolerance(program, outputs_a, outputs_b, f32_ulp_tolerance(program))
}

/// Compare output buffers using the tolerance owned by `op_id`.
#[must_use]
pub fn compare_operation_outputs(
    op_id: &str,
    program: &Program,
    outputs_a: &[Vec<u8>],
    outputs_b: &[Vec<u8>],
) -> BufferParity {
    compare_output_buffers_with_tolerance(
        program,
        outputs_a,
        outputs_b,
        effective_tolerance(op_id, program),
    )
}

fn compare_output_buffers_with_tolerance(
    program: &Program,
    outputs_a: &[Vec<u8>],
    outputs_b: &[Vec<u8>],
    tolerance: u32,
) -> BufferParity {
    if outputs_a.len() != outputs_b.len() {
        return BufferParity::Mismatch(format!(
            "output buffer count mismatch: {} vs {}; left={} right={}",
            outputs_a.len(),
            outputs_b.len(),
            summarize_buffers(outputs_a),
            summarize_buffers(outputs_b)
        ));
    }

    let output_indices = program.output_buffer_indices();
    if output_indices.len() != outputs_a.len() {
        return BufferParity::Mismatch(format!(
            "program declares {} output buffer(s), compared {} result buffer(s)",
            output_indices.len(),
            outputs_a.len()
        ));
    }

    for (slot, ((bytes_a, bytes_b), buffer_index)) in outputs_a
        .iter()
        .zip(outputs_b.iter())
        .zip(output_indices.iter().copied())
        .enumerate()
    {
        if bytes_a.len() != bytes_b.len() {
            return BufferParity::Mismatch(format!(
                "output buffer {slot} length mismatch: {} vs {}; left={} right={}",
                bytes_a.len(),
                bytes_b.len(),
                summarize_bytes(bytes_a),
                summarize_bytes(bytes_b)
            ));
        }
        let element = program.buffers()[buffer_index as usize].element();
        if element == DataType::F32 {
            if !f32_buffer_matches(bytes_a, bytes_b, tolerance) {
                return BufferParity::Mismatch(format!(
                    "output buffer {slot} (F32) exceeded the {tolerance}-ULP window; left={} right={}",
                    summarize_bytes(bytes_a),
                    summarize_bytes(bytes_b)
                ));
            }
        } else if bytes_a != bytes_b {
            return BufferParity::Mismatch(format!(
                "output buffer {slot} ({element:?}) is not byte-identical; left={} right={}",
                summarize_bytes(bytes_a),
                summarize_bytes(bytes_b)
            ));
        }
    }

    BufferParity::Ok
}

fn summarize_buffers(buffers: &[Vec<u8>]) -> String {
    buffers
        .iter()
        .enumerate()
        .map(|(slot, bytes)| format!("{slot}:{}", summarize_bytes(bytes)))
        .collect::<Vec<_>>()
        .join(",")
}

fn summarize_bytes(bytes: &[u8]) -> String {
    const MAX_BYTES: usize = 32;
    let mut summary = format!("len={} hex=", bytes.len());
    for byte in bytes.iter().take(MAX_BYTES) {
        summary.push_str(&format!("{byte:02x}"));
    }
    if bytes.len() > MAX_BYTES {
        summary.push_str("...");
    }
    summary
}

/// Compare two `[u8]` views as packed little-endian f32 arrays under a
/// ULP window. Returns `false` if lengths differ or any element falls
/// outside the window. NaN inputs only match bitwise.
pub fn f32_buffer_matches(bytes_a: &[u8], bytes_b: &[u8], tolerance: u32) -> bool {
    if bytes_a.len() != bytes_b.len() || bytes_a.len() % 4 != 0 {
        return false;
    }
    if tolerance == 0 {
        return bytes_a == bytes_b;
    }
    bytes_a
        .chunks_exact(4)
        .zip(bytes_b.chunks_exact(4))
        .all(|(left, right)| {
            let left = f32::from_bits(u32::from_le_bytes([left[0], left[1], left[2], left[3]]));
            let right =
                f32::from_bits(u32::from_le_bytes([right[0], right[1], right[2], right[3]]));
            left.to_bits() == right.to_bits()
                || ulp_distance(left, right).is_some_and(|ulp| ulp <= tolerance)
        })
}

/// Sign-aware ULP distance between two same-signed finite f32 values.
/// Returns `None` for NaN on either side.
pub fn ulp_distance(left: f32, right: f32) -> Option<u32> {
    if left.is_nan() || right.is_nan() {
        return None;
    }
    let left = ordered_f32_bits(left);
    let right = ordered_f32_bits(right);
    Some(left.abs_diff(right))
}

fn ordered_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}
