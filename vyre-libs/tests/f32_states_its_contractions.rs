//! A registered f32 op states its fused multiply-adds instead of leaving them
//! for a backend to discover.
//!
//! # WHY
//!
//! `a * b + c` written as a multiply and an add grants every backend a choice
//! the reference evaluator does not take: a device may contract the pair into
//! one rounding, the reference always takes two, and the two answers differ.
//! One step of that stays inside the elementary parity window. A chain does
//! not, and a cancelling reduction amplifies it:
//!
//! - five iterations of the Newton-Schulz quintic, written as five multiplies
//!   and two adds per step, put `givens_rotate_pair -> newton_schulz_poly5_f32`
//!   46 ULP from the reference on a live device;
//! - a Gram matrix built from a multiply feeding an add put
//!   `gqa_attention -> tensor_train_decompose` 503 ULP from the reference
//!   evaluated on the device's own intermediate, four times that operation's
//!   budget.
//!
//! Widening a window would accept an unbounded amount of drift. Stating `Fma`
//! removes the choice, because the reference, the naga emitter and the PTX
//! emitter all answer an `Fma` node with a single fused rounding.
//!
//! # What this gate holds
//!
//! The op set comes from the registry at run time, so an op added tomorrow is
//! judged without anyone listing it here. [`UNSTATED_CONTRACTIONS`] is a debt
//! ledger, not an oracle: it records the operations that still leave the choice
//! open, and every entry was read off this gate rather than derived
//! independently. The gate fails in BOTH directions, so the ledger cannot go
//! stale in silence:
//!
//! - an operation that is not listed and leaves a float multiply feeding an add
//!   fails, which is what closes the class against a new operation;
//! - a listed operation whose count changed fails, including when it improved,
//!   so paying the debt down means editing the ledger in the same change.
//!
//! What this does not catch: a chain of adds or multiplies whose *association*
//! a backend reorders, and an approximate transcendental. Those are the
//! transcendental budget's concern, not contraction.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::ir::{BinOp, DataType, Expr, Node, Program};
use vyre_foundation::operation::OperationRegistry;
use vyre_foundation::visit::{walk_exprs, walk_nodes};

/// Names whose value is floating point: float buffers, plus every `let` and
/// `assign` that binds a float expression, to a fixed point so a binding that
/// reads an earlier one is classified too.
struct FloatNames {
    buffers: BTreeSet<String>,
    vars: BTreeSet<String>,
}

impl FloatNames {
    fn of(program: &Program) -> Self {
        let buffers = program
            .buffers()
            .iter()
            .filter(|decl| is_float_type(&decl.element()))
            .map(|decl| decl.name().to_owned())
            .collect();
        let mut names = Self {
            buffers,
            vars: BTreeSet::new(),
        };
        loop {
            let before = names.vars.len();
            walk_nodes(program, |node| {
                let (name, value) = match node {
                    Node::Let { name, value } | Node::Assign { name, value } => (name, value),
                    _ => return,
                };
                if names.is_float(value) {
                    names.vars.insert(name.as_str().to_owned());
                }
            });
            if names.vars.len() == before {
                return names;
            }
        }
    }

    /// Whether an expression produces a floating-point value. Unknown shapes
    /// answer `false`: this gate reports a pair only when it can show the pair
    /// is floating point, so an op it cannot classify is never accused.
    fn is_float(&self, expr: &Expr) -> bool {
        match expr {
            Expr::LitF32(_) => true,
            Expr::Var(name) => self.vars.contains(name.as_str()),
            Expr::Load { buffer, .. } => self.buffers.contains(buffer.as_str()),
            Expr::Cast { target, .. } => is_float_type(target),
            Expr::Fma { .. } => true,
            Expr::BinOp { op, left, right } => {
                !matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                ) && (self.is_float(left) || self.is_float(right))
            }
            Expr::UnOp { operand, .. } => self.is_float(operand),
            Expr::Select {
                true_val,
                false_val,
                ..
            } => self.is_float(true_val) || self.is_float(false_val),
            Expr::SubgroupShuffle { value, .. } | Expr::SubgroupReduce { value, .. } => {
                self.is_float(value)
            }
            Expr::Atomic { buffer, .. } => self.buffers.contains(buffer.as_str()),
            _ => false,
        }
    }

    /// A floating-point addition one of whose operands is a floating-point
    /// multiplication: the shape a device may fuse and the reference may not.
    fn fusable_pair(&self, expr: &Expr) -> bool {
        let Expr::BinOp { op, left, right } = expr else {
            return false;
        };
        if !matches!(op, BinOp::Add | BinOp::Sub) {
            return false;
        }
        [left.as_ref(), right.as_ref()]
            .into_iter()
            .any(|side| matches!(side, Expr::BinOp { op: BinOp::Mul, .. }) && self.is_float(side))
    }
}

fn is_float_type(ty: &DataType) -> bool {
    matches!(
        ty,
        DataType::F32 | DataType::F16 | DataType::BF16 | DataType::F64
    )
}

#[test]
fn every_registered_float_op_states_its_fused_multiply_adds() {
    let catalog = vyre_libs::operation_catalog::all_entries().count();
    assert!(
        catalog > 0,
        "Fix: the library catalog is empty, so no library registration reached the registry."
    );
    let mut offenders: BTreeMap<String, usize> = BTreeMap::new();
    let mut judged = 0usize;
    for entry in OperationRegistry::global().iter() {
        let Some(program) = entry.program() else {
            continue;
        };
        let names = FloatNames::of(&program);
        if names.buffers.is_empty() {
            continue;
        }
        judged += 1;
        let mut count = 0usize;
        walk_exprs(&program, |expr| {
            if names.fusable_pair(expr) {
                count += 1;
            }
        });
        if count > 0 {
            offenders.insert(entry.id.to_owned(), count);
        }
    }
    assert!(
        judged > 0,
        "Fix: the registry produced no floating-point program, so this gate judged nothing."
    );

    let pinned: BTreeMap<String, usize> = UNSTATED_CONTRACTIONS
        .iter()
        .map(|(id, count)| ((*id).to_owned(), *count))
        .collect();
    assert_eq!(
        pinned.len(),
        UNSTATED_CONTRACTIONS.len(),
        "Fix: UNSTATED_CONTRACTIONS lists an operation twice; one row per operation."
    );

    let added: Vec<_> = offenders
        .iter()
        .filter(|(id, _)| !pinned.contains_key(*id))
        .collect();
    assert!(
        added.is_empty(),
        "Fix: {} registered floating-point op(s) leave a multiply feeding an add for the backend \
         to contract, so the device and the reference round a different number of times. State \
         the pair as `Expr::fma(a, b, c)`, which every arm answers with one rounding: {added:?}",
        added.len()
    );

    let resolved: Vec<_> = pinned
        .keys()
        .filter(|id| !offenders.contains_key(*id))
        .collect();
    assert!(
        resolved.is_empty(),
        "Fix: {} operation(s) no longer leave an unstated contraction. Delete their rows from `UNSTATED_CONTRACTIONS` so the ledger keeps naming the real debt: {resolved:?}",
        resolved.len()
    );

    let moved: Vec<_> = offenders
        .iter()
        .filter(|(id, count)| pinned.get(*id).is_some_and(|pin| pin != *count))
        .map(|(id, count)| (id, pinned[id], *count))
        .collect();
    assert!(
        moved.is_empty(),
        "Fix: the unstated-contraction count moved for {} operation(s), listed as (op, pinned, measured). Update `UNSTATED_CONTRACTIONS` in the same change that moved it: {moved:?}",
        moved.len()
    );
}

/// Operations that still leave a float multiply feeding an add, and how many
/// sites each leaves. Shrink-only: see the module comment.
const UNSTATED_CONTRACTIONS: &[(&str, usize)] = &[
    ("vyre-libs::math::fft::fft_convolve_circular_complex", 32),
    ("vyre-libs::math::fft::fft_radix2", 8),
    (
        "vyre-libs::math::fft::pointwise_complex_multiply_conjugate",
        8,
    ),
    ("vyre-libs::math::givens_rotate_pair", 2),
    ("vyre-libs::math::jacobi_apply_rotation", 8),
    (
        "vyre-libs::math::quantized::i4x8_batched_matmul_f32_scaled",
        1,
    ),
    (
        "vyre-libs::math::quantized::i4x8_batched_matmul_top1_f32_scaled",
        1,
    ),
    (
        "vyre-libs::math::quantized::i4x8_batched_matvec_f32_scaled",
        1,
    ),
    ("vyre-libs::math::quantized::i4x8_dot_f32_scaled", 1),
    ("vyre-libs::math::quantized::i4x8_matvec_f32_scaled", 1),
    ("vyre-libs::math::reduce_variance", 17),
    ("vyre-libs::math::symmetric_eigen_jacobi", 8),
    ("vyre-libs::math::tensor_train_decompose", 8),
    ("vyre-libs::nn::gelu", 1),
    ("vyre-libs::nn::layer_norm", 2),
    ("vyre-libs::nn::linear_relu", 1),
    ("vyre-libs::nn::linear_silu", 1),
    ("vyre-libs::nn::mlp_4x_leaky_sq", 2),
    ("vyre-libs::nn::mlp_4x_leaky_sq::hidden_projection", 1),
    ("vyre-libs::nn::mlp_4x_leaky_sq::output_projection", 1),
    ("vyre-libs::nn::mlp_backward", 3),
    ("vyre-libs::nn::partial_rope_backward", 3),
    ("vyre-libs::nn::rms_norm", 1),
    ("vyre-libs::nn::rms_norm_linear", 2),
    ("vyre-libs::nn::skip_gate", 2),
    ("vyre-libs::optim::adamw_step", 10),
    ("vyre-libs::optim::ema_apply", 2),
    ("vyre-libs::optim::muon_update", 8),
    ("vyre-libs::optim::muoneq_r", 8),
    ("vyre-libs::quant::int4_batched_matmul_f32_scaled", 1),
    ("vyre-libs::quant::int4_batched_matmul_top1_f32_scaled", 1),
    ("vyre-libs::quant::int4_batched_matvec_f32_scaled", 1),
    ("vyre-libs::quant::int4_dot_f32_scaled", 1),
    ("vyre-libs::quant::int4_matvec_f32_scaled", 1),
    ("vyre-libs::quant::int6_unpack", 1),
];
