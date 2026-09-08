//! Shared f32 backend-parity contract for Cat-A and conform gates.
//!
//! Integer and boolean outputs remain byte-identical. F32 outputs use a
//! bounded ULP window because GPU backends may contract multiply-add
//! sequences and may use native approximate transcendental instructions.

use core::ops::ControlFlow;
use std::collections::BTreeSet;

use crate::ir::{DataType, Expr, Program, UnOp};
use crate::operation::OperationRegistry;
use crate::visit::{for_each_expr, try_for_each_expr};

/// Maximum accepted reference-oracle error against correctly-rounded f32
/// transcendentals.
pub const REFERENCE_TRANSCENDENTAL_ULP_BUDGET: u32 = 4;

/// Maximum accepted backend-vs-reference error for programs containing f32
/// transcendentals.
pub const BACKEND_TRANSCENDENTAL_ULP_BUDGET: u32 = 128;

/// Maximum accepted backend-vs-reference error for elementary f32 programs.
///
/// This is the contraction contract: a backend is allowed to fuse `a*b+c` into
/// one FMA while the reference evaluates it as two operations. The budget is
/// program-level, not an op-id whitelist.
pub const BACKEND_ELEMENTARY_F32_ULP_BUDGET: u32 = 4;

/// Rounding a target is permitted to apply to a chain of f32 arithmetic.
///
/// The mode is the neutral half of the contraction contract stated above: the
/// budget says how far a contracted backend may drift, and this says whether
/// contraction is permitted at all. It names no instruction and no shader
/// dialect, so every emitter reads the same policy and expresses it in its own
/// text.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum FloatLoweringMode {
    /// A target may fold `a * b + c` into a single rounding.
    ///
    /// Every shipped backend does, which is why
    /// [`BACKEND_ELEMENTARY_F32_ULP_BUDGET`] is not zero.
    #[default]
    Contracted,
    /// Every f32 operation rounds separately, as IEEE-754 specifies.
    ///
    /// An emitter honors this by forcing each intermediate f32 result to be
    /// materialized before it is consumed, which is what denies a target the
    /// multiply-add pair it would otherwise fuse. Two chains emitted under
    /// different modes are different modules, so anything that caches an
    /// emitted artifact carries the mode in its identity.
    StrictIeee,
}

/// Identity of the emitted-artifact contract this mode participates in.
///
/// A cached artifact recorded before the mode existed carries no mode, and
/// serving it to a strict-mode dispatch would return contracted arithmetic
/// under a strict request. Every cache key that admits an emitted artifact
/// mixes this value in, so a stale entry misses rather than being served.
///
/// Version 2 added the strict expansion in
/// [`fp_expansion`](crate::fp_expansion): a v1 strict-mode artifact has its
/// multiply-adds separated but still calls the target's approximate native
/// transcendental, so serving it under v2 would answer a bit-identity request
/// with backend-dependent bits.
pub const FLOAT_LOWERING_CONTRACT_VERSION: u32 = 2;

impl FloatLoweringMode {
    /// True when the mode denies a target the fused multiply-add.
    #[must_use]
    pub const fn blocks_contraction(self) -> bool {
        matches!(self, Self::StrictIeee)
    }

    /// Every mode, so a consumer sweeps the set instead of naming one member.
    ///
    /// [`Self::roster_index`] is what keeps this list in step with the enum: a
    /// new variant makes that match non-exhaustive, and the index its new arm
    /// returns addresses this array, so a variant added without extending the
    /// array indexes past the end.
    pub const EVERY: &'static [Self] = &[Self::Contracted, Self::StrictIeee];

    /// Position of this mode in [`Self::EVERY`].
    #[must_use]
    pub const fn roster_index(self) -> usize {
        match self {
            Self::Contracted => 0,
            Self::StrictIeee => 1,
        }
    }

    /// Stable label a cache key and a diagnostic both name the mode by.
    #[must_use]
    pub const fn cache_label(self) -> &'static str {
        match self {
            Self::Contracted => "contracted",
            Self::StrictIeee => "strict-ieee",
        }
    }
}

/// Normalize an f32 so two backends that agree numerically agree bitwise.
///
/// Every NaN payload collapses to one quiet NaN and every subnormal flushes to
/// a zero of its own sign. A signed zero is preserved: `-0.0` and `+0.0` are
/// numerically equal and every backend distinguishes their bits, so collapsing
/// them here would hide a real difference. The wire encoder applies this same
/// rule rather than one of its own, because the encoded form is the key an
/// artifact is cached under: an encoder that folded `-0.0` into `+0.0` served
/// one program the other one's device code, while the reference interpreter
/// read the literal as written and disagreed with it.
///
/// This is the one definition. Four identical bodies stood in
/// `scalar_ops`, two reference evaluators and one reference test, and the whole
/// f32 parity story is that they agree: a copy that flushed a subnormal to
/// `+0.0` instead of its own sign would make one evaluator's output differ from
/// another's on a value neither considers exceptional, and the ULP window above
/// would report the difference as drift.
#[must_use]
pub fn canonical_f32(value: f32) -> f32 {
    if value.is_nan() {
        f32::from_bits(0x7FC0_0000)
    } else if value.is_subnormal() {
        f32::from_bits(value.to_bits() & 0x8000_0000)
    } else {
        value
    }
}

/// Return the allowed f32 ULP tolerance for backend-vs-reference parity checks.
///
/// Every caller compares a backend against the reference oracle, so the window
/// can never be zero: contraction is a backend right, stated at the top of this
/// module, and every shipped backend folds `a*b+c` into one FMA. A `strict-fp`
/// feature used to force 0 here. It forbade nothing, because no emitter
/// consulted it; its only effect was to fail every elementary f32 op that
/// contracts, so `cargo test --workspace --all-features`, which the release
/// procedure requires, could not pass. `newton_schulz_poly5_f32` drifted 4 ULP,
/// `newton_schulz_5step` 2 and `ema_apply` 1, with the backends agreeing
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

/// Combine an op-id-specific budget with the program-level f32 policy.
///
/// A registration whose contract states an absolute bound has no ULP reading,
/// so the comparison falls back to the program-level policy rather than
/// treating an unreadable bound as byte identity.
#[must_use]
pub fn effective_tolerance(op_id: &str, program: &Program) -> u32 {
    OperationRegistry::global()
        .get(op_id)
        .and_then(|entry| entry.ulp_budget())
        .unwrap_or(0)
        .max(f32_ulp_tolerance(program))
}

/// The operations in `program` a backend may lower to an approximate native
/// instruction, named by their IR variant, in ascending order without repeats.
///
/// The set is the one `is_transcendental_op` classifies, so a lowering record
/// and the parity window are answers to the same question. An op added to the
/// policy appears here without anyone editing a second list.
#[must_use]
pub fn approximable_operations(program: &Program) -> Vec<String> {
    let mut named = BTreeSet::new();
    for_each_expr(program.entry(), |expr| {
        if let Expr::UnOp { op, .. } = expr {
            if is_transcendental_op(expr) {
                named.insert(format!("{op:?}"));
            }
        }
    });
    named.into_iter().collect()
}

/// True when any expression in `program` reaches an approximable f32 op.
///
/// Two hand-written enumerations used to stand here, one over `Node` and one
/// over `Expr`, each recursing into the positions it happened to name and each
/// ending in a catch-all that read as "nothing here". Between them they decided
/// what the ULP budget applies to, so a `Node` variant that gained a body or an
/// `Expr` variant that gained an operand silently narrowed the tolerance policy:
/// a program whose only transcendental sat in the new position was judged
/// elementary and held to a 4 ULP budget it cannot meet.
///
/// Position enumeration is [`try_for_each_expr`]'s, which reaches every operand
/// of every node and every sub-expression of every operand, so what is left here
/// is the policy itself: which ops are approximable. It also stops at the first
/// hit rather than walking the whole program, which the recursive pair could not
/// do.
fn program_has_transcendental(program: &Program) -> bool {
    try_for_each_expr(program.entry(), |expr| {
        if is_transcendental_op(expr) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_break()
}

/// Whether one unary operator is an f32 op a backend may lower to an
/// approximate native instruction.
///
/// The set is the policy: an op left out of it asserts that backends agree
/// with the reference to [`BACKEND_ELEMENTARY_F32_ULP_BUDGET`] on that op.
/// `UnOp::Reciprocal` is deliberately outside: a division rather than an
/// approximate reciprocal instruction is the usual lowering, so it stays in the
/// elementary window.
///
/// `UnOp` is `#[non_exhaustive]`, so the closure cannot be a compile-time one
/// here. [`recorded_approximability`] answers `None` for an operator no arm
/// names, and the closure test in this module reads that: it walks the frozen
/// builtin tag table and fails on a tag with no recorded decision.
#[must_use]
pub fn is_approximable_unary_op(op: &UnOp) -> bool {
    recorded_approximability(op).unwrap_or(false)
}

/// The recorded approximability decision for one unary operator, or `None`
/// when no arm names it.
///
/// The value of the third answer is that an unrecorded builtin is
/// distinguishable from a recorded elementary one. Two hand-typed lists stood
/// in the closure test instead, restating this match a third time so the count
/// of recorded operators could be compared against the tag table. Three copies
/// of one decision drift, and a list that omitted an operator read as an
/// absent decision rather than as an omission.
fn recorded_approximability(op: &UnOp) -> Option<bool> {
    match op {
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
        | UnOp::Tanh => Some(true),
        UnOp::Negate
        | UnOp::BitNot
        | UnOp::LogicalNot
        | UnOp::Popcount
        | UnOp::Clz
        | UnOp::Ctz
        | UnOp::ReverseBits
        | UnOp::Abs
        | UnOp::Floor
        | UnOp::Ceil
        | UnOp::Round
        | UnOp::Trunc
        | UnOp::Sign
        | UnOp::IsNan
        | UnOp::IsInf
        | UnOp::IsFinite
        | UnOp::Unpack4Low
        | UnOp::Unpack4High
        | UnOp::Unpack8Low
        | UnOp::Unpack8High
        | UnOp::Reciprocal
        | UnOp::BitcastF32ToU32
        | UnOp::BitcastU32ToF32 => Some(false),
        // An extension operator declares its own lowering contract, and this
        // module cannot read it. It is graded by the elementary window until
        // the registration states otherwise through `effective_tolerance`.
        UnOp::Opaque(_) => Some(false),
        // A builtin operator added to `vyre-spec` after this arm was written.
        // The closure test names it as soon as it exists.
        _ => None,
    }
}

/// Every builtin unary operator [`is_approximable_unary_op`] admits, in wire
/// tag order.
///
/// Derived by filtering the frozen builtin tag table through the classifier, so
/// a caller that needs the set never carries a second copy of it and a variant
/// added to the policy appears here with nothing else edited.
#[must_use]
pub fn approximable_unary_ops() -> Vec<UnOp> {
    crate::serial::wire::tags::builtin_un_ops()
        .into_iter()
        .filter(|op| is_approximable_unary_op(op))
        .collect()
}

/// Whether `expr` is itself an f32 op a backend may lower to an approximate
/// native instruction.
///
/// Shallow: sub-expressions are the walk's job.
fn is_transcendental_op(expr: &Expr) -> bool {
    matches!(expr, Expr::UnOp { op, .. } if is_approximable_unary_op(op))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType, Node};

    /// WHY: every consumer that separates cached artifacts by rounding mode
    /// sweeps `FloatLoweringMode::EVERY`. A mode added to the enum and left out
    /// of that roster would be silently exempt from cache separation, so the
    /// roster and the enum are pinned to each other here: the index each
    /// variant reports must address its own slot, and a missing slot indexes
    /// past the end.
    #[test]
    fn every_float_lowering_mode_holds_its_own_roster_slot() {
        for (slot, mode) in FloatLoweringMode::EVERY.iter().enumerate() {
            assert_eq!(
                mode.roster_index(),
                slot,
                "Fix: keep FloatLoweringMode::EVERY in the order roster_index reports."
            );
        }
        let mut labels: Vec<&str> = FloatLoweringMode::EVERY
            .iter()
            .map(|mode| mode.cache_label())
            .collect();
        labels.sort_unstable();
        let count = labels.len();
        labels.dedup();
        assert_eq!(
            labels.len(),
            count,
            "Fix: give every float lowering mode its own cache label."
        );
    }

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

    /// Every builtin unary tag carries a recorded approximability decision.
    ///
    /// `UnOp` is `#[non_exhaustive]`, so `recorded_approximability` cannot
    /// close over the variant space at compile time from this crate: a new
    /// builtin falls into its catch-all. The closure is enforced here instead,
    /// over the frozen wire tag table, so a tag added to `vyre-spec` turns
    /// this red until its decision is recorded.
    ///
    /// The partition assertion is what stops the test passing vacuously. A
    /// classifier that answered one way for every operator would satisfy the
    /// closure and grade every program under one budget.
    ///
    /// What it does not catch: an extension operator, which declares its own
    /// contract through `effective_tolerance` and has no wire tag to enumerate.
    #[test]
    fn every_builtin_unary_tag_has_a_recorded_approximability_decision() {
        let builtins = crate::serial::wire::tags::builtin_un_ops();
        let unrecorded: Vec<&UnOp> = builtins
            .iter()
            .filter(|op| recorded_approximability(op).is_none())
            .collect();
        assert!(
            unrecorded.is_empty(),
            "a builtin unary wire tag has no recorded approximability decision. \
             Fix: give each of {unrecorded:?} an arm in `recorded_approximability`, \
             deciding whether a backend may lower it to an approximate instruction."
        );
        let approximable = approximable_unary_ops();
        assert!(
            !approximable.is_empty() && approximable.len() < builtins.len(),
            "the policy must split the builtin unary operators. \
             approximable={approximable:?} builtins={builtins:?}"
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

/// One output buffer seen from both sides, with the element type the
/// program declares for it.
struct OutputSlot<'a> {
    slot: usize,
    element: DataType,
    left: &'a [u8],
    right: &'a [u8],
}

/// Align two result-buffer lists against the program's declared outputs.
///
/// The single owner of "which bytes belong to which declared output, and
/// are the two sides even comparable". Every parity check walks outputs
/// through this so none of them can disagree about slot identity or about
/// what counts as an unalignable pair. `Err` carries the caller-facing
/// reason already formatted.
fn align_output_slots<'a>(
    program: &Program,
    outputs_a: &'a [Vec<u8>],
    outputs_b: &'a [Vec<u8>],
) -> Result<Vec<OutputSlot<'a>>, String> {
    if outputs_a.len() != outputs_b.len() {
        return Err(format!(
            "output buffer count mismatch: {} vs {}; left={} right={}",
            outputs_a.len(),
            outputs_b.len(),
            summarize_buffers(outputs_a),
            summarize_buffers(outputs_b)
        ));
    }

    let output_indices = program.output_buffer_indices();
    if output_indices.len() != outputs_a.len() {
        return Err(format!(
            "program declares {} output buffer(s), compared {} result buffer(s)",
            output_indices.len(),
            outputs_a.len()
        ));
    }

    let mut slots = Vec::with_capacity(outputs_a.len());
    for (slot, ((bytes_a, bytes_b), buffer_index)) in outputs_a
        .iter()
        .zip(outputs_b.iter())
        .zip(output_indices.iter().copied())
        .enumerate()
    {
        if bytes_a.len() != bytes_b.len() {
            return Err(format!(
                "output buffer {slot} length mismatch: {} vs {}; left={} right={}",
                bytes_a.len(),
                bytes_b.len(),
                summarize_bytes(bytes_a),
                summarize_bytes(bytes_b)
            ));
        }
        slots.push(OutputSlot {
            slot,
            element: program.buffers()[buffer_index as usize].element(),
            left: bytes_a,
            right: bytes_b,
        });
    }
    Ok(slots)
}

fn compare_output_buffers_with_tolerance(
    program: &Program,
    outputs_a: &[Vec<u8>],
    outputs_b: &[Vec<u8>],
    tolerance: u32,
) -> BufferParity {
    let slots = match align_output_slots(program, outputs_a, outputs_b) {
        Ok(slots) => slots,
        Err(reason) => return BufferParity::Mismatch(reason),
    };

    for OutputSlot {
        slot,
        element,
        left,
        right,
    } in slots
    {
        if element == DataType::F32 {
            if !f32_buffer_matches(left, right, tolerance) {
                return BufferParity::Mismatch(format!(
                    "output buffer {slot} (F32) exceeded the {tolerance}-ULP window; left={} right={}",
                    summarize_bytes(left),
                    summarize_bytes(right)
                ));
            }
        } else if left != right {
            return BufferParity::Mismatch(format!(
                "output buffer {slot} ({element:?}) is not byte-identical; left={} right={}",
                summarize_bytes(left),
                summarize_bytes(right)
            ));
        }
    }

    BufferParity::Ok
}

/// Largest ULP distance across every declared F32 output slot.
///
/// Reports the measured divergence instead of judging it against a
/// tolerance, so an audit can rank results. Non-F32 slots are skipped:
/// they are byte-exact or they are not comparable at all, and neither is
/// a ULP figure.
///
/// Returns `None` when the two sides cannot be aligned against the
/// program's outputs, or when an F32 slot is not a whole number of f32
/// values. Returns `Some(u32::MAX)` for a pair that is incomparable
/// rather than merely distant: NaN against a number, or two non-finite
/// values of different class or sign. Same-signed infinities and
/// NaN-against-NaN are treated as agreeing, because a backend is allowed
/// to reach them by a different route.
#[must_use]
pub fn max_output_ulp(
    program: &Program,
    outputs_a: &[Vec<u8>],
    outputs_b: &[Vec<u8>],
) -> Option<u32> {
    let slots = align_output_slots(program, outputs_a, outputs_b).ok()?;
    let mut max_ulp = 0u32;
    for slot in slots {
        if slot.element != DataType::F32 {
            continue;
        }
        if slot.left.len() % 4 != 0 {
            return None;
        }
        for (a, b) in slot.left.chunks_exact(4).zip(slot.right.chunks_exact(4)) {
            let left = f32::from_bits(u32::from_le_bytes([a[0], a[1], a[2], a[3]]));
            let right = f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            match slot_pair_ulp(left, right) {
                Some(0) => {}
                Some(ulp) => max_ulp = max_ulp.max(ulp),
                None => return Some(u32::MAX),
            }
        }
    }
    Some(max_ulp)
}

/// ULP distance for one f32 pair, or `None` when the pair is
/// incomparable. Sole owner of the non-finite classification used by
/// [`max_output_ulp`].
fn slot_pair_ulp(left: f32, right: f32) -> Option<u32> {
    if left.to_bits() == right.to_bits() {
        return Some(0);
    }
    if left.is_nan() && right.is_nan() {
        return Some(0);
    }
    if !left.is_finite() && !right.is_finite() {
        if left.is_infinite()
            && right.is_infinite()
            && left.is_sign_positive() == right.is_sign_positive()
        {
            return Some(0);
        }
        return None;
    }
    if left.is_nan() || right.is_nan() {
        return None;
    }
    ulp_distance(left, right)
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

#[cfg(test)]
mod output_ulp_tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType};

    fn one_f32_output_program() -> Program {
        Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::F32).with_count(2)],
            [1, 1, 1],
            vec![],
        )
    }

    fn f32_bytes(values: [f32; 2]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn identical_outputs_measure_zero_ulp() {
        let program = one_f32_output_program();
        let bytes = vec![f32_bytes([1.0, -2.5])];
        assert_eq!(max_output_ulp(&program, &bytes, &bytes), Some(0));
    }

    #[test]
    fn the_reported_figure_is_the_worst_slot_element_not_the_first() {
        let program = one_f32_output_program();
        let left = vec![f32_bytes([1.0, 100.0])];
        let right = vec![f32_bytes([
            f32::from_bits(1.0f32.to_bits() + 1),
            f32::from_bits(100.0f32.to_bits() + 7),
        ])];
        assert_eq!(max_output_ulp(&program, &left, &right), Some(7));
    }

    #[test]
    fn same_signed_infinities_and_nans_agree_but_a_nan_against_a_number_does_not() {
        let program = one_f32_output_program();
        let agreeing_left = vec![f32_bytes([f32::INFINITY, f32::NAN])];
        let agreeing_right = vec![f32_bytes([f32::INFINITY, -f32::NAN])];
        assert_eq!(
            max_output_ulp(&program, &agreeing_left, &agreeing_right),
            Some(0)
        );

        let crossed_left = vec![f32_bytes([f32::NAN, 0.0])];
        let crossed_right = vec![f32_bytes([1.0, 0.0])];
        assert_eq!(
            max_output_ulp(&program, &crossed_left, &crossed_right),
            Some(u32::MAX)
        );

        let opposed_left = vec![f32_bytes([f32::INFINITY, 0.0])];
        let opposed_right = vec![f32_bytes([f32::NEG_INFINITY, 0.0])];
        assert_eq!(
            max_output_ulp(&program, &opposed_left, &opposed_right),
            Some(u32::MAX)
        );
    }

    #[test]
    fn unalignable_output_lists_report_no_measurement() {
        let program = one_f32_output_program();
        let one = vec![f32_bytes([1.0, 1.0])];
        assert_eq!(max_output_ulp(&program, &one, &[]), None);
        assert_eq!(
            max_output_ulp(&program, &one, &[vec![0u8; 4]]),
            None,
            "a length-mismatched slot is not a distance"
        );
        assert_eq!(
            max_output_ulp(&program, &[vec![0u8; 6]], &[vec![0u8; 6]]),
            None,
            "an F32 slot that is not a whole number of f32 values is not a distance"
        );
    }

    #[test]
    fn a_non_f32_output_slot_contributes_no_distance() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(2)],
            [1, 1, 1],
            vec![],
        );
        let left = vec![vec![1u8, 0, 0, 0, 2, 0, 0, 0]];
        let right = vec![vec![9u8, 0, 0, 0, 9, 0, 0, 0]];
        assert_eq!(max_output_ulp(&program, &left, &right), Some(0));
    }
}
