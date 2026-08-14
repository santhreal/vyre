//! Operand tables and the expectation generator for the u32 binop parity gates.
//!
//! # Why this has one owner
//!
//! Every backend that lowers `mulhi`, `abs_diff`, the saturating forms, the
//! rotates, and the total div/mod/shift contract owes the same gate: dispatch
//! the op on real silicon over the operands that sit on its overflow and
//! masking boundaries, and compare byte-for-byte against the CPU reference.
//! The operand table and the comparison are the same question on every
//! backend, so a copy per backend means the wgpu suite and the CUDA suite can
//! disagree about which operands the boundary even is. Adding an operand here
//! adds it to every backend at once.
//!
//! # What this crate must NOT own
//!
//! The CPU reference arm stays in each backend's suite, and so does the
//! dispatch. A shared reference would make one arm answer for both, and the
//! whole point of a per-backend gate is that naga's multi-step `select`
//! synthesis and the PTX native instructions are different implementations of
//! the same contract: each needs its own live proof against the reference, and
//! neither may ever be compared against the other in place of it.

use vyre_foundation::ir::Expr;

/// Operands spanning the overflow and identity boundaries every u32 binop cares
/// about: zero, one, the max, the sign bit, a 16-bit boundary, and mid-range.
const EXTREMES: &[(u32, u32)] = &[
    (0, 0),
    (1, 1),
    (u32::MAX, 1),
    (1, u32::MAX),
    (u32::MAX, u32::MAX),
    (0x8000_0000, 0x8000_0000),
    (0x1_0000, 0x1_0000),
    (100, 50),
    (50, 100),
    (0x7FFF_FFFF, 2),
];

/// Multiplicative-overflow operands, on top of [`EXTREMES`], for the saturating
/// product: `2^16 * 2^16` and `2^15 * 2^17` both cross `u32::MAX` exactly.
const MULTIPLICATIVE_OVERFLOW: &[(u32, u32)] = &[
    (0x1_0000, 0x1_0000),
    (0x8000, 0x2_0000),
    (1000, 1000),
    (3, 4),
];

/// (value, amount) pairs spanning the rotate-mask boundary: 0, mid, 31, 32, 33.
const ROTATE_AMOUNTS: &[(u32, u32)] = &[
    (1, 0),
    (1, 1),
    (1, 31),
    (1, 32),
    (1, 33),
    (0x8000_0000, 1),
    (0xDEAD_BEEF, 4),
    (0xDEAD_BEEF, 8),
    (0xDEAD_BEEF, 16),
    (0xFFFF_FFFF, 17),
];

/// Divisors including the zero-divisor sentinels and normal control values.
const DIVISORS: &[(u32, u32)] = &[
    (10, 0),
    (0, 0),
    (u32::MAX, 0),
    (123, 0),
    (100, 7),
    (0, 5),
    (u32::MAX, 1),
    (4096, 4096),
];

/// (value, shift amount); the amounts at or above 32 exercise the `& 31` mask a
/// non-masking lowering gets wrong, e.g. `1 << 32` becoming 0 instead of 1.
const SHIFT_AMOUNTS: &[(u32, u32)] = &[
    (1, 0),
    (1, 31),
    (1, 32),
    (1, 33),
    (0xFF, 36),
    (0xFFFF_FFFF, 32),
    (0x1, 63),
    (0xDEAD_BEEF, 4),
];

/// One u32 binop under parity test: what to build, and over which operands.
#[derive(Debug, Clone, Copy)]
pub struct U32BinopCase {
    /// Op name, as it appears in a failure message.
    pub op: &'static str,
    /// IR builder for the op, applied to the two loaded lanes.
    pub build: fn(Expr, Expr) -> Expr,
    base: &'static [(u32, u32)],
    extra: &'static [(u32, u32)],
}

impl U32BinopCase {
    /// Operands for this case: the shared boundary table plus whatever extra
    /// operands only this op has a boundary at.
    #[must_use]
    pub fn pairs(&self) -> Vec<(u32, u32)> {
        self.base.iter().chain(self.extra).copied().collect()
    }
}

/// The ops a backend synthesizes from a multi-step expression rather than from
/// one native instruction.
///
/// naga has no instruction for any of these and builds each from arithmetic
/// plus `select`; PTX has `mul.hi.u32`, `vabsdiff`, the `*.sat` forms and
/// funnel-shift `shf` for most of them. Two unrelated lowerings of one
/// contract, which is why the table is shared and the proof is not.
pub const SYNTHETIC_U32_BINOPS: &[U32BinopCase] = &[
    U32BinopCase {
        op: "mulhi",
        build: Expr::mulhi,
        base: EXTREMES,
        extra: &[],
    },
    U32BinopCase {
        op: "abs_diff",
        build: Expr::abs_diff,
        base: EXTREMES,
        extra: &[],
    },
    U32BinopCase {
        op: "saturating_add",
        build: Expr::saturating_add,
        base: EXTREMES,
        extra: &[],
    },
    U32BinopCase {
        op: "saturating_sub",
        build: Expr::saturating_sub,
        base: EXTREMES,
        extra: &[],
    },
    U32BinopCase {
        op: "saturating_mul",
        build: Expr::saturating_mul,
        base: EXTREMES,
        extra: MULTIPLICATIVE_OVERFLOW,
    },
    U32BinopCase {
        op: "rotate_left",
        build: Expr::rotate_left,
        base: ROTATE_AMOUNTS,
        extra: &[],
    },
    U32BinopCase {
        op: "rotate_right",
        build: Expr::rotate_right,
        base: ROTATE_AMOUNTS,
        extra: &[],
    },
];

/// One op whose hardware behavior is undefined but whose oracle contract is
/// total, with the oracle's answer pinned as data.
///
/// `oracle` is what the total contract requires for `pairs`, written out rather
/// than computed, so a reference arm that drifts has something to disagree
/// with. Each backend still derives the same values from its own reference
/// closure and compares against this pin: that comparison is the arm that
/// catches a drifting reference, and it stays per backend.
#[derive(Debug, Clone, Copy)]
pub struct TotalU32Case {
    /// Op name, as it appears in a failure message.
    pub op: &'static str,
    /// IR builder for the op, applied to the two loaded lanes.
    pub build: fn(Expr, Expr) -> Expr,
    /// Operands, including the cases hardware leaves undefined.
    pub pairs: &'static [(u32, u32)],
    /// The total contract's answer for `pairs`, one value per pair.
    pub oracle: &'static [u32],
}

/// The unsigned ops the oracle defines totally where hardware does not.
///
/// Signed `i32 / 0` and `i32::MIN / -1` are rejected upstream as undefined, so
/// they are not emittable and have no row here.
pub const TOTAL_U32_CASES: &[TotalU32Case] = &[
    TotalU32Case {
        op: "div",
        build: Expr::div,
        pairs: DIVISORS,
        // `x / 0 == u32::MAX`, never `x`, which is what a bare naga Divide gives.
        oracle: &[u32::MAX, u32::MAX, u32::MAX, u32::MAX, 14, 0, u32::MAX, 1],
    },
    TotalU32Case {
        op: "rem",
        build: Expr::rem,
        pairs: DIVISORS,
        // `x % 0 == 0`.
        oracle: &[0, 0, 0, 0, 2, 0, 0, 0],
    },
    TotalU32Case {
        op: "shl",
        build: Expr::shl,
        pairs: SHIFT_AMOUNTS,
        // `1 << 32 == 1` (NOT 0), `0xFFFFFFFF << 32 == 0xFFFFFFFF`, `1 << 63 == 2^31`.
        oracle: &[
            1,
            0x8000_0000,
            1,
            2,
            0xFF0,
            0xFFFF_FFFF,
            0x8000_0000,
            0xEADB_EEF0,
        ],
    },
    TotalU32Case {
        op: "shr",
        build: Expr::shr,
        pairs: SHIFT_AMOUNTS,
        // `1 >> 32 == 1` (32 & 31 == 0), `0xFF >> 36 == 0xFF >> 4 == 0xF`.
        oracle: &[1, 0, 1, 0, 0xF, 0xFFFF_FFFF, 0, 0x0DEA_DBEE],
    },
];

/// The synthetic case named `op`.
///
/// # Panics
/// Panics naming every available op when `op` is not in the table, so a
/// renamed op fails at the lookup instead of silently testing nothing.
#[must_use]
pub fn synthetic_u32_case(op: &str) -> &'static U32BinopCase {
    SYNTHETIC_U32_BINOPS
        .iter()
        .find(|case| case.op == op)
        .unwrap_or_else(|| {
            let available: Vec<&str> = SYNTHETIC_U32_BINOPS.iter().map(|c| c.op).collect();
            panic!(
                "no synthetic u32 binop case named {op:?}. Fix: use one of {available:?} or add \
                 the case to SYNTHETIC_U32_BINOPS."
            )
        })
}

/// The total-contract case named `op`.
///
/// # Panics
/// Panics naming every available op when `op` is not in the table.
#[must_use]
pub fn total_u32_case(op: &str) -> &'static TotalU32Case {
    TOTAL_U32_CASES
        .iter()
        .find(|case| case.op == op)
        .unwrap_or_else(|| {
            let available: Vec<&str> = TOTAL_U32_CASES.iter().map(|c| c.op).collect();
            panic!(
                "no total u32 case named {op:?}. Fix: use one of {available:?} or add the case to \
                 TOTAL_U32_CASES."
            )
        })
}

/// Apply a CPU reference to every operand pair.
#[must_use]
pub fn expected_u32(reference: impl Fn(u32, u32) -> u32, pairs: &[(u32, u32)]) -> Vec<u32> {
    pairs.iter().map(|&(a, b)| reference(a, b)).collect()
}

/// Assert a dispatched result equals what the caller's CPU reference computes.
///
/// `backend` names the arm under test and `lowering` names what that arm
/// lowered the op to, because the useful part of the failure is which of the
/// two unrelated lowerings of this contract broke.
///
/// # Panics
/// Panics with the operands, the reference answer and the dispatched answer
/// when they differ.
pub fn assert_matches_reference(
    backend: &str,
    lowering: &str,
    op: &str,
    pairs: &[(u32, u32)],
    dispatched: &[u32],
    reference: impl Fn(u32, u32) -> u32,
) {
    let expected = expected_u32(reference, pairs);
    assert_eq!(
        dispatched, expected,
        "{backend} `{op}` diverged from the Rust/oracle reference ({lowering} miscompiles on \
         hardware).\n  pairs:    {pairs:?}\n  expected: {expected:?}\n  got:      {dispatched:?}"
    );
}
