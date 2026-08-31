//! Generated live CUDA/reference differential matrix for f32 IR semantics.

#![cfg(feature = "device-tests")]

mod harness;

use harness::{
    assert_f32_matrix_sweep, assert_u32_matrix_sweep, eq_word, f32_bytes, ge_word,
    generated_lane_program, gt_word, guarded_generated_store, le_word, live_backend, lt_word,
    ne_word, GeneratedMatrixCase, GENERATED_LANE_COUNT as LANE_COUNT,
};
use vyre_foundation::ir::{DataType, Expr, Program};

const MAX_ARITH_ULP: u32 = 1;

const F32_ARITH_BITS: &[u32] = &[
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x3f80_0000, // +1
    0xbf80_0000, // -1
    0x4000_0000, // +2
    0xc000_0000, // -2
    0x3f00_0000, // +0.5
    0xbf00_0000, // -0.5
    0x0080_0000, // smallest positive normal
    0x8080_0000, // largest-magnitude negative just-normal boundary
    0x7f7f_ffff, // max finite
    0xff7f_ffff, // min finite
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // canonical quiet NaN
    0xffc0_0000, // negative quiet NaN
    0x7fa0_0001, // payload NaN
    0x3eaa_aaab, // 1/3 rounded
    0xbeaa_aaab, // -1/3 rounded
    0x4120_0000, // 10
    0xc120_0000, // -10
    0x447a_0000, // 1000
    0xc47a_0000, // -1000
];

const F32_CLASSIFY_BITS: &[u32] = &[
    0x0000_0000,
    0x8000_0000,
    0x0000_0001, // positive subnormal
    0x8000_0001, // negative subnormal
    0x007f_ffff, // largest positive subnormal
    0x807f_ffff, // largest negative subnormal
    0x0080_0000,
    0x8080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0000,
    0x7fa0_0001,
    0x7fff_ffff,
    0xffff_ffff,
];

/// A binary f32 case. `output` is F32 for arithmetic and U32 for the comparison
/// cases, whose predicate words are the whole point of the sweep.
#[derive(Clone)]
struct F32BinaryCase {
    name: &'static str,
    rhs: F32RhsKind,
    output: DataType,
    build: fn(Expr, Expr) -> Expr,
}

/// A unary f32 case. `output` is F32 for arithmetic and U32 for classification.
#[derive(Clone)]
struct F32UnaryCase {
    name: &'static str,
    inputs: F32InputKind,
    output: DataType,
    build: fn(Expr) -> Expr,
}

#[derive(Clone, Copy)]
enum F32RhsKind {
    Mixed,
    NonZero,
}

#[derive(Clone, Copy)]
enum F32InputKind {
    Mixed,
    SqrtDomain,
    Classification,
}

fn isnan_word(value: Expr) -> Expr {
    Expr::select(Expr::is_nan(value), Expr::u32(1), Expr::u32(0))
}

fn isinf_word(value: Expr) -> Expr {
    Expr::select(Expr::is_inf(value), Expr::u32(1), Expr::u32(0))
}

fn isfinite_word(value: Expr) -> Expr {
    Expr::select(Expr::is_finite(value), Expr::u32(1), Expr::u32(0))
}

const F32_BINARY_CASES: &[F32BinaryCase] = &[
    F32BinaryCase {
        name: "f32_add",
        rhs: F32RhsKind::Mixed,
        output: DataType::F32,
        build: Expr::add,
    },
    F32BinaryCase {
        name: "f32_sub",
        rhs: F32RhsKind::Mixed,
        output: DataType::F32,
        build: Expr::sub,
    },
    F32BinaryCase {
        name: "f32_mul",
        rhs: F32RhsKind::Mixed,
        output: DataType::F32,
        build: Expr::mul,
    },
    F32BinaryCase {
        name: "f32_div_nonzero",
        rhs: F32RhsKind::NonZero,
        output: DataType::F32,
        build: Expr::div,
    },
    F32BinaryCase {
        name: "f32_min",
        rhs: F32RhsKind::Mixed,
        output: DataType::F32,
        build: Expr::min,
    },
    F32BinaryCase {
        name: "f32_max",
        rhs: F32RhsKind::Mixed,
        output: DataType::F32,
        build: Expr::max,
    },
];

const F32_COMPARE_CASES: &[F32BinaryCase] = &[
    F32BinaryCase {
        name: "f32_eq",
        rhs: F32RhsKind::Mixed,
        output: DataType::U32,
        build: eq_word,
    },
    F32BinaryCase {
        name: "f32_ne",
        rhs: F32RhsKind::Mixed,
        output: DataType::U32,
        build: ne_word,
    },
    F32BinaryCase {
        name: "f32_lt",
        rhs: F32RhsKind::Mixed,
        output: DataType::U32,
        build: lt_word,
    },
    F32BinaryCase {
        name: "f32_le",
        rhs: F32RhsKind::Mixed,
        output: DataType::U32,
        build: le_word,
    },
    F32BinaryCase {
        name: "f32_gt",
        rhs: F32RhsKind::Mixed,
        output: DataType::U32,
        build: gt_word,
    },
    F32BinaryCase {
        name: "f32_ge",
        rhs: F32RhsKind::Mixed,
        output: DataType::U32,
        build: ge_word,
    },
];

const F32_UNARY_CASES: &[F32UnaryCase] = &[
    F32UnaryCase {
        name: "f32_negate",
        inputs: F32InputKind::Mixed,
        output: DataType::F32,
        build: Expr::negate,
    },
    F32UnaryCase {
        name: "f32_abs",
        inputs: F32InputKind::Mixed,
        output: DataType::F32,
        build: Expr::abs,
    },
    F32UnaryCase {
        name: "f32_sqrt",
        inputs: F32InputKind::SqrtDomain,
        output: DataType::F32,
        build: Expr::sqrt,
    },
    F32UnaryCase {
        name: "f32_reciprocal",
        inputs: F32InputKind::Mixed,
        output: DataType::F32,
        build: Expr::reciprocal,
    },
    F32UnaryCase {
        name: "f32_floor",
        inputs: F32InputKind::Mixed,
        output: DataType::F32,
        build: Expr::floor,
    },
    F32UnaryCase {
        name: "f32_ceil",
        inputs: F32InputKind::Mixed,
        output: DataType::F32,
        build: Expr::ceil,
    },
    F32UnaryCase {
        name: "f32_trunc",
        inputs: F32InputKind::Mixed,
        output: DataType::F32,
        build: Expr::trunc,
    },
];

const F32_CLASSIFY_CASES: &[F32UnaryCase] = &[
    F32UnaryCase {
        name: "f32_is_nan",
        inputs: F32InputKind::Classification,
        output: DataType::U32,
        build: isnan_word,
    },
    F32UnaryCase {
        name: "f32_is_inf",
        inputs: F32InputKind::Classification,
        output: DataType::U32,
        build: isinf_word,
    },
    F32UnaryCase {
        name: "f32_is_finite",
        inputs: F32InputKind::Classification,
        output: DataType::U32,
        build: isfinite_word,
    },
];

#[test]
fn generated_f32_binary_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let lhs = generated_f32_values(F32InputKind::Mixed, 0x1357_9bdf);

    assert_f32_matrix_sweep(
        &backend,
        "f32 binary",
        "every adversarial lane active",
        MAX_ARITH_ULP,
        F32_BINARY_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: f32_binary_program(case),
            inputs: vec![
                f32_bytes(&lhs),
                f32_bytes(&generated_f32_rhs(case.rhs, 0xf00d_cafe)),
            ],
        }),
    );
}

#[test]
fn generated_f32_unary_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();

    assert_f32_matrix_sweep(
        &backend,
        "f32 unary",
        "every adversarial lane active",
        MAX_ARITH_ULP,
        F32_UNARY_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: f32_unary_program(case),
            inputs: vec![f32_bytes(&generated_f32_values(case.inputs, 0x2468_ace0))],
        }),
    );
}

#[test]
fn generated_f32_classification_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();

    assert_u32_matrix_sweep(
        &backend,
        "f32 classification",
        "every adversarial lane active",
        F32_CLASSIFY_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: f32_unary_program(case),
            inputs: vec![f32_bytes(&generated_f32_values(case.inputs, 0x2468_ace0))],
        }),
    );
}

#[test]
fn generated_f32_comparison_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let lhs = generated_f32_values(F32InputKind::Mixed, 0x55aa_1234);
    let rhs = generated_f32_rhs(F32RhsKind::Mixed, 0xaa55_4321);

    assert_u32_matrix_sweep(
        &backend,
        "f32 comparison",
        "NaN/Inf edge lanes active",
        F32_COMPARE_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: f32_binary_program(case),
            inputs: vec![f32_bytes(&lhs), f32_bytes(&rhs)],
        }),
    );
}

/// `out[idx] = build(lhs[idx], rhs[idx])` over two f32 buffers.
fn f32_binary_program(case: &F32BinaryCase) -> Program {
    let idx = Expr::var("idx");
    let value = (case.build)(
        Expr::load("lhs", idx.clone()),
        Expr::load("rhs", idx.clone()),
    );
    generated_lane_program(
        &[("lhs", DataType::F32), ("rhs", DataType::F32)],
        case.output.clone(),
        guarded_generated_store(value),
    )
}

/// `out[idx] = build(input[idx])` over one f32 buffer.
fn f32_unary_program(case: &F32UnaryCase) -> Program {
    let value = (case.build)(Expr::load("input", Expr::var("idx")));
    generated_lane_program(
        &[("input", DataType::F32)],
        case.output.clone(),
        guarded_generated_store(value),
    )
}

fn generated_f32_values(kind: F32InputKind, salt: u32) -> Vec<f32> {
    (0..LANE_COUNT)
        .map(|lane| match kind {
            // Exact class boundaries: perturbing the mantissa would move a lane
            // off the very class it was chosen to name.
            F32InputKind::Classification => {
                f32::from_bits(F32_CLASSIFY_BITS[lane % F32_CLASSIFY_BITS.len()])
            }
            F32InputKind::Mixed => perturbed_arith_value(lane, salt, 0xffff_ffff),
            F32InputKind::SqrtDomain => perturbed_arith_value(lane, salt, 0x7fff_ffff),
        })
        .collect()
}

/// One arithmetic-corpus lane: the seed under `bit_mask`, mantissa-perturbed on
/// every lane that is not an exact seed lane.
fn perturbed_arith_value(lane: usize, salt: u32, bit_mask: u32) -> f32 {
    let bits = F32_ARITH_BITS[lane % F32_ARITH_BITS.len()] & bit_mask;
    let lane_word = lane as u32;
    let mixed = lane_word
        .wrapping_mul(0x045d_9f3b)
        .rotate_left((lane_word & 15) + 1)
        ^ salt.rotate_right(lane_word & 31);
    f32::from_bits(if lane % 5 == 0 {
        bits
    } else {
        bits ^ (mixed & 0x007f_ffff)
    })
}

fn generated_f32_rhs(kind: F32RhsKind, salt: u32) -> Vec<f32> {
    generated_f32_values(F32InputKind::Mixed, salt)
        .into_iter()
        .enumerate()
        .map(|(lane, value)| match kind {
            F32RhsKind::Mixed => value,
            F32RhsKind::NonZero if value == 0.0 || lane % 17 == 0 => {
                f32::from_bits(0x3f80_0000 ^ ((lane as u32) << 12))
            }
            F32RhsKind::NonZero => value,
        })
        .collect()
}
